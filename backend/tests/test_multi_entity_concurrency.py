"""Multi-entity chat turns must gather their per-entity legs, and bound them (#260).

``_prepare_multi_entity_turn`` used to ``await`` each entity's retrieval leg
inside a ``for`` loop, so an N-entity comparison ran N full pipelines back to
back (up to 2N once the auto-facet fallback retries). The legs are independent,
so they now run under a bounded gather.

The equivalence oracle throughout is the SAME function run with
``CHAT_MULTI_ENTITY_CONCURRENCY = 1``, which admits one leg at a time and is
therefore the old sequential loop. It is preferred over a copied-out reference
implementation because it exercises the real prompt assembly, not a copy of it
that can drift.

What is pinned here, all through the real ``_prepare_multi_entity_turn`` /
``_prepare_turn`` with retrieval stubbed:

- the legs actually OVERLAP -- a barrier every leg must reach, so a sequential
  implementation deadlocks and fails rather than merely being slower;
- the gathered result is IDENTICAL to the sequential one: same articles, same
  order, same per-article "Entities:" annotation, same prompt, same retrieval
  arguments, for both comparison and intersection;
- the fan-out is BOUNDED: the semaphore caps legs in flight, and a question
  naming more entities than ``CHAT_MAX_MULTI_ENTITIES`` never reaches the
  multi-entity path at all.
"""

import asyncio
import time

import pytest

from app import chat as chat_module
from app import main as main_module
from app.main import SourceArticle
from app.query_intent import MultiEntityQuery

# Long enough that source_context() truncates it, so the compared prompt is
# sensitive to the body budget as well as to article identity and order.
BODY = "b" * 400


def _run(coro):
    return asyncio.run(coro)


class _Leg:
    """Stubbed retrieval pipeline, with the knobs the tests turn.

    One instance stands in for every entity leg, so it can count how many are
    in flight at once and record exactly what each was asked for.
    """

    def __init__(
        self,
        *,
        entity_articles: dict[str, list[SourceArticle]],
        delay: float = 0.0,
        facts: dict[str, str] | None = None,
    ):
        self.articles = entity_articles
        self.delay = delay
        self.facts = facts or {}
        self.started: list[str] = []
        self.finished: list[str] = []
        self.calls: list[tuple] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._barrier: tuple[asyncio.Event, object] | None = None
        self._slow_entities: frozenset[str] = frozenset()

    def arm_barrier(self, parties: int) -> None:
        """No leg may leave until ``parties`` legs have arrived.

        Under a sequential loop the first leg waits for arrivals that can only
        happen after it returns, so the barrier is never satisfied and the wait
        times out: the test fails on the absence of overlap, not on a wall-clock
        threshold that a slow machine could trip.
        """
        arrived = asyncio.Event()
        seen = 0

        def _mark() -> None:
            nonlocal seen
            seen += 1
            if seen >= parties:
                arrived.set()

        self._barrier = (arrived, _mark)

    def slow_for(self, *entities: str) -> None:
        """These entities sleep before returning; everything else does not.

        Combined with the default ``delay=0`` (a bare checkpoint), this is what
        gives the fast legs something to overlap with.
        """
        self._slow_entities = frozenset(entities)

    async def retrieve(self, rq, top_k, **kwargs):
        entity = self._entity_of(rq)
        self.calls.append((entity, top_k, kwargs))
        self.started.append(entity)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._barrier is not None:
                arrived, mark = self._barrier
                mark()
                await asyncio.wait_for(arrived.wait(), timeout=2.0)
            # Always yield once, so two legs with no delay can still overlap.
            await asyncio.sleep(self.delay if entity in self._slow_entities else 0)
            # Copy, don't alias: body_rescue rewrites a.score in place, so a
            # shared fixture object would carry one test's rescue into the next.
            return (
                [a.model_copy(deep=True) for a in self.articles.get(entity, [])],
                self.facts.get(f"{entity}.industry"),
                None,
                None,
            )
        finally:
            self.in_flight -= 1
            self.finished.append(entity)

    async def rescue(self, query, articles):
        # A rescued article gets a distinct score, so a test can tell a rescue
        # that ran from one that did not.
        for a in articles:
            a.score = 0.99
        return articles

    def _entity_of(self, rq: str) -> str:
        return rq.split(" ")[0]


def _article(aid: int, title: str, score: float) -> SourceArticle:
    return SourceArticle(
        id=aid,
        title=title,
        url=f"u{aid}",
        published_date="2025-01-01",
        summary=f"summary {aid}",
        body=f"{title} {BODY}",
        score=score,
    )


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Install a _Leg-backed pipeline; the factory takes this test's knobs."""
    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(main_module, "_effective_intent", lambda q, f, t: (q, None, None, None, None))
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)

    def install(**kwargs) -> _Leg:
        leg = _Leg(**kwargs)
        monkeypatch.setattr(main_module, "retrieve_with_auto_facet_fallback", leg.retrieve)
        monkeypatch.setattr(main_module, "body_rescue", leg.rescue)
        return leg

    return install


def _multi(entities: list[str], mode: str = "comparison", scaffold: str = "funding") -> MultiEntityQuery:
    return MultiEntityQuery(mode=mode, entities=entities, scaffold=scaffold)


# The article set every equivalence test shares: a shared id (3) so the dedupe
# and the per-article entity list are exercised, and an id (1) matched by two
# entities with different scores so the rank key has something to order.
_ARTICLES = {
    "alpha": [_article(1, "alpha one", 0.9), _article(3, "shared", 0.7)],
    "bravo": [_article(2, "bravo two", 0.9), _article(3, "shared", 0.7)],
    "charlie": [_article(1, "alpha one", 0.8), _article(4, "charlie four", 0.85)],
}


# --- overlap ---


def test_legs_run_concurrently(stub_pipeline, monkeypatch):
    """Every leg must be in flight at the same time.

    A barrier, not a stopwatch: no leg may leave until all have arrived, so a
    sequential implementation deadlocks and the wait times out.
    """
    entities = ["alpha", "bravo", "charlie"]
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )
    leg.arm_barrier(len(entities))

    turn = _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))

    assert leg.max_in_flight == len(entities)
    assert sorted(leg.started) == sorted(entities)
    assert turn.sources, "the barrier must not have short-circuited the turn"


def test_body_rescue_leg_also_overlaps(stub_pipeline, monkeypatch):
    """The rescue is inside the gathered leg, so its awaits overlap too.

    body_rescue only runs on a weak result set, so each leg is given a
    below-gate score; the rescue is what lifts the articles into the answer,
    which also proves each leg's rescue saw ITS OWN results.
    """
    entities = ["alpha", "bravo"]
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", True)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.01)] for i, e in enumerate(entities)}
    )
    leg.arm_barrier(len(entities))

    turn = _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))

    assert leg.max_in_flight == len(entities)
    # Unrescued these would score 0.01 and be gated out entirely.
    assert [s["title"] for s in turn.sources] == ["alpha deal", "bravo deal"]
    assert all(s["score"] == pytest.approx(0.99) for s in turn.sources)


# --- equivalence with the sequential implementation ---


@pytest.mark.parametrize("mode", ["comparison", "intersection"])
def test_gathered_result_identical_to_sequential(stub_pipeline, monkeypatch, mode):
    """Same articles, order, annotation and prompt as the sequential loop.

    The oracle is the real function with concurrency pinned to 1, so the
    prompt assembly, the dedupe, the per-article entity list (which feeds both
    the "Entities:" line and the rank key) and the retrieval arguments are all
    compared -- not just the source ids.
    """
    entities = ["alpha", "bravo", "charlie"]
    facts = {"alpha.industry": "Finance", "bravo.industry": "Tech", "charlie.industry": "Tech"}
    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 4)
    leg = stub_pipeline(entity_articles=_ARTICLES, facts=facts)
    gathered = _run(chat_module._prepare_multi_entity_turn(_multi(entities, mode), "q", []))

    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 1)
    leg_seq = stub_pipeline(entity_articles=_ARTICLES, facts=facts)
    sequential = _run(chat_module._prepare_multi_entity_turn(_multi(entities, mode), "q", []))

    assert leg.max_in_flight == len(entities), "the gathered run must really have overlapped"
    assert leg_seq.max_in_flight == 1, "concurrency 1 must be the sequential oracle"
    assert leg_seq.started == entities

    # Sources, order, and the full prompt/system/note.
    assert gathered.sources == sequential.sources
    assert [s["id"] for s in gathered.sources] == [s["id"] for s in sequential.sources]
    assert gathered.answer == sequential.answer
    assert gathered.system == sequential.system
    assert gathered.note == sequential.note

    # What each leg was asked for, so a change to the per-entity arguments
    # (top_k, auto facets, need_body) cannot pass as "same result".
    assert leg.calls == leg_seq.calls
    assert [c[0] for c in leg.calls] == entities
    assert all(c[2]["need_body"] is True for c in leg.calls)
    # The shared article is annotated with both of its entities, in entity order.
    assert "Entities: alpha, bravo" in gathered.answer
    assert "Entities: alpha, charlie" in gathered.answer


def _entities_by_article(answer: str) -> dict[str, str]:
    """Map each article's title to the entity list annotated on its own block.

    Asserting "Entities: alpha, bravo" appears somewhere is symmetric: a
    mispairing still produces it, just on the wrong articles. Binding the
    annotation to the article it names is what actually detects a swap.
    """
    # A block is "<<<ARTICLE n>>>\n[n] <title> (<date>)\n<body>\nEntities: <...>",
    # so the "n>>>" that closes the ARTICLE header is its own line and the
    # title line follows it.
    out = {}
    for block in answer.split("<<<ARTICLE ")[1:]:
        title = block.splitlines()[1].split("] ", 1)[1].rsplit(" (", 1)[0]
        ents = block.rsplit("Entities: ", 1)
        out[title] = ents[1].split("\n", 1)[0].strip() if len(ents) == 2 else ""
    return out


def test_each_article_is_annotated_with_its_own_entities(stub_pipeline, monkeypatch):
    """A completion-order swap must be visible on the articles themselves.

    Pairing bravo's results to alpha is not detectable by asserting the
    annotation text exists; it is detectable by checking WHICH article carries
    WHICH annotation.
    """
    entities = ["alpha", "bravo", "charlie"]
    leg = stub_pipeline(entity_articles={
        "alpha": [_article(1, "alpha one", 0.9), _article(4, "shared ab", 0.7)],
        "bravo": [_article(2, "bravo two", 0.9), _article(4, "shared ab", 0.7)],
        "charlie": [_article(3, "charlie three", 0.9)],
    })
    leg.slow_for("charlie")

    turn = _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))
    got = _entities_by_article(turn.answer)

    assert got["alpha one"] == "alpha"
    assert got["bravo two"] == "bravo"
    assert got["charlie three"] == "charlie"
    assert got["shared ab"] == "alpha, bravo", "the shared article lists both, in entity order"


def test_repeated_runs_are_byte_identical(stub_pipeline, monkeypatch):
    """Determinism: the same question produces the same prompt every time."""
    entities = ["alpha", "bravo", "charlie"]
    stub_pipeline(entity_articles=_ARTICLES)

    runs = [_run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", [])) for _ in range(5)]

    assert len({(r.answer, r.system, r.note, tuple(s["id"] for s in r.sources)) for r in runs}) == 1


# --- the fan-out is bounded ---


def test_concurrency_capped(stub_pipeline, monkeypatch):
    """More entities than the concurrency setting must not all run at once."""
    entities = ["e0", "e1", "e2", "e3", "e4", "e5"]
    monkeypatch.setattr(chat_module.config, "CHAT_MAX_MULTI_ENTITIES", 6)
    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 2)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )
    leg.slow_for(*entities)
    leg.delay = 0.01

    _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))

    assert leg.max_in_flight == 2, "the semaphore must admit exactly CHAT_MULTI_ENTITY_CONCURRENCY legs"
    assert sorted(leg.started) == sorted(entities), "capping must not drop legs"
    assert sorted(leg.finished) == sorted(entities), "every capped leg must still run to completion"


def test_concurrency_cap_still_overlaps(stub_pipeline, monkeypatch):
    """Cap 1 degenerates to the old sequential order; cap 2 overlaps."""
    articles = {"a": [_article(1, "a deal", 0.9)], "b": [_article(2, "b deal", 0.9)]}

    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 1)
    serial = stub_pipeline(entity_articles=articles)
    _run(chat_module._prepare_multi_entity_turn(_multi(["a", "b"]), "q", []))
    assert serial.max_in_flight == 1
    assert serial.finished == ["a", "b"]

    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 2)
    parallel = stub_pipeline(entity_articles=articles)
    parallel.slow_for("b")
    _run(chat_module._prepare_multi_entity_turn(_multi(["a", "b"]), "q", []))
    assert parallel.max_in_flight == 2


def test_oversized_entity_list_never_fans_out(stub_pipeline, monkeypatch):
    """A question naming more entities than the cap takes the single-query path.

    The question below yields far more entities than the cap, so the
    multi-entity expansion -- one pipeline per entity -- must not run at all.
    """
    from app.query_intent import detect_multi_entity

    monkeypatch.setattr(chat_module.config, "CHAT_MAX_MULTI_ENTITIES", 6)
    entities = [f"Acme{i}" for i in range(120)]
    question = "compare " + " and ".join(entities) + " funding rounds"
    assert len(question) < chat_module.MAX_CONTENT_LEN, "the question must fit the API's own limit"

    detected = detect_multi_entity(question)
    assert detected is not None
    assert len(detected.entities) > chat_module.config.CHAT_MAX_MULTI_ENTITIES
    monkeypatch.setattr(chat_module, "detect_multi_entity", lambda q: detected)

    calls: list[str] = []
    leg = stub_pipeline(entity_articles=_ARTICLES)

    async def counting(rq, top_k, **kwargs):
        calls.append(rq)
        return await leg.retrieve(rq, top_k, **kwargs)

    from app import main

    main.retrieve_with_auto_facet_fallback = counting
    try:
        turn = _run(chat_module._prepare_turn(question, []))
    finally:
        main.retrieve_with_auto_facet_fallback = leg.retrieve

    assert len(calls) == 1, f"one retrieval for the whole question, not one per entity (got {len(calls)})"
    assert calls[0] == question
    assert "## Multi-entity comparison" not in (turn.system or "")


def test_entity_list_at_the_cap_still_compares(stub_pipeline, monkeypatch):
    """The cap is inclusive: exactly CHAT_MAX_MULTI_ENTITIES entities compares."""
    monkeypatch.setattr(chat_module.config, "CHAT_MAX_MULTI_ENTITIES", 3)
    entities = ["alpha", "bravo", "charlie"]
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )
    monkeypatch.setattr(chat_module, "detect_multi_entity", lambda q: _multi(entities))

    turn = _run(chat_module._prepare_turn("compare alpha and bravo and charlie funding", []))

    assert sorted(leg.started) == sorted(entities)
    assert "## Multi-entity comparison" in (turn.system or "")


# --- wall clock ---


def test_gathered_turn_beats_the_sequential_turn(stub_pipeline, monkeypatch):
    """Measure the win the gather actually buys, against the sequential oracle.

    The stub's delay stands in for the Qdrant I/O each leg does; the CPU rerank
    it stands in for is serialized by the shared ``inference_lock``, so the
    honest ceiling here is the I/O overlap alone. Asserted with a margin, since
    the number is a property of the stub rather than of the production path.
    """
    entities = [f"e{i}" for i in range(4)]
    articles = {e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 4)
    leg = stub_pipeline(entity_articles=articles)
    leg.slow_for(*entities)
    leg.delay = 0.05

    start = time.perf_counter()
    _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))
    gathered = time.perf_counter() - start

    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 1)
    leg_seq = stub_pipeline(entity_articles=articles)
    leg_seq.slow_for(*entities)
    leg_seq.delay = 0.05

    start = time.perf_counter()
    _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))
    sequential = time.perf_counter() - start

    assert sequential >= len(entities) * 0.05, "the sequential turn must pay every delay"
    assert gathered < sequential, f"gathered {gathered:.3f}s should beat sequential {sequential:.3f}s"
    # Never better than one delay plus changeover: the legs cannot overlap
    # more than the slowest one.
    assert gathered >= 0.05


# --- failure behaviour matches the sequential loop ---


def test_error_is_the_first_entity_s_not_the_race_winner(stub_pipeline, monkeypatch):
    """The surfaced error is entity 0's, not whichever leg happened to lose the race.

    Plain gather() raises whichever leg failed first in wall-clock time; the
    sequential loop this replaced raised the first ENTITY's. Alpha is first in
    entity order but raises last in wall-clock, and the two failing legs raise
    different exception types, so a race surfaces ValueError("bravo failed")
    where the code must surface RuntimeError("alpha failed"). Repeated three
    times, since a race could not be relied on to reproduce.

    The await-all property -- that no leg keeps running against Qdrant after
    the turn has already failed -- is NOT asserted here. It was tried and
    dropped as vacuous: a coroutine's ``finally`` runs during asyncio.run's
    shutdown cancellation too, so the bookkeeping showed every leg finished
    under plain gather as well. That property rests on the implementation
    (return_exceptions=True awaits all legs before the re-raise), not on a test.
    """
    entities = ["alpha", "bravo", "charlie"]
    _Leg(entity_articles=_ARTICLES)  # installs the fixture patch we then replace

    async def flaky(rq, top_k, **kwargs):
        entity = rq.split(" ")[0]
        if entity == "alpha":
            await asyncio.sleep(0.02)   # first in entity order, last to fail
            raise RuntimeError("alpha failed")
        if entity == "bravo":
            raise ValueError("bravo failed")
        await asyncio.sleep(0.01)
        return ([_article(1, f"{entity} deal", 0.9)], None, None, None)

    from app import main

    saved = main.retrieve_with_auto_facet_fallback
    main.retrieve_with_auto_facet_fallback = flaky
    try:
        for _ in range(3):
            with pytest.raises(RuntimeError, match="alpha failed"):
                _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))
    finally:
        main.retrieve_with_auto_facet_fallback = saved


# --- the config knobs cannot be misconfigured into a hang or a dead feature ---


def test_zero_concurrency_knob_does_not_hang(stub_pipeline, monkeypatch):
    """CHAT_MULTI_ENTITY_CONCURRENCY=0 would make the semaphore never release.

    asyncio.Semaphore(0) blocks every leg forever, which is strictly worse than
    the sequential bug being fixed, so the value is floored at 1.
    """
    entities = ["alpha", "bravo"]
    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 0)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )

    turn = _run(asyncio.wait_for(
        chat_module._prepare_multi_entity_turn(_multi(entities), "q", []), timeout=5.0
    ))

    assert leg.max_in_flight == 1, "0 must floor to 1, not deadlock the turn"
    assert [s["title"] for s in turn.sources] == ["alpha deal", "bravo deal"]


@pytest.mark.parametrize("cap", [0, 1, -5])
def test_tiny_entity_cap_keeps_the_feature_alive(stub_pipeline, monkeypatch, cap):
    """A comparison needs at least two entities, so a cap below that floors to 2.

    Otherwise a typo in the env would silently disable multi-entity turns
    entirely rather than mean anything.
    """
    entities = ["alpha", "bravo"]
    monkeypatch.setattr(chat_module.config, "CHAT_MAX_MULTI_ENTITIES", cap)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )
    monkeypatch.setattr(chat_module, "detect_multi_entity", lambda q: _multi(entities))

    turn = _run(chat_module._prepare_turn("compare alpha and bravo funding", []))

    assert sorted(leg.started) == sorted(entities), "a two-entity comparison must still run"
    assert "## Multi-entity comparison" in (turn.system or "")
