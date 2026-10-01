"""Multi-entity chat turns gather their per-entity legs under a bounded gather.

The equivalence oracle throughout is the SAME function run with
``CHAT_MULTI_ENTITY_CONCURRENCY = 1``, which admits one leg at a time and is
therefore the old sequential loop -- preferred over a copied-out reference
implementation because it exercises the real prompt assembly.
"""

import asyncio
import json
import time

import pytest
from conftest import auth_cookie
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth as auth_module
from app import chat as chat_module
from app import main as main_module
from app.auth import AuthStore
from app.chat import ChatStore
from app.main import SourceArticle
from app.query_intent import MultiEntityQuery, detect_multi_entity

# Long enough that source_context() truncates it, so the compared prompt is sensitive to the body budget too.
BODY = "b" * 400


def _run(coro):
    return asyncio.run(coro)


class _Leg:
    """Stubbed retrieval pipeline; one instance stands in for every entity leg."""

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
        self.rescue_delay = 0.0
        self.rescues = 0
        self.rescue_in_flight = 0
        self.max_rescue_in_flight = 0
        self._slow_entities: frozenset[str] = frozenset()

    def arm_barrier(self, parties: int) -> None:
        """No leg may leave until ``parties`` legs have arrived.

        A sequential loop never satisfies the barrier, so the wait times out: the
        test fails on the absence of overlap, not on a wall-clock threshold.
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
        """These entities sleep for ``delay``; every other leg is a bare checkpoint."""
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
            # shared object would carry one test's rescue into the next.
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
        # The cap is held across both of a leg's awaits, so the rescue needs its
        # own in-flight counters: bounding only the retrieval would leave it unbounded.
        self.rescues += 1
        self.rescue_in_flight += 1
        self.max_rescue_in_flight = max(self.max_rescue_in_flight, self.rescue_in_flight)
        try:
            if self.rescue_delay:
                await asyncio.sleep(self.rescue_delay)
            for a in articles:
                a.score = 0.99
            return articles
        finally:
            self.rescue_in_flight -= 1

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


# A shared id (3) so the dedupe and the per-article entity list are exercised,
# and an id (1) matched by two entities with different scores so the rank key has
# something to order.
_ARTICLES = {
    "alpha": [_article(1, "alpha one", 0.9), _article(3, "shared", 0.7)],
    "bravo": [_article(2, "bravo two", 0.9), _article(3, "shared", 0.7)],
    "charlie": [_article(1, "alpha one", 0.8), _article(4, "charlie four", 0.85)],
}


# --- overlap ---


def test_legs_run_concurrently(stub_pipeline, monkeypatch):
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
    """Each leg gets a below-gate score, so only its own body rescue can lift its
    articles into the answer."""
    entities = ["alpha", "bravo"]
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", True)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.01)] for i, e in enumerate(entities)}
    )
    leg.arm_barrier(len(entities))

    turn = _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))

    assert leg.max_in_flight == len(entities)
    assert [s["title"] for s in turn.sources] == ["alpha deal", "bravo deal"]
    assert all(s["score"] == pytest.approx(0.99) for s in turn.sources)


# --- equivalence with the sequential implementation ---


@pytest.mark.parametrize("mode", ["comparison", "intersection"])
def test_gathered_result_identical_to_sequential(stub_pipeline, monkeypatch, mode):
    """Same articles, order, annotation and prompt as the sequential loop."""
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

    assert gathered.sources == sequential.sources
    assert [s["id"] for s in gathered.sources] == [s["id"] for s in sequential.sources]
    assert gathered.answer == sequential.answer
    assert gathered.system == sequential.system
    assert gathered.note == sequential.note

    # Per-entity arguments too, so a change there cannot pass as "same result".
    assert leg.calls == leg_seq.calls
    assert [c[0] for c in leg.calls] == entities
    assert all(c[2]["need_body"] is True for c in leg.calls)
    assert "Entities: alpha, bravo" in gathered.answer
    assert "Entities: alpha, charlie" in gathered.answer


def _entities_by_article(answer: str) -> dict[str, str]:
    """Map each article's title to the entity list annotated on its own block.

    Asserting the annotation text appears somewhere is symmetric: a mispairing
    still produces it, just on the wrong articles.
    """
    # Block shape: "<<<ARTICLE n>>>\n[n] <title> (<date>)\n<body>\nEntities: <...>",
    # so the title line is the one after the ARTICLE header.
    out = {}
    for block in answer.split("<<<ARTICLE ")[1:]:
        title = block.splitlines()[1].split("] ", 1)[1].rsplit(" (", 1)[0]
        ents = block.rsplit("Entities: ", 1)
        out[title] = ents[1].split("\n", 1)[0].strip() if len(ents) == 2 else ""
    return out


def test_each_article_is_annotated_with_its_own_entities(stub_pipeline, monkeypatch):
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
    """The question yields far more entities than the cap, so the per-entity
    expansion must not run at all."""
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
    """The stub's delay stands in for the Qdrant I/O each leg does; the CPU rerank
    it stands in for is serialized by the shared ``inference_lock``, so the honest
    ceiling here is the I/O overlap alone."""
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
    # Never better than one delay plus changeover: the legs cannot overlap past the slowest one.
    assert gathered >= 0.05


# --- failure behaviour matches the sequential loop ---


def test_error_is_the_first_entity_s_not_the_race_winner(stub_pipeline, monkeypatch):
    """Plain gather() raises whichever leg failed first in wall-clock time; the
    sequential loop raised the first ENTITY's. Alpha is first in entity order but
    raises last in wall-clock, and the two failing legs raise different types.

    The await-all property -- that no leg keeps running against Qdrant after the
    turn has failed -- is not asserted here: a coroutine's ``finally`` also runs
    during asyncio.run's shutdown cancellation, so bookkeeping of it would pass
    under plain gather too.
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


# --- the config knobs ---


def test_zero_concurrency_knob_does_not_hang(stub_pipeline, monkeypatch):
    """``asyncio.Semaphore(0)`` blocks every leg forever, so the knob is floored at 1."""
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


def test_concurrency_cap_covers_body_rescue(stub_pipeline, monkeypatch):
    """The semaphore is held across both of a leg's awaits, so bounding only the
    retrieval would leave the body fetch -- the other half of a leg's cost --
    unbounded. The last assertion pins that rescues really do overlap.

    The rescue delay is the longer of the two on purpose: an unbounded rescue
    only piles up if a leg is still fetching bodies when the next one starts.
    """
    entities = ["alpha", "bravo", "charlie", "delta"]
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", True)
    monkeypatch.setattr(chat_module.config, "CHAT_MULTI_ENTITY_CONCURRENCY", 2)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.01)] for i, e in enumerate(entities)}
    )
    leg.slow_for(*entities)
    leg.delay = 0.01
    leg.rescue_delay = 0.05

    _run(chat_module._prepare_multi_entity_turn(_multi(entities), "q", []))

    assert leg.rescues == len(entities), "capping must not skip a leg's rescue"
    assert leg.max_in_flight == 2
    assert leg.max_rescue_in_flight == 2, "rescues must overlap, and stay under the cap"


@pytest.mark.parametrize("cap", [0, 1, -5])
def test_tiny_entity_cap_keeps_the_feature_alive(stub_pipeline, monkeypatch, cap):
    """A comparison needs at least two entities, so a cap below that floors to 2."""
    entities = ["alpha", "bravo"]
    monkeypatch.setattr(chat_module.config, "CHAT_MAX_MULTI_ENTITIES", cap)
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )
    monkeypatch.setattr(chat_module, "detect_multi_entity", lambda q: _multi(entities))

    turn = _run(chat_module._prepare_turn("compare alpha and bravo funding", []))

    assert sorted(leg.started) == sorted(entities), "a two-entity comparison must still run"
    assert "## Multi-entity comparison" in (turn.system or "")


# --- the wire: event order and what the turn is charged ---


_QUESTION = "compare Acme and Bravo and Cirrus funding rounds"
_EMAIL = "user-a@example.com"


def _events(body: str) -> list[str]:
    """The event names of an SSE body, in the order the client receives them."""
    return [line[len("event: "):] for line in body.splitlines() if line.startswith("event: ")]


def _done_payload(body: str) -> dict:
    return json.loads(body.split("event: done\ndata: ", 1)[1])


class _Budget:
    """The daily cap, standing in for the Redis one and counting what it is asked:
    one hold taken, one settle recorded, no release."""

    def __init__(self):
        self.reserved: list[float] = []
        self.settled: list[float] = []
        self.released: list[list[str]] = []

    async def reserve(self, estimate_usd: float = 0.0) -> str:
        self.reserved.append(estimate_usd)
        return f"hold-{len(self.reserved)}"

    async def settle(self, ids, actual_usd: float) -> None:
        self.settled.append(actual_usd)

    async def release(self, ids) -> None:
        self.released.append(list(ids))

    def install(self, monkeypatch) -> "_Budget":
        monkeypatch.setattr(chat_module, "reserve", self.reserve)
        monkeypatch.setattr(chat_module, "settle", self.settle)
        monkeypatch.setattr(chat_module, "release", self.release)
        return self


def _stream_client(tmp_path):
    """A TestClient over the real chat router, on real stores in tmp_path."""
    chat_store = ChatStore(str(tmp_path / "chat.db"))
    auth_store = AuthStore(str(tmp_path / "auth.db"))
    _run(chat_store.connect())
    _run(auth_store.connect())
    app = FastAPI()
    app.include_router(chat_module.router)
    chat_module.store = chat_store
    auth_module.store = auth_store
    return TestClient(app), chat_store, auth_store


def _auth_cookie(auth_store) -> dict[str, str]:
    """Create the account and return the session cookie the browser would send: the
    credential is an HttpOnly cookie, so there is no ``Authorization`` header."""
    user = _run(auth_store.get_user_by_email(_EMAIL))
    if user is None:
        user = _run(auth_store.create_user(_EMAIL, "secret1", "user-a", "user"))
    return auth_cookie(_run(auth_store.issue_token(user.id, 7)))


def _post_turn(client, cookie, question: str) -> str:
    """Drive one SSE turn through the real route and hand back the raw body."""
    sid = client.post("/api/chat/sessions", cookies=cookie).json()["id"]
    with client.stream(
        "POST", f"/api/chat/sessions/{sid}/messages/stream",
        cookies=cookie, json={"content": question},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        return "".join(r.iter_text())


def _fake_llm(pieces, calls):
    """A stand-in provider that records the prompts it was actually sent."""

    async def stream(client, prompt, model, usage_holder=None, system_prompt=None):
        calls.append(prompt)
        for piece in pieces:
            yield piece
        if usage_holder is not None:
            usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=4, completion_tokens=2))

    return stream


def test_stream_event_order_is_unchanged(stub_pipeline, monkeypatch, tmp_path):
    """The frontend switches on the event name and on its position in the stream,
    so "the legs ran" is not the property at stake -- the sequence is."""
    # The detector's own order and spelling, so the expansion is not a monkeypatched stand-in.
    entities = ["cirrus", "bravo", "acme"]
    assert detect_multi_entity(_QUESTION).entities == entities
    leg = stub_pipeline(entity_articles={
        # A shared id puts the dedupe and the per-article entity list on the wire
        # too, and the middle entity finishing LAST means the output order is not
        # completion order.
        "cirrus": [_article(1, "cirrus deal", 0.9), _article(9, "shared", 0.6)],
        "bravo": [_article(2, "bravo deal", 0.9)],
        "acme": [_article(3, "acme deal", 0.9), _article(9, "shared", 0.6)],
    })
    leg.arm_barrier(len(entities))
    leg.slow_for("bravo")
    leg.delay = 0.05

    monkeypatch.setattr(chat_module, "stream_answer", _fake_llm(["Both ", "raised."], []))
    _Budget().install(monkeypatch)

    client, chat_store, auth_store = _stream_client(tmp_path)
    try:
        body = _post_turn(client, _auth_cookie(auth_store), _QUESTION)
    finally:
        _run(auth_store.close())
        _run(chat_store.close())

    assert leg.max_in_flight == len(entities), "the legs must really have overlapped"
    assert _events(body) == ["start", "delta", "delta", "done"]
    # The two-entity article sorts first (the most entities match it), then by score.
    assert [s["title"] for s in _done_payload(body)["message"]["sources"]] == [
        "shared", "cirrus deal", "bravo deal", "acme deal",
    ]


def test_entity_count_does_not_multiply_billed_calls(stub_pipeline, monkeypatch, tmp_path):
    """A leg is retrieval only -- it never reaches the provider -- so gathering them
    must not multiply what the turn costs."""
    entities = ["cirrus", "bravo", "acme"]
    leg = stub_pipeline(
        entity_articles={e: [_article(i + 1, f"{e} deal", 0.9)] for i, e in enumerate(entities)}
    )
    leg.arm_barrier(len(entities))

    calls: list[str] = []
    monkeypatch.setattr(chat_module, "stream_answer", _fake_llm(["One answer."], calls))
    budget = _Budget().install(monkeypatch)

    client, chat_store, auth_store = _stream_client(tmp_path)
    try:
        body = _post_turn(client, _auth_cookie(auth_store), _QUESTION)
    finally:
        _run(auth_store.close())
        _run(chat_store.close())

    assert leg.max_in_flight == len(entities)
    assert len(calls) == 1, f"one billed LLM call for the turn, not one per entity (got {len(calls)})"
    assert len(budget.reserved) == 1, "one hold taken up front, before the billed call"
    assert len(budget.settled) == 1, "one settle, recording what the turn really cost"
    assert not budget.released, "an answered turn settles its hold, it never releases it"
    assert budget.settled[0] > 0
    assert _events(body) == ["start", "delta", "done"]
