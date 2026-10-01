"""A /search cache key must be a (query, filter, config) triple, and both keys must be read in one MGET."""
import asyncio

import pytest

from app import main
from app.config import config
from app.redis_cache import HybridCache

# Config the retrieval/rerank pipeline reads: each changes the cached set, so each must be in the key.
RETRIEVAL_KNOBS = [
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBED_MODEL",
    "SPARSE_MODEL",
    "EMBED_DEVICE",
    "RERANK_BACKEND",
    "RERANK_MODEL",
    "RERANK_CANDIDATES",
    "RECENCY_STRENGTH",
    "RECENCY_DECAY_DAYS",
    "RECENCY_BOOST_STRENGTH",
    "RECENCY_BOOST_DECAY_DAYS",
    "ENABLE_QUERY_EXPANSION",
    "ENABLE_ENTITY_BOOST",
    "REDIS_URL",
    "ANALYTICS_REDIS_DB",
]

# Config /search reads after retrieval: it changes the summary page, not the retrieval entry.
SEARCH_KNOBS = [
    "ENABLE_CLICK_BOOST",
    "CLICK_BOOST_MIN_CLICKS",
    "CLICK_BOOST_MIN_ARTICLE_CLICKS",
    "CLICK_BOOST_MIN_SHARE",
    "CLICK_BOOST_MULT",
    "CLICK_QUERY_MAX_LEN",
    "ENABLE_DIVERSITY",
    "DIVERSITY_LAMBDA",
    "DIVERSITY_SIM_THRESHOLD",
    "ASK_MIN_SCORE",
]

# Config that cannot change a result; folding it in would invalidate every entry on any deploy.
IRRELEVANT_KNOBS = [
    "CACHE_TTL_SECONDS",
    "VECTOR_CACHE_TTL_SECONDS",
    "CACHE_MAX_SIZE",
    "AUTH_DB_PATH",
    "CHAT_DB_PATH",
    "CORS_ORIGINS",
    "PUBLIC_SEARCH_RATE_PER_MIN",
]

# Round trips a /search miss cost before the two search-layer reads were merged into one MGET.
PRE_FIX_ROUND_TRIPS = 6


def _flipped(value):
    """A different value of the same type (sequences replaced wholesale, not mutated)."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-other"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)([*value, "extra"])
    return "changed"


def _article(id_, score=0.9):
    return main.SourceArticle(
        id=id_, title=f"t{id_}", url=f"u{id_}", summary="s", body="", score=score
    )


def _summary(id_, score=0.9):
    return main.SourceSummary(
        id=id_, title=f"t{id_}", url=f"u{id_}", published_date="2025-01-10",
        category="News", summary="s", score=score,
        author_names=["A"], industry_names=["F"], dealtype_names=["D"],
    ).model_dump()


class _CountingRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.commands: list[tuple[str, object]] = []

    async def get(self, key):
        self.commands.append(("GET", key))
        return self.store.get(key)

    async def mget(self, keys, *_rest):

        # Production calls mget both as mget(keys) and mget(*keys); accept either.

        if isinstance(keys, str):

            keys = [keys, *_rest]

        else:

            keys = [*keys, *_rest]
        self.commands.append(("MGET", tuple(keys)))
        return [self.store.get(key) for key in keys]

    async def set(self, key, value, ex=None):
        self.commands.append(("SET", key))
        self.store[key] = value

    async def aclose(self):
        return None


def _wire_search(monkeypatch, articles=None, boosted=2.0, real_hybrid=False):
    """``real_hybrid`` keeps the real ``hybrid_search``, so the ``vec:`` cache lookup is counted too."""
    articles = list(articles) if articles is not None else [_article(1, 0.9), _article(2, 0.5)]

    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)

    async def fake_rerank(q, results):
        return results

    monkeypatch.setattr(main, "rerank", fake_rerank)

    async def fake_hybrid_search(rq, n, qfilter=None):
        return [_article(a.id, a.score) for a in articles]

    async def fake_record_search(*args, **kwargs):
        return None

    async def fake_click_boost(q, results):
        if config.ENABLE_CLICK_BOOST:
            for r in results:
                r.score = r.score * boosted
        return results

    if real_hybrid:
        def fake_embed_sparse(model, q):
            return _SparseEmbedding()

        monkeypatch.setattr(main, "_embed_sparse", fake_embed_sparse)
        monkeypatch.setitem(main.state, "model", _FakeDenseEncoder())
        monkeypatch.setitem(main.state, "sparse_model", object())
        monkeypatch.setitem(main.state, "qdrant", _FakeQdrant(articles))
    else:
        monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main, "record_search", fake_record_search)
    monkeypatch.setattr(main, "apply_click_boost", fake_click_boost)

    cache = HybridCache("redis://unused", 300, 64)
    cache._redis = _CountingRedis()
    monkeypatch.setattr(main, "cache", cache)
    return cache


class _FakeQdrant:
    def __init__(self, articles):
        self._articles = articles

    async def query_points(self, **kwargs):
        class _Point:
            def __init__(self, a):
                self.id = a.id
                self.payload = a.model_dump()
                self.score = a.score

        class _Response:
            def __init__(self, points):
                self.points = points

        return _Response([_Point(a) for a in self._articles])


class _FakeDenseEncoder:
    """Encoder double; ``encode`` runs via ``asyncio.to_thread``, so it must be sync with ``.tolist()``."""

    class _Vector:
        @staticmethod
        def tolist():
            return [0.1, 0.2, 0.3]

    def encode(self, query):
        return self._Vector()


class _SparseEmbedding:
    """Sparse encoder output; ``hybrid_search`` calls ``.tolist()`` on both fields."""

    class _Vec:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    def __init__(self):
        self.indices = self._Vec([1, 4])
        self.values = self._Vec([0.5, 0.25])


def _search(**kwargs):
    params = {"q": "fintech funding", "top_k": 8, "industry": None, "dealtype": None,
              "author": None, "content_type": None, "tag": None, "from_date": None, "to_date": None}
    params.update(kwargs)
    return asyncio.run(main.search(**params))


def test_the_fingerprint_covers_exactly_the_knobs_under_test():
    """The knob lists are hand-written; nothing else ties them to the fingerprint."""
    covered = {attr for _, attr in main._RETRIEVAL_CONFIG_INPUTS}
    assert covered == set(RETRIEVAL_KNOBS) | set(SEARCH_KNOBS), (
        "a config input is in the fingerprint but not exercised here, or is "
        "exercised here but missing from the fingerprint"
    )
    assert not covered & set(IRRELEVANT_KNOBS), (
        "a knob that cannot change a result is also in the fingerprint"
    )


def test_a_fully_cold_search_miss_costs_fewer_round_trips_than_the_base(monkeypatch):
    """Counted through the real ``hybrid_search``, so the ``vec:`` lookup a miss pays for is included."""
    cache = _wire_search(monkeypatch, real_hybrid=True)
    _search()

    kinds = [c[0] for c in cache._redis.commands]
    assert kinds == ["MGET", "GET", "SET", "SET", "SET"], f"unexpected traffic: {kinds}"
    assert len(cache._redis.commands) < PRE_FIX_ROUND_TRIPS
    assert len(cache._redis.commands) == PRE_FIX_ROUND_TRIPS - 1
    assert kinds.count("GET") == 1, "the only remaining GET is the vector cache"


@pytest.mark.parametrize("knob", RETRIEVAL_KNOBS + SEARCH_KNOBS)
def test_every_retrieval_config_knob_changes_the_retrieve_key(monkeypatch, knob):
    before = main.retrieve_cache_key("q", 8, None)
    monkeypatch.setattr(config, knob, _flipped(getattr(config, knob)))
    assert main.retrieve_cache_key("q", 8, None) != before, (
        f"{knob} changes what is retrieved but not the cache key: an entry "
        f"computed under the old value would be replayed for the whole TTL"
    )


@pytest.mark.parametrize("knob", RETRIEVAL_KNOBS + SEARCH_KNOBS)
def test_every_retrieval_config_knob_changes_the_search_key(monkeypatch, knob):
    before = main.search_cache_key("q", 8, "facets")
    monkeypatch.setattr(config, knob, _flipped(getattr(config, knob)))
    assert main.search_cache_key("q", 8, "facets") != before, (
        f"{knob} changes what /search returns but not the cache key"
    )


@pytest.mark.parametrize("knob", IRRELEVANT_KNOBS)
def test_config_knobs_that_cannot_change_results_keep_the_same_key(monkeypatch, knob):
    """An over-broad key (TTL, size cap, CORS allowlist) would split one cache into many."""
    before = main.retrieve_cache_key("q", 8, None)
    monkeypatch.setattr(config, knob, _flipped(getattr(config, knob)))
    assert main.retrieve_cache_key("q", 8, None) == before, (
        f"{knob} cannot change a retrieval result, so it must not invalidate it"
    )


def test_config_knobs_do_not_grow_the_key_without_bound():
    key = main.retrieve_cache_key("q", 8, None)
    assert len(key) < 160, f"cache key grew unexpectedly long: {key!r}"


def test_query_top_k_and_filter_still_separate_entries():
    """Distinct queries, top_k and filters must not collide; the digest must not swallow them."""
    f = main.build_facet_filter("Fintech", None, None, None, None)
    assert main.retrieve_cache_key("a", 8, None) != main.retrieve_cache_key("b", 8, None)
    assert main.retrieve_cache_key("a", 8, None) != main.retrieve_cache_key("a", 9, None)
    assert main.retrieve_cache_key("a", 8, None) != main.retrieve_cache_key("a", 8, f)
    assert main.search_cache_key("a", 8, "f") != main.search_cache_key("a", 8, "g")


def test_flipping_a_config_cannot_serve_a_stale_entry(monkeypatch):
    """The stubbed boost doubles scores, so a stale entry shows in the payload, not only in ``cached``."""
    _wire_search(monkeypatch)

    first = _search()
    assert first.cached is False
    boosted_score = first.results[0].score

    repeat = _search()
    assert repeat.cached is True
    assert [r.score for r in repeat.results] == [r.score for r in first.results]

    monkeypatch.setattr(config, "ENABLE_CLICK_BOOST", False)
    after_flip = _search()
    assert after_flip.cached is False, (
        "an entry computed with the boost enabled was replayed after the toggle "
        "was flipped"
    )
    assert after_flip.results[0].score == pytest.approx(boosted_score / 2.0), (
        "the response is still the stale, boosted one"
    )


@pytest.mark.parametrize(
    "knob", ["ENABLE_ENTITY_BOOST", "ENABLE_QUERY_EXPANSION", "DIVERSITY_LAMBDA", "RERANK_CANDIDATES"]
)
def test_every_retrieval_knob_invalidates_a_populated_cache(monkeypatch, knob):
    _wire_search(monkeypatch)
    assert _search().cached is False
    assert _search().cached is True

    monkeypatch.setattr(config, knob, _flipped(getattr(config, knob)))
    assert _search().cached is False, (
        f"{knob} was flipped but the previous entry was still served"
    )


def test_a_search_miss_costs_three_redis_round_trips(monkeypatch):
    cache = _wire_search(monkeypatch)
    _search()

    kinds = [c[0] for c in cache._redis.commands]
    assert kinds == ["MGET", "SET", "SET"], f"unexpected Redis traffic: {kinds}"
    assert len(cache._redis.commands) < PRE_FIX_ROUND_TRIPS

    mget_keys = next(c[1] for c in cache._redis.commands if c[0] == "MGET")
    assert len(mget_keys) == 2
    assert sorted(k.split(":", 1)[0] for k in mget_keys) == ["retrieve", "search"]


def test_the_inner_leg_does_not_re_read_its_own_key(monkeypatch):
    cache = _wire_search(monkeypatch)
    _search()
    assert not [c for c in cache._redis.commands if c[0] == "GET"], (
        "retrieve_and_rerank re-read the key /search had already prefetched"
    )


def test_a_search_hit_costs_a_single_round_trip(monkeypatch):
    cache = _wire_search(monkeypatch)
    _search()
    cache._redis.commands.clear()

    hit = _search()
    assert hit.cached is True
    assert [c[0] for c in cache._redis.commands] == ["MGET"]


def test_an_empty_result_set_is_not_written(monkeypatch):
    cache = _wire_search(monkeypatch, articles=[])
    _search()
    assert [c[0] for c in cache._redis.commands] == ["MGET"]


def test_both_entries_are_still_written_with_their_own_payloads(monkeypatch):
    cache = _wire_search(monkeypatch)
    _search()
    written = {k.split(":", 1)[0]: v for k, v in cache._redis.store.items()}
    assert set(written) == {"search", "retrieve"}
    assert "body" not in written["retrieve"][0]
    assert "body" not in written["search"][0]
    assert written["search"] != written["retrieve"]


def test_results_are_identical_with_and_without_a_warm_cache(monkeypatch):
    cache = _wire_search(monkeypatch)
    cold = _search()
    warm = _search()
    assert warm.cached is True
    assert [(r.id, r.score) for r in warm.results] == [
        (r.id, r.score) for r in cold.results
    ]

    cache._redis.store.clear()
    cache._mem.clear()
    again = _search()
    assert [(r.id, r.score) for r in again.results] == [
        (r.id, r.score) for r in cold.results
    ]


def test_a_malformed_date_is_rejected_even_on_a_warm_entry(monkeypatch):
    """Warmed under the malformed request's own key, proving validation precedes the cache-hit return."""
    from fastapi import HTTPException

    cache = _wire_search(monkeypatch)

    q_fixed, _ = main.fix_query("fintech funding")
    retrieval_q, eff_from, eff_to, auto_dealtype, _auto_industry = main._effective_intent(
        q_fixed, "not-a-date", None
    )
    content_type = main.extract_content_type(q_fixed)
    warm_key = main.search_cache_key(
        retrieval_q,
        8,
        main.facet_cache_token(
            None, auto_dealtype, None, eff_from, eff_to, content_type
        ),
    )
    asyncio.run(cache.set(warm_key, [_summary(1, 0.9)]))

    with pytest.raises(HTTPException) as exc:
        _search(from_date="not-a-date")
    assert exc.value.status_code == 400
