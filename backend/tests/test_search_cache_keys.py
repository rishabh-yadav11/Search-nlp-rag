"""Cache-key completeness and round-trip cost for /search (#266).

The retrieval cache is documented as "deterministic for a (query, filter) pair".
It is only deterministic for a *(query, filter, configuration)* triple: the
pipeline reads ~15 config values, and any of them changes what comes back. These
tests pin both halves of that: a config change must not be able to serve a stale
entry, and the two cache keys one /search request needs must be read in a single
round trip.
"""
import asyncio

import pytest

from app import main
from app.config import config
from app.redis_cache import HybridCache

# Config values read by the retrieval/rerank pipeline. Each one changes the
# cached article set, so each one must be part of the cache key.
RETRIEVAL_KNOBS = [
    "QDRANT_COLLECTION",
    "EMBED_MODEL",
    "SPARSE_MODEL",
    "RERANK_BACKEND",
    "RERANK_MODEL",
    "RERANK_CANDIDATES",
    "RECENCY_STRENGTH",
    "RECENCY_DECAY_DAYS",
    "ENABLE_QUERY_EXPANSION",
    "ENABLE_ENTITY_BOOST",
]

# Config values read after retrieval, by /search itself. They change the summary
# page but not the underlying retrieval entry.
SEARCH_KNOBS = [
    "ENABLE_CLICK_BOOST",
    "CLICK_BOOST_MIN_CLICKS",
    "CLICK_BOOST_MIN_ARTICLE_CLICKS",
    "CLICK_BOOST_MIN_SHARE",
    "CLICK_BOOST_MULT",
    "ENABLE_DIVERSITY",
    "DIVERSITY_LAMBDA",
    "DIVERSITY_SIM_THRESHOLD",
    "ASK_MIN_SCORE",
]

# Config values that cannot change a retrieval result; they must stay out of the
# key or every entry would be invalidated by an unrelated deploy.
IRRELEVANT_KNOBS = [
    "CACHE_TTL_SECONDS",
    "VECTOR_CACHE_TTL_SECONDS",
    "REDIS_URL",
    "AUTH_DB_PATH",
    "PUBLIC_SEARCH_RATE_PER_MIN",
]

# Round trips a /search miss cost on one connection before #266:
# GET(search:...), then GET(retrieve:...) from inside the retrieval leg, then
# SET(retrieve:...) and SET(search:...).
PRE_FIX_ROUND_TRIPS = 4


def _flipped(value):
    """A different value of the same shape as ``value``."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-other"
    if isinstance(value, int):
        return value + 1
    return value + 0.5


def _article(id_, score=0.9):
    return main.SourceArticle(
        id=id_, title=f"t{id_}", url=f"u{id_}", summary="s", body="", score=score
    )


class _CountingRedis:
    """Redis double recording every command, i.e. every round trip."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.commands: list[tuple[str, object]] = []

    async def get(self, key):
        self.commands.append(("GET", key))
        return self.store.get(key)

    async def mget(self, keys):
        self.commands.append(("MGET", tuple(keys)))
        return [self.store.get(key) for key in keys]

    async def set(self, key, value, ex=None):
        self.commands.append(("SET", key))
        self.store[key] = value

    async def aclose(self):
        return None


def _wire_search(monkeypatch, articles=None, boosted=2.0):
    """Run the *real* /search -> retrieve_and_rerank path with only the external
    services (Qdrant, the cross-encoder, analytics) stubbed out, so these tests
    exercise the production key building and cache plumbing."""
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
        # A visible, deterministic effect: with the boost on, /search returns
        # different scores than the retrieval it was handed.
        if config.ENABLE_CLICK_BOOST:
            for r in results:
                r.score = r.score * boosted
        return results

    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main, "record_search", fake_record_search)
    monkeypatch.setattr(main, "apply_click_boost", fake_click_boost)

    cache = HybridCache("redis://unused", 300, 64)
    cache._redis = _CountingRedis()
    monkeypatch.setattr(main, "cache", cache)
    return cache


def _search(**kwargs):
    params = {"q": "fintech funding", "top_k": 8, "industry": None, "dealtype": None,
              "author": None, "content_type": None, "from_date": None, "to_date": None}
    params.update(kwargs)
    return asyncio.run(main.search(**params))


# --- key completeness -------------------------------------------------------


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
    """The digest must cover exactly the retrieval-affecting config, no more.

    An over-broad key (folding in the cache TTL or the Redis URL) would silently
    split one cache into many and make every entry colder.
    """
    before = main.retrieve_cache_key("q", 8, None)
    monkeypatch.setattr(config, knob, _flipped(getattr(config, knob)))
    assert main.retrieve_cache_key("q", 8, None) == before, (
        f"{knob} cannot change a retrieval result, so it must not invalidate it"
    )


def test_config_knobs_do_not_grow_the_key_without_bound():
    """The digest keeps the key short however many knobs are added."""
    key = main.retrieve_cache_key("q", 8, None)
    assert len(key) < 160, f"cache key grew unexpectedly long: {key!r}"


def test_query_top_k_and_filter_still_separate_entries():
    """The digest must not swallow the pre-existing key components."""
    f = main.build_facet_filter("Fintech", None, None, None, None)
    assert main.retrieve_cache_key("a", 8, None) != main.retrieve_cache_key("b", 8, None)
    assert main.retrieve_cache_key("a", 8, None) != main.retrieve_cache_key("a", 9, None)
    assert main.retrieve_cache_key("a", 8, None) != main.retrieve_cache_key("a", 8, f)
    assert main.search_cache_key("a", 8, "f") != main.search_cache_key("a", 8, "g")


# --- a stale entry must not be servable -------------------------------------


def test_flipping_a_config_cannot_serve_a_stale_entry(monkeypatch):
    """A config change must invalidate the entry, not merely rename its key.

    The stubbed click boost doubles scores while it is enabled, so a stale entry
    would be visible in the payload (wrong scores) and not only in the `cached`
    flag.
    """
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
    """End-to-end version of the same guarantee, through the real cache object."""
    _wire_search(monkeypatch)
    assert _search().cached is False
    assert _search().cached is True

    monkeypatch.setattr(config, knob, _flipped(getattr(config, knob)))
    assert _search().cached is False, (
        f"{knob} was flipped but the previous entry was still served"
    )


# --- round trips ------------------------------------------------------------


def test_a_search_miss_costs_three_redis_round_trips(monkeypatch):
    """A miss reads both keys in one MGET and writes each entry once.

    Before #266 the same request issued GET(search:...), then GET(retrieve:...)
    from inside the retrieval leg, then SET(retrieve:...) and SET(search:...) --
    four sequential round trips for the same work on one connection.
    """
    cache = _wire_search(monkeypatch)
    _search()

    kinds = [c[0] for c in cache._redis.commands]
    assert kinds == ["MGET", "SET", "SET"], f"unexpected Redis traffic: {kinds}"
    assert len(cache._redis.commands) < PRE_FIX_ROUND_TRIPS

    # The two reads were merged: the MGET carries one key of each kind.
    mget_keys = next(c[1] for c in cache._redis.commands if c[0] == "MGET")
    assert len(mget_keys) == 2
    assert sorted(k.split(":", 1)[0] for k in mget_keys) == ["retrieve", "search"]


def test_the_inner_leg_does_not_re_read_its_own_key(monkeypatch):
    """The prefetched value is used as-is, including when it is a miss."""
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
    """Unchanged behaviour: no empty entry, so only the MGET is issued."""
    cache = _wire_search(monkeypatch, articles=[])
    _search()
    assert [c[0] for c in cache._redis.commands] == ["MGET"]


def test_both_entries_are_still_written_with_their_own_payloads(monkeypatch):
    """The two keys are read together, not merged: they hold different values.

    The outer entry is the post-boost summary slice; the inner one is the full
    reranked article set with bodies excluded.
    """
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

    # ...and the same query against a fully cold cache reproduces them exactly.
    cache._redis.store.clear()
    cache._mem.clear()
    again = _search()
    assert [(r.id, r.score) for r in again.results] == [
        (r.id, r.score) for r in cold.results
    ]
