"""Bounds on a user-supplied search query (#241).

A megabyte of `q` used to reach the transformer tokenizers and be embedded
verbatim in a Redis key, so an unauthenticated caller could burn encode CPU
and create multi-KB cache keys. The fix has three separately-falsifiable parts,
and each is asserted here on its own:

  * the HTTP surface rejects an over-long `q` with 422 instead of truncating,
  * the retrieval cache keys stay short whatever the query is,
  * nothing handed to the dense encoder or the cross-encoder is unbounded.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FusionQuery, SparseVector

from app import auth, main
from app.config import config
from app.main import SourceArticle


def _run(coro):
    return asyncio.run(coro)


_client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _public_rate_limiter(monkeypatch):
    """The public endpoints fail closed (503) without a limiter store, so give
    them a counting stub; the limiter itself is covered in test_main_http."""
    counters: dict[str, int] = {}

    class _FakeRateRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            counters[key] = counters.get(key, 0) + 1
            return counters[key]

    monkeypatch.setattr(auth, "_rate_client", _FakeRateRedis())
    return counters


_MISS = object()


class _FakeCache:
    """Records every key it is asked for and every key it is handed."""

    def __init__(self, get_result=_MISS):
        self.get_result = get_result
        self.store: dict = {}
        self.gets: list[str] = []
        self.sets: list[str] = []

    async def get(self, key):
        self.gets.append(key)
        if self.get_result is not _MISS:
            return self.get_result
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append(key)


class _Arr:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _RecordingDense:
    """Dense encoder stand-in that records the exact text it was given."""

    def __init__(self, vec=(0.1, 0.2)):
        self.vec = list(vec)
        self.seen: list[str] = []

    def encode(self, text):
        self.seen.append(text)
        return _Arr(self.vec)


class _SparseEmb:
    def __init__(self):
        self.indices = _Arr([1])
        self.values = _Arr([0.5])


class _RecordingSparse:
    def __init__(self):
        self.emb = _SparseEmb()
        self.seen: list[list[str]] = []

    def embed(self, texts):
        self.seen.append(list(texts))
        return iter([self.emb])


class _RecordingReranker:
    """Cross-encoder stand-in that records the (query, passage) pairs."""

    def __init__(self, logits=(1.0, 1.0)):
        self.logits = list(logits)
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.logits[: len(self.pairs)]


class _Point:
    def __init__(self, id_, payload, score=1.0):
        self.id = id_
        self.payload = payload
        self.score = score


class _QueryResult:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    def __init__(self, points=None):
        self.points = points or []
        self.query_points_calls = []

    async def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return _QueryResult(self.points)

    async def retrieve(self, **kwargs):
        return _QueryResult([])

    async def scroll(self, **kwargs):
        return [], None


def _article(id_: int, score: float) -> SourceArticle:
    return SourceArticle(
        id=id_, title=f"T{id_}", url=f"https://example.com/{id_}", score=score
    )


def _body_article(id_: int, body: str, score: float = 0.1) -> SourceArticle:
    return SourceArticle(
        id=id_, title=f"T{id_}", url=f"https://example.com/{id_}",
        body=body, score=score,
    )


def _wire_encoders(monkeypatch, dense=None, sparse=None, reranker=None):
    """Install recording encoders and a Qdrant stand-in on the app state."""
    dense = dense or _RecordingDense()
    sparse = sparse or _RecordingSparse()
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    monkeypatch.setitem(main.state, "qdrant", _FakeQdrant([_Point(1, {"title": "T", "url": "u"})]))
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    return dense, sparse


# --- the HTTP bound --------------------------------------------------------


def _detail_message(response) -> str:
    return json.dumps(response.json())


@pytest.mark.parametrize("length", [config.SEARCH_QUERY_MAX_CHARS + 1, 1000, 8000, 32000])
def test_search_rejects_an_over_long_query_and_names_the_limit(length):
    """A long q must be refused rather than truncated, and the response has to
    name the limit so the UI can explain the refusal instead of showing a
    generic failure. (1MB is not exercised through the HTTP client: httpx
    refuses to even build a URL that long, so the 1MB case is proven against
    hybrid_search/retrieve_and_rerank directly below.)"""
    resp = _client.get("/search", params={"q": "A" * length})

    assert resp.status_code == 422
    body = _detail_message(resp)
    assert str(config.SEARCH_QUERY_MAX_CHARS) in body
    # The error must point at `q` itself, not some unrelated field.
    assert any("q" in loc for err in resp.json()["detail"] for loc in err["loc"])


def test_search_accepts_a_query_at_the_limit(monkeypatch):
    """Boundary: the limit is inclusive. A query exactly at it — and one just
    under it — must still be served normally rather than refused."""
    _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setattr(main, "record_search", _noop_record_search)
    monkeypatch.setattr(main, "apply_click_boost", _passthrough_boost)
    monkeypatch.setattr(main, "diversify", lambda results, k, **kw: results)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)
    monkeypatch.setattr(main, "retrieve_with_auto_facet_fallback", _fake_retrieve)

    at_limit = _client.get("/search", params={"q": "A" * config.SEARCH_QUERY_MAX_CHARS})
    assert at_limit.status_code == 200

    under = _client.get("/search", params={"q": "A" * (config.SEARCH_QUERY_MAX_CHARS - 1)})
    assert under.status_code == 200


async def _fake_retrieve(q, top_k, qfilter=None, **kwargs):
    # retrieve_with_auto_facet_fallback returns the reranked set plus the
    # facets it actually applied (used to detect a relaxed auto facet).
    return ([_article(1, 0.9)], kwargs.get("industry"), kwargs.get("dealtype"),
            kwargs.get("content_type"))


def test_search_refuses_one_char_over_the_limit():
    resp = _client.get("/search", params={"q": "A" * (config.SEARCH_QUERY_MAX_CHARS + 1)})

    assert resp.status_code == 422


async def _noop_record_search(*args, **kwargs):
    return None


async def _passthrough_boost(q, results):
    return results


async def _passthrough_rerank(query, results):
    return results


# --- the cache-key bound ---------------------------------------------------


def test_a_short_query_still_names_itself_in_the_key():
    """The bound must not rewrite normal keys: a 5-char query is unchanged, so
    the cache entries already in Redis keep hitting."""
    assert main._cache_query_component("deals") == "deals"


def test_a_long_query_is_replaced_by_a_digest_not_cut_short():
    long_q = "A" * 10_000
    component = main._cache_query_component(long_q)

    assert len(component) < len(long_q)
    assert component.startswith("h:")
    assert "A" * config.CACHE_KEY_QUERY_MAX_CHARS not in component


def test_two_different_long_queries_do_not_share_a_cache_key():
    """Truncating the text instead of digesting it would collide every query
    with the same long prefix onto one key and serve the wrong results."""
    prefix = "A" * (config.CACHE_KEY_QUERY_MAX_CHARS + 1)
    a = main._cache_query_component(prefix + "one")
    b = main._cache_query_component(prefix + "two")

    assert a != b


def test_the_same_long_query_always_maps_to_the_same_component():
    long_q = "B" * 50_000

    assert main._cache_query_component(long_q) == main._cache_query_component(long_q)


def test_the_digest_does_not_depend_on_the_process():
    """A per-process salt (e.g. hash()) would make the key non-deterministic
    and every request would miss the cache."""
    import hashlib

    long_q = "C" * 20_000
    expected = f"h:{hashlib.sha256(long_q.encode('utf-8')).hexdigest()[:32]}"

    assert main._cache_query_component(long_q) == expected


def test_megabyte_query_produces_a_bounded_vec_key(monkeypatch):
    """The key actually handed to Redis, measured — not just "it still works"."""
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main.config, "EMBED_MODEL", "dense-model")
    monkeypatch.setattr(main.config, "SPARSE_MODEL", "sparse-model")
    dense, _sparse = _wire_encoders(monkeypatch)

    _run(main.hybrid_search("A" * 1_000_000, 4))

    assert cache.gets and cache.sets
    for key in (cache.gets[0], cache.sets[0]):
        assert len(key) < 300, f"cache key grew to {len(key)} chars: {key[:80]!r}"
        assert key.startswith("vec:dense-model|sparse-model:h:")
    # The key is bounded because the text is digested, not because the text was
    # quietly dropped: the encoder still got a full-length (clamped) query.
    assert dense.seen and len(dense.seen[0]) == config.SEARCH_QUERY_MAX_CHARS


def test_megabyte_query_produces_a_bounded_retrieve_key(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "rerank", _passthrough_rerank)
    monkeypatch.setattr(main.config, "ENABLE_ENTITY_BOOST", False)

    _run(main.retrieve_and_rerank("A" * 1_000_000, 4, None))

    assert cache.gets
    assert len(cache.gets[0]) < 300, f"cache key grew to {len(cache.gets[0])} chars"
    assert cache.gets[0].startswith("retrieve:h:")


def test_long_but_accepted_query_produces_a_bounded_search_key(monkeypatch):
    """A 300-char q is legal (under the 512 ceiling) yet long enough that its
    key must still be digested rather than spelled out."""
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "record_search", _noop_record_search)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)
    _wire_encoders(monkeypatch)

    monkeypatch.setattr(main, "retrieve_with_auto_facet_fallback", _fake_retrieve)

    resp = _client.get("/search", params={"q": "A" * 300})

    assert resp.status_code == 200
    assert cache.gets
    key = cache.gets[0]
    assert len(key) < 300, f"cache key grew to {len(key)} chars"
    assert key.startswith("search:h:")


# --- nothing unbounded reaches a tokenizer --------------------------------


def test_dense_encode_never_sees_more_than_the_limit(monkeypatch):
    dense, sparse = _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "cache", _FakeCache())

    _run(main.hybrid_search("A" * 1_000_000, 4))

    assert dense.seen == ["A" * config.SEARCH_QUERY_MAX_CHARS]
    # The sparse encoder tokenizes the same text on the same miss.
    assert sparse.seen[0] == ["A" * config.SEARCH_QUERY_MAX_CHARS]


def test_cross_encoder_never_sees_more_than_the_limit(monkeypatch):
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)

    _run(main.rerank("A" * 1_000_000, [_article(1, 0.9), _article(2, 0.8)]))

    assert reranker.pairs
    for query_side, _passage in reranker.pairs:
        assert len(query_side) == config.SEARCH_QUERY_MAX_CHARS


def test_body_rescue_never_sees_more_than_the_limit(monkeypatch):
    """body_rescue is a SECOND cross-encoder call site, driven by chat with a
    message of up to MAX_CONTENT_LEN (8000). The clamp in rerank() is a local
    and cannot reach it, so it needs its own bound and its own assertion."""
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.9)
    # Only articles with a body are scored, and the top score must be under the
    # threshold or body_rescue returns before it ever builds a pair.
    articles = [_body_article(1, "alpha body"), _body_article(2, "beta body")]

    _run(main.body_rescue("alpha " + "A" * 8000, articles))

    assert reranker.pairs, "body_rescue must actually have reached the reranker"
    for query_side, passage in reranker.pairs:
        assert len(query_side) <= config.SEARCH_QUERY_MAX_CHARS
        # The passage side is the article window and is never truncated by this
        # bound — clamping the query must not cost the window its context.
        assert "body" in passage


def test_a_normal_query_reaches_the_encoders_unharmed(monkeypatch):
    """Guard against over-correcting: the clamp must not disturb a short query,
    and the passage side of the pair must never be truncated."""
    dense, _sparse = _wire_encoders(monkeypatch)
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)
    monkeypatch.setattr(main, "cache", _FakeCache())

    _run(main.hybrid_search("fintech funding", 4))
    _run(main.rerank("fintech funding", [_article(1, 0.9), _article(2, 0.8)]))

    assert dense.seen == ["fintech funding"]
    assert [q for q, _ in reranker.pairs] == ["fintech funding", "fintech funding"]


def test_the_vec_key_actually_still_round_trips(monkeypatch):
    """A digested key must remain a usable key: the value written under it has
    to be readable back on the next identical query, and a different long query
    must still miss rather than read someone else's vectors."""
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    _wire_encoders(monkeypatch)
    long_a = "A" * 200 + " alpha"
    long_b = "A" * 200 + " beta"

    _run(main.hybrid_search(long_a, 4))
    _run(main.hybrid_search(long_a, 4))
    dense_a, _ = _wire_encoders(monkeypatch)
    _run(main.hybrid_search(long_b, 4))

    assert cache.gets[0] == cache.gets[1], "an identical long query must reuse its key"
    assert cache.gets[0] != cache.gets[2], "a different long query must not reuse it"
    assert len(dense_a.seen) == 1, "only the distinct query needed a fresh encode"


def test_prefetch_still_receives_the_cached_vector(monkeypatch):
    """The digested key must not break the retrieval itself: the Qdrant call
    still needs a real dense vector and a real sparse vector."""
    qdrant = _FakeQdrant([_Point(1, {"title": "T", "url": "u"})])
    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setitem(main.state, "model", _RecordingDense((0.4, 0.5)))
    monkeypatch.setitem(main.state, "sparse_model", _RecordingSparse())
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    articles = _run(main.hybrid_search("Z" * 900, 4))

    assert [a.id for a in articles] == [1]
    kwargs = qdrant.query_points_calls[0]
    assert isinstance(kwargs["query"], FusionQuery)
    assert kwargs["prefetch"][0].query == [0.4, 0.5]
    assert isinstance(kwargs["prefetch"][1].query, SparseVector)
