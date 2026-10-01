"""Bounds on a user-supplied search query.

An unbounded `q` reaches the transformer tokenizers and is embedded verbatim in
a Redis key, so an unauthenticated caller can burn encode CPU and create
multi-KB cache keys.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FusionQuery, SparseVector

from app import auth, main
from app.config import config
from app.input_hygiene import MAX_FACET_VALUE_LEN, MAX_FACET_VALUES
from app.main import SourceArticle


def _run(coro):
    return asyncio.run(coro)


_client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _public_rate_limiter(monkeypatch):
    """Public endpoints fail closed (503) without a limiter store; the limiter
    itself is covered in test_main_http."""
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

    async def get_many(self, keys):
        return [await self.get(key) for key in keys]

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append(key)


class _Arr:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _RecordingDense:
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
    dense = dense or _RecordingDense()
    sparse = sparse or _RecordingSparse()
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    monkeypatch.setitem(main.state, "qdrant", _FakeQdrant([_Point(1, {"title": "T", "url": "u"})]))
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    return dense, sparse


def _detail_message(response) -> str:
    return json.dumps(response.json())


@pytest.mark.parametrize("length", [config.SEARCH_QUERY_MAX_CHARS + 1, 1000, 8000, 32000])
def test_search_rejects_an_over_long_query_and_names_the_limit(length):
    """1MB is not exercised here: httpx refuses to build a URL that long, so
    that case is proven directly against hybrid_search/retrieve_and_rerank."""
    resp = _client.get("/search", params={"q": "A" * length})

    assert resp.status_code == 422
    body = _detail_message(resp)
    assert str(config.SEARCH_QUERY_MAX_CHARS) in body
    assert any("q" in loc for err in resp.json()["detail"] for loc in err["loc"])
    # Frontend isQueryTooLong() keys on type == "string_too_long" AND a loc
    # naming q; a change to the validation-error shape silently downgrades every
    # over-long-query UI message to a generic 422, and nothing backend-side fails.
    assert any(
        err.get("type") == "string_too_long" and "q" in err.get("loc", [])
        for err in resp.json()["detail"]
    ), resp.json()["detail"]


def test_search_accepts_a_query_at_the_limit(monkeypatch):
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
    # Mirrors retrieve_with_auto_facet_fallback's return: results plus the
    # facets it actually applied (a relaxed auto facet shows up here).
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


def test_a_short_query_still_names_itself_in_the_key():
    assert main._cache_key_component("deals") == "deals"


def test_a_long_query_is_replaced_by_a_digest_not_cut_short():
    long_q = "A" * 10_000
    component = main._cache_key_component(long_q)

    assert len(component) < len(long_q)
    assert component.startswith("h:")
    assert "A" * config.CACHE_KEY_QUERY_MAX_CHARS not in component


def test_two_different_long_queries_do_not_share_a_cache_key():
    """Truncating instead of digesting would collide same-prefix queries onto one
    key and serve the wrong results."""
    prefix = "A" * (config.CACHE_KEY_QUERY_MAX_CHARS + 1)
    a = main._cache_key_component(prefix + "one")
    b = main._cache_key_component(prefix + "two")

    assert a != b


def test_the_same_long_query_always_maps_to_the_same_component():
    long_q = "B" * 50_000

    assert main._cache_key_component(long_q) == main._cache_key_component(long_q)


def test_the_digest_does_not_depend_on_the_process():
    """A per-process salt (e.g. hash()) would make the key non-deterministic, so
    every request would miss."""
    import hashlib

    long_q = "C" * 20_000
    expected = f"h:{hashlib.sha256(long_q.encode('utf-8')).hexdigest()[:32]}"

    assert main._cache_key_component(long_q) == expected


def test_megabyte_query_produces_a_bounded_vec_key(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main.config, "EMBED_MODEL", "dense-model")
    monkeypatch.setattr(main.config, "SPARSE_MODEL", "sparse-model")
    dense, _sparse = _wire_encoders(monkeypatch)

    _run(main.hybrid_search("A" * 1_000_000, 4))

    assert cache.gets and cache.sets
    for key in (cache.gets[0], cache.sets[0]):
        assert len(key) < 300, f"cache key grew to {len(key)} chars: {key[:80]!r}"
        assert key.startswith("vec:")
        assert "dense-model" in key and "sparse-model" in key
        # Key parts are length-prefixed, so the digest is not the next segment;
        # what matters is that the key carries the clamped text's digest, not
        # the text itself.
        assert main._cache_key_component("A" * config.RETRIEVAL_QUERY_MAX_CHARS) in key
        assert "A" * 200 not in key, "the raw query reached the key"
    # Bounded because the text is digested, not dropped: the encoder still got
    # a full-length (clamped) query.
    assert dense.seen and len(dense.seen[0]) == config.RETRIEVAL_QUERY_MAX_CHARS


def test_megabyte_query_produces_a_bounded_retrieve_key(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "rerank", _passthrough_rerank)
    monkeypatch.setattr(main.config, "ENABLE_ENTITY_BOOST", False)

    _run(main.retrieve_and_rerank("A" * 1_000_000, 4, None))

    assert cache.gets
    assert len(cache.gets[0]) < 300, f"cache key grew to {len(cache.gets[0])} chars"
    assert cache.gets[0].startswith("retrieve:")
    assert main._cache_key_component("A" * 1_000_000) in cache.gets[0]
    assert "A" * 200 not in cache.gets[0], "the raw query reached the key"


def test_long_but_accepted_query_produces_a_bounded_search_key(monkeypatch):
    """300 chars is legal under the 512 ceiling yet long enough that its key must
    still be digested rather than spelled out."""
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
    assert key.startswith("search:")
    assert main._cache_key_component("A" * 300) in key
    assert "A" * 200 not in key, "the raw query reached the key"


def _max_sized_facet() -> str:
    """The longest facet that reaches a key or a Qdrant MatchAny at all; beyond
    it ``input_hygiene.split_facet_values`` returns 400 and no key is built."""
    return ",".join("B" * MAX_FACET_VALUE_LEN for _ in range(MAX_FACET_VALUES))


def test_the_query_segment_is_still_digested_when_a_facet_is_long(monkeypatch):
    """Only the QUERY segment is asserted here: with a facet at the cap the query
    must still be a digest. The facet is at the cap, not over it, because over
    the cap the request is refused."""
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "record_search", _noop_record_search)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)
    _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "retrieve_with_auto_facet_fallback", _fake_retrieve)

    resp = _client.get("/search", params={"q": "A" * 300, "author": _max_sized_facet()})

    assert resp.status_code == 200
    assert cache.gets
    key = cache.gets[0]
    expected = main._cache_key_component("A" * 300)
    assert expected.startswith("h:") and len(expected) == 34
    assert expected in key, f"key does not carry the query digest: {key[:80]!r}"
    assert "A" * 200 not in key, "the raw query reached the key"
    assert "B" * 200 not in key, "the raw facet reached the key"
    assert len(key) <= 256, f"cache key grew to {len(key)} chars: {key[:80]!r}"


def test_the_retrieve_key_digests_its_query_segment_under_a_long_filter(monkeypatch):
    """The `retrieve:` key is built at a different call site from the `search:`
    key, so the query segment needs its own assertion."""
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "rerank", _passthrough_rerank)
    monkeypatch.setattr(main.config, "ENABLE_ENTITY_BOOST", False)
    qfilter = main.build_facet_filter(industry=None, dealtype=None, author=_max_sized_facet(),
                                      from_date=None, to_date=None, content_type=None)

    _run(main.retrieve_and_rerank("A" * 300, 4, qfilter))

    assert cache.gets
    key = cache.gets[0]
    # The filter is ~1 KB, so the whole key body is digested: no caller text
    # survives and the key stays bounded.
    assert key.startswith("retrieve:sha256:")
    assert len(key) == len("retrieve:sha256:") + 64
    assert "A" * 200 not in key, "the raw query reached the key"
    assert "B" * 200 not in key, "the raw facet reached the key"


def test_dense_encode_never_sees_more_than_the_limit(monkeypatch):
    dense, sparse = _wire_encoders(monkeypatch)
    monkeypatch.setattr(main, "cache", _FakeCache())

    _run(main.hybrid_search("A" * 1_000_000, 4))

    assert dense.seen == ["A" * config.RETRIEVAL_QUERY_MAX_CHARS]
    assert sparse.seen[0] == ["A" * config.RETRIEVAL_QUERY_MAX_CHARS]


def test_cross_encoder_never_sees_more_than_the_limit(monkeypatch):
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)

    _run(main.rerank("A" * 1_000_000, [_article(1, 0.9), _article(2, 0.8)]))

    assert reranker.pairs
    for query_side, _passage in reranker.pairs:
        assert len(query_side) == config.RETRIEVAL_QUERY_MAX_CHARS


def test_body_rescue_never_sees_more_than_the_limit(monkeypatch):
    """A SECOND cross-encoder call site driven by chat's MAX_CONTENT_LEN; the
    clamp in rerank() is a local and cannot reach it."""
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.9)
    # Only articles with a body are scored, and the top score must be under the
    # threshold or body_rescue returns before it builds a pair.
    articles = [_body_article(1, "alpha body"), _body_article(2, "beta body")]

    _run(main.body_rescue("alpha " + "A" * 8000, articles))

    assert reranker.pairs, "body_rescue must actually have reached the reranker"
    for query_side, passage in reranker.pairs:
        assert len(query_side) <= config.RETRIEVAL_QUERY_MAX_CHARS
        # The passage side is the article window and is never truncated by this
        # bound — clamping the query must not cost the window its context.
        assert "body" in passage


def test_chats_accepted_message_length_is_not_silently_cut(monkeypatch):
    """Clamping chat to /search's 512 would hide terms the LLM prompt still
    contains — a relevance bug, not a performance trade."""
    from app.chat import MAX_CONTENT_LEN

    assert config.RETRIEVAL_QUERY_MAX_CHARS == MAX_CONTENT_LEN
    assert config.RETRIEVAL_QUERY_MAX_CHARS != config.SEARCH_QUERY_MAX_CHARS


def test_an_8000_char_chat_message_reaches_every_encoder_in_full(monkeypatch):
    from app.chat import MAX_CONTENT_LEN

    dense, sparse = _wire_encoders(monkeypatch)
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)
    monkeypatch.setattr(main, "cache", _FakeCache())
    # Filler ends mid-word so the message is exactly MAX_CONTENT_LEN with no
    # trailing whitespace, which canonicalisation would strip before the clamp
    # is measured. MAX_CONTENT_LEN % 5 == 0.
    message = ("zeta " * (MAX_CONTENT_LEN // 5 - 1)) + "zetas"
    assert len(message) == MAX_CONTENT_LEN

    _run(main.hybrid_search(message, 4))
    _run(main.rerank(message, [_article(1, 0.9), _article(2, 0.8)]))

    assert dense.seen == [message]
    assert sparse.seen[0] == [message]
    assert [q for q, _ in reranker.pairs] == [message, message]


def test_a_normal_query_reaches_the_encoders_unharmed(monkeypatch):
    dense, _sparse = _wire_encoders(monkeypatch)
    reranker = _RecordingReranker()
    monkeypatch.setitem(main.state, "reranker", reranker)
    monkeypatch.setattr(main, "cache", _FakeCache())

    _run(main.hybrid_search("fintech funding", 4))
    _run(main.rerank("fintech funding", [_article(1, 0.9), _article(2, 0.8)]))

    assert dense.seen == ["fintech funding"]
    assert [q for q, _ in reranker.pairs] == ["fintech funding", "fintech funding"]


def test_the_vec_key_actually_still_round_trips(monkeypatch):
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
