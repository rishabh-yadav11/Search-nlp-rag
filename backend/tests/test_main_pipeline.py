"""Unit tests for the internal (non-HTTP) pipeline functions in app.main:
embedding, intent rewrite, retrieval, body rescue/attach, and facet values."""

import asyncio
import math
import types

import pytest
from qdrant_client.models import Fusion, FusionQuery, SparseVector

from app import main
from app.main import SourceArticle

_MISS = object()


def _run(coro):
    return asyncio.run(coro)


def _article(id_: int, score: float, **kwargs) -> SourceArticle:
    defaults = {"title": f"Title {id_}", "url": f"https://example.com/{id_}", "score": score}
    defaults.update(kwargs)
    return SourceArticle(id=id_, **defaults)


class _Arr:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeDense:
    def __init__(self, vec):
        self.vec = vec
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return _Arr(self.vec)


class _SparseEmb:
    def __init__(self, indices, values):
        self.indices = _Arr(indices)
        self.values = _Arr(values)


class _FakeSparse:
    def __init__(self, indices, values):
        self.emb = _SparseEmb(indices, values)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return iter([self.emb])


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
        self.retrieve_result = []
        self.query_points_calls = []
        self.retrieve_calls = []

    async def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return _QueryResult(self.points)

    async def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return self.retrieve_result


class _FakeCache:
    def __init__(self, get_result=_MISS):
        self.get_result = get_result
        self.store = {}
        self.sets = []
        self.gets = []

    async def get(self, key):
        self.gets.append(key)
        if self.get_result is not _MISS:
            return self.get_result
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append((key, value, ttl))


class _ProbeLock:
    def __init__(self):
        self.entries = 0
        self.exits = 0

    async def __aenter__(self):
        self.entries += 1
        return self

    async def __aexit__(self, *exc):
        self.exits += 1
        return False


class _FakeReranker:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self.logits


class _FakeFacetClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _with_facet(qdrant: _FakeQdrant, result):
    qdrant.http = types.SimpleNamespace(
        collections_api=types.SimpleNamespace(api_client=_FakeFacetClient(result))
    )
    return qdrant


# --- _embed_sparse ---


def test_embed_sparse_returns_first_element():
    fake = _FakeSparse([1, 3], [0.9, 0.4])
    out = main._embed_sparse(fake, "query")
    assert fake.calls == 1
    assert out.indices.tolist() == [1, 3]
    assert out.values.tolist() == [0.9, 0.4]


# --- _effective_intent ---


def test_effective_intent_month_scoped_branch():
    rq, fd, td, dt, ind = main._effective_intent("top pharma deals of month january 2025", None, None)
    assert (rq, fd, td) == ("pharma deals", "2025-01-01", "2025-01-31")
    assert dt is None and ind is None


def test_effective_intent_user_dates_win():
    rq, fd, td, dt, ind = main._effective_intent("deals in 2025", "2024-01-01", "2024-12-31")
    assert (rq, fd, td) == ("deals in 2025", "2024-01-01", "2024-12-31")
    assert dt is None and ind is None


def test_effective_intent_no_intent_passthrough():
    rq, fd, td, dt, ind = main._effective_intent("latest deals", None, None)
    # Generic noise ('latest') is stripped so the embedding focuses on 'deals'.
    assert (rq, fd, td) == ("deals", None, None)
    assert dt is None and ind is None


def test_effective_intent_normalizes_word_numbers():
    """'top ten ipo' must retrieve exactly like 'top 10 ipo': the literal word
    'ten' would otherwise dilute the embedding/rerank match against titles like
    'Ten Sports'."""
    assert main._effective_intent("top ten ipo", None, None) == ("top 10 ipo", None, None, None, None)
    assert main._effective_intent("top ten deals", None, None) == ("top 10 deals", None, None, None, None)
    assert main._effective_intent("top ten ipo", "2024-01-01", "2024-12-31") == (
        "top 10 ipo", "2024-01-01", "2024-12-31", None, None,
    )


# --- _retrieval_queries ---


def test_retrieval_queries_year_in_review_two_legs():
    assert main._retrieval_queries("top 3 unicorns created in 2025") == [
        "Flashback 2025 unicorns created",
        "unicorns created",
    ]


def test_retrieval_queries_month_scoped_single_leg():
    assert main._retrieval_queries("top pharma deals of month january 2025") == ["pharma deals"]


def test_retrieval_queries_plain_single():
    assert main._retrieval_queries("latest deals") == ["latest deals"]


def test_retrieval_queries_dedup_when_flashback_equals_topic():
    assert main._retrieval_queries("top deals") == ["top deals"]


# --- hybrid_search ---


def test_hybrid_search_cache_miss(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    dense = _FakeDense([0.1, 0.2, 0.3])
    sparse = _FakeSparse([1, 3], [0.9, 0.4])
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    qdrant = _FakeQdrant(
        points=[
            _Point(
                1,
                {
                    "title": "T1",
                    "url": "u1",
                    "summary": "s1",
                    "body": "b1",
                    "author_names": ["A"],
                    "industry_names": ["Fin"],
                    "dealtype_names": ["M&A"],
                },
                score=0.8,
            )
        ]
    )
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "EMBED_MODEL", "dense-model")
    monkeypatch.setattr(main.config, "SPARSE_MODEL", "sparse-model")
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    monkeypatch.setattr(main.config, "VECTOR_CACHE_TTL_SECONDS", 123)

    articles = _run(main.hybrid_search("query", 8))

    assert dense.calls == 1
    assert sparse.calls == 1
    assert len(cache.sets) == 1
    key, value, ttl = cache.sets[0]
    assert key == "vec:dense-model|sparse-model:query"
    assert ttl == 123
    assert value["dense"] == [0.1, 0.2, 0.3]
    assert value["si"] == [1, 3]
    assert value["sv"] == [0.9, 0.4]
    assert [a.id for a in articles] == [1]
    assert articles[0].title == "T1"
    assert articles[0].body == "b1"
    assert articles[0].author_names == ["A"]

    kwargs = qdrant.query_points_calls[0]
    assert kwargs["collection_name"] == "col"
    assert kwargs["limit"] == 8
    assert kwargs["with_payload"] == main._PAYLOAD_FIELDS
    assert kwargs["query_filter"] is None
    assert isinstance(kwargs["query"], FusionQuery)
    assert kwargs["query"].fusion == Fusion.RRF
    assert len(kwargs["prefetch"]) == 2
    assert kwargs["prefetch"][0].using == "dense"
    assert kwargs["prefetch"][1].using == "sparse"
    assert kwargs["prefetch"][0].limit == 32


def test_hybrid_search_cache_hit_skips_encoding(monkeypatch):
    vec = {"dense": [0.1, 0.2], "si": [1], "sv": [0.7]}
    monkeypatch.setattr(main, "cache", _FakeCache(get_result=vec))
    dense = _FakeDense([9.9, 9.9])
    sparse = _FakeSparse([9], [9.9])
    monkeypatch.setitem(main.state, "model", dense)
    monkeypatch.setitem(main.state, "sparse_model", sparse)
    qdrant = _FakeQdrant(points=[_Point(2, {"title": "T2", "url": "u2"}, score=0.5)])
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    articles = _run(main.hybrid_search("query", 4, with_body=True))

    assert dense.calls == 0
    assert sparse.calls == 0
    assert [a.id for a in articles] == [2]
    kwargs = qdrant.query_points_calls[0]
    assert kwargs["with_payload"] is True
    assert kwargs["query"].fusion == Fusion.RRF
    assert isinstance(kwargs["prefetch"][1].query, SparseVector)


def test_hybrid_search_acquires_inference_lock_once_on_miss(monkeypatch):
    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setitem(main.state, "model", _FakeDense([0.1]))
    monkeypatch.setitem(main.state, "sparse_model", _FakeSparse([0], [0.5]))
    monkeypatch.setitem(main.state, "qdrant", _FakeQdrant(points=[_Point(1, {"title": "T", "url": "u"})]))
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    lock = _ProbeLock()
    monkeypatch.setattr(main, "inference_lock", lock)

    _run(main.hybrid_search("query", 4))

    assert lock.entries == 1
    assert lock.exits == 1


# --- body_rescue ---


def test_body_rescue_empty_articles(monkeypatch):
    fake = _FakeReranker([1.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    assert _run(main.body_rescue("query", [])) == []
    assert fake.calls == 0


def test_body_rescue_skips_when_top_score_strong(monkeypatch):
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a = _article(1, 0.5, body="has body")
    out = _run(main.body_rescue("some query", [a]))
    assert fake.calls == 0
    assert out[0].score == 0.5


def test_body_rescue_skips_stopword_only_query(monkeypatch):
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a = _article(1, 0.1, body="has body")
    out = _run(main.body_rescue("a the of and", [a]))
    assert fake.calls == 0
    assert out[0].score == 0.1


def test_body_rescue_skips_when_bodies_empty(monkeypatch):
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a1 = _article(1, 0.1)
    a2 = _article(2, 0.1)
    out = _run(main.body_rescue("funding deals", [a1, a2]))
    assert fake.calls == 0
    assert [x.id for x in out] == [1, 2]


def test_body_rescue_lifts_deep_body_match_and_reorders(monkeypatch):
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)
    fake = _FakeReranker([5.0, -2.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    matching = _article(1, 0.1, body=("filler " * 100) + ("2008 crisis central banks lessons " * 40))
    unrelated = _article(2, 0.1, body=("weather and markets " * 200))
    out = _run(main.body_rescue("lessons 2008 crisis central banks", [matching, unrelated]))
    assert fake.calls == 1
    assert [a.id for a in out] == [1, 2]
    assert abs(out[0].score - 1.0 / (1.0 + math.exp(-5.0))) < 1e-9
    assert abs(out[1].score - max(0.1, 1.0 / (1.0 + math.exp(2.0)))) < 1e-9


def test_body_rescue_reranker_error_propagates(monkeypatch):
    monkeypatch.setattr(main.config, "BODY_RESCUE_THRESHOLD", 0.2)

    class _Boom:
        def predict(self, pairs):
            raise RuntimeError("model load failed")

    monkeypatch.setitem(main.state, "reranker", _Boom())
    a = _article(1, 0.1, body="some body with funding deals")
    with pytest.raises(RuntimeError):
        _run(main.body_rescue("funding deals", [a]))


# --- _attach_bodies ---


def test_attach_bodies_empty_skips_retrieve(monkeypatch):
    qdrant = _FakeQdrant()
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    _run(main._attach_bodies([]))
    assert qdrant.retrieve_calls == []


def test_attach_bodies_sets_payloads_for_found_ids(monkeypatch):
    qdrant = _FakeQdrant()
    qdrant.retrieve_result = [_Point(1, {"body": "body1"}), _Point(3, {"body": "body3"})]
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")
    arts = [_article(1, 0.5), _article(2, 0.5), _article(3, 0.5)]
    _run(main._attach_bodies(arts))
    assert qdrant.retrieve_calls[0]["collection_name"] == "col"
    assert qdrant.retrieve_calls[0]["ids"] == [1, 2, 3]
    assert qdrant.retrieve_calls[0]["with_payload"] == ["body"]
    assert arts[0].body == "body1"
    assert arts[1].body == ""
    assert arts[2].body == "body3"


def test_attach_bodies_none_payload_gets_empty_body(monkeypatch):
    qdrant = _FakeQdrant()
    qdrant.retrieve_result = [_Point(1, None)]
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    arts = [_article(1, 0.5)]
    _run(main._attach_bodies(arts))
    assert arts[0].body == ""


def test_attach_bodies_retrieve_error_propagates(monkeypatch):
    class _Boom:
        async def retrieve(self, **kwargs):
            raise RuntimeError("qdrant down")

    monkeypatch.setitem(main.state, "qdrant", _Boom())
    with pytest.raises(RuntimeError):
        _run(main._attach_bodies([_article(1, 0.5)]))


# --- _retrieval_leg ---


def test_retrieval_leg_expands_query_and_uses_rerank_candidates(monkeypatch):
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["qfilter"] = qfilter
        return []

    monkeypatch.setattr(main, "expand_query", lambda q: "EXPANDED " + q)
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main.config, "ENABLE_QUERY_EXPANSION", True)
    monkeypatch.setattr(main.config, "RERANK_CANDIDATES", 12)

    _run(main._retrieval_leg("funding deals", 5, None))

    assert captured["query"] == "EXPANDED funding deals"
    assert captured["top_k"] == 12
    assert captured["qfilter"] is None


def test_retrieval_leg_skips_expansion_for_flashback(monkeypatch):
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["query"] = query
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(main, "expand_query", lambda q: "EXPANDED " + q)
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main.config, "ENABLE_QUERY_EXPANSION", True)
    monkeypatch.setattr(main.config, "RERANK_CANDIDATES", 12)

    _run(main._retrieval_leg("Flashback 2025 deals", 5, None))

    assert captured["query"] == "Flashback 2025 deals"


def test_retrieval_leg_no_expansion_when_disabled(monkeypatch):
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["query"] = query
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(main, "expand_query", lambda q: "EXPANDED " + q)
    monkeypatch.setattr(main, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(main.config, "ENABLE_QUERY_EXPANSION", False)
    monkeypatch.setattr(main.config, "RERANK_CANDIDATES", 12)

    _run(main._retrieval_leg("funding deals", 20, None))

    assert captured["query"] == "funding deals"
    assert captured["top_k"] == 20


# --- source_context ---


def test_source_context_all_facets_and_body():
    a = SourceArticle(
        id=1,
        title="T",
        url="u",
        published_date="2024-01-01",
        summary="sum",
        body="body text",
        author_names=["Alice", "Bob"],
        industry_names=["Fintech"],
        dealtype_names=["Funding"],
        score=0.5,
    )
    out = main.source_context(a, 1)
    assert "[1] T (2024-01-01 | Authors: Alice, Bob | Industry: Fintech | Dealtype: Funding)" in out
    assert "sum" in out
    assert "body text" in out


def test_source_context_no_facets_no_summary():
    a = SourceArticle(id=2, title="T", url="u", published_date=None, score=0.5)
    out = main.source_context(a, 3)
    assert out == "[3] T (n/a)"


def test_source_context_truncates_body_to_body_limit():
    a = SourceArticle(id=1, title="T", url="u", summary="s", body="x" * 100, score=0.5)
    out = main.source_context(a, 1, body_limit=20)
    assert len(out.split("\n")[-1]) == 20


# --- _facet_values ---


def test_facet_values_sorted_and_request_kwargs(monkeypatch):
    qdrant = _with_facet(
        _FakeQdrant(), {"result": {"hits": [{"value": "Finance"}, {"value": "TMT"}, {"value": "General"}]}}
    )
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    monkeypatch.setattr(main.config, "QDRANT_COLLECTION", "col")

    out = _run(main._facet_values("industry_names"))

    assert out == ["Finance", "General", "TMT"]
    req = qdrant.http.collections_api.api_client.calls[0]
    assert req["type_"] is dict
    assert req["method"] == "POST"
    assert req["url"] == "/collections/{collection_name}/facet"
    assert req["path_params"] == {"collection_name": "col"}
    assert req["json"] == {"key": "industry_names", "limit": main.FACETS_LIMIT}


def test_facet_values_filters_non_string_hits(monkeypatch):
    qdrant = _with_facet(_FakeQdrant(), {"result": {"hits": [{"value": "Finance"}, {"value": 42}, {"value": None}]}})
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    assert _run(main._facet_values("industry_names")) == ["Finance"]


def test_facet_values_empty_or_none_result(monkeypatch):
    qdrant = _with_facet(_FakeQdrant(), {})
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    assert _run(main._facet_values("industry_names")) == []
    qdrant2 = _with_facet(_FakeQdrant(), None)
    monkeypatch.setitem(main.state, "qdrant", qdrant2)
    assert _run(main._facet_values("industry_names")) == []


def test_facet_values_error_propagates(monkeypatch):
    class _Boom:
        async def request(self, **kwargs):
            raise RuntimeError("qdrant down")

    qdrant = _FakeQdrant()
    qdrant.http = types.SimpleNamespace(collections_api=types.SimpleNamespace(api_client=_Boom()))
    monkeypatch.setitem(main.state, "qdrant", qdrant)
    with pytest.raises(RuntimeError):
        _run(main._facet_values("industry_names"))


# --- _best_body_window tail branch ---


def test_best_body_window_tail_wins_on_tie():
    body = ("filler " * 60) + "alpha beta gamma"
    tokens = {"alpha", "beta", "gamma"}
    out = main._best_body_window(body, tokens, 50, 50)
    assert out == body[-50:]


# --- lifespan ---


def _stub_lifespan_deps(monkeypatch, chat_connect_error=None, auth_connect_error=None):
    """Fake every startup dependency of app.main.lifespan. Returns the fakes so
    tests can assert on their lifecycle state."""

    class FakeChatStore:
        def __init__(self):
            self.closed = False

        async def connect(self):
            if chat_connect_error:
                raise chat_connect_error
            self.connected = True

        async def close(self):
            self.closed = True

    class FakeAuthStore:
        def __init__(self):
            self.closed = False

        async def connect(self):
            if auth_connect_error:
                raise auth_connect_error
            self.connected = True

        async def close(self):
            self.closed = True

    class FakeQdrant:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        async def close(self):
            self.closed = True

    class FakeCache:
        def __init__(self):
            self.closed = False

        async def get(self, key):
            return None

        async def set(self, key, value, ttl=None):
            pass

        async def close(self):
            self.closed = True

    chat_store = FakeChatStore()
    auth_store = FakeAuthStore()
    qdrant = FakeQdrant()
    cache = FakeCache()

    monkeypatch.setattr(main, "DenseEncoder", lambda *a, **k: object())
    monkeypatch.setattr(main, "SparseTextEmbedding", lambda *a, **k: object())
    monkeypatch.setattr(main, "Reranker", lambda *a, **k: object())
    monkeypatch.setattr(main, "AsyncQdrantClient", lambda *a, **k: qdrant)
    monkeypatch.setattr(main, "AsyncOpenAI", lambda *a, **k: object())
    monkeypatch.setattr(main, "cache", cache)

    monkeypatch.setattr(main.chat_module, "ChatStore", lambda *a, **k: chat_store)
    monkeypatch.setattr(main.chat_module, "store", None)

    async def _retention_loop():
        await asyncio.sleep(3600)

    monkeypatch.setattr(main.chat_module, "retention_loop", _retention_loop)

    monkeypatch.setattr(main.auth_module, "AuthStore", lambda *a, **k: auth_store)
    monkeypatch.setattr(main.auth_module, "store", None)

    async def _bootstrap_admin():
        return None

    monkeypatch.setattr(main.auth_module, "bootstrap_admin", _bootstrap_admin)

    fixer_calls = []
    monkeypatch.setattr(main, "init_fixer", lambda *a, **k: fixer_calls.append((a, k)))

    closed = []

    async def _close_analytics():
        closed.append("analytics")

    async def _close_cost():
        closed.append("cost")

    monkeypatch.setattr(main, "close_analytics", _close_analytics)
    monkeypatch.setattr(main, "close_cost_budget", _close_cost)

    return {
        "chat_store": chat_store,
        "auth_store": auth_store,
        "qdrant": qdrant,
        "cache": cache,
        "fixer_calls": fixer_calls,
        "closed": closed,
    }


def _restore_state(orig):
    main.state.clear()
    main.state.update(orig)


def test_lifespan_startup_and_teardown(monkeypatch):
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch)

    async def scenario():
        async with main.lifespan(None):
            assert main.state["model"] is not None
            assert main.state["sparse_model"] is not None
            assert main.state["reranker"] is not None
            assert main.state["qdrant"] is deps["qdrant"]
            assert main.state["llm"] is not None
            assert main.chat_module.store is deps["chat_store"]
            assert main.auth_module.store is deps["auth_store"]
            assert deps["fixer_calls"][0][1]["max_edit"] == main.config.QUERY_FIX_MAX_EDIT
            assert "chat_retention" in main.state

    try:
        _run(scenario())
    finally:
        _restore_state(orig)

    assert deps["chat_store"].closed is True
    assert deps["auth_store"].closed is True
    assert deps["qdrant"].closed is True
    assert deps["cache"].closed is True
    assert deps["closed"] == ["analytics", "cost"]


def test_lifespan_llm_none_without_api_key(monkeypatch):
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "")
    deps = _stub_lifespan_deps(monkeypatch)

    async def scenario():
        async with main.lifespan(None):
            assert main.state["llm"] is None

    try:
        _run(scenario())
    finally:
        _restore_state(orig)

    assert deps["chat_store"].closed is True


def test_lifespan_startup_failure_propagates(monkeypatch):
    orig = dict(main.state)
    monkeypatch.setattr(main.config, "GEMINI_API_KEY", "sk-test")
    deps = _stub_lifespan_deps(monkeypatch, chat_connect_error=RuntimeError("sqlite locked"))

    async def scenario():
        async with main.lifespan(None):
            pass

    try:
        with pytest.raises(RuntimeError):
            _run(scenario())
    finally:
        _restore_state(orig)

    assert deps["chat_store"].closed is False
