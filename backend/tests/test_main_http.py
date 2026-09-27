"""HTTP-level tests for the /search, /facets, /analytics/click and
/analytics/summary endpoints of app.main (cache hit/miss wiring, qdrant/redis
error mapping, and analytics beacons)."""

import asyncio
import os

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Filter

from app import auth, main
from app.config import config
from app.main import SourceArticle, SourceSummary


async def _noop_async(*args, **kwargs):
    return None



def _via_local_proxy(app):
    """Present requests to `app` as if they arrived from a reverse proxy on this
    host -- nginx forwarding to 127.0.0.1, as the reference deploy in setup.sh
    does -- instead of the TestClient's default non-IP peer."""

    async def wrapper(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": ("127.0.0.1", 40000)}
        await app(scope, receive, send)

    return wrapper


_client = TestClient(main.app, raise_server_exceptions=False)

_MISS = object()


@pytest.fixture(autouse=True)
def _public_rate_limiter(monkeypatch):
    """Install a working in-memory limiter store for the public endpoints.

    /search, /facets and /analytics/click now fail CLOSED (503) when the
    limiter's Redis is unreachable, so the tests that are about search wiring
    rather than rate limiting get a counting stub instead of a real Redis. It is
    rebuilt per test, so no counter leaks between cases.
    """
    counters: dict[str, int] = {}

    class _FakeRateRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            counters[key] = counters.get(key, 0) + 1
            return counters[key]

    monkeypatch.setattr(auth, "_rate_client", _FakeRateRedis())

    return counters


def _cached_search_client(monkeypatch):
    """Wire /search to a pure cache hit so the limiter is the only variable."""
    monkeypatch.setattr(main, "cache", _FakeCache(get_result=[_summary_dict(1, 0.9)]))
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)

    async def fake_record_search(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "record_search", fake_record_search)


def test_search_over_the_limit_is_rejected_with_429(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 2)
    _cached_search_client(monkeypatch)

    assert _client.get("/search", params={"q": "test"}).status_code == 200
    assert _client.get("/search", params={"q": "test"}).status_code == 200
    over = _client.get("/search", params={"q": "test"})

    assert over.status_code == 429
    assert over.headers["Retry-After"] == str(config.PUBLIC_RATE_WINDOW_SECONDS)


def test_search_rejects_over_long_q_and_accepts_a_normal_one(monkeypatch):
    """`q` reaches retrieval, query expansion and the reranker verbatim, so an
    unbounded q is an unbounded amount of work per request. The bound is a
    422 from validation, before any of that runs."""
    _cached_search_client(monkeypatch)

    ok = _client.get("/search", params={"q": "a" * config.SEARCH_QUERY_MAX_CHARS})
    too_long = _client.get("/search", params={"q": "a" * (config.SEARCH_QUERY_MAX_CHARS + 1)})

    assert ok.status_code == 200
    assert too_long.status_code == 422
    assert [e["loc"][-1] for e in too_long.json()["detail"]] == ["q"]


def test_search_limit_is_per_client_ip_not_one_global_bucket(monkeypatch):
    """Two clients behind the reference proxy each get their own bucket.

    AUTH_TRUST_X_FORWARDED_FOR is deliberately NOT stubbed: this asserts the
    SHIPPED default, which is what a real host runs. Behind the loopback peer
    nginx presents, the forwarded client IP is honored without any .env edit;
    stubbing the flag true here would only prove the code works under a
    deployment the operator has to configure by hand, and would hide the
    single-bucket collapse that a default of "off" actually caused.
    """
    # Precondition, so this cannot silently degrade into asserting whatever the
    # ambient config happens to be: the shipped value is "auto" (.env.example)
    # or absent (config.py's default), and any explicit true/false override
    # would make the assertion below prove something else.
    assert os.environ.get("AUTH_TRUST_X_FORWARDED_FOR", "auto").lower() == "auto", (
        "this test asserts the shipped default; unset AUTH_TRUST_X_FORWARDED_FOR or set it to 'auto'"
    )
    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 1)
    _cached_search_client(monkeypatch)
    proxied = _via_local_proxy(main.app)
    client_a = TestClient(proxied, raise_server_exceptions=False, headers={"x-forwarded-for": "1.1.1.1"})
    client_b = TestClient(proxied, raise_server_exceptions=False, headers={"x-forwarded-for": "2.2.2.2"})

    assert client_a.get("/search", params={"q": "test"}).status_code == 200
    assert client_a.get("/search", params={"q": "test"}).status_code == 429
    # A different client IP must still be served: its own first request.
    assert client_b.get("/search", params={"q": "test"}).status_code == 200


def test_search_limit_ignores_xff_from_a_client_that_is_not_behind_a_proxy(monkeypatch):
    """A direct caller cannot forge X-Forwarded-For to escape its rate-limit
    bucket -- the other side of trusting that header only for a loopback peer.

    Each request carries a DIFFERENT forged address. Reusing one forged value
    would prove nothing, since a client that always claims the same IP lands in
    the same bucket whether or not the header is honored at all."""
    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 1)
    _cached_search_client(monkeypatch)
    first = TestClient(main.app, raise_server_exceptions=False, headers={"x-forwarded-for": "9.9.9.9"})
    second = TestClient(main.app, raise_server_exceptions=False, headers={"x-forwarded-for": "8.8.8.8"})

    assert first.get("/search", params={"q": "test"}).status_code == 200
    # Same socket peer, so still the same bucket despite the new claimed IP.
    assert second.get("/search", params={"q": "test"}).status_code == 429


def test_search_fails_closed_with_503_when_the_limiter_store_is_down(monkeypatch):
    """An unrated /search is the scraping vector the limit exists to close."""
    _cached_search_client(monkeypatch)

    class _BrokenRedis:
        async def set(self, *args, **kwargs):
            raise ConnectionError("redis down")

        async def incr(self, *args, **kwargs):
            raise ConnectionError("redis down")

    monkeypatch.setattr(auth, "_rate_client", _BrokenRedis())
    assert _client.get("/search", params={"q": "test"}).status_code == 503


def test_facets_over_the_limit_is_rejected_with_429(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_FACETS_RATE_PER_MIN", 1)
    monkeypatch.setattr(main, "cache", _FakeCache(get_result={"industry": [], "dealtype": []}))

    assert _client.get("/facets").status_code == 200
    assert _client.get("/facets").status_code == 429


def test_exhausting_the_search_limit_does_not_spend_the_facets_budget(monkeypatch):
    """The per-endpoint limits are separate budgets, not one shared counter.

    The limiter key is public:rl:<action>:<client ip>. If the action segment
    were ever dropped, a client that burned its /search allowance would also
    be throttled on /facets and its click beacons, and a runaway /ready prober
    could throttle search -- with every other limit test still green, since
    each of them only ever exhausts one endpoint at a time.
    """

    class _BothEndpointsCache:
        """One cache serving both routes: facets wants a mapping, search a list."""

        async def get(self, key):
            if key == main.FACETS_CACHE_KEY:
                return {"industry": [], "dealtype": []}
            return [_summary_dict(1, 0.9)]

        async def set(self, key, value, ttl=None):
            return None

    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 1)
    monkeypatch.setattr(config, "PUBLIC_FACETS_RATE_PER_MIN", 1)
    _cached_search_client(monkeypatch)
    monkeypatch.setattr(main, "cache", _BothEndpointsCache())

    assert _client.get("/search", params={"q": "test"}).status_code == 200
    assert _client.get("/search", params={"q": "test"}).status_code == 429
    # The search allowance is spent, but facets has its own.
    assert _client.get("/facets").status_code == 200


def test_analytics_click_over_the_limit_is_rejected_with_429(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_CLICK_RATE_PER_MIN", 1)
    monkeypatch.setattr(main, "record_click", _noop_async)

    assert _client.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 200
    assert _client.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 429


def _run(coro):
    return asyncio.run(coro)


def _summary_dict(id_: int, score: float = 0.5) -> dict:
    return {
        "id": id_,
        "title": f"Title {id_}",
        "url": f"https://example.com/{id_}",
        "published_date": "2025-01-10",
        "category": "News",
        "summary": f"summary {id_}",
        "score": score,
        "author_names": ["A"],
        "industry_names": ["Fintech"],
        "dealtype_names": ["Funding"],
    }


def _article(id_: int, score: float) -> SourceArticle:
    return SourceArticle(
        id=id_,
        title=f"Title {id_}",
        url=f"https://example.com/{id_}",
        summary=f"summary {id_}",
        score=score,
    )


class _FakeCache:
    """In-memory stand-in for the HybridCache: async get/set, with a fixed
    get() result or a get() error optional."""

    def __init__(self, get_result=_MISS, get_error=None):
        self.get_result = get_result
        self.get_error = get_error
        self.store: dict = {}
        self.sets: list = []

    async def get(self, key):
        if self.get_error is not None:
            raise self.get_error
        if self.get_result is not _MISS:
            return self.get_result
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append((key, value))


# --- /search ---


async def _passthrough_boost(q, results):
    return results


def test_search_cache_hit_returns_cached_summaries(monkeypatch):
    records = []
    cached = [_summary_dict(1, 0.9), _summary_dict(2, 0.7)]

    async def fake_record_search(*args, **kwargs):
        records.append((args, kwargs))

    monkeypatch.setattr(main, "cache", _FakeCache(get_result=cached))
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: "weak note")
    monkeypatch.setattr(main, "record_search", fake_record_search)

    resp = _run(main.search(q="fintech funding", top_k=8, industry=None, dealtype=None,
                            author=None, content_type=None, from_date=None, to_date=None))

    assert resp.cached is True
    assert [r.id for r in resp.results] == [1, 2]
    assert all(isinstance(r, SourceSummary) for r in resp.results)
    assert resp.note == "weak note"
    assert len(records) == 1
    assert records[0][0][0] == "fintech funding"
    assert records[0][0][1] == 2
    assert records[0][1]["cached"] is True
    assert records[0][1]["filtered"] is False


def test_search_cache_miss_runs_full_pipeline(monkeypatch):
    records = []
    boost_calls = []
    div_calls = []
    cache = _FakeCache()
    articles = [_article(1, 0.9), _article(2, 0.7)]

    async def fake_retrieve(q, top_k, qfilter, need_body=False):
        return articles

    async def fake_boost(q, results):
        boost_calls.append((q, results))
        return results

    def fake_diversify(results, eff_top_k, **kwargs):
        div_calls.append((results, eff_top_k))
        return results

    async def fake_record_search(*args, **kwargs):
        records.append((args, kwargs))

    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "apply_click_boost", fake_boost)
    monkeypatch.setattr(main, "diversify", fake_diversify)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)

    resp = _run(main.search(q="fintech funding", top_k=8, industry=None, dealtype=None,
                            author=None, content_type=None, from_date=None, to_date=None))

    assert boost_calls, "apply_click_boost should have been called"
    assert div_calls, "diversify should have been called"
    assert resp.cached is False
    assert [r.id for r in resp.results] == [1, 2]
    assert all(isinstance(r, SourceSummary) for r in resp.results)
    assert all("body" not in r.model_dump() for r in resp.results)
    assert len(cache.sets) == 1
    (stored_key, stored_value) = cache.sets[0]
    assert stored_key.startswith("search:")
    assert all("body" not in d for d in stored_value)
    assert records[0][1]["cached"] is False
    assert records[0][0][1] == 2


def test_search_cache_miss_skips_boost_and_diversity_when_disabled(monkeypatch):
    boost_calls = []
    div_calls = []

    async def fake_retrieve(q, top_k, qfilter, need_body=False):
        return [_article(1, 0.9)]

    async def fake_boost(q, results):
        boost_calls.append(q)
        return results

    def fake_diversify(results, eff_top_k, **kwargs):
        div_calls.append(eff_top_k)
        return results

    async def fake_record_search(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "apply_click_boost", fake_boost)
    monkeypatch.setattr(main, "diversify", fake_diversify)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)

    resp = _run(main.search(q="fintech funding", top_k=8, industry=None, dealtype=None,
                            author=None, content_type=None, from_date=None, to_date=None))

    assert resp.cached is False
    assert boost_calls == []
    assert div_calls == []
    assert [r.id for r in resp.results] == [1]


def test_search_passes_built_facet_filter_to_retrieve(monkeypatch):
    captured = {}

    async def fake_retrieve(q, top_k, qfilter, need_body=False):
        captured["qfilter"] = qfilter
        return [_article(1, 0.9), _article(2, 0.7)]

    async def fake_record_search(*args, **kwargs):
        pass

    async def fake_temporal_fallback(results, top_k, from_date, to_date, industry, dealtype, author, need_body):
        return results

    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "_temporal_date_fallback", fake_temporal_fallback)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)

    resp = _run(main.search(q="fintech funding", top_k=8, industry="Fintech", dealtype=None,
                            author=None, content_type=None, from_date="2025-01-01", to_date=None))

    qfilter = captured["qfilter"]
    assert isinstance(qfilter, Filter)
    assert {c.key for c in qfilter.must} == {"industry_names", "published_date"}
    assert resp.cached is False
    assert [r.id for r in resp.results] == [1, 2]


def test_search_cache_miss_does_not_cache_empty_results(monkeypatch):
    """Regression: an empty result set was cached and then replayed as
    authoritative 'no results' for the whole TTL, so a date-filtered query that
    transiently retrieved nothing kept returning nothing for minutes."""
    cache = _FakeCache()

    async def fake_retrieve(q, top_k, qfilter, need_body=False):
        return []

    async def fake_record_search(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "apply_click_boost", _passthrough_boost)
    monkeypatch.setattr(main, "diversify", lambda results, eff_top_k, **kw: results)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)

    resp = _run(main.search(q="edtech startups 2020", top_k=8, industry=None, dealtype=None,
                            author=None, content_type=None, from_date=None, to_date=None))

    assert resp.results == []
    assert resp.cached is False
    assert cache.sets == [], "empty result sets must never be written to the cache"


def test_search_empty_results_are_not_served_from_cache(monkeypatch):
    """The whole point of the guard: a second identical query must re-run
    retrieval instead of being answered from a poisoned empty cache entry."""
    cache = _FakeCache()
    calls = []

    async def fake_retrieve(q, top_k, qfilter, need_body=False):
        calls.append(q)
        return [_article(1, 0.9)] if len(calls) > 1 else []

    async def fake_record_search(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "apply_click_boost", _passthrough_boost)
    monkeypatch.setattr(main, "diversify", lambda results, eff_top_k, **kw: results)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)

    kwargs = {"top_k": 8, "industry": None, "dealtype": None, "author": None, "content_type": None,
              "from_date": None, "to_date": None}
    first = _run(main.search(q="edtech startups 2020", **kwargs))
    second = _run(main.search(q="edtech startups 2020", **kwargs))

    assert first.results == []
    assert len(calls) == 2, "the second query must re-run retrieval, not hit the cache"
    assert [r.id for r in second.results] == [1]
    assert second.cached is False


def test_search_non_empty_results_are_still_cached(monkeypatch):
    """Guard against over-correcting: the cache must stay enabled for real
    result sets, only empty ones are skipped."""
    cache = _FakeCache()
    articles = [_article(1, 0.9)]

    async def fake_retrieve(q, top_k, qfilter, need_body=False):
        return articles

    async def fake_record_search(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "apply_click_boost", _passthrough_boost)
    monkeypatch.setattr(main, "diversify", lambda results, eff_top_k, **kw: results)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)

    _run(main.search(q="fintech funding", top_k=8, industry=None, dealtype=None,
                     author=None, content_type=None, from_date=None, to_date=None))

    assert len(cache.sets) == 1
    assert cache.sets[0][0].startswith("search:")
    assert [d["id"] for d in cache.sets[0][1]] == [1]


# --- retrieve_and_rerank caching ---


def _patch_retrieval_pipeline(monkeypatch, articles):
    """Wires retrieve_and_rerank's collaborators to fakes returning `articles`."""

    async def fake_leg(rq, top_k, qfilter):
        return list(articles)

    async def fake_rerank(q, results):
        return list(results)

    monkeypatch.setattr(main, "_retrieval_queries", lambda q: [q])
    monkeypatch.setattr(main, "_retrieval_leg", fake_leg)
    monkeypatch.setattr(main, "rerank", fake_rerank)
    monkeypatch.setattr(main, "sort_results", lambda r, recency_boost=False: r)
    monkeypatch.setattr(main, "apply_entity_boost", lambda q, r: r)


def test_retrieve_and_rerank_does_not_cache_empty_results(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    _patch_retrieval_pipeline(monkeypatch, [])

    out = _run(main.retrieve_and_rerank("edtech startups 2020", 8, None))

    assert out == []
    assert cache.sets == [], "empty result sets must never be written to the cache"
    assert cache.store == {}


def test_retrieve_and_rerank_non_empty_results_are_still_cached(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)
    _patch_retrieval_pipeline(monkeypatch, [_article(1, 0.9)])

    out = _run(main.retrieve_and_rerank("fintech funding", 8, None))

    assert [a.id for a in out] == [1]
    assert len(cache.sets) == 1
    assert cache.sets[0][0].startswith("retrieve:")
    assert "body" not in cache.sets[0][1][0]


def test_search_retrieve_error_returns_500(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setattr(main, "retrieve_and_rerank", boom)

    r = _client.get("/search", params={"q": "test"})
    assert r.status_code == 500


def test_search_cache_error_returns_500(monkeypatch):
    monkeypatch.setattr(main, "cache", _FakeCache(get_error=RuntimeError("redis down")))

    r = _client.get("/search", params={"q": "test"})
    assert r.status_code == 500


# --- /facets ---


def test_facets_cache_hit(monkeypatch):
    cached = {"industry": ["Fintech", "Healthtech"], "dealtype": ["M&A", "Funding"]}
    monkeypatch.setattr(main, "cache", _FakeCache(get_result=cached))

    async def fake_facet_values(key):
        raise AssertionError("_facet_values must not run on a cache hit")

    monkeypatch.setattr(main, "_facet_values", fake_facet_values)

    r = _client.get("/facets")
    assert r.status_code == 200
    assert r.json() == cached


def test_facets_cache_miss(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(main, "cache", cache)

    async def fake_facet_values(key):
        return {"industry_names": ["Fintech", "Healthtech"], "dealtype_names": ["M&A"]}[key]

    monkeypatch.setattr(main, "_facet_values", fake_facet_values)

    r = _client.get("/facets")
    assert r.status_code == 200
    assert r.json() == {"industry": ["Fintech", "Healthtech"], "dealtype": ["M&A"]}
    assert cache.sets == [
        (main.FACETS_CACHE_KEY, {"industry": ["Fintech", "Healthtech"], "dealtype": ["M&A"]})
    ]


def test_facets_qdrant_error_returns_500(monkeypatch):
    async def fake_facet_values(key):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setattr(main, "_facet_values", fake_facet_values)

    r = _client.get("/facets")
    assert r.status_code == 500


# --- /analytics/click ---


@pytest.mark.parametrize(
    "event,expected",
    [
        (main.ClickEvent(query="fintech", position=2, id=42), ("fintech", 2, 42)),
        (main.ClickEvent(query="fintech", position=2, id=None), ("fintech", 2, None)),
    ],
)
def test_analytics_click(monkeypatch, event, expected):
    calls = []

    async def fake_record_click(*args):
        calls.append(args)

    monkeypatch.setattr(main, "record_click", fake_record_click)

    resp = _run(main.analytics_click(event))
    assert resp == {"ok": True}
    assert calls == [expected]


# --- /analytics/summary ---


def test_analytics_summary(monkeypatch):
    async def fake_analytics_data():
        return {"searches_total": 5}

    monkeypatch.setattr(main, "analytics_data", fake_analytics_data)

    assert _run(main.get_analytics_summary()) == {"searches_total": 5}
