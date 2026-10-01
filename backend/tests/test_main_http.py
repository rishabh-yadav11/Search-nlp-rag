"""HTTP-level tests for the /search, /facets, /analytics/click and
/analytics/summary endpoints of app.main."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from _support import FakeCache, make_article
from _support import run_sync as _run
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from qdrant_client.models import Filter
from rate_limit_fake import RateLimitRedisFake

from app import auth, main
from app.analytics import AnalyticsUnavailableError
from app.config import config
from app.main import SourceSummary


async def _noop_async(*args, **kwargs):
    return None



def _via_local_proxy(app):
    """Present requests to `app` as if from a reverse proxy on this host -- nginx
    forwarding to 127.0.0.1, as setup.sh does -- not the TestClient default."""

    async def wrapper(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": ("127.0.0.1", 40000)}
        await app(scope, receive, send)

    return wrapper


_client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _public_rate_limiter(monkeypatch):
    """Install a working in-memory limiter store for the public endpoints.

    /search, /facets and /analytics/click fail CLOSED (503) when the limiter's
    Redis is unreachable, so these tests get a counting stub that models
    SET NX EX / INCR for real. Rebuilt per test, so no counter leaks.
    """
    fake = RateLimitRedisFake()
    monkeypatch.setattr(auth, "_rate_client", fake)

    return fake.counters


def _cached_search_client(monkeypatch):
    """Wire /search to a pure cache hit so the limiter is the only variable."""
    monkeypatch.setattr(main, "cache", FakeCache(get_result=[_summary_dict(1, 0.9)]))
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
    """`q` reaches retrieval, expansion and the reranker verbatim, so an
    unbounded q is unbounded work; the bound is a 422 from validation."""
    _cached_search_client(monkeypatch)

    ok = _client.get("/search", params={"q": "a" * config.SEARCH_QUERY_MAX_CHARS})
    too_long = _client.get("/search", params={"q": "a" * (config.SEARCH_QUERY_MAX_CHARS + 1)})

    assert ok.status_code == 200
    assert too_long.status_code == 422
    assert [e["loc"][-1] for e in too_long.json()["detail"]] == ["q"]


def test_search_limit_is_per_client_ip_not_one_global_bucket(monkeypatch):
    """Two clients behind the reference proxy each get their own bucket.

    AUTH_TRUST_X_FORWARDED_FOR is deliberately NOT stubbed: this asserts the
    SHIPPED default, which is what a real host runs.
    """
    # Precondition, so this cannot silently degrade into asserting whatever the
    # ambient config happens to be.
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
    assert client_b.get("/search", params={"q": "test"}).status_code == 200


def test_search_limit_ignores_xff_from_a_client_that_is_not_behind_a_proxy(monkeypatch):
    """A direct caller cannot forge X-Forwarded-For to escape its rate-limit
    bucket -- the other side of trusting that header only for a loopback peer.

    Each request forges a DIFFERENT address; a reused claim would land in the
    same bucket either way."""
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
    monkeypatch.setattr(main, "cache", FakeCache(get_result={"industry": [], "dealtype": []}))

    assert _client.get("/facets").status_code == 200
    assert _client.get("/facets").status_code == 429


def test_exhausting_the_search_limit_does_not_spend_the_facets_budget(monkeypatch):
    """The per-endpoint limits are separate budgets, not one shared counter.

    The limiter key is public:rl:<action>:<client ip>: dropping the action
    segment would let one endpoint's exhausted budget throttle the others.
    """

    class _BothEndpointsCache:
        """One cache serving both routes: facets wants a mapping, search a list."""

        async def get(self, key):
            if key == main.FACETS_CACHE_KEY:
                return {"industry": [], "dealtype": []}
            return [_summary_dict(1, 0.9)]

        async def get_many(self, keys):
            return [await self.get(key) for key in keys]

        async def set(self, key, value, ttl=None):
            return None

    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 1)
    monkeypatch.setattr(config, "PUBLIC_FACETS_RATE_PER_MIN", 1)
    _cached_search_client(monkeypatch)
    monkeypatch.setattr(main, "cache", _BothEndpointsCache())

    assert _client.get("/search", params={"q": "test"}).status_code == 200
    assert _client.get("/search", params={"q": "test"}).status_code == 429
    assert _client.get("/facets").status_code == 200


def test_analytics_click_over_the_limit_is_rejected_with_429(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_CLICK_RATE_PER_MIN", 1)
    monkeypatch.setattr(main, "record_click", _noop_async)

    assert _client.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 200
    assert _client.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 429


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


_article = make_article


# --- /search ---


async def _passthrough_boost(q, results):
    return results


def test_search_cache_hit_returns_cached_summaries(monkeypatch):
    records = []
    cached = [_summary_dict(1, 0.9), _summary_dict(2, 0.7)]

    async def fake_record_search(*args, **kwargs):
        records.append((args, kwargs))

    monkeypatch.setattr(main, "cache", FakeCache(get_result=cached))
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: "weak note")
    monkeypatch.setattr(main, "record_search", fake_record_search)

    resp = _run(main.search(q="fintech funding", top_k=8, industry=None, dealtype=None,
                            author=None, content_type=None, from_date=None, to_date=None, tag=None))

    assert resp.cached is True
    assert [r.id for r in resp.results] == [1, 2]
    assert all(isinstance(r, SourceSummary) for r in resp.results)
    assert resp.note == "weak note"
    assert len(records) == 1
    assert records[0][0][0] == "fintech funding"
    assert records[0][0][1] == 2
    assert records[0][1]["cached"] is True
    assert records[0][1]["filtered"] is False


def test_search_cache_miss_runs_full_pipeline(monkeypatch, fake_cache):
    records = []
    boost_calls = []
    div_calls = []
    cache = fake_cache()
    articles = [_article(1, 0.9), _article(2, 0.7)]

    async def fake_retrieve(q, top_k, qfilter, need_body=False, prefetched=None):
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
                            author=None, content_type=None, from_date=None, to_date=None, tag=None))

    assert boost_calls, "apply_click_boost should have been called"
    assert div_calls, "diversify should have been called"
    assert resp.cached is False
    assert [r.id for r in resp.results] == [1, 2]
    assert all(isinstance(r, SourceSummary) for r in resp.results)
    assert all("body" not in r.model_dump() for r in resp.results)
    assert len(cache.sets) == 1
    (stored_key, stored_value, _ttl) = cache.sets[0]
    assert stored_key.startswith("search:")
    assert all("body" not in d for d in stored_value)
    assert records[0][1]["cached"] is False
    assert records[0][0][1] == 2


def test_search_cache_miss_skips_boost_and_diversity_when_disabled(monkeypatch, fake_cache):
    boost_calls = []
    div_calls = []

    async def fake_retrieve(q, top_k, qfilter, need_body=False, prefetched=None):
        return [_article(1, 0.9)]

    async def fake_boost(q, results):
        boost_calls.append(q)
        return results

    def fake_diversify(results, eff_top_k, **kwargs):
        div_calls.append(eff_top_k)
        return results

    async def fake_record_search(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "cache", fake_cache())
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
                            author=None, content_type=None, from_date=None, to_date=None, tag=None))

    assert resp.cached is False
    assert boost_calls == []
    assert div_calls == []
    assert [r.id for r in resp.results] == [1]


def test_search_passes_built_facet_filter_to_retrieve(monkeypatch, fake_cache):
    captured = {}

    async def fake_retrieve(q, top_k, qfilter, need_body=False, prefetched=None):
        captured["qfilter"] = qfilter
        return [_article(1, 0.9), _article(2, 0.7)]

    async def fake_record_search(*args, **kwargs):
        pass

    async def fake_temporal_fallback(results, top_k, from_date, to_date, industry, dealtype, author, tag, need_body):
        return results

    monkeypatch.setattr(main, "cache", fake_cache())
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "_temporal_date_fallback", fake_temporal_fallback)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", fake_record_search)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)

    resp = _run(main.search(q="fintech funding", top_k=8, industry="Fintech", dealtype=None,
                            author=None, content_type=None, from_date="2025-01-01", to_date=None, tag=None))

    qfilter = captured["qfilter"]
    assert isinstance(qfilter, Filter)
    assert {c.key for c in qfilter.must} == {"industry_names", "published_date"}
    assert resp.cached is False
    assert [r.id for r in resp.results] == [1, 2]


def test_search_cache_miss_does_not_cache_empty_results(monkeypatch, fake_cache):
    """An empty result set must never be cached: replayed as authoritative 'no
    results' for the TTL, a transiently empty query keeps returning nothing."""
    cache = fake_cache()

    async def fake_retrieve(q, top_k, qfilter, need_body=False, prefetched=None):
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
                            author=None, content_type=None, from_date=None, to_date=None, tag=None))

    assert resp.results == []
    assert resp.cached is False
    assert cache.sets == [], "empty result sets must never be written to the cache"


def test_search_empty_results_are_not_served_from_cache(monkeypatch, fake_cache):
    """A second identical query must re-run retrieval, not be answered from the poisoned entry."""
    cache = fake_cache()
    calls = []

    async def fake_retrieve(q, top_k, qfilter, need_body=False, prefetched=None):
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
              "from_date": None, "to_date": None, "tag": None}
    first = _run(main.search(q="edtech startups 2020", **kwargs))
    second = _run(main.search(q="edtech startups 2020", **kwargs))

    assert first.results == []
    assert len(calls) == 2, "the second query must re-run retrieval, not hit the cache"
    assert [r.id for r in second.results] == [1]
    assert second.cached is False


def test_search_non_empty_results_are_still_cached(monkeypatch, fake_cache):
    """Guard against over-correcting: the cache stays enabled for real result sets."""
    cache = fake_cache()
    articles = [_article(1, 0.9)]

    async def fake_retrieve(q, top_k, qfilter, need_body=False, prefetched=None):
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
                     author=None, content_type=None, from_date=None, to_date=None, tag=None))

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


def test_retrieve_and_rerank_does_not_cache_empty_results(monkeypatch, fake_cache):
    cache = fake_cache()
    monkeypatch.setattr(main, "cache", cache)
    _patch_retrieval_pipeline(monkeypatch, [])

    out = _run(main.retrieve_and_rerank("edtech startups 2020", 8, None))

    assert out == []
    assert cache.sets == [], "empty result sets must never be written to the cache"
    assert cache.store == {}


def test_retrieve_and_rerank_non_empty_results_are_still_cached(monkeypatch, fake_cache):
    cache = fake_cache()
    monkeypatch.setattr(main, "cache", cache)
    _patch_retrieval_pipeline(monkeypatch, [_article(1, 0.9)])

    out = _run(main.retrieve_and_rerank("fintech funding", 8, None))

    assert [a.id for a in out] == [1]
    assert len(cache.sets) == 1
    assert cache.sets[0][0].startswith("retrieve:")
    assert "body" not in cache.sets[0][1][0]


def test_search_retrieve_error_returns_500(monkeypatch, fake_cache):
    async def boom(*args, **kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "cache", fake_cache())
    monkeypatch.setattr(main, "retrieve_and_rerank", boom)

    r = _client.get("/search", params={"q": "test"})
    assert r.status_code == 500


def test_search_cache_error_returns_500(monkeypatch, fake_cache):
    """A cache read that raises must surface as a 500.

    Retrieval is stubbed to succeed, so only the cache can turn this into a 500;
    the same wiring answers 200 with a working cache.
    """

    async def fake_retrieve(*args, **kwargs):
        return [_article(1, 0.9)]

    monkeypatch.setattr(main, "cache", fake_cache(get_error=RuntimeError("redis down")))
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "_effective_intent", lambda q, fd, td: (q, None, None, None, None))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "apply_click_boost", _passthrough_boost)
    monkeypatch.setattr(main, "diversify", lambda results, **kwargs: results)
    monkeypatch.setattr(main, "weak_results_note", lambda scores, label: None)
    monkeypatch.setattr(main, "record_search", _noop_async)
    monkeypatch.setattr(main.config, "ENABLE_CLICK_BOOST", False)
    monkeypatch.setattr(main.config, "ENABLE_DIVERSITY", False)

    r = _client.get("/search", params={"q": "test"})
    assert r.status_code == 500


# --- /facets ---


def test_facets_cache_hit(monkeypatch, fake_cache):
    cached = {"industry": ["Fintech", "Healthtech"], "dealtype": ["M&A", "Funding"]}
    monkeypatch.setattr(main, "cache", fake_cache(get_result=cached))

    async def fake_facet_values(key):
        raise AssertionError("_facet_values must not run on a cache hit")

    monkeypatch.setattr(main, "_facet_values", fake_facet_values)

    r = _client.get("/facets")
    assert r.status_code == 200
    assert r.json() == cached


def test_facets_cache_miss(monkeypatch, fake_cache):
    cache = fake_cache()
    monkeypatch.setattr(main, "cache", cache)

    async def fake_facet_values(key):
        return {"industry_names": ["Fintech", "Healthtech"], "dealtype_names": ["M&A"]}[key]

    async def fake_top_facet_values(key, limit):
        return ["IPO", "VCC Startups"][:limit]

    monkeypatch.setattr(main, "_facet_values", fake_facet_values)
    monkeypatch.setattr(main, "_top_facet_values", fake_top_facet_values)

    expected = {"industry": ["Fintech", "Healthtech"], "dealtype": ["M&A"],
                "tags": ["IPO", "VCC Startups"]}
    r = _client.get("/facets")
    assert r.status_code == 200
    assert r.json() == expected
    assert cache.sets == [(main.FACETS_CACHE_KEY, expected, None)]


def test_facets_qdrant_error_returns_500(monkeypatch, fake_cache):
    async def fake_facet_values(key):
        raise RuntimeError("qdrant down")

    async def fake_top_facet_values(key, limit):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(main, "cache", fake_cache())
    monkeypatch.setattr(main, "_facet_values", fake_facet_values)
    monkeypatch.setattr(main, "_top_facet_values", fake_top_facet_values)

    r = _client.get("/facets")
    assert r.status_code == 500

def test_facets_survives_a_failed_tag_scan(monkeypatch, fake_cache):
    """The tag walk is the slowest and only optional scan, so its timeout must
    not 500 the controlled vocabularies the UI's autocomplete depends on; a
    typed tag still filters correctly against an empty suggestion list."""
    async def fake_facet_values(key):
        return {"industry_names": ["Fintech"], "dealtype_names": ["M&A"]}[key]

    async def failing_top_facet_values(key, limit):
        raise TimeoutError("tag walk exceeded the client timeout")

    cache = fake_cache()
    monkeypatch.setattr(main, "cache", cache)
    monkeypatch.setattr(main, "_facet_values", fake_facet_values)
    monkeypatch.setattr(main, "_top_facet_values", failing_top_facet_values)

    r = _client.get("/facets")

    assert r.status_code == 200
    expected = {"industry": ["Fintech"], "dealtype": ["M&A"], "tags": []}
    assert r.json() == expected
    assert cache.sets == [(main.FACETS_CACHE_KEY, expected, None)]


def test_facets_concurrent_misses_share_one_scan_and_one_cache_write(monkeypatch):
    """K callers that miss together must cost one scan per key, not one each.

    ``cache.get`` then ``cache.set`` is a check-then-act; counted on the scan
    seam, never on elapsed time -- each scan refuses to finish until all K
    callers have been through the cache.
    """
    callers = 6
    state: dict = {}

    class _ArrivalCache(FakeCache):
        """Records how many callers have been through the cache."""

        async def get(self, key):
            value = await super().get(key)
            if key == main.FACETS_CACHE_KEY:
                state["arrived"] += 1
                if state["arrived"] == callers:
                    state["everyone_here"].set()
            return value

    async def scenario():
        state["arrived"] = 0
        state["everyone_here"] = asyncio.Event()
        scans: list[str] = []

        async def parking_facet_values(key):
            scans.append(key)
            await asyncio.wait_for(state["everyone_here"].wait(), timeout=10)
            return {"industry_names": ["Fintech"], "dealtype_names": ["M&A"]}[key]

        async def parking_top_facet_values(key, limit):
            scans.append(key)
            await asyncio.wait_for(state["everyone_here"].wait(), timeout=10)
            return ["IPO"]

        cache = _ArrivalCache()
        monkeypatch.setattr(main, "cache", cache)
        monkeypatch.setattr(main, "_facet_values", parking_facet_values)
        monkeypatch.setattr(main, "_top_facet_values", parking_top_facet_values)
        results = await asyncio.gather(*(main.facets() for _ in range(callers)))
        return scans, results, cache.sets

    scans, results, sets = _run(scenario())

    expected = {"industry": ["Fintech"], "dealtype": ["M&A"], "tags": ["IPO"]}
    assert sorted(scans) == ["dealtype_names", "industry_names", "tag_names"]
    assert results == [expected] * callers
    assert sets == [(main.FACETS_CACHE_KEY, expected, None)]


def test_facets_failed_scan_releases_the_single_flight_and_caches_nothing(monkeypatch):
    """A scan that raises must release the guard and cache nothing."""
    attempts: list[str] = []

    async def down_facet_values(key):
        attempts.append(key)
        raise RuntimeError("qdrant down")

    async def working_top_facet_values(key, limit):
        # The tag scan degrades to [] instead of propagating, so it is kept
        # healthy here: this is about the guard being released.
        attempts.append(key)
        return ["IPO"]


    async def scenario():
        cache = FakeCache()
        monkeypatch.setattr(main, "cache", cache)
        monkeypatch.setattr(main, "_facet_values", down_facet_values)
        monkeypatch.setattr(main, "_top_facet_values", working_top_facet_values)
        outcomes = await asyncio.gather(main.facets(), main.facets(), return_exceptions=True)
        # Let the scan's done callback run, so the guard is observably released.
        await asyncio.sleep(0)
        return cache, outcomes, main._facet_scan_task

    cache, outcomes, guard = _run(scenario())

    assert [type(outcome) for outcome in outcomes] == [RuntimeError, RuntimeError]
    assert sorted(attempts) == ["dealtype_names", "industry_names", "tag_names"]
    assert cache.sets == []
    assert guard is None

    async def working_facet_values(key):
        return {"industry_names": ["Fintech"], "dealtype_names": ["M&A"]}[key]

    monkeypatch.setattr(main, "_facet_values", working_facet_values)
    monkeypatch.setattr(main, "_top_facet_values", working_top_facet_values)
    expected = {"industry": ["Fintech"], "dealtype": ["M&A"], "tags": ["IPO"]}
    assert _run(main.facets()) == expected
    assert cache.sets == [(main.FACETS_CACHE_KEY, expected, None)]


def test_facets_scans_the_vocabularies_concurrently(monkeypatch):
    """The keys are independent, so the second scan must start before the first
    finishes; each refuses to finish until the others have started, so the
    awaited-back-to-back version times out here."""
    scans_expected = 3

    async def scenario():
        started: list[str] = []
        all_started = asyncio.Event()

        async def barrier_facet_values(key):
            started.append(key)
            if len(started) == scans_expected:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=10)
            return [key]

        async def barrier_top_facet_values(key, limit):
            return await barrier_facet_values(key)

        cache = FakeCache()
        monkeypatch.setattr(main, "cache", cache)
        monkeypatch.setattr(main, "_facet_values", barrier_facet_values)
        monkeypatch.setattr(main, "_top_facet_values", barrier_top_facet_values)
        return started, await main.facets()

    started, result = _run(scenario())

    assert sorted(started) == ["dealtype_names", "industry_names", "tag_names"]
    assert result == {"industry": ["industry_names"], "dealtype": ["dealtype_names"],
                      "tags": ["tag_names"]}

    # No scan ran to completion before the others started, so industry need not be first.
    assert len(started) == scans_expected



@pytest.mark.parametrize("cancels", ["creator", "waiter"])
def test_facets_cancelled_caller_does_not_abort_the_scan_the_others_share(monkeypatch, cancels):
    """A request that goes away mid-scan must not take the shared scan down for
    the callers still waiting.

    Both roles are covered -- the one that started the scan and the one that
    joined it -- because a cancelled waiter must let go rather than tear it down.
    """

    async def scenario():
        release = asyncio.Event()
        scans: list[str] = []

        async def parked_facet_values(key):
            scans.append(key)
            await release.wait()
            return {"industry_names": ["Fintech"], "dealtype_names": ["M&A"]}[key]

        async def parked_top_facet_values(key, limit):
            scans.append(key)
            await release.wait()
            return ["IPO"]

        cache = FakeCache()
        monkeypatch.setattr(main, "cache", cache)
        monkeypatch.setattr(main, "_facet_values", parked_facet_values)
        monkeypatch.setattr(main, "_top_facet_values", parked_top_facet_values)

        creator = asyncio.create_task(main.facets())
        await asyncio.sleep(0)
        waiter = asyncio.create_task(main.facets())
        await asyncio.sleep(0)
        leaving, staying = (creator, waiter) if cancels == "creator" else (waiter, creator)
        leaving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leaving
        release.set()
        return scans, await staying

    scans, result = _run(scenario())

    assert sorted(scans) == ["dealtype_names", "industry_names", "tag_names"]
    assert result == {"industry": ["Fintech"], "dealtype": ["M&A"], "tags": ["IPO"]}


# --- /analytics/summary ---


class _NoopStore:
    async def record_admin_audit(self, actor_id, action):
        return None


class _AuditReq:
    """Minimal Request stand-in: the handler reads only ``state.user_id``."""

    state = SimpleNamespace(user_id="admin-1")


def test_analytics_summary(monkeypatch):
    # The handler takes a Request and resolves a store because it records the
    # read in the admin audit trail; a bare call supplies neither.
    monkeypatch.setattr(main.chat_module, "_require_store", lambda: _NoopStore())

    async def fake_analytics_data():
        return {"searches_total": 5}

    monkeypatch.setattr(main, "analytics_data", fake_analytics_data)

    assert _run(main.get_analytics_summary(_AuditReq(), None, None)) == {"searches_total": 5}


def test_analytics_summary_endpoint_maps_store_failure_to_503(monkeypatch):
    """A failing analytics store must return a 503 response carrying the error --
    not the error dict as a 200, which the dashboard rendered as a real report."""
    async def failing_analytics_data():
        raise AnalyticsUnavailableError("analytics unavailable")

    monkeypatch.setattr(main, "analytics_data", failing_analytics_data)
    monkeypatch.setattr(main.chat_module, "_require_store", lambda: _NoopStore())

    res = _run(main.get_analytics_summary(_AuditReq(), None, None))

    assert isinstance(res, JSONResponse)
    assert res.status_code == 503
    assert json.loads(res.body)["error"] == "analytics unavailable"
