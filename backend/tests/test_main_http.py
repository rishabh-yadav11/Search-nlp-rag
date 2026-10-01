"""HTTP-level coverage of the /search, /facets and /analytics routes of app.main."""

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
    """Fake a loopback peer, so the proxy-trust paths see what setup.sh's nginx sends."""

    async def wrapper(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": ("127.0.0.1", 40000)}
        await app(scope, receive, send)

    return wrapper


_client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _public_rate_limiter(monkeypatch):
    """/search, /facets and /analytics/click fail CLOSED (503) when the limiter's Redis is unreachable."""
    fake = RateLimitRedisFake()
    monkeypatch.setattr(auth, "_rate_client", fake)

    return fake.counters


def _cached_search_client(monkeypatch):
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
    """An unbounded q is unbounded work per request, so validation rejects it with 422 before retrieval runs."""
    _cached_search_client(monkeypatch)

    ok = _client.get("/search", params={"q": "a" * config.SEARCH_QUERY_MAX_CHARS})
    too_long = _client.get("/search", params={"q": "a" * (config.SEARCH_QUERY_MAX_CHARS + 1)})

    assert ok.status_code == 200
    assert too_long.status_code == 422
    assert [e["loc"][-1] for e in too_long.json()["detail"]] == ["q"]


def test_search_limit_is_per_client_ip_not_one_global_bucket(monkeypatch):
    """Two proxied clients get their own bucket; the XFF trust flag is deliberately left at the shipped default."""
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
    """XFF is trusted only from a loopback peer; the two requests forge different IPs, so reusing one would prove nothing."""
    monkeypatch.setattr(config, "PUBLIC_SEARCH_RATE_PER_MIN", 1)
    _cached_search_client(monkeypatch)
    first = TestClient(main.app, raise_server_exceptions=False, headers={"x-forwarded-for": "9.9.9.9"})
    second = TestClient(main.app, raise_server_exceptions=False, headers={"x-forwarded-for": "8.8.8.8"})

    assert first.get("/search", params={"q": "test"}).status_code == 200
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
    """Limiter key is public:rl:<action>:<ip>; other limit tests exhaust one endpoint, so a dropped action stays green."""

    class _BothEndpointsCache:
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
    """An empty set replayed as 'no results' pins a date-filtered query to nothing for the whole TTL."""
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


def _patch_retrieval_pipeline(monkeypatch, articles):
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
    """Retrieval is stubbed to succeed, so the raising cache read is the only path to this 500."""

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
    """A failed tag walk must still 200 with the two controlled vocabularies the UI autocomplete needs."""
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
    """Counted on the scan seam, never on elapsed time: each scan waits for every caller, so a later scan cannot hide."""
    callers = 6
    state: dict = {}

    class _ArrivalCache(FakeCache):
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
    """A raising scan must release the single-flight guard, or the next caller wedges on it forever."""
    attempts: list[str] = []

    async def down_facet_values(key):
        attempts.append(key)
        raise RuntimeError("qdrant down")

    async def working_top_facet_values(key, limit):
        attempts.append(key)
        return ["IPO"]


    async def scenario():
        cache = FakeCache()
        monkeypatch.setattr(main, "cache", cache)
        monkeypatch.setattr(main, "_facet_values", down_facet_values)
        monkeypatch.setattr(main, "_top_facet_values", working_top_facet_values)
        outcomes = await asyncio.gather(main.facets(), main.facets(), return_exceptions=True)
        # Yield so the scan task's done callback runs and the guard is observably released.
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
    """Each scan blocks until all three start, so a serial implementation times out here, not merely slower."""
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

    assert len(started) == scans_expected



@pytest.mark.parametrize("cancels", ["creator", "waiter"])
def test_facets_cancelled_caller_does_not_abort_the_scan_the_others_share(monkeypatch, cancels):
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


class _NoopStore:
    async def record_admin_audit(self, actor_id, action):
        return None


class _AuditReq:
    """Minimal Request stand-in: the handler reads only ``state.user_id``."""

    state = SimpleNamespace(user_id="admin-1")


def test_analytics_summary(monkeypatch):
    monkeypatch.setattr(main.chat_module, "_require_store", lambda: _NoopStore())

    async def fake_analytics_data():
        return {"searches_total": 5}

    monkeypatch.setattr(main, "analytics_data", fake_analytics_data)

    assert _run(main.get_analytics_summary(_AuditReq(), None, None)) == {"searches_total": 5}


def test_analytics_summary_endpoint_maps_store_failure_to_503(monkeypatch):
    """503, not an error-shaped 200: the dashboard read the error dict as an all-zero report."""
    async def failing_analytics_data():
        raise AnalyticsUnavailableError("analytics unavailable")

    monkeypatch.setattr(main, "analytics_data", failing_analytics_data)
    monkeypatch.setattr(main.chat_module, "_require_store", lambda: _NoopStore())

    res = _run(main.get_analytics_summary(_AuditReq(), None, None))

    assert isinstance(res, JSONResponse)
    assert res.status_code == 503
    assert json.loads(res.body)["error"] == "analytics unavailable"
