"""HTTP-level tests for the /search, /facets, /analytics/click and
/analytics/summary endpoints of app.main (cache hit/miss wiring, qdrant/redis
error mapping, and analytics beacons)."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Filter

from app import main
from app.main import SourceArticle, SourceSummary

_client = TestClient(main.app, raise_server_exceptions=False)

_MISS = object()


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

    monkeypatch.setattr(main, "cache", _FakeCache())
    monkeypatch.setattr(main, "fix_query", lambda q: (q, "fixed"))
    monkeypatch.setattr(main, "expand_query", lambda q: q)
    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
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
    monkeypatch.setattr(main, "sort_results", lambda r: r)
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
