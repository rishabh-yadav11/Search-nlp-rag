import asyncio
import math
from datetime import UTC
from datetime import datetime as _dt

import pytest
from fastapi import HTTPException
from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchAny

from app import main
from app.main import SourceArticle, build_facet_filter, sort_results


def _article(id_: int, score: float, published_date: str | None = None, title: str = "") -> SourceArticle:
    return SourceArticle(
        id=id_,
        title=title,
        url=f"https://example.com/{id_}",
        published_date=published_date,
        score=score,
    )


def _conditions(qfilter: Filter) -> dict[str, list[FieldCondition]]:
    assert isinstance(qfilter, Filter)
    by_key: dict[str, list[FieldCondition]] = {}
    for c in qfilter.must:
        by_key.setdefault(c.key, []).append(c)
    return by_key


def _only(conds: list[FieldCondition]) -> FieldCondition:
    assert len(conds) == 1
    return conds[0]


class _FrozenNow(_dt):
    """datetime subclass pinned to a fixed 'now' for deterministic recency math."""

    _FIXED = _dt(2026, 8, 13, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls._FIXED if tz is not None else cls._FIXED.replace(tzinfo=None)


def test_build_facet_filter_match_any():
    f = build_facet_filter("Fintech, Healthtech", "M&A", "Alice Bob", None, None)
    conds = _conditions(f)
    assert isinstance(_only(conds["industry_names"]).match, MatchAny)
    assert _only(conds["industry_names"]).match.any == ["Fintech", "Healthtech"]
    assert _only(conds["dealtype_names"]).match.any == ["M&A"]
    assert _only(conds["author_names"]).match.any == ["Alice Bob"]


def test_build_facet_filter_dates():
    f = build_facet_filter(None, None, None, "2025-01-01", "2025-12-31")
    conds = _conditions(f)
    date_conds = conds["published_date"]
    assert all(isinstance(c.range, DatetimeRange) for c in date_conds)
    gtes = [c.range.gte for c in date_conds]
    ltes = [c.range.lte for c in date_conds]
    assert _dt(2025, 1, 1, tzinfo=UTC) in gtes
    assert _dt(2025, 12, 31, 23, 59, 59, 999999, tzinfo=UTC) in ltes


def test_build_facet_filter_none_when_unfiltered():
    assert build_facet_filter(None, None, None, None, None) is None


@pytest.mark.parametrize("field", ["from_date", "to_date"])
def test_build_facet_filter_invalid_date_raises_400(field):
    kwargs = {"industry": None, "dealtype": None, "author": None, "from_date": None, "to_date": None}
    kwargs[field] = "not-a-date"
    with pytest.raises(HTTPException) as exc_info:
        build_facet_filter(**kwargs)
    assert exc_info.value.status_code == 400


def test_effective_intent_explicit_user_dates_win():
    rq, fd, td = main._effective_intent("deals in 2025", "2024-01-01", "2024-12-31")
    assert (fd, td) == ("2024-01-01", "2024-12-31")
    assert rq == "deals in 2025"


def test_effective_intent_auto_year_range():
    rq, fd, td = main._effective_intent("deals in 2025", None, None)
    assert (fd, td) == ("2025-01-01", "2025-12-31")
    assert rq == "deals in 2025"


def test_effective_intent_no_year_no_dates():
    rq, fd, td = main._effective_intent("latest deals", None, None)
    assert (fd, td) == (None, None)
    assert rq == "latest deals"


def test_sort_results_recency_ordering(monkeypatch):
    monkeypatch.setattr(main, "datetime", _FrozenNow)
    recent = _article(1, 1.0, "2026-08-01")
    old = _article(2, 1.0, "2015-01-01")
    out = sort_results([old, recent])
    assert [a.id for a in out] == [1, 2]


def test_sort_results_missing_date_last_on_tie(monkeypatch):
    monkeypatch.setattr(main, "datetime", _FrozenNow)
    # Dated article is 12 days old on the frozen 'now' (2026-08-13).
    mult = 1.0 - main.config.RECENCY_STRENGTH * (1.0 - math.exp(-12.0 / main.config.RECENCY_DECAY_DAYS))
    dated = _article(1, 1.0, "2026-08-01")
    missing = _article(2, mult, None)
    out = sort_results([missing, dated])
    assert [a.id for a in out] == [1, 2]


def test_sort_results_full_ordering(monkeypatch):
    monkeypatch.setattr(main, "datetime", _FrozenNow)
    recent = _article(1, 1.0, "2026-08-01")
    old = _article(2, 0.9, "2025-01-01")
    missing = _article(3, 0.5, None)
    out = sort_results([missing, old, recent])
    assert [a.id for a in out] == [1, 2, 3]


class _FakeReranker:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self.logits


def test_rerank_sigmoid_and_ordering(monkeypatch):
    fake = _FakeReranker([1.0, -1.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    a1 = _article(1, 0.5)
    a2 = _article(2, 0.5)
    out = asyncio.run(main.rerank("q", [a1, a2]))
    assert fake.calls == 1
    assert [a.id for a in out] == [1, 2]
    assert abs(out[0].score - 1.0 / (1.0 + math.exp(-1.0))) < 1e-9
    assert abs(out[1].score - 1.0 / (1.0 + math.exp(1.0))) < 1e-9


def test_rerank_single_result_short_circuit(monkeypatch):
    fake = _FakeReranker([99.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    one = _article(1, 0.5)
    out = asyncio.run(main.rerank("q", [one]))
    assert fake.calls == 0
    assert out == [one]


def test_merge_results_dedupes_keeping_highest_score():
    from app.main import _merge_results

    a1 = _article(1, 0.5, title="x")
    a2 = _article(1, 0.9, title="x")
    b = _article(2, 0.7, title="y")
    merged = _merge_results([a1, b], [a2])
    assert [x.id for x in merged] == [1, 2]
    assert abs(merged[0].score - 0.9) < 1e-9


def test_merge_results_empty_and_disjoint():
    from app.main import _merge_results

    a = _article(1, 0.4)
    c = _article(3, 0.6)
    assert _merge_results() == []
    assert [x.id for x in _merge_results([a], [c])] == [1, 3]


def test_retrieval_queries_dual_for_year_top_intent():
    from app.main import _retrieval_queries

    qs = _retrieval_queries("top 3 unicorns created in 2025")
    assert qs == ["Flashback 2025 unicorns created", "unicorns created"]


def test_retrieval_queries_single_for_non_year_top():
    from app.main import _retrieval_queries

    assert _retrieval_queries("fintech funding") == ["fintech funding"]


def test_retrieval_queries_no_dup_when_topic_equals_rewrite():
    from app.main import _retrieval_queries

    # A query that's already a bare 'top <topic>' with no year: no rewrite -> single
    qs = _retrieval_queries("top deals")
    assert qs == ["top deals"]


class _FakeCache:
    """Minimal in-memory stand-in for the HybridCache used by retrieve_and_rerank."""

    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value):
        self.store[key] = value


def test_filter_token_deterministic_and_json_serializable():
    """The retrieve-cache key from a Qdrant filter must be a stable string
    (regression: model_dump_json(sort_keys=...) is unsupported in pydantic)."""
    f = Filter(must=[FieldCondition(key="industry_names", match=MatchAny(any=["Fintech"]))])
    assert main._filter_token(None) == ""
    t1 = main._filter_token(f)
    t2 = main._filter_token(f)
    assert t1 == t2
    assert isinstance(t1, str)
    assert "Fintech" in t1


def test_retrieve_and_rerank_caches_without_body(monkeypatch):
    """The retrieve cache round-trips SourceArticles but never stores bodies;
    a chat request re-fetches them from Qdrant after a cache hit."""
    from app.main import SourceArticle

    fake_cache = _FakeCache()
    monkeypatch.setattr(main, "cache", fake_cache)

    def make_article(id_: int, score: float) -> SourceArticle:
        return SourceArticle(id=id_, title="t", url="u", summary="s", body="", score=score)

    async def fake_leg(rq, top_k, qfilter):
        return [make_article(1, 0.9), make_article(2, 0.7)]

    async def fake_rerank(q, results):
        results.sort(key=lambda a: a.score, reverse=True)
        return results

    async def fake_bodies(articles):
        for a in articles:
            a.body = "full body"

    monkeypatch.setattr(main, "_retrieval_queries", lambda q: [q])
    monkeypatch.setattr(main, "_retrieval_leg", fake_leg)
    monkeypatch.setattr(main, "rerank", fake_rerank)
    monkeypatch.setattr(main, "sort_results", lambda r: r)
    monkeypatch.setattr(main, "apply_entity_boost", lambda q, r: r)
    monkeypatch.setattr(main, "_attach_bodies", fake_bodies)
    monkeypatch.setattr(main.config, "ENABLE_ENTITY_BOOST", True)

    out1 = asyncio.run(main.retrieve_and_rerank("q", 8, None, need_body=True))
    assert out1[0].body == "full body"
    (_, cached), = list(fake_cache.store.items())
    assert "body" not in cached[0]
    assert cached[0]["id"] == 1

    async def refetch_bodies(articles):
        for a in articles:
            a.body = "refetched"

    monkeypatch.setattr(main, "_attach_bodies", refetch_bodies)
    out2 = asyncio.run(main.retrieve_and_rerank("q", 8, None, need_body=True))
    assert out2[0].body == "refetched"
    assert len(fake_cache.store) == 1  # still a single cache entry


def test_source_context_includes_whole_body():
    """Chat prompt context must carry the full article body, not a fixed excerpt."""
    from app.main import SourceArticle, source_context

    body = "x" * 4000
    a = SourceArticle(id=1, title="t", url="u", published_date="2024-01-01",
                      summary="s", body=body, score=0.9)
    out = source_context(a, 1)
    assert body in out
    assert "x" * 3000 in out


def test_analytics_dashboard_is_frontend_owned():
    """The dashboard UI is a Next.js page now (frontend/app/analytics/dashboard);
    the backend only serves the JSON data endpoints, both admin-gated."""
    paths = {r.path for r in main.app.routes}
    assert "/analytics/dashboard" not in paths
    assert "/analytics/summary" in paths
    assert "/analytics/chat" in paths


def test_best_body_window_finds_query_token_dense_region():
    from app.main import _best_body_window, _query_content_tokens

    body = ("intro filler " * 200) + ("2008 crisis central banks lessons Subbarao " * 30) + ("tail filler " * 100)
    tokens = _query_content_tokens("lessons RBI governor Subbarao central banks learned 2008 crisis")
    assert "2008" in tokens and "crisis" in tokens and "subbarao" in tokens
    win = _best_body_window(body, tokens, 1500, 500)
    low = win.lower()
    assert "2008" in low and "subbarao" in low
    assert low.find("2008") < 1500  # picked the dense region, not the filler intro


def test_body_rescue_lifts_deep_body_match_and_reorders(monkeypatch):
    """A weak title+summary score is rescued when the body region matches the
    query; the score becomes max(baseline, body-window score)."""
    from app.main import body_rescue

    def make(id_: int, body: str) -> SourceArticle:
        a = SourceArticle(id=id_, title="t", url=f"https://example.com/{id_}",
                          summary="s", body=body, score=0.1)
        return a

    matching = make(1, "filler words here " * 100 + "2008 crisis central banks lessons learned " * 40)
    unrelated = make(2, "completely unrelated filler content about weather and markets " * 200)
    fake = _FakeReranker([5.0, -2.0])
    monkeypatch.setitem(main.state, "reranker", fake)

    out = asyncio.run(body_rescue("lessons RBI governor Subbarao central banks learned 2008 crisis", [matching, unrelated]))
    assert fake.calls == 1
    assert [a.id for a in out] == [1, 2]
    assert abs(out[0].score - 1.0 / (1.0 + math.exp(-5.0))) < 1e-9
    assert abs(out[1].score - max(0.1, 1.0 / (1.0 + math.exp(2.0)))) < 1e-9


def test_body_rescue_skips_when_top_score_strong(monkeypatch):
    from app.main import body_rescue

    a = _article(1, 0.8)
    a.body = "has body"
    fake = _FakeReranker([9.0])
    monkeypatch.setitem(main.state, "reranker", fake)
    out = asyncio.run(body_rescue("some query", [a]))
    assert fake.calls == 0
    assert out[0].id == 1
