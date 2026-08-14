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
