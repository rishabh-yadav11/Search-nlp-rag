import asyncio
import math
from datetime import UTC
from datetime import datetime as _dt
from functools import partial

import pytest
from _support import OMIT, make_article
from fastapi import HTTPException
from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchAny

from app import main
from app.main import SourceArticle, build_facet_filter, sort_results

# No summary, and a title that stays empty unless a test sets one.
_article = partial(make_article, title="", summary=OMIT)


def _conditions(qfilter: Filter) -> dict[str, list[FieldCondition]]:
    assert isinstance(qfilter, Filter)
    by_key: dict[str, list[FieldCondition]] = {}
    for c in qfilter.must:
        by_key.setdefault(c.key, []).append(c)
    return by_key


def _only(conds: list[FieldCondition]) -> FieldCondition:
    assert len(conds) == 1
    return conds[0]


def _route_paths(routes) -> set[str]:
    """Every path the app serves, descending into ``include_router`` wrappers.

    fastapi 0.141 keeps included routes under the wrapper rather than the
    top-level list, so a flat walk both misses them and raises on the wrapper.
    """
    paths: set[str] = set()
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            paths.add(path)
        nested = getattr(route, "original_router", None)
        children = getattr(nested, "routes", None) if nested is not None else None
        if children:
            paths |= _route_paths(children)
    return paths


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
    # DatetimeRange re-parses the ISO strings into datetimes of its own tz class,
    # so compare isoformat rather than the datetimes.
    assert _dt(2025, 1, 1, tzinfo=UTC).isoformat() in [g.isoformat() for g in gtes if g is not None]
    assert _dt(2025, 12, 31, 23, 59, 59, 999999, tzinfo=UTC).isoformat() in [l.isoformat() for l in ltes if l is not None]


def test_build_facet_filter_none_when_unfiltered():
    assert build_facet_filter(None, None, None, None, None) is None


def test_build_facet_filter_tag():
    """The tag param is a MatchAny on tag_names, like the other name facets."""
    f = build_facet_filter(None, None, None, None, None, None, "IPO,Flipkart")
    conds = _conditions(f)
    assert _only(conds["tag_names"]).match.any == ["IPO", "Flipkart"]
    assert list(conds) == ["tag_names"], "the tag param must not drag in another facet"
    assert build_facet_filter(None, None, None, None, None, None, None) is None


def test_the_auto_facet_retry_never_drops_an_explicit_tag(monkeypatch):
    """A tag survives the relaxation retry, which recomputes only the auto facets."""
    filters = []

    async def fake_retrieve_and_rerank(q, top_k, qfilter, **kwargs):
        filters.append(qfilter)
        return []

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve_and_rerank)

    asyncio.run(main.retrieve_with_auto_facet_fallback(
        "edtech", 5,
        industry=None, dealtype=None, author=None, content_type=None, tag="IPO",
        eff_from=None, eff_to=None, auto_industry="TMT", auto_dealtype=None,
    ))

    assert len(filters) == 2, "the relaxation retry never ran"
    assert {c.key for c in filters[0].must} == {"industry_names", "tag_names"}
    assert {c.key for c in filters[1].must} == {"tag_names"}, \
        "the retry dropped the explicit tag filter"


class _ScoredPoint:
    def __init__(self, tags):
        self.payload = {"tag_names": tags}


def _tag_scroll_qdrant():
    """A scroll source over three pages, re-served from the start per instance.

    Sensex is seen first and IPO last, so a scan that stopped once it had
    enough would rank differently from one that counts every point.
    """
    pages = [
        ([_ScoredPoint(["Sensex"]), _ScoredPoint(["Sensex"])], 2),
        ([_ScoredPoint(["IPO"]), _ScoredPoint(["IPO"]), _ScoredPoint(["IPO"])], 5),
        ([_ScoredPoint(["Flipkart"]), _ScoredPoint(["VCC Startups"])], None),
    ]

    class _Qdrant:
        def __init__(self):
            self.calls: list[dict] = []

        async def scroll(self, *, collection_name, limit, with_payload, with_vectors, offset):
            self.calls.append({"collection_name": collection_name, "limit": limit,
                               "with_payload": with_payload, "offset": offset})
            return pages[len(self.calls) - 1]

    return _Qdrant()


def test_the_tag_vocabulary_is_ranked_by_frequency_over_every_point(monkeypatch):
    """The cap truncates a finished ranking, it does not end the walk.

    The top N by frequency is unknowable until every value on every point has
    been counted, so all three pages are read and the tail dropped.
    """
    qdrant = _tag_scroll_qdrant()
    monkeypatch.setitem(main.state, "qdrant", qdrant)

    out = asyncio.run(main._top_facet_values("tag_names", 2))

    assert out == ["IPO", "Sensex"]
    assert [c["offset"] for c in qdrant.calls] == [None, 2, 5], \
        "the walk stopped before the collection was exhausted"
    # One keyword field per page: the whole payload is ~6KB of body per point.
    assert all(c["with_payload"] == ["tag_names"] for c in qdrant.calls)


def test_equal_tag_counts_break_alphabetically(monkeypatch):
    """Equal counts have no frequency order, so ties break by name; otherwise
    the cached payload would depend on the order the pages arrived in."""
    qdrant = _tag_scroll_qdrant()
    monkeypatch.setitem(main.state, "qdrant", qdrant)

    out = asyncio.run(main._top_facet_values("tag_names", 4))

    assert out == ["IPO", "Sensex", "Flipkart", "VCC Startups"]


@pytest.mark.parametrize("field", ["from_date", "to_date"])
def test_build_facet_filter_invalid_date_raises_400(field):
    kwargs = {"industry": None, "dealtype": None, "author": None, "from_date": None, "to_date": None}
    kwargs[field] = "not-a-date"
    with pytest.raises(HTTPException) as exc_info:
        build_facet_filter(**kwargs)
    assert exc_info.value.status_code == 400


def test_effective_intent_explicit_user_dates_win():
    rq, fd, td, dt, ind = main._effective_intent("deals in 2025", "2024-01-01", "2024-12-31")
    assert (fd, td) == ("2024-01-01", "2024-12-31")
    assert rq == "deals in 2025"
    assert dt is None and ind is None


def test_effective_intent_auto_year_range():
    rq, fd, td, dt, ind = main._effective_intent("deals in 2025", None, None)
    assert (fd, td) == ("2025-01-01", "2025-12-31")
    assert rq == "deals in 2025"
    assert dt is None and ind is None


def test_effective_intent_no_year_no_dates():
    rq, fd, td, dt, ind = main._effective_intent("latest deals", None, None)
    assert (fd, td) == (None, None)
    assert rq == "deals"
    assert dt is None and ind is None


def test_sort_results_recency_ordering(monkeypatch):
    monkeypatch.setattr(main, "datetime", _FrozenNow)
    recent = _article(1, 1.0, "2026-08-01")
    old = _article(2, 1.0, "2015-01-01")
    out = sort_results([old, recent])
    assert [a.id for a in out] == [1, 2]


def test_sort_results_missing_date_last_on_tie(monkeypatch):
    monkeypatch.setattr(main, "datetime", _FrozenNow)
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


def _blend_key(article: SourceArticle, strength: float, decay: float) -> tuple[float, str]:
    """The blended-score half of the key sort_results ranks on, recomputed here.

    Limited to dated, naive, already-past articles, where the tz-stripped
    tiebreak and the future-date clamp in _recency_multiplier cannot differ.
    """
    dt = _dt.fromisoformat(article.published_date).replace(tzinfo=UTC)
    age_days = (_dt(2026, 8, 13, tzinfo=UTC) - dt).total_seconds() / 86400.0
    return (
        article.score * (1.0 - strength * (1.0 - math.exp(-age_days / decay))),
        article.published_date,
    )


def _pin_shipped_recency_boost(monkeypatch):
    """Pin the boost knobs to the shipped values, not to whatever .env holds."""
    monkeypatch.setattr(main.config, "RECENCY_BOOST_STRENGTH", 0.85)
    monkeypatch.setattr(main.config, "RECENCY_BOOST_DECAY_DAYS", 30.0)
    monkeypatch.setattr(main, "datetime", _FrozenNow)


def test_recency_boost_knobs_ship_the_values_the_pre_knob_constants_had(parse_config):
    """The shipped defaults must stay the weights the ranking tests pin."""
    shipped = parse_config()
    assert shipped.RECENCY_BOOST_STRENGTH == 0.85
    assert shipped.RECENCY_BOOST_DECAY_DAYS == 30.0


def test_shipped_recency_boost_reproduces_the_pre_knob_ranking(monkeypatch):
    """The shipped boost weights must reproduce the ranking sort_results gave
    while they were hardcoded in app/main.py."""
    _pin_shipped_recency_boost(monkeypatch)

    articles = [
        _article(1, 0.25, "2026-08-01"),
        _article(2, 1.0, "2025-01-01"),
        _article(3, 0.9, "2015-01-01"),
        _article(4, 0.1, "2026-08-10"),
    ]
    expected = sorted(articles, key=lambda a: _blend_key(a, 0.85, 30.0), reverse=True)
    out = sort_results(articles, recency_boost=True)
    assert [a.id for a in out] == [a.id for a in expected]
    # A no-op boost would satisfy the comparison above, so pin that it reorders.
    assert [a.id for a in out] != [
        a.id for a in sorted(articles, key=lambda a: (a.score, a.published_date), reverse=True)
    ]


def test_recency_boost_strength_knob_moves_the_ranking(monkeypatch):
    """Turning strength off must hand ranking back to raw relevance."""
    _pin_shipped_recency_boost(monkeypatch)
    recent = _article(1, 0.25, "2026-08-01")
    old = _article(2, 1.0, "2025-01-01")
    assert [a.id for a in sort_results([old, recent], recency_boost=True)] == [1, 2]

    monkeypatch.setattr(main.config, "RECENCY_BOOST_STRENGTH", 0.0)
    assert [a.id for a in sort_results([old, recent], recency_boost=True)] == [2, 1]


def test_recency_boost_decay_knob_moves_the_ranking(monkeypatch):
    """A longer decay keeps a moderately old hit competitive; both knobs must
    reach sort_results."""
    _pin_shipped_recency_boost(monkeypatch)
    recent = _article(1, 0.2, "2026-08-01")
    old = _article(2, 1.0, "2025-01-01")
    assert [a.id for a in sort_results([old, recent], recency_boost=True)] == [2, 1]

    monkeypatch.setattr(main.config, "RECENCY_BOOST_DECAY_DAYS", 90.0)
    assert [a.id for a in sort_results([old, recent], recency_boost=True)] == [1, 2]


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

    qs = _retrieval_queries("top deals")
    assert qs == ["top deals"]


def test_filter_token_deterministic_and_json_serializable():
    """Stable string cache key: pydantic Filter has no model_dump_json(sort_keys=...)."""
    f = Filter(must=[FieldCondition(key="industry_names", match=MatchAny(any=["Fintech"]))])
    assert main._filter_token(None) == ""
    t1 = main._filter_token(f)
    t2 = main._filter_token(f)
    assert t1 == t2
    assert isinstance(t1, str)
    assert "Fintech" in t1


def test_retrieve_and_rerank_caches_without_body(monkeypatch, fake_cache):
    """The retrieve cache round-trips SourceArticles but never stores bodies;
    a chat request re-fetches them from Qdrant after a cache hit."""
    from app.main import SourceArticle

    cache = fake_cache()
    monkeypatch.setattr(main, "cache", cache)

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
    monkeypatch.setattr(main, "sort_results", lambda r, recency_boost=False: r)
    monkeypatch.setattr(main, "apply_entity_boost", lambda q, r: r)
    monkeypatch.setattr(main, "_attach_bodies", fake_bodies)
    monkeypatch.setattr(main.config, "ENABLE_ENTITY_BOOST", True)

    out1 = asyncio.run(main.retrieve_and_rerank("q", 8, None, need_body=True))
    assert out1[0].body == "full body"
    (_, cached), = list(cache.store.items())
    assert "body" not in cached[0]
    assert cached[0]["id"] == 1

    async def refetch_bodies(articles):
        for a in articles:
            a.body = "refetched"

    monkeypatch.setattr(main, "_attach_bodies", refetch_bodies)
    out2 = asyncio.run(main.retrieve_and_rerank("q", 8, None, need_body=True))
    assert out2[0].body == "refetched"
    assert len(cache.store) == 1


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
    """The dashboard UI is a Next.js page; the backend serves only the JSON data
    endpoints."""
    paths = _route_paths(main.app.routes)
    assert "/analytics/dashboard" not in paths
    assert "/analytics/summary" in paths
    assert "/analytics/chat" in paths


def test_route_paths_walks_routes_nested_in_included_routers():
    """The walk must reach routes mounted via include_router, not just the
    top level.

    Every path asserted by ``test_analytics_dashboard_is_frontend_owned`` is a
    top-level route, so those three assertions stay green even if the descent
    is deleted. The paths used here are reachable only through it: fastapi
    0.141 keeps the health, auth and chat routes under their wrappers.
    """
    paths = _route_paths(main.app.routes)
    flat = {r.path for r in main.app.routes if getattr(r, "path", None) is not None}

    for nested_only in ("/health", "/ready", "/api/auth/login", "/api/chat/sessions"):
        assert nested_only not in flat, "guard is stale: this route is top-level now"
        assert nested_only in paths

    assert paths - flat, "the walk added nothing beyond the top level"


def test_best_body_window_finds_query_token_dense_region():
    from app.main import _best_body_window, _query_content_tokens

    body = ("intro filler " * 200) + ("2008 crisis central banks lessons Subbarao " * 30) + ("tail filler " * 100)
    tokens = _query_content_tokens("lessons RBI governor Subbarao central banks learned 2008 crisis")
    assert "2008" in tokens and "crisis" in tokens and "subbarao" in tokens
    win = _best_body_window(body, tokens, 1500, 500)
    low = win.lower()
    assert "2008" in low and "subbarao" in low
    assert low.find("2008") < 1500


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
