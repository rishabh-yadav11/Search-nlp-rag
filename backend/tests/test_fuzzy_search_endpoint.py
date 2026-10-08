"""Randomized /search endpoint tests through the real app (faked stores).

Every request must answer 200 with well-formed JSON whose results carry
title/url/score. Explicit date windows must actually reach the fake Qdrant: a
window that excludes the whole corpus returns an honest empty list, never stale
rows, and every returned row sits inside the requested window.
"""

from __future__ import annotations

from datetime import date

from fakes import seed_articles
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_PAYLOAD_BANK = [
    "funding", "news", "startup", "ipo", "deals", "mumbai", "india", "acquisition",
    "earnings", "report", "round", "valuation", "investors", "capital", "market",
    "growth", "technology", "latest", "Ola", "PhonePe", "Zomato", "Swiggy",
]
_TOP_HINT = st.one_of(
    st.just(""),
    st.just("top 5"),
    st.just("best 10"),
    st.just("top"),
)
_QUERY = st.lists(st.sampled_from(_PAYLOAD_BANK), min_size=1, max_size=5).map(
    lambda ws: " ".join(ws)
)

_INDUSTRIES = ["Finance", "Mobility", "Consumer", "Industrials"]
_DEALTYPES = ["Venture Capital", "IPO", "M&A", "Earnings", "Stake Sale"]

_PARAMS = {
    "industry": st.one_of(st.none(), *[st.just(v) for v in _INDUSTRIES]),
    "dealtype": st.one_of(st.none(), *[st.just(v) for v in _DEALTYPES]),
    "author": st.one_of(st.none(), st.just("Alice Rao"), st.just("Nobody")),
    "content_type": st.one_of(st.none(), st.just("article")),
    "tag": st.one_of(st.none(), st.just("funding"), st.just("ipo")),
    "top_k": st.integers(min_value=1, max_value=10),
}


def _date_window() -> st.SearchStrategy[tuple[str | None, str | None]]:
    base = st.dates(min_value=date(2018, 1, 1), max_value=date(2032, 12, 31))
    return st.one_of(
        st.tuples(st.none(), st.none()),
        st.tuples(base, base).map(lambda t: (min(t[0], t[1]).isoformat(), max(t[0], t[1]).isoformat())),
    )


@settings(
    max_examples=60,
    deadline=None,
    # frozen_clock patches query_intent._now once for the whole test call; we
    # deliberately keep one frozen instant across every generated input.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    q=_QUERY,
    hint=_TOP_HINT,
    industry=_PARAMS["industry"],
    dealtype=_PARAMS["dealtype"],
    author=_PARAMS["author"],
    content_type=_PARAMS["content_type"],
    tag=_PARAMS["tag"],
    top_k=_PARAMS["top_k"],
    window=_date_window(),
)
def test_search_always_well_formed(app_client, frozen_clock, q, hint, industry, dealtype,
                                   author, content_type, tag, top_k, window) -> None:
    from_date, to_date = window
    params = {
        "q": f"{hint} {q}".strip() if hint else q,
        "top_k": top_k,
    }
    if industry:
        params["industry"] = industry
    if dealtype:
        params["dealtype"] = dealtype
    if author:
        params["author"] = author
    if content_type:
        params["content_type"] = content_type
    if tag:
        params["tag"] = tag
    if from_date:
        params["from_date"] = from_date
    if to_date:
        params["to_date"] = to_date

    resp = app_client.get("/search", params=params)
    assert resp.status_code == 200, f"{params} -> {resp.status_code}: {resp.text[:300]}"
    body = resp.json()
    assert "results" in body
    results = body["results"]
    assert isinstance(results, list)
    for item in results:
        assert "title" in item and isinstance(item["title"], str)
        assert "url" in item and isinstance(item["url"], str)
        assert "score" in item and isinstance(item["score"], float)
        assert "published_date" in item
    # An explicit date window must be honoured by every returned row.
    if from_date and to_date:
        for item in results:
            pd = item.get("published_date")
            assert pd is not None
            assert from_date <= pd.split("T")[0] <= to_date, (pd, from_date, to_date)
    # An explicit industry facet must be honoured.
    if industry:
        for item in results:
            assert industry in (item.get("industry_names") or []), (industry, item.get("title"))


def test_search_empty_window_is_honest_not_stale(app_client, frozen_clock) -> None:
    # The corpus has no 2030 articles, so a 2030 window must be empty — even
    # though the un-filtered corpus has plenty of rows the dimmer could serve.
    resp = app_client.get(
        "/search",
        params={"q": "funding round unique-token-x7q", "from_date": "2030-01-01", "to_date": "2030-12-31"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_window_returns_only_in_window(app_client, frozen_clock) -> None:
    # March 2025 in the corpus holds exactly one article (published 2025-03-02).
    resp = app_client.get(
        "/search",
        params={"q": "nestle unique-window-token-a1", "from_date": "2025-03-01", "to_date": "2025-03-31"},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    ids = {int(r["id"]) for r in results}
    assert ids == {2}
    for r in results:
        assert "2025-03-02" == r["published_date"]


def test_search_industry_facet_filters(app_client, frozen_clock) -> None:
    resp = app_client.get("/search", params={"q": "coffee unique-ind-token-b2", "industry": "Finance"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results
    for r in results:
        assert "Finance" in (r.get("industry_names") or [])


def test_search_top_k_bounds_result_count(app_client, frozen_clock) -> None:
    # NB: a "top"-hinted query is scaled UP by suggested_top_k; use a plain
    # query so the explicit top_k is the effective cap.
    resp = app_client.get("/search", params={"q": "generic liquidity query", "top_k": 3})
    assert resp.status_code == 200
    assert 0 <= len(resp.json()["results"]) <= 3


def test_search_seed_data_has_expected_span() -> None:
    # The fixture corpus intentionally spans years so date windows can select.
    articles = seed_articles()
    years = {int(a["published_date"][:4]) for a in articles.values()}
    assert max(years) >= 2025
    assert min(years) <= 2021
