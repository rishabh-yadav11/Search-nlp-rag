"""Smoke tests for /search and /facets over the real app (faked stores)."""

from __future__ import annotations


def test_search_returns_structured_payload(app_client, frozen_clock) -> None:
    resp = app_client.get("/search", params={"q": "funding round"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"query", "results", "cached", "latency_ms"}
    assert body["query"] == "funding round"
    assert isinstance(body["results"], list)
    assert body["results"], "expected at least one seeded result for a funding query"
    first = body["results"][0]
    assert {"title", "url", "score"} <= set(first)
    assert isinstance(first["score"], float)


def test_search_top_k_honored(app_client, frozen_clock) -> None:
    # "top"-hinted queries are scaled up by suggested_top_k; use a plain query
    # so the explicit top_k is the effective cap.
    resp = app_client.get("/search", params={"q": "anything capital general", "top_k": 2})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 2


def test_search_facets_are_filtered(app_client, frozen_clock) -> None:
    resp = app_client.get("/search", params={"q": "compare facet filter", "industry": "Finance"})
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert "Finance" in (item.get("industry_names") or [])


def test_search_date_window_honest_empty(app_client, frozen_clock) -> None:
    resp = app_client.get(
        "/search",
        params={"q": "unique empty-window smoke", "from_date": "2035-01-01", "to_date": "2035-12-31"},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_rejects_empty_after_normalise(app_client) -> None:
    # A query that normalises away to nothing is a 400, never a silent search.
    resp = app_client.get("/search", params={"q": "\u0000\u0001\u0002"})
    assert resp.status_code == 400


def test_facets_returns_controlled_vocabularies(app_client) -> None:
    resp = app_client.get("/facets")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"industry", "dealtype", "tags"}
    # Every value is a non-empty string.
    for values in body.values():
        assert all(isinstance(v, str) and v for v in values)
    # The seed corpus's controlled values must be discoverable.
    assert "Finance" in body["industry"]
    assert "Venture Capital" in body["dealtype"]


def test_facets_reachable_unauthenticated(app_client) -> None:
    # /facets is a public autocomplete surface (no auth dependency).
    resp = app_client.get("/facets")
    assert resp.status_code == 200
