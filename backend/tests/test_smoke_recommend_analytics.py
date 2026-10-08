"""Smoke tests for the recommender family and the analytics endpoints.

Recommenders run against the fake Qdrant; analytics use the in-memory fake
Redis. Auth gating is asserted for the admin-read surfaces.
"""

from __future__ import annotations

import itertools
import os

_PASSWORD = "Password1"
_EMAIL_COUNTER = itertools.count()


def _email() -> str:
    return f"rec-{next(_EMAIL_COUNTER)}-{os.getpid()}@example.test"


def _authed(app_client):
    app_client.cookies.clear()
    email = _email()
    app_client.post("/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "Rec"})
    r = app_client.post("/api/auth/login", json={"email": email, "password": _PASSWORD})
    assert r.status_code == 200, r.text
    return email


def test_interaction_recorded(app_client) -> None:
    _authed(app_client)
    r = app_client.post("/recommend/interaction", json={"article_id": 1, "interaction_type": "click"})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok", "article_id": 1}


def test_interaction_unknown_article_404(app_client) -> None:
    _authed(app_client)
    r = app_client.post("/recommend/interaction", json={"article_id": 9999, "interaction_type": "click"})
    assert r.status_code == 404


def test_interaction_invalid_type_422(app_client) -> None:
    _authed(app_client)
    r = app_client.post("/recommend/interaction", json={"article_id": 1, "interaction_type": "hover"})
    assert r.status_code == 422


def test_interaction_requires_auth(app_client) -> None:
    app_client.cookies.clear()
    r = app_client.post("/recommend/interaction", json={"article_id": 1, "interaction_type": "click"})
    assert r.status_code == 401


def test_trending_reflects_recorded_click(app_client) -> None:
    _authed(app_client)
    assert app_client.post("/recommend/interaction",
                           json={"article_id": 1, "interaction_type": "click"}).status_code == 200
    r = app_client.get("/recommend/trending", params={"limit": 3})
    assert r.status_code == 200
    articles = r.json().get("articles", r.json().get("results", []))
    ids = [int(a.get("id")) for a in articles if isinstance(a, dict)]
    assert 1 in ids, articles


def test_recommend_family(app_client) -> None:
    _authed(app_client)

    r = app_client.get("/recommend/for-you", params={"limit": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    results = body.get("results", [])
    assert isinstance(results, list)

    r = app_client.get("/recommend/similar/1", params={"limit": 3})
    assert r.status_code == 200, r.text
    similar = r.json().get("results", [])
    assert isinstance(similar, list)
    assert all((a or {}).get("id") != 1 for a in similar)

    r = app_client.post("/recommend/similar/batch", json={"article_ids": [1, 2]})
    assert r.status_code == 200, r.text
    batch = r.json().get("results", [])
    assert isinstance(batch, list)
    assert len(batch) == 2


def test_recommend_for_you_requires_auth(app_client) -> None:
    app_client.cookies.clear()
    assert app_client.get("/recommend/for-you").status_code == 401


def test_trending_requires_auth(app_client) -> None:
    # /recommend/trending is behind require_auth (a 401 without a session).
    app_client.cookies.clear()
    assert app_client.get("/recommend/trending").status_code == 401


def test_analytics_click_anonymous_beacon(app_client) -> None:
    app_client.cookies.clear()
    r = app_client.post("/analytics/click", json={"query": "funding", "position": 1, "id": 1})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    # An unknown article id is still a tolerated beacon (no ranking vote).
    r2 = app_client.post("/analytics/click", json={"query": "funding", "position": 2, "id": 9999})
    assert r2.status_code == 200


def test_analytics_admin_reads_are_gated(app_client, admin_headers) -> None:
    app_client.cookies.clear()
    # Admin (service token, analytics:read) can read both reports.
    summary = app_client.get("/analytics/summary", headers=admin_headers)
    assert summary.status_code == 200
    assert isinstance(summary.json(), dict)
    chat_report = app_client.get("/analytics/chat", headers=admin_headers)
    assert chat_report.status_code == 200

    # A plain user (cookie only) is authenticated but not authorized.
    _authed(app_client)
    assert app_client.get("/analytics/summary").status_code == 403
    assert app_client.get("/analytics/chat").status_code == 403

    # No credential at all -> 401.
    app_client.cookies.clear()
    assert app_client.get("/analytics/summary").status_code == 401
