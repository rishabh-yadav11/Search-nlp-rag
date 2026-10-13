

from __future__ import annotations

import itertools
import os

from app.config import config

_PASSWORD = "Password1"
_EMAIL_COUNTER = itertools.count()


def _email() -> str:
    return f"feed-{next(_EMAIL_COUNTER)}-{os.getpid()}@example.test"


def _signup_login(app_client) -> None:

    app_client.cookies.clear()
    email = _email()
    r = app_client.post(
        "/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "Feed"}
    )
    assert r.status_code == 200, r.text
    r = app_client.post("/api/auth/login", json={"email": email, "password": _PASSWORD})
    assert r.status_code == 200, r.text


def test_feed_requires_auth(app_client) -> None:
    app_client.cookies.clear()
    assert app_client.get("/api/feed/subscriptions").status_code == 401
    assert app_client.get("/api/feed").status_code == 401


def test_subscription_add_list_remove_idempotent(app_client) -> None:
    _signup_login(app_client)


    r = app_client.get("/api/feed/subscriptions")
    assert r.status_code == 200, r.text
    assert r.json() == {"subscriptions": []}


    r = app_client.post(
        "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "added": True}


    r = app_client.get("/api/feed/subscriptions")
    assert r.status_code == 200, r.text
    subs = r.json()["subscriptions"]
    assert len(subs) == 1
    assert subs[0]["kind"] == "tag"
    assert subs[0]["value"] == "funding"
    assert isinstance(subs[0]["created_at"], float)


    r = app_client.post(
        "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "added": False}
    assert len(app_client.get("/api/feed/subscriptions").json()["subscriptions"]) == 1


    r = app_client.request(
        "DELETE", "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "removed": True}
    assert app_client.get("/api/feed/subscriptions").json()["subscriptions"] == []


    r = app_client.request(
        "DELETE", "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "removed": False}


def test_subscriptions_validation(app_client) -> None:
    _signup_login(app_client)


    r = app_client.post(
        "/api/feed/subscriptions", json={"kind": "author", "value": "Alice"}
    )
    assert r.status_code == 422, r.text


    r = app_client.post("/api/feed/subscriptions", json={"kind": "tag", "value": "   "})
    assert r.status_code == 422, r.text
    r = app_client.post("/api/feed/subscriptions", json={"kind": "tag", "value": ""})
    assert r.status_code == 422, r.text


    from app.input_hygiene import MAX_FACET_VALUE_LEN

    r = app_client.post(
        "/api/feed/subscriptions",
        json={"kind": "tag", "value": "x" * (MAX_FACET_VALUE_LEN + 1)},
    )
    assert r.status_code == 422, r.text


    assert app_client.get("/api/feed/subscriptions").json()["subscriptions"] == []


def test_feed_empty_subscriptions_honest_note(app_client) -> None:
    _signup_login(app_client)
    r = app_client.get("/api/feed")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"] == []
    assert isinstance(body["note"], str)
    assert "subscribe" in body["note"].lower()


def test_feed_matching_subscription_returns_recency_desc(app_client) -> None:
    _signup_login(app_client)
    r = app_client.post(
        "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    )
    assert r.status_code == 200, r.text

    r = app_client.get("/api/feed?limit=20")
    assert r.status_code == 200, r.text
    body = r.json()
    results = body["results"]


    assert [a["id"] for a in results] == [1, 5]
    assert body["note"] is None
    for a in results:
        assert a["score"] == 0.0
        for field in (
            "id",
            "title",
            "url",
            "published_date",
            "category",
            "summary",
            "author_names",
            "industry_names",
            "dealtype_names",
            "tag_names",
            "content_type",
            "score",
        ):
            assert field in a, a


def test_feed_industry_subscription(app_client) -> None:
    _signup_login(app_client)
    r = app_client.post(
        "/api/feed/subscriptions", json={"kind": "industry", "value": "Mobility"}
    )
    assert r.status_code == 200, r.text
    r = app_client.get("/api/feed")
    assert r.status_code == 200, r.text
    assert [a["id"] for a in r.json()["results"]] == [2]


def test_feed_any_match_union(app_client) -> None:

    _signup_login(app_client)
    assert app_client.post(
        "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    ).status_code == 200
    assert app_client.post(
        "/api/feed/subscriptions", json={"kind": "industry", "value": "Mobility"}
    ).status_code == 200

    r = app_client.get("/api/feed")
    assert r.status_code == 200, r.text

    assert [a["id"] for a in r.json()["results"]] == [1, 2, 5]


def test_feed_no_match_honest_empty(app_client) -> None:
    _signup_login(app_client)
    assert app_client.post(
        "/api/feed/subscriptions", json={"kind": "tag", "value": "does-not-exist"}
    ).status_code == 200
    r = app_client.get("/api/feed")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"] == []
    assert isinstance(body["note"], str)
    assert body["note"]


def test_feed_limit_param_bounds(app_client) -> None:
    _signup_login(app_client)

    assert app_client.get("/api/feed?limit=0").status_code == 422
    assert app_client.get("/api/feed?limit=51").status_code == 422

    assert app_client.post(
        "/api/feed/subscriptions", json={"kind": "tag", "value": "funding"}
    ).status_code == 200
    r = app_client.get("/api/feed?limit=1")
    assert r.status_code == 200, r.text
    assert [a["id"] for a in r.json()["results"]] == [1]


def test_feed_subscription_cap_400(app_client, monkeypatch) -> None:
    _signup_login(app_client)
    monkeypatch.setattr(config, "FEED_MAX_SUBSCRIPTIONS", 2)


    for value in ("funding", "fintech"):
        r = app_client.post("/api/feed/subscriptions", json={"kind": "tag", "value": value})
        assert r.status_code == 200, r.text
        assert r.json()["added"] is True


    r = app_client.post("/api/feed/subscriptions", json={"kind": "tag", "value": "ipo"})
    assert r.status_code == 400, r.text


    r = app_client.post("/api/feed/subscriptions", json={"kind": "tag", "value": "funding"})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "added": False}


    subs = app_client.get("/api/feed/subscriptions").json()["subscriptions"]
    assert len(subs) == 2
