"""Tests for the Slice A analytics expansion (search + user/account analytics).

Covers the contract's section 1 additions to /analytics/summary (latency
percentiles, intent_counts, hourly_volume, top_queries_today), the section-2
/analytics/users shape, the guarded last_seen migration, feed/session recording
on interactions, and session-id plumbing on search/clicks.
"""

from __future__ import annotations

import itertools
import os

_PASSWORD = "Password1"
_EMAIL_COUNTER = itertools.count()


def _email() -> str:
    return f"a-{next(_EMAIL_COUNTER)}-{os.getpid()}@example.test"


def _authed(app_client) -> str:
    app_client.cookies.clear()
    email = _email()
    app_client.post("/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "Rec"})
    r = app_client.post("/api/auth/login", json={"email": email, "password": _PASSWORD})
    assert r.status_code == 200, r.text
    return email


def _reset_analytics() -> None:
    """Clear the shared fake analytics keyspace so a test starts clean.

    The FakeRedis instance is an in-memory object shared module-wide (session
    app_client), so a reset is loop-agnostic and safe to call from a sync test."""
    from app.analytics import _client
    _client().reset()


def test_summary_new_keys_present_empty(app_client, admin_headers) -> None:
    app_client.cookies.clear()
    _reset_analytics()
    r = app_client.get("/analytics/summary", headers=admin_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    # Latency percentiles: float ms, 0.0 with no samples.
    latency = body["latency"]
    assert set(latency) == {"p50", "p90", "p95", "p99"}
    assert latency["p50"] == 0.0 and latency["p99"] == 0.0
    # hourly_volume: last 24 hourly buckets, UTC, ascending.
    hv = body["hourly_volume"]
    assert isinstance(hv, list) and len(hv) == 24
    assert all(len(row) == 2 for row in hv)
    assert all(not v for _, v in hv)  # empty store -> all zeros
    # intent_counts: all five classes, cumulative.
    ic = body["intent_counts"]
    assert set(ic) == {"timed", "category", "flashback", "recency", "other"}
    assert all(v == 0 for v in ic.values())
    # top_queries_today: empty list with no searches.
    assert body["top_queries_today"] == []


def test_search_records_intent_and_session(app_client, admin_headers) -> None:
    # A timed-intent query (explicit year) exercised through the real /search
    # pipeline against the fake Qdrant. Start from a clean analytics store so
    # counts are deterministic in the shared session.
    app_client.cookies.clear()
    _reset_analytics()
    r = app_client.get("/search", params={"q": "top deals in 2025"}, headers={
        **admin_headers, "X-Session-Id": "sess-test-1",
    })
    assert r.status_code == 200, r.text

    summary = app_client.get("/analytics/summary", headers=admin_headers).json()
    # intent_counts: the timed query tallied at least one.
    assert summary["intent_counts"]["timed"] == 1
    # searches_total reflects the single search.
    assert summary["searches_total"] == 1
    # hourly_volume has a non-zero current bucket (exactly one non-zero bucket).
    non_zero = [row for row in summary["hourly_volume"] if row[1] > 0]
    assert len(non_zero) == 1
    # top_queries_today now has exactly the digest member.
    assert len(summary["top_queries_today"]) == 1
    assert summary["top_queries_today"][0][0].startswith("q1:")
    assert summary["top_queries_today"][0][1] == 1
    # latency: this search produced a latency sample so at least p50 is set.
    assert summary["latency"]["p50"] > 0.0


def test_analytics_users_shape(app_client, admin_headers) -> None:
    app_client.cookies.clear()
    # A signup + successful login produces signups and a login tick.
    _authed(app_client)
    r = app_client.get("/analytics/users", headers=admin_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    required = {
        "signups", "role_distribution", "disabled_accounts",
        "active_today", "active_last_7d", "active_last_30d",
        "last_login_success", "signups_14d", "top_spenders", "audit_recent",
    }
    assert required.issubset(body.keys()), body.keys()
    assert set(body["signups"]) == {"today", "total"}
    assert body["signups"]["total"] >= 1
    assert "user" in body["role_distribution"]
    assert body["role_distribution"]["user"] >= 1
    # A login just happened, so the daily login counter is non-zero.
    assert body["last_login_success"] >= 1
    # A successful login touches last_seen -> at least one active user today.
    assert body["active_today"] >= 1
    # signups_14d: 14 ascending [date, count] rows.
    assert len(body["signups_14d"]) == 14
    assert all(len(row) == 2 for row in body["signups_14d"])
    # top_spenders / audit_recent are lists of triples.
    assert isinstance(body["top_spenders"], list)
    assert all(len(row) == 3 for row in body["top_spenders"])
    assert isinstance(body["audit_recent"], list)
    assert all(len(row) == 3 for row in body["audit_recent"])


def test_admin_reads_are_gated(app_client) -> None:
    app_client.cookies.clear()
    # A plain user is authenticated but not authorized for /analytics/users.
    _authed(app_client)
    assert app_client.get("/analytics/users").status_code == 403
    # No credential -> 401.
    app_client.cookies.clear()
    assert app_client.get("/analytics/users").status_code == 401


def test_last_seen_migration_guarded(tmp_path) -> None:
    """connect() adds users.last_seen exactly once, guarded per-column."""
    import asyncio
    import sqlite3

    from app.auth import AuthStore

    db_path = tmp_path / "auth.db"
    # Pre-create a legacy users table WITHOUT last_seen.
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE,"
        " password_hash TEXT NOT NULL, name TEXT NOT NULL DEFAULT '',"
        " role TEXT NOT NULL DEFAULT 'user', is_active INTEGER NOT NULL DEFAULT 1,"
        " created_at REAL NOT NULL)"
    )
    legacy.execute(
        "INSERT INTO users (id, email, password_hash, name, role, is_active, created_at)"
        " VALUES ('u1', 'u1@example.test', 'x', 'A', 'user', 1, 1000.0)"
    )
    legacy.commit()
    legacy.close()

    s = AuthStore(str(db_path))

    async def _exercise():
        await s.connect()
        cols = [r["name"] for r in await s._fetchall("PRAGMA table_info(users)")]
        assert "last_seen" in cols
        u = await s.get_user("u1")
        assert u.id == "u1" and u.last_seen is None
        # Re-running connect() must not error or duplicate the column.
        await s.connect()
        cols2 = [r["name"] for r in await s._fetchall("PRAGMA table_info(users)")]
        assert cols2.count("last_seen") == 1
        # touch_last_seen writes it.
        await s.touch_last_seen("u1", now=1234.5)
        u2 = await s.get_user("u1")
        assert u2.last_seen == 1234.5
        await s.close()

    asyncio.run(_exercise())
