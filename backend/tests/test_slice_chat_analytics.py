"""Slice B: chat analytics (new global_stats keys) + rating endpoint tests.

Covers the four required areas: byte-compatibility of existing global_stats
keys, the rating endpoint round-trip and non-owner 404, the guarded model/rating
column migration (idempotent), and the percentile-offset helper.
"""

from __future__ import annotations

import itertools
import os

import pytest

from app.chat import ChatStore, _percentile_offset
from app.config import config

_PASSWORD = "Password1"
_EMAIL_COUNTER = itertools.count()
_QUESTION = "what happened with Ola Electric funding news"

EXPECTED_STATS_KEYS = {
    # existing keys (must stay byte-compatible)
    "sessions", "users", "messages", "total_tokens", "total_cost",
    "avg_latency_ms", "top_by_cost", "top_by_tokens", "sessions_today",
    "daily_sessions",
    # new keys from the frozen contract section 3
    "latency", "failed_turn_rate", "abandon_rate", "citation_rate",
    "avg_sources_per_cited", "avg_tokens_per_message", "avg_cost_per_message",
    "budget", "model_usage", "daily_budget", "non_llm_answer_rate",
}


def _email(tag: str) -> str:
    return f"{tag}-{next(_EMAIL_COUNTER)}-{os.getpid()}@example.test"


def _authed(app_client, tag: str = "rating") -> None:
    app_client.cookies.clear()
    email = _email(tag)
    r = app_client.post("/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "Chat"})
    assert r.status_code == 200, r.text
    r = app_client.post("/api/auth/login", json={"email": email, "password": _PASSWORD})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------------
# ChatStore-level helpers (async store, run each await via a fresh loop)
# ---------------------------------------------------------------------------


def run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    store = ChatStore(str(tmp_path / "chat.db"))
    run(store.connect())
    yield store
    run(store.close())


def _append(store, session: object, role: str, content: str, **kw):
    return run(store._append_authorized(session, role, content, **kw))


# ---------------------------------------------------------------------------
# global_stats: new keys + byte-compatibility of the existing ones
# ---------------------------------------------------------------------------


def test_global_stats_new_keys_byte_compat(store):
    s1 = run(store.create_session("alice"))
    _append(store, s1, "user", "hello")
    # Small-talk answer: no LLM -> no tokens, no cost, no sources.
    _append(store, s1, "assistant", "Hi! How can I help?", latency_ms=10.0)

    s2 = run(store.create_session("bob"))
    _append(store, s2, "user", "Ola funding?")
    _append(store, s2, "assistant", "Ola raised a round.", sources=[{"id": 1}, {"id": 2}],
             prompt_tokens=100, completion_tokens=200, cost=0.5, latency_ms=30.0)
    _append(store, s2, "assistant", "More context.", sources=[{"id": 9}],
             prompt_tokens=50, completion_tokens=50, cost=0.1, latency_ms=20.0,
             aborted=True)

    # A session that was abandoned after exactly ONE message (never answered).
    s3 = run(store.create_session("carol"))
    _append(store, s3, "user", "abandoned question")

    stats = run(store.global_stats())
    assert set(stats.keys()) == EXPECTED_STATS_KEYS

    # --- existing keys: byte-compatible values (unchanged semantics) ---
    assert stats["sessions"] == 3
    assert stats["users"] == 3
    assert stats["messages"] == 6
    # assistant-only token/cost sums
    assert stats["total_tokens"] == 100 + 200 + 50 + 50
    assert stats["total_cost"] == 0.6
    # assistant latency > 0 only, avg
    assert stats["avg_latency_ms"] == round((30.0 + 20.0 + 10.0) / 3, 1)
    assert stats["sessions_today"] == 3
    assert len(stats["daily_sessions"]) == 1

    # --- new keys ---
    # assistant latencies [10, 20, 30]; nearest-rank p50 -> offset 1 -> 20
    assert stats["latency"]["p50"] == 20.0
    assert stats["latency"]["p90"] == 30.0
    assert stats["latency"]["p95"] == 30.0
    # one aborted assistant row out of 6 messages (rounded to 4 dp, as served)
    assert stats["failed_turn_rate"] == round(1 / 6, 4)
    # exactly one session (s3) holds a single message
    assert stats["abandon_rate"] == round(1 / 3, 4)
    # 2 of 3 assistant messages cite >= 1 source
    assert stats["citation_rate"] == round(2 / 3, 4)
    # cited messages have 2 and 1 sources -> avg 1.5
    assert stats["avg_sources_per_cited"] == round(1.5, 4)
    # tokens 400 / 3 assistant; cost 0.6 / 3 assistant
    assert stats["avg_tokens_per_message"] == round(400 / 3, 4)
    assert stats["avg_cost_per_message"] == round(0.6 / 3, 4)
    # one non-LLM assistant answer (zero tokens+cost: the small-talk one)
    assert stats["non_llm_answer_rate"] == round(1 / 3, 4)
    # model_usage: all assistant rows carry config.LLM_MODEL
    assert stats["model_usage"] == [[config.LLM_MODEL, 3, 400, round(0.6, 4)]]
    assert stats["budget"]["daily_limit_usd"] == config.LLM_DAILY_BUDGET_USD
    assert stats["budget"]["today_spent_usd"] == round(0.6, 4)
    assert len(stats["daily_budget"]) == 1


def test_global_stats_empty_store_zero_defaults(store):
    stats = run(store.global_stats())
    assert stats["latency"] == {"p50": 0.0, "p90": 0.0, "p95": 0.0}
    assert stats["failed_turn_rate"] == 0.0
    assert stats["abandon_rate"] == 0.0
    assert stats["citation_rate"] == 0.0
    assert stats["avg_sources_per_cited"] == 0.0
    assert stats["avg_tokens_per_message"] == 0.0
    assert stats["avg_cost_per_message"] == 0.0
    assert stats["non_llm_answer_rate"] == 0.0
    assert stats["model_usage"] == []
    assert stats["daily_budget"] == []
    assert stats["messages"] == 0


def test_percentile_offset_helper():
    # n=0 -> 0 (no sample, caller returns 0.0)
    assert _percentile_offset(0, 50) == 0
    # single sample: every percentile = that sample (offset 0)
    assert _percentile_offset(1, 50) == 0
    assert _percentile_offset(1, 95) == 0
    # four samples [10,20,30,40]: p50 -> index 1 (20), p90 -> index 3 (40)
    assert _percentile_offset(4, 50) == 1
    assert _percentile_offset(4, 90) == 3
    # p95 with only 4 samples clamps to the last element
    assert _percentile_offset(4, 95) == 3
    # even n p50 picks index n/2 - 1
    assert _percentile_offset(6, 50) == 2


# ---------------------------------------------------------------------------
# Guarded migration: model/rating columns added idempotently, backfilled
# ---------------------------------------------------------------------------


def test_migration_adds_model_and_rating_idempotently(tmp_path):
    import aiosqlite

    path = str(tmp_path / "legacy.db")
    # A database from before the model/rating columns existed.
    async def make_legacy():
        db = await aiosqlite.connect(path)
        await db.execute(
            "CREATE TABLE messages ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
            " role TEXT NOT NULL, content TEXT NOT NULL,"
            " sources TEXT NOT NULL DEFAULT '[]',"
            " created_at REAL NOT NULL,"
            " prompt_tokens INTEGER NOT NULL DEFAULT 0,"
            " completion_tokens INTEGER NOT NULL DEFAULT 0,"
            " cost REAL NOT NULL DEFAULT 0,"
            " latency_ms REAL NOT NULL DEFAULT 0,"
            " aborted INTEGER NOT NULL DEFAULT 0)"
        )
        await db.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,"
            " title TEXT NOT NULL DEFAULT 'New chat', created_at REAL NOT NULL,"
            " updated_at REAL NOT NULL)"
        )
        await db.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at)"
            " VALUES ('legacy-session', 'alice', 'New chat', 1.0, 1.0)"
        )
        await db.execute(
            "INSERT INTO messages (session_id, role, content, created_at)"
            " VALUES ('legacy-session', 'assistant', 'old answer', 1.0)"
        )
        await db.commit()
        await db.close()

    run(make_legacy())

    store = ChatStore(path)
    run(store.connect())

    async def _cols():
        db = await aiosqlite.connect(path)
        cols = [r[1] for r in await db.execute_fetchall("PRAGMA table_info(messages)")]
        legacy_model = (
            await (await db.execute("SELECT model FROM messages WHERE id = 1")).fetchone()
        )[0]
        await db.close()
        return cols, legacy_model

    cols, legacy_model = run(_cols())
    assert "model" in cols
    assert "rating" in cols
    # Legacy rows are backfilled with the configured default model.
    assert legacy_model == config.LLM_MODEL

    # Reconnect (migration must be idempotent -- no duplicate-column error).
    run(store.close())
    run(store.connect())
    cols, _ = run(_cols())
    assert "model" in cols and "rating" in cols

    # A fresh message inserted through the store carries the real model.
    async def _session_and_row():
        session = await store.create_session("alice2")
        row = await store._append_authorized(session, "assistant", "new answer")
        return row

    row = run(_session_and_row())
    async def _row_model():
        db = await aiosqlite.connect(path)
        val = (await (await db.execute("SELECT model FROM messages WHERE id = ?", (row.id,))).fetchone())[0]
        await db.close()
        return val

    assert run(_row_model()) == config.LLM_MODEL
    run(store.close())


def test_store_adds_new_columns_on_connect(tmp_path):
    """A brand-new store's schema includes model + rating without ALTER (CREATE)."""
    store = ChatStore(str(tmp_path / "fresh.db"))
    run(store.connect())
    run(store.close())


# ---------------------------------------------------------------------------
# Rating endpoint: round-trip + non-owner 404
# ---------------------------------------------------------------------------


def test_rating_round_trip_and_non_owner(app_client):
    _authed(app_client, "owner")
    session_id = app_client.post("/api/chat/sessions").json()["id"]
    turn = app_client.post(f"/api/chat/sessions/{session_id}/messages", json={"content": _QUESTION})
    assert turn.status_code == 200, turn.text[:500]
    msg_id = turn.json()["assistant"]["id"]

    # Round-trip: rate +1, clear with 0, then a negative rating.
    for rating in (1, 0, -1):
        r = app_client.post(f"/api/chat/messages/{msg_id}/rating", json={"rating": rating})
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True}

    # Out of range -> 422, never a partial write.
    assert app_client.post(f"/api/chat/messages/{msg_id}/rating", json={"rating": 2}).status_code == 422
    assert app_client.post(f"/api/chat/messages/{msg_id}/rating", json={"rating": -2}).status_code == 422

    # Non-owner: a second account cannot rate the first account's message.
    _authed(app_client, "other")
    assert app_client.post(f"/api/chat/messages/{msg_id}/rating", json={"rating": 1}).status_code == 404
    # Nonexistent message id -> 404.
    assert app_client.post("/api/chat/messages/999999/rating", json={"rating": 1}).status_code == 404


def test_rating_requires_auth(app_client):
    app_client.cookies.clear()
    r = app_client.post("/api/chat/messages/1/rating", json={"rating": 1})
    assert r.status_code == 401
