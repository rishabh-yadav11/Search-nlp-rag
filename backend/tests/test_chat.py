import asyncio
import json
import logging
import pathlib
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from functools import partial
from typing import ClassVar

import anyio
import httpx
import openai
import pytest
from _support import run_sync as _run
from conftest import auth_cookie
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import auth as auth_module
from app import chat as chat_module
from app import cost_budget as cost_budget_module
from app.auth import AuthStore
from app.chat import ChatStore, _smalltalk_reply

USER_A = "user-a-device-id-0001"
USER_B = "user-b-device-id-0002"
EMAIL_A = "user-a@example.com"
EMAIL_B = "user-b@example.com"


@pytest.fixture(autouse=True)
def _stub_temporal_date_window(monkeypatch):
    """retrieve_by_date_window needs a live Qdrant client; these tests stub retrieval above it."""
    from app import main as _main

    async def _noop(*args, **kwargs):
        return []

    monkeypatch.setattr(_main, "retrieve_by_date_window", _noop)


@pytest.fixture(autouse=True)
def _shipped_weak_gate(monkeypatch):
    """Deployment settings: a machine whose .env retunes them would decide whether these tests take the weak-fallback branch."""
    monkeypatch.setattr(chat_module.config, "WEAK_RESULT_SCORE", 0.3)
    monkeypatch.setattr(chat_module.config, "WEAK_RESULT_MIN_STRONG", 3)


def _store(tmp_path):
    s = ChatStore(str(tmp_path / "chat.db"))
    _run(s.connect())
    return s


def _auth_store(tmp_path):
    s = AuthStore(str(tmp_path / "auth.db"))
    _run(s.connect())
    return s


def _auth_cookies(auth_store, email=EMAIL_A, role="user"):
    """The credential is an HttpOnly cookie, so a test authenticates as a browser does: by cookie, never by an ``Authorization`` header."""
    user = _run(auth_store.get_user_by_email(email))
    if user is None:
        user = _run(auth_store.create_user(email, "secret1", email.split("@")[0], role))
    elif user.role != role:
        _run(auth_store.update_user(user.id, None, role, None))
    token = _run(auth_store.issue_token(user.id, 7))
    return auth_cookie(token)


def test_create_and_list_sessions(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        b = _run(store.create_session(USER_A))
        listed = _run(store.list_sessions(USER_A))
        assert [s.id for s in listed] == [b.id, a.id]
        assert _run(store.list_sessions(USER_B)) == []
    finally:
        _run(store.close())


def test_ownership_isolation(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        assert _run(store.get_session(a.id, USER_A)) is not None
        assert _run(store.get_session(a.id, USER_B)) is None
        with pytest.raises(HTTPException) as exc:
            _run(store.messages_page(a.id, USER_B))
        assert exc.value.status_code == 404
        with pytest.raises(HTTPException) as exc:
            _run(store.append_message(a.id, USER_B, "user", "hi"))
        assert exc.value.status_code == 404
    finally:
        _run(store.close())


def test_append_and_read_messages(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        u = _run(store.append_message(a.id, USER_A, "user", "Hello"))
        m = _run(store.append_message(a.id, USER_A, "assistant", "Hi there", [{"id": 1, "title": "Src"}]))
        msgs, total = _run(store.messages_page(a.id, USER_A))
        assert total == 2
        assert [x.content for x in msgs] == ["Hello", "Hi there"]
        assert msgs[1].sources == [{"id": 1, "title": "Src"}]
        assert msgs[1].role == "assistant"
        assert u.id < m.id
    finally:
        _run(store.close())


def test_recent_turns_order(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        for role, text in [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]:
            _run(store.append_message(a.id, USER_A, role, text))
        turns = _run(store.recent_turns(a.id, USER_A, max_turns=2))
        assert [t.content for t in turns] == ["q1", "a1", "q2", "a2"]
    finally:
        _run(store.close())


def test_rename_and_delete(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        renamed = _run(store.rename_session(a.id, USER_A, "My title"))
        assert renamed.title == "My title"
        _run(store.append_message(a.id, USER_A, "user", "x"))
        _run(store.delete_session(a.id, USER_A))
        assert _run(store.get_session(a.id, USER_A)) is None
    finally:
        _run(store.close())


def test_purge_expired(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        _run(store.append_message(a.id, USER_A, "user", "old"))
        _run(store._db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time() - 200 * 86400, a.id)))
        _run(store._db.commit())
        assert _run(store.purge_expired()) == 1
        assert _run(store.get_session(a.id, USER_A)) is None
    finally:
        _run(store.close())


def _legacy_db(path, messages_schema):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE sessions ("
        " id TEXT PRIMARY KEY, user_id TEXT NOT NULL,"
        " title TEXT NOT NULL DEFAULT 'New chat',"
        " created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.execute(messages_schema)
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES ('s1', 'u1', 'Legacy', 0, 0)"
    )
    conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES ('s1', 'user', 'hello', 0)")
    conn.commit()
    conn.close()


def test_connect_migrates_legacy_messages_schema(tmp_path):
    db_path = tmp_path / "chat.db"
    _legacy_db(
        db_path,
        "CREATE TABLE messages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
        " role TEXT NOT NULL, content TEXT NOT NULL,"
        " sources TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL)",
    )
    store = ChatStore(str(db_path))
    _run(store.connect())
    try:
        cols = _run(store._db.execute_fetchall("PRAGMA table_info(messages)"))
        names = {c["name"] for c in cols}
        for name in ("prompt_tokens", "completion_tokens", "cost", "latency_ms"):
            assert name in names
        rows = _run(store._db.execute_fetchall("SELECT * FROM messages"))
        assert rows[0]["prompt_tokens"] == 0
        assert rows[0]["latency_ms"] == 0
    finally:
        _run(store.close())


def test_connect_adds_missing_latency_ms_only(tmp_path):
    db_path = tmp_path / "chat.db"
    _legacy_db(
        db_path,
        "CREATE TABLE messages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
        " role TEXT NOT NULL, content TEXT NOT NULL,"
        " sources TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL,"
        " prompt_tokens INTEGER NOT NULL DEFAULT 0,"
        " completion_tokens INTEGER NOT NULL DEFAULT 0,"
        " cost REAL NOT NULL DEFAULT 0)",
    )
    store = ChatStore(str(db_path))
    _run(store.connect())
    try:
        cols = _run(store._db.execute_fetchall("PRAGMA table_info(messages)"))
        assert "latency_ms" in {c["name"] for c in cols}
    finally:
        _run(store.close())


def test_close_is_idempotent(tmp_path):
    store = _store(tmp_path)
    _run(store.close())
    assert store._db is None
    _run(store.close())
    assert store._db is None


def test_rename_delete_missing_session_404(tmp_path):
    store = _store(tmp_path)
    try:
        with pytest.raises(HTTPException) as exc:
            _run(store.rename_session("missing", USER_A, "title"))
        assert exc.value.status_code == 404
        with pytest.raises(HTTPException) as exc:
            _run(store.delete_session("missing", USER_A))
        assert exc.value.status_code == 404
    finally:
        _run(store.close())


def test_global_stats_raises_on_error(tmp_path, monkeypatch):
    """Raise rather than degrade, so the endpoint can answer 503 instead of a 200 indistinguishable from an empty store."""
    store = _store(tmp_path)
    try:
        async def boom(*args, **kwargs):
            raise RuntimeError("db gone")

        monkeypatch.setattr(store, "_fetchone", boom)
        monkeypatch.setattr(store, "_fetchall", boom)
        with pytest.raises(chat_module.ChatAnalyticsUnavailableError):
            _run(store.global_stats())
    finally:
        _run(store.close())


def _make_client(tmp_path):
    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    app = FastAPI()
    app.include_router(chat_module.router)
    chat_module.store = chat_store
    auth_module.store = auth_store
    client = TestClient(app)
    return client, chat_store, auth_store


def test_api_requires_auth(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        assert client.post("/api/chat/sessions").status_code == 401
        assert client.get("/api/chat/sessions").status_code == 401
        # X-User-Id is not a credential
        assert client.post("/api/chat/sessions", headers={"X-User-Id": USER_A}).status_code == 401
        assert client.post("/api/chat/sessions", cookies=auth_cookie("garbage")).status_code == 401
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_create_and_list(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        created = client.post("/api/chat/sessions", cookies=h).json()
        assert created["id"]
        listed = client.get("/api/chat/sessions", cookies=h).json()
        assert [s["id"] for s in listed] == [created["id"]]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_get_rename_delete_flow(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        h_b = _auth_cookies(auth_store, email=EMAIL_B)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert detail["messages"] == []

        renamed = client.patch(f"/api/chat/sessions/{sid}", cookies=h, json={"content": "Renamed"}).json()
        assert renamed["title"] == "Renamed"

        assert client.get(f"/api/chat/sessions/{sid}", cookies=h_b).status_code == 404

        assert client.delete(f"/api/chat/sessions/{sid}", cookies=h).status_code == 200
        assert client.get(f"/api/chat/sessions/{sid}", cookies=h).status_code == 404
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_runs_turn(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_turn(question, history):
            assert question == "Who invested in fintech?"
            assert [m.role for m in history] == ["user"]
            return "A fintech investor is [1].", [{"id": 1, "title": "Fintech funding"}], None, 120, 45, 0.0012

        monkeypatch.setattr(chat_module, "_run_turn", fake_turn)

        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "Who invested in fintech?"})
        assert r.status_code == 200
        body = r.json()
        assert body["user"]["content"] == "Who invested in fintech?"
        assert body["assistant"]["content"] == "A fintech investor is [1]."
        assert body["assistant"]["sources"][0]["title"] == "Fintech funding"
        assert body["assistant"]["prompt_tokens"] == 120
        assert body["assistant"]["completion_tokens"] == 45
        assert body["assistant"]["cost"] == 0.0012

        assert client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["title"] == "Who invested in fintech?"

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert len(detail["messages"]) == 2
        assert detail["messages"][1]["prompt_tokens"] == 120
        assert detail["messages"][1]["cost"] == 0.0012
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


class _RoundTripCounter:
    """Recording round trips, each a serialized await on aiosqlite's single worker thread."""

    def __init__(self, inner):
        self._inner = inner
        self.log = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def execute(self, sql, *a, **kw):
        self.log.append(("execute", " ".join(sql.split())))
        return await self._inner.execute(sql, *a, **kw)

    async def execute_fetchall(self, sql, *a, **kw):
        self.log.append(("select", " ".join(sql.split())))
        return await self._inner.execute_fetchall(sql, *a, **kw)

    async def commit(self):
        self.log.append(("commit", ""))
        return await self._inner.commit()


SESSION_AUTH_SELECT = "FROM sessions WHERE id = ? AND user_id = ?"
# A turn once made 13 serialized round trips, FOUR of them this exact SELECT: once per
# append_message, once in _auto_title, once inside rename_session.
TURN_ROUND_TRIPS_BEFORE_259 = 13


async def _ok_turn(question, history):
    return f"Answer to {question} [1].", [{"id": 1, "title": "Src"}], None, 10, 5, 0.001


def test_one_turn_makes_fewer_round_trips_and_authorises_once(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        monkeypatch.setattr(chat_module, "_run_turn", _ok_turn)

        counter = _RoundTripCounter(chat_store._db)
        chat_store._db = counter

        r = client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h,
            json={"content": "Who invested in fintech?"},
        )
        assert r.status_code == 200

        assert len(counter.log) < TURN_ROUND_TRIPS_BEFORE_259
        session_selects = [e for e in counter.log if SESSION_AUTH_SELECT in e[1]]
        assert len(session_selects) == 1
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_a_failed_turn_rolls_back_without_a_second_authorisation(tmp_path, monkeypatch):
    """A failed turn has already proved it owns the row, so the rollback reuses that proof instead of re-SELECTing."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def boom(question, history):
            raise RuntimeError("the provider is on fire")

        monkeypatch.setattr(chat_module, "_run_turn", boom)

        counter = _RoundTripCounter(chat_store._db)
        chat_store._db = counter

        with pytest.raises(RuntimeError):
            client.post(
                f"/api/chat/sessions/{sid}/messages", cookies=h,
                json={"content": "Who invested in fintech?"},
            )

        # Counted before the GET below, which is itself a session-authorising read.
        session_selects = [e for e in counter.log if SESSION_AUTH_SELECT in e[1]]
        assert len(session_selects) == 1, [e[1] for e in session_selects]

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert detail["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())



@pytest.mark.parametrize(
    "parked_at", ["user_insert", "history_read", "in_the_turn"]
)
def test_a_cancelled_turn_rolls_back_without_a_second_authorisation(
    tmp_path, monkeypatch, parked_at
):
    store, sid = _store_with_session(tmp_path)
    try:
        _pin_budget_disabled(monkeypatch)
        ready = asyncio.Event()
        parked = {"done": False}
        real_append = store._append_authorized
        real_recent = store.recent_turns

        async def park():
            parked["done"] = True
            ready.set()
            await asyncio.sleep(3600)

        if parked_at == "user_insert":
            async def slow_append(session, role, *args, **kwargs):
                result = await real_append(session, role, *args, **kwargs)
                if role == "user" and not parked["done"]:
                    await park()
                return result

            monkeypatch.setattr(store, "_append_authorized", slow_append)
        elif parked_at == "history_read":
            async def slow_recent(session_id, user_id, max_turns):
                if not parked["done"]:
                    await park()
                return await real_recent(session_id, user_id, max_turns)

            monkeypatch.setattr(store, "recent_turns", slow_recent)
        else:
            async def stuck_turn(question, history):
                await park()

            monkeypatch.setattr(chat_module, "_run_turn", stuck_turn)

        counter = _RoundTripCounter(store._db)
        store._db = counter
        request = _cancel_request()

        async def run_it():
            task = asyncio.create_task(
                chat_module.send_message(sid, chat_module.MessageIn(content="what deals happened"), request)
            )
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True

        # Counted before reading the rows back, which is itself a session-authorising read.
        session_selects = [e for e in counter.log if SESSION_AUTH_SELECT in e[1]]
        assert len(session_selects) == 1, [e[1] for e in session_selects]
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def test_cancel_at_the_authorisation_read_writes_nothing(tmp_path, monkeypatch):
    """The authorisation is the one await the cancellation guards do not wrap, sound only because the INSERT has not been issued yet."""
    store, sid = _store_with_session(tmp_path)
    try:
        _pin_budget_disabled(monkeypatch)
        ready = asyncio.Event()
        parked = {"done": False}
        real_get = store.get_session

        async def slow_get(session_id, user_id):
            # Parks after the read resolves, as aiosqlite does on its worker thread, so the SELECT is still logged.
            result = await real_get(session_id, user_id)
            if not parked["done"]:
                parked["done"] = True
                ready.set()
                await asyncio.sleep(3600)
            return result

        monkeypatch.setattr(store, "get_session", slow_get)

        counter = _RoundTripCounter(store._db)
        store._db = counter

        async def run_it():
            task = asyncio.create_task(
                chat_module.send_message(
                    sid, chat_module.MessageIn(content="what deals happened"), _cancel_request()
                )
            )
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        # Swallowing the cancel would end the task normally and hide the disconnect.
        assert _run(run_it()) is True
        assert [kind for kind, _sql in counter.log] == ["select"]
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def test_turn_on_a_session_the_user_does_not_own_is_rejected(tmp_path, monkeypatch):
    """The surviving `id AND user_id` SELECT is the only thing keeping a user out of another user's conversation, so it must run before any write on both turn paths."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h_a = _auth_cookies(auth_store, email=EMAIL_A)
        h_b = _auth_cookies(auth_store, email=EMAIL_B)
        sid = client.post("/api/chat/sessions", cookies=h_a).json()["id"]

        async def must_not_run(question, history):
            raise AssertionError("the LLM was reached for a session the user does not own")

        monkeypatch.setattr(chat_module, "_run_turn", must_not_run)

        assert client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h_b,
            json={"content": "what did they invest in?"},
        ).status_code == 404
        assert client.post(
            f"/api/chat/sessions/{sid}/messages/stream", cookies=h_b,
            json={"content": "what did they invest in?"},
        ).status_code == 404

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h_a).json()
        assert detail["messages"] == []
        assert detail["title"] == "New chat"

        monkeypatch.setattr(chat_module, "_run_turn", _ok_turn)
        assert client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h_a,
            json={"content": "Who invested in fintech?"},
        ).status_code == 200
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_turn_still_persists_messages_titles_and_history(tmp_path, monkeypatch):
    """The turn's observable behaviour is pinned: history carries forward, the
    conversation is named after the FIRST question, and a second turn must not
    clobber that name."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        seen = []

        async def fake_turn(question, history):
            seen.append((question, [m.content for m in history]))
            return f"Answer to {question} [1].", [{"id": 1, "title": "Src"}], None, 10, 5, 0.001

        monkeypatch.setattr(chat_module, "_run_turn", fake_turn)

        assert client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h,
            json={"content": "Who invested in fintech?"},
        ).status_code == 200

        assert client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h,
            json={"content": "And in mobility?"},
        ).status_code == 200

        assert seen == [
            ("Who invested in fintech?", ["Who invested in fintech?"]),
            (
                "And in mobility?",
                [
                    "Who invested in fintech?",
                    "Answer to Who invested in fintech? [1].",
                    "And in mobility?",
                ],
            ),
        ]

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert detail["title"] == "Who invested in fintech?"
        assert [m["role"] for m in detail["messages"]] == [
            "user", "assistant", "user", "assistant",
        ]
        assert [m["content"] for m in detail["messages"]] == [
            "Who invested in fintech?",
            "Answer to Who invested in fintech? [1].",
            "And in mobility?",
            "Answer to And in mobility? [1].",
        ]
        assert detail["messages"][1]["prompt_tokens"] == 10
        assert detail["messages"][1]["cost"] == 0.001
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_auto_title_does_not_clobber_a_rename_made_during_the_turn(tmp_path, monkeypatch):
    """Auto-titling no longer re-reads the session, so the untitled test has to
    come from the UPDATE itself: a rename made while the answer is in flight
    must win over the question text."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        user_id = _run(auth_store.get_user_by_email(EMAIL_A)).id
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def rename_mid_turn(question, history):
            await chat_store.rename_session(sid, user_id, "My carefully chosen name")
            return "An answer [1].", [], None, 1, 1, 0.0

        monkeypatch.setattr(chat_module, "_run_turn", rename_mid_turn)

        r = client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h,
            json={"content": "Who invested in fintech?"},
        )
        assert r.status_code == 200

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert detail["title"] == "My carefully chosen name"
        assert [m["content"] for m in detail["messages"]] == [
            "Who invested in fintech?", "An answer [1].",
        ]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_turn_on_a_conversation_deleted_mid_turn_is_a_clean_404(tmp_path, monkeypatch):
    """Existence is now enforced by the INSERT's own WHERE clause, so a mid-turn
    delete must still answer a clean 404, not a FOREIGN KEY error."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        user_id = _run(auth_store.get_user_by_email(EMAIL_A)).id
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def delete_mid_turn(question, history):
            await chat_store.delete_session(sid, user_id)
            return "An answer [1].", [], None, 1, 1, 0.0

        monkeypatch.setattr(chat_module, "_run_turn", delete_mid_turn)

        r = client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h,
            json={"content": "Who invested in fintech?"},
        )
        assert r.status_code == 404
        assert "conversation not found" in r.text
        assert _run(chat_store.get_session(sid, user_id)) is None
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_usage_stats(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        h_b = _auth_cookies(auth_store, email=EMAIL_B)
        assert client.get("/api/chat/usage", cookies=h).json() == {
            "sessions": 0, "messages": 0, "total_tokens": 0, "total_cost": 0.0
        }

        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_turn(question, history):
            return "answer", [], None, 100, 50, 0.0005

        monkeypatch.setattr(chat_module, "_run_turn", fake_turn)
        client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "query one"})
        client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "query two"})

        usage = client.get("/api/chat/usage", cookies=h).json()
        assert usage["sessions"] == 1
        assert usage["messages"] == 4
        assert usage["total_tokens"] == 300
        assert abs(usage["total_cost"] - 0.001) < 1e-9

        assert client.get("/api/chat/usage", cookies=h_b).json()["total_tokens"] == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_rejects_empty(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        assert client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "   "}).status_code == 400
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_smalltalk_returns_canned_reply():
    for greeting in ["hi", "hello", "hey", "good morning", "Good Afternoon!", "namaste", "how are you?", "thanks", "thank you", "bye", "who are you?"]:
        reply = _smalltalk_reply(greeting)
        assert reply is not None, greeting
        assert "ASK VCCircle" in reply or "archive" in reply

def test_smalltalk_ignores_real_queries():
    for q in ["who invested in Ola Electric?", "top 10 fintech deals 2025", "Ola Electric IPO", "what is the latest funding news", "how many deals did Sequoia do last year?"]:
        assert _smalltalk_reply(q) is None, q

def test_smalltalk_short_circuits_rag(monkeypatch):
    from app import main

    async def boom(*args, **kwargs):
        raise AssertionError("retrieval should not run for small talk")

    monkeypatch.setattr(main, "retrieve_and_rerank", boom)
    answer, sources, note, pt, ct, cost = _run(chat_module._run_turn("good morning", []))
    assert answer.startswith("Hello!")
    assert sources == []
    assert note is None
    assert (pt, ct, cost) == (0, 0, 0.0)


def test_api_stream_smalltalk_short_circuits(tmp_path, monkeypatch):
    from app import main

    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def boom(*args, **kwargs):
            raise AssertionError("retrieval should not run for small talk")

        monkeypatch.setattr(main, "retrieve_and_rerank", boom)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", cookies=h, json={"content": "good morning"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())

        assert "event: start" in body
        assert "event: done" in body
        assert "Hello!" in body
        assert "event: error" not in body

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert len(detail["messages"]) == 2
        assert detail["messages"][1]["role"] == "assistant"
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_full_turn(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(
                answer="prompt-text",
                sources=[{"id": 1, "title": "Src"}],
                note=None,
                needs_llm=True,
            )

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            for piece in ["Hello ", "world", "!"]:
                yield piece
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        _pin_budget_disabled(monkeypatch)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", cookies=h, json={"content": "Who invested in fintech?"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())

        assert body.count("event: delta") == 3
        assert "Hello world!" in body
        assert "event: done" in body
        assert "prompt_tokens" in body

        detail = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()
        assert detail["messages"][1]["content"] == "Hello world!"
        assert detail["messages"][1]["prompt_tokens"] == 50
        assert detail["messages"][1]["completion_tokens"] == 10
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_budget_exceeded(tmp_path, monkeypatch):
    """SSE stream fails closed with an error event when the daily budget is hit."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="prompt-text", sources=[{"id": 1}], note=None, needs_llm=True)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=10.0)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", cookies=h, json={"content": "question"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())

        assert "event: error" in body
        assert "Daily AI budget reached" in body
        assert "event: done" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_global_stats_aggregates(tmp_path):
    store = _store(tmp_path)
    try:
        for user, q, a, pt, ct, cost in [
            (USER_A, "q1", "a1", 100, 20, 0.01),
            (USER_A, "q2", "a2", 200, 30, 0.02),
            (USER_B, "q3", "a3", 300, 40, 0.03),
        ]:
            sid = _run(store.create_session(user, "Conversation")).id
            _run(store.append_message(sid, user, "user", q))
            _run(store.append_message(
                sid, user, "assistant", a,
                prompt_tokens=pt, completion_tokens=ct, cost=cost, latency_ms=250.0,
            ))

        g = _run(store.global_stats())
        assert g["sessions"] == 3
        assert g["users"] == 2
        assert g["messages"] == 6
        assert g["total_tokens"] == 690
        assert g["total_cost"] == pytest.approx(0.06)
        assert g["avg_latency_ms"] == pytest.approx(250.0)
        assert len(g["top_by_cost"]) == 3
        assert g["top_by_cost"][0][2] == pytest.approx(0.03)
        assert len(g["top_by_tokens"]) == 3
        assert g["top_by_tokens"][0][2] == 340
        assert g["daily_sessions"][0][1] == 3
    finally:
        _run(store.close())


def test_analytics_chat_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from app import main

    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    chat_module.store = chat_store
    auth_module.store = auth_store
    client = TestClient(main.app)
    try:
        admin_h = _auth_cookies(auth_store, email="admin@example.com", role="admin")
        sid = client.post("/api/chat/sessions", cookies=admin_h).json()["id"]
        client.post(f"/api/chat/sessions/{sid}/messages", cookies=admin_h, json={"content": "hello"})

        user_h = _auth_cookies(auth_store, email=EMAIL_A)
        assert client.get("/analytics/chat", cookies=user_h).status_code == 403
        assert client.get("/analytics/chat").status_code == 401

        res = client.get("/analytics/chat", cookies=admin_h)
        assert res.status_code == 200
        d = res.json()
        assert d["sessions"] >= 1
        assert d["messages"] >= 2
        assert d["users"] == 1
        assert "total_tokens" in d and "total_cost" in d
    finally:
        chat_module.store = None
        auth_module.store = None
        _run(auth_store.close())
        _run(chat_store.close())


# global_stats is cross-user, so a title in it leaks one user's text to every admin.
PRIVATE_QUESTION = "my doctor prescribed 40mg of sertraline for my bipolar, should i stop"


def _seeded_titled_sessions(store):
    """Titles come from the real _auto_title path, not a hand-set string."""
    ids, titles = [], []
    for user in (USER_A, USER_B):
        session = _run(store.create_session(user))
        sid = session.id
        _run(chat_module._auto_title(store, session, PRIVATE_QUESTION))
        _run(store.append_message(sid, user, "user", PRIVATE_QUESTION))
        _run(store.append_message(
            sid, user, "assistant", "here is a generic answer",
            prompt_tokens=120, completion_tokens=60, cost=0.02, latency_ms=200.0,
        ))
        ids.append(sid)
        titles.append(_run(store.get_session(sid, user)).title)
    return ids, titles


def test_global_stats_omits_user_question_text(tmp_path):
    store = _store(tmp_path)
    try:
        _ids, titles = _seeded_titled_sessions(store)
        assert titles == [PRIVATE_QUESTION[:60], PRIVATE_QUESTION[:60]]

        blob = json.dumps(_run(store.global_stats()))
        assert PRIVATE_QUESTION not in blob
        for title in titles:
            assert title not in blob
            for row in _run(store.global_stats())["top_by_cost"]:
                assert title not in row
            for row in _run(store.global_stats())["top_by_tokens"]:
                assert title not in row
    finally:
        _run(store.close())


def test_global_stats_top_rows_use_session_ids(tmp_path):
    store = _store(tmp_path)
    try:
        ids, titles = _seeded_titled_sessions(store)
        g = _run(store.global_stats())
        for key in ("top_by_cost", "top_by_tokens"):
            assert g[key], f"{key} unexpectedly empty"
            for row in g[key]:
                assert isinstance(row, list) and len(row) == 4
                assert row[0] in ids
                assert row[0] not in titles
    finally:
        _run(store.close())


def test_analytics_chat_endpoint_omits_titles(tmp_path, monkeypatch):
    from app import main

    async def fake_turn(question, history):
        return "a generic reply", [], None, 120, 45, 0.0012

    monkeypatch.setattr(chat_module, "_run_turn", fake_turn)

    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    chat_module.store = chat_store
    auth_module.store = auth_store
    client = TestClient(main.app)
    try:
        admin_h = _auth_cookies(auth_store, email="admin@example.com", role="admin")
        admin_id = _run(auth_store.get_user_by_email("admin@example.com")).id
        sid = client.post("/api/chat/sessions", cookies=admin_h).json()["id"]
        client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=admin_h,
            json={"content": PRIVATE_QUESTION},
        )
        title = _run(chat_store.get_session(sid, admin_id)).title
        assert title == PRIVATE_QUESTION[:60]

        res = client.get("/analytics/chat", cookies=admin_h)
        assert res.status_code == 200
        assert PRIVATE_QUESTION not in res.text
        assert title not in res.text
        payload = res.json()
        for key in ("top_by_cost", "top_by_tokens"):
            assert json.dumps(payload[key]).count('"title"') == 0
            assert [row[0] for row in payload[key]] == [sid]
    finally:
        chat_module.store = None
        auth_module.store = None
        _run(auth_store.close())
        _run(chat_store.close())


def test_analytics_chat_records_admin_audit(tmp_path):
    from app import main

    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    chat_module.store = chat_store
    auth_module.store = auth_store
    client = TestClient(main.app)
    try:
        admin_h = _auth_cookies(auth_store, email="admin@example.com", role="admin")
        admin = _run(auth_store.get_user_by_email("admin@example.com"))
        user_h = _auth_cookies(auth_store, email=EMAIL_A)

        assert _run(chat_store.admin_audit_log()) == []
        assert client.get("/analytics/chat", cookies=admin_h).status_code == 200

        log = _run(chat_store.admin_audit_log())
        assert log[0]["actor_id"] == admin.id
        assert log[0]["action"] == "analytics.chat.read"
        assert log[0]["created_at"] > 0

        before = len(log)
        assert client.get("/analytics/chat", cookies=user_h).status_code == 403
        after = _run(chat_store.admin_audit_log())
        assert len(after) == before
        assert all(r["actor_id"] == admin.id for r in after)
    finally:
        chat_module.store = None
        auth_module.store = None
        _run(auth_store.close())
        _run(chat_store.close())


def test_analytics_chat_survives_a_failing_audit_write(tmp_path, monkeypatch):
    from app import main

    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    chat_module.store = chat_store
    auth_module.store = auth_store
    client = TestClient(main.app)

    async def exploding_audit(actor_id, action):
        raise RuntimeError("audit table unavailable")

    try:
        admin_h = _auth_cookies(auth_store, email="admin@example.com", role="admin")
        admin = _run(auth_store.get_user_by_email("admin@example.com"))
        sid = client.post("/api/chat/sessions", cookies=admin_h).json()["id"]
        # top_by_* joins messages, so an empty session would not appear at all.
        _run(chat_store.append_message(sid, admin.id, "user", "a question"))
        monkeypatch.setattr(chat_store, "record_admin_audit", exploding_audit)

        res = client.get("/analytics/chat", cookies=admin_h)
        assert res.status_code == 200
        assert res.json()["sessions"] >= 1
        assert [row[0] for row in res.json()["top_by_cost"]] == [sid]
    finally:
        chat_module.store = None
        auth_module.store = None
        _run(auth_store.close())
        _run(chat_store.close())


def test_admin_audit_expires_via_retention_sweep(tmp_path):
    store = _store(tmp_path)
    try:
        stale = time.time() - (chat_module.AUDIT_RETENTION_DAYS + 1) * 86400
        _run(store._db.execute(
            "INSERT INTO admin_audit (actor_id, action, created_at) VALUES (?, ?, ?)",
            ("old-admin", "analytics.chat.read", stale),
        ))
        _run(store._db.execute(
            "INSERT INTO admin_audit (actor_id, action, created_at) VALUES (?, ?, ?)",
            ("recent-admin", "analytics.chat.read", time.time() - 60),
        ))
        _run(store._db.commit())
        assert len(_run(store.admin_audit_log())) == 2

        _run(store.record_admin_audit("current-admin", "analytics.chat.read"))
        assert len(_run(store.admin_audit_log())) == 3

        _run(store.purge_expired())

        actors = {r["actor_id"] for r in _run(store.admin_audit_log())}
        assert "old-admin" not in actors, "expired audit row was never pruned"
        assert actors == {"recent-admin", "current-admin"}
    finally:
        _run(store.close())


def test_global_stats_docstring_makes_no_false_safety_claim():
    """Negative-only on purpose: pinning the replacement wording would fail on any harmless rewording."""
    doc = ChatStore.global_stats.__doc__
    assert doc is not None
    low = doc.lower()
    assert "privacy-safe" not in low
    assert "privacy safe" not in low
    assert "only counts/aggregates" not in low
    assert "no message contents" not in low


def test_prepare_turn_passes_intent_date_filter_to_retrieval(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: ("q", "2024-01-01", "2024-12-31", None, None))

    captured = {}

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        captured["qfilter"] = qfilter
        return [SourceArticle(id=1, title="t", url="u", published_date="2024-03-01",
                              summary="s", body="b", score=0.9)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("Top startup funding deals of 2024", []))
    assert turn.needs_llm
    qf = captured["qfilter"]
    assert qf is not None
    keys = {c.key for c in qf.must}
    assert "published_date" in keys
    assert len(turn.sources) == 1


def test_prepare_turn_no_note_when_sources_score_gated_empty(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", True)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: ("q", None, None, None, None))

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-06-01",
                              summary="s", body="b", score=0.1)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("edtech startups 2020", []))
    assert turn.sources == []
    assert "No sufficiently relevant articles" in turn.answer
    assert turn.note is None


def test_prepare_turn_weak_nonempty_sources_keep_note(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", True)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: ("q", None, None, None, None))

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-06-01",
                              summary="s", body="b", score=0.25)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("some niche topic", []))
    assert len(turn.sources) == 1
    assert isinstance(turn.note, str)


def test_prepare_turn_vague_followup_inherits_previous_retrieval(monkeypatch):
    from app import main
    from app.chat import MessageOut
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)

    intents = {
        "top ipo in 2025": ("Flashback 2025 IPO", "2025-01-01", "2025-12-31", None, None),
        "make this into a table": ("make this into a table", None, None, None, None),
    }
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: intents[q])

    captured = {}

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        captured["rq"] = rq
        captured["top_k"] = top_k
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-06-01",
                              summary="s", body="b", score=0.9)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    def msg(i, role, content):
        return MessageOut(id=i, role=role, content=content, sources=[], created_at=float(i),
                          prompt_tokens=0, completion_tokens=0, cost=0.0, latency_ms=0.0)

    history = [
        msg(1, "user", "top ipo in 2025"),
        msg(2, "assistant", "Here are the IPOs."),
        msg(3, "user", "make this into a table"),
    ]
    turn = _run(chat_module._prepare_turn("make this into a table", history))
    assert turn.needs_llm
    assert captured["rq"] == "Flashback 2025 IPO"
    assert captured["top_k"] == 10
    assert "make this into a table" in turn.answer
    assert "top ipo in 2025" in turn.answer


def test_is_vague_followup_treats_prior_result_reference_as_vague():
    from app.chat import _is_vague_followup

    assert _is_vague_followup("share the data in chart") is True
    assert _is_vague_followup("share the last result data in chart") is True
    assert _is_vague_followup("show that data as a chart") is True
    assert _is_vague_followup("give the previous answer in a table") is True
    # 'this deals' + format words references the prior result, not a new topic.
    assert _is_vague_followup("share this deals in tabular format") is True
    assert _is_vague_followup("share this deals in table format") is True
    assert chat_module._requested_view("share this deals in tabular format") == "table"
    assert chat_module._requested_view("share this deals in table format") == "table"
    assert _is_vague_followup("m&a deals in 2025") is False
    assert _is_vague_followup("make a table of top 15 deals in 2024-25") is False
    # A new predication on the noun is a standalone topic, not a reference.
    assert _is_vague_followup("this table shows Q3 deals") is False
    assert _is_vague_followup("this week deals in fintech") is False
    # Trailing politeness does not make a reference topical.
    assert _is_vague_followup("share the last result thanks") is True
    assert _is_vague_followup("share the last result, thanks") is True


def test_previous_user_question_skips_chained_vague_followups():
    from app.chat import MessageOut, _previous_user_question

    def msg(i, role, content):
        return MessageOut(id=i, role=role, content=content, sources=[], created_at=float(i),
                          prompt_tokens=0, completion_tokens=0, cost=0.0, latency_ms=0.0)

    history = [
        msg(1, "user", "share list of IPO companies in table format"),
        msg(2, "assistant", "Here are the IPOs."),
        msg(3, "user", "share the data in chart"),          # vague follow-up
        msg(4, "assistant", "No relevant articles."),
        msg(5, "user", "share the last result data in chart"),  # current turn
    ]
    assert _previous_user_question(history) == "share list of IPO companies in table format"


def test_prepare_turn_prior_result_followup_inherits_real_previous_query(monkeypatch):
    """A prior-result follow-up must inherit the real preceding query, not an
    earlier degenerate one, or retrieval dead-ends on 'No relevant articles'."""
    from app import main
    from app.chat import MessageOut
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)

    intents = {
        "share list of ipo companies in table format": ("IPO companies table", None, None, None, None),
        "share the data in chart": ("share the data in chart", None, None, None, None),
        "share the last result data in chart": ("share the last result data in chart", None, None, None, None),
    }
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: intents[q.lower()])

    captured = {}

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        captured["rq"] = rq
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-06-01",
                              summary="s", body="b", score=0.9)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    def msg(i, role, content):
        return MessageOut(id=i, role=role, content=content, sources=[], created_at=float(i),
                          prompt_tokens=0, completion_tokens=0, cost=0.0, latency_ms=0.0)

    history = [
        msg(1, "user", "share list of IPO companies in table format"),
        msg(2, "assistant", "Here are the IPOs."),
        msg(3, "user", "share the data in chart"),
        msg(4, "assistant", "No relevant articles."),
        msg(5, "user", "share the last result data in chart"),
    ]
    turn = _run(chat_module._prepare_turn("share the last result data in chart", history))
    assert turn.needs_llm
    assert captured["rq"] == "IPO companies table"
    assert "share the last result data in chart" in turn.answer
    assert "share list of IPO companies in table format" in turn.answer


def test_prepare_turn_real_question_does_not_inherit_previous_retrieval(monkeypatch):
    """A standalone question -- even one asking for a table -- uses its own retrieval topic."""
    from app import main
    from app.chat import MessageOut
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    captured = {}

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        captured["rq"] = rq
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-06-01",
                              summary="s", body="b", score=0.9)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    def msg(i, role, content):
        return MessageOut(id=i, role=role, content=content, sources=[], created_at=float(i),
                          prompt_tokens=0, completion_tokens=0, cost=0.0, latency_ms=0.0)

    history = [
        msg(1, "user", "top ipo in 2025"),
        msg(2, "assistant", "Here are the IPOs."),
        msg(3, "user", "make a table of top 15 deals in 2024-25"),
    ]
    turn = _run(chat_module._prepare_turn("make a table of top 15 deals in 2024-25", history))
    assert turn.needs_llm
    assert captured["rq"] == "make a table of top 15 deals in 2024-25"


def test_chat_imports_shared_retrieval_helpers():
    from app import main

    for name in ("_effective_intent", "retrieve_and_rerank", "body_rescue", "source_context", "to_summary"):
        assert hasattr(main, name), f"app.main.{name} missing (needed by chat._prepare_turn)"


def test_parse_dataviz_valid_block():
    text = (
        "Top deals:\n\n```dataviz\n"
        '{"title": "Top 2025 deals", "columns": ["Deal", "Value ($B)"], '
        '"rows": [["Zepto raise", 1.0], ["Shriram stake", 4.4]], "value_column": 1, "format": "$B"}\n'
        "```\n"
    )
    data = chat_module.parse_dataviz(text)
    assert data is not None
    assert data["title"] == "Top 2025 deals"
    assert data["value_column"] == 1
    assert len(data["rows"]) == 2


def test_parse_dataviz_missing_returns_none():
    assert chat_module.parse_dataviz("Just some prose [1].") is None
    assert chat_module.parse_dataviz("") is None


def test_parse_dataviz_malformed_json_returns_none():
    assert chat_module.parse_dataviz("```dataviz\n{not json}\n```") is None


def test_parse_dataviz_invalid_shape_returns_none():
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A","B"], "rows": [["x", 1], ["y"]]}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A","B"], "rows": [["x", "y"], ["z", "w"]]}\n```') is None


def test_parse_dataviz_infers_numeric_column():
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Deal", "Value"], "rows": [["Zepto", 1.0], ["MUFG", 4.4]]}\n```'
    )
    assert data is not None
    assert data["value_column"] == 1


def test_parse_dataviz_allows_missing_values():
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Proceeds (₹ Cr)"], '
        '"rows": [["Wakefit", ""], ["Groww", 1200], ["Meesho", ""]], "value_column": 1}\n```'
    )
    assert data is not None
    assert data["value_column"] == 1

    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Proceeds (₹ Cr)"], '
        '"rows": [["Wakefit", null], ["Groww", 1200]], "value_column": 1}\n```'
    )
    assert data is not None
    assert len(data["rows"]) == 2

    # A common 'not stated' token is treated as missing, like "".
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", "value not stated"], ["Groww", 1200]], "value_column": 1}\n```'
    )
    assert data is not None
    assert data["rows"][0][1] == "value not stated"

    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", "abc"], ["Groww", 1200]], "value_column": 1}\n```'
    ) is None
    # No numeric cell invalidates a chart block, but a plain text table (value_column null) is valid.
    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", ""], ["Groww", "value not stated"]], "value_column": 1}\n```'
    ) is None
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Status"], '
        '"rows": [["Wakefit", "not stated"], ["Groww", "not stated"]], "value_column": null, "view": "table"}\n```'
    )
    assert data is not None
    assert data["value_column"] is None
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", ""], ["Groww", 1200]]}\n```'
    )
    assert data is not None
    assert data["value_column"] == 1


def test_sanitize_dataviz_keeps_valid_strips_malformed():
    valid = "Prose [1].\n\n```dataviz\n{\"columns\": [\"Deal\", \"Value\"], \"rows\": [[\"Zepto\", 1.0]]}\n```"
    assert chat_module._sanitize_dataviz(valid) == valid

    bad = "Prose [1].\n\n```dataviz\n{not json}\n```"
    out = chat_module._sanitize_dataviz(bad)
    assert "```dataviz" not in out
    assert "Prose [1]" in out

    # A bare ```dataviz``` tag in prose matches the newline-optional grammar, so it is stripped too.
    plain = "Just prose with a ```dataviz``` mention."
    out = chat_module._sanitize_dataviz(plain)
    assert "```dataviz" not in out
    # The trailing \s* of the fence grammar eats the space before "mention."
    assert out == "Just prose with a mention."


def test_parse_dataviz_rejects_all_empty_label_cells():
    """An all-blank label column is treated as malformed so the nudge retry gets a chance to rebuild it."""
    empty_labels = (
        '```dataviz\n{"columns": ["Deal", "Value ($B)"], '
        '"rows": [["", 8.5], ["", 4.3], ["", 0.35]], "value_column": 1, "format": "$B"}\n```'
    )
    assert chat_module.parse_dataviz(empty_labels) is None
    out = chat_module._sanitize_dataviz("Prose [1].\n\n" + empty_labels)
    assert "```dataviz" not in out
    assert "Prose [1]" in out

    partly = (
        '```dataviz\n{"columns": ["Deal", "Value ($B)"], '
        '"rows": [["Reliance", 8.5], ["", 4.3], ["Zepto", 0.35]], "value_column": 1}\n```'
    )
    assert chat_module.parse_dataviz(partly) is not None


def test_parse_dataviz_accepts_numeric_identifier_labels():
    explicit = (
        '```dataviz\n{"columns": ["Year", "Revenue"], '
        '"rows": [[2024, 100], [2025, 150]], "value_column": 1}\n```'
    )
    data = chat_module.parse_dataviz(explicit)
    assert data is not None
    assert data["value_column"] == 1
    assert data["rows"] == [[2024, 100], [2025, 150]]

    auto = (
        '```dataviz\n{"columns": ["Year", "Revenue"], '
        '"rows": [[2024, 100], [2025, 150]]}\n```'
    )
    data = chat_module.parse_dataviz(auto)
    assert data is not None
    assert data["value_column"] == 0  # no value_column, so the first numeric one wins
    assert data["rows"] == [[2024, 100], [2025, 150]]


def test_finalize_answer_only_keeps_charts_on_explicit_request():
    """Charts must never appear unasked: model emission of a dataviz block is non-deterministic."""
    with_block = (
        "Prose [1].\n\n```dataviz\n"
        '{"columns": ["Company", "Value"], "rows": [["Wakefit", 1.0]], "value_column": 1}\n'
        "```"
    )
    out = chat_module._finalize_answer(with_block, "top 10 ipo deals in 2025")
    assert "dataviz" not in out
    assert "Prose [1]" in out
    out = chat_module._finalize_answer(with_block, "make a table of top 10 ipo deals")
    assert "dataviz" in out
    assert chat_module.parse_dataviz(out)["view"] == "table"
    assert chat_module._finalize_answer("Just prose [1].", "top deals") == "Just prose [1]."


def test_chart_intent_regex():
    for q in [
        "show me a chart of top deals",
        "show me a bar chart",
        "plot the deals as a graph",
        "make a pie chart of the sectors",
        "give me a line graph of funding by year",
        "as a table",
        "graph it",
        "visualize the top 10 ipo deals",
        "share this deals in tabular format",
        "share this deals in table format",
        "in tabular format",
    ]:
        assert chat_module._CHART_INTENT_RE.search(q), q
    assert not chat_module._CHART_INTENT_RE.search("in table tennis"), "in table tennis"
    assert not chat_module._CHART_INTENT_RE.search("present table tennis scores")
    assert not chat_module._CHART_INTENT_RE.search("in the plot of the story")
    for q in [
        "top 5 deals in 2025",
        "top 10 ipo deals in 2025",
        "biggest funding rounds",
        "how many IPOs this year",
        "yearly breakdown of deals",
        "market share of fintech",
        "2008 crisis",
        "who invested in Ola Electric?",
        "a plot of land in Gurgaon",
    ]:
        assert not chat_module._CHART_INTENT_RE.search(q), q


def test_answer_with_dataviz_retries_when_block_missing(monkeypatch):
    calls = []
    systems = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        systems.append(system_prompt or "")
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]]}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_budget_disabled(monkeypatch)

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", [], "SYSTEM"))
    assert len(calls) == 2
    # The retry instruction rides in the system role: the user message is the one the system prompt declares untrusted.
    assert chat_module._dataviz_nudge("show me a chart of top 5 deals") in systems[1]
    assert chat_module._dataviz_nudge("show me a chart of top 5 deals") not in calls[1]
    assert calls[1] == "PROMPT"
    assert systems[1].startswith("SYSTEM")
    assert systems[0] == "SYSTEM"
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 13
    assert "dataviz" in result.content


def test_answer_with_dataviz_single_call_when_block_present(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]]}\n```',
            prompt_tokens=10,
            completion_tokens=5,
        )

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 1
    assert result.prompt_tokens == 10


def test_answer_with_dataviz_no_retry_for_non_numeric_question(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        return chat_module.LLMResult(content="Plain answer [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("who invested in Ola Electric?", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "Plain answer [1]."


def test_answer_with_dataviz_no_retry_for_ranked_question_without_chart_ask(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        return chat_module.LLMResult(content="Top deal is Zepto [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1
    assert "dataviz" not in result.content


def test_effective_chat_k_dynamic():
    assert chat_module._effective_chat_k("top 10 ipo deals in 2025") == 10
    assert chat_module._effective_chat_k("top ipo deals in 2025") == 10  # bare top -> list default
    assert chat_module._effective_chat_k("who invested in Ola Electric?") == chat_module.config.TOP_K
    assert chat_module._effective_chat_k("top 50 ipo deals") == chat_module.config.CHAT_MAX_SOURCES


def test_dataviz_nudge_row_cap_scales():
    assert "max 10 rows" in chat_module._dataviz_nudge("top 10 ipo deals in 2025")
    assert f"max {chat_module.config.TOP_K} rows" in chat_module._dataviz_nudge("who invested in Ola Electric?")


def test_chat_prompt_instructs_constructing_top_n_lists():
    assert "Build the list only from items the articles actually name" in chat_module.CHAT_PROMPT
    assert "Never refuse" in chat_module.CHAT_PROMPT
    assert "just because the articles lack exact values or a pre-made ranking" in chat_module.CHAT_PROMPT
    assert "always beats a refusal" in chat_module.CHAT_PROMPT
    assert "are COMPANIES that went public or filed for an IPO" in chat_module.CHAT_PROMPT
    assert "never private funding rounds, stake sales, or M&A" in chat_module.CHAT_PROMPT
    assert "write \"value not stated\"" in chat_module.CHAT_PROMPT
    assert "every item mentioned in your prose answer must appear as a row" in chat_module.CHAT_PROMPT
    assert "never drop the row" in chat_module.CHAT_PROMPT
    assert "set `\"value_column\"` to `null`" in chat_module.CHAT_PROMPT


def test_requested_view_detection():
    assert chat_module._requested_view("show me a table of top deals") == "table"
    assert chat_module._requested_view("give me a bar chart") == "bar"
    assert chat_module._requested_view("graph the top deals") == "bar"
    assert chat_module._requested_view("show me a line chart of funding") == "line"
    assert chat_module._requested_view("pie chart of sectors") == "pie"
    assert chat_module._requested_view("make a pictogram of deals") == "picto"
    assert chat_module._requested_view("show me a chart of top deals") is None
    assert chat_module._requested_view("who invested in Ola Electric?") is None


def test_dataviz_view_instruction():
    assert "pie" in chat_module._dataviz_view_instruction("show me a pie chart of sectors")
    assert chat_module._dataviz_view_instruction("show me a chart of deals") == ""
    assert chat_module._dataviz_view_instruction("top 10 ipo deals in 2025") == ""


def test_apply_requested_view_pins_block_view():
    text = (
        "Here they are [1].\n\n```dataviz\n"
        '{"columns": ["Deal", "Value ($B)"], "rows": [["Zepto", 1.0], ["MUFG", 4.4]], "value_column": 1}\n'
        "```"
    )
    out = chat_module._apply_requested_view(text, "show me a pie chart of deals")
    block = chat_module.parse_dataviz(out)
    assert block["view"] == "pie"
    assert block["kind"] == "pie"
    assert "```dataviz" in out

    out = chat_module._apply_requested_view(text, "show me a table of deals")
    block = chat_module.parse_dataviz(out)
    assert block["view"] == "table"

    text2 = (
        "Top IPOs [1].\n\n```dataviz\n"
        '{"columns": ["Company", "Status"], "rows": [["Wakefit", "not stated"], ["Groww", "not stated"]]}\n'
        "```"
    )
    out = chat_module._apply_requested_view(text2, "make a table of top 10 ipos")
    block = chat_module.parse_dataviz(out)
    assert block is not None
    assert block["view"] == "table"
    assert block["value_column"] is None
    assert "not stated" in out

    assert chat_module._apply_requested_view(text, "show me a chart of deals") == text
    assert chat_module._apply_requested_view("Just prose [1].", "show me a pie chart") == "Just prose [1]."


def test_prepare_turn_scales_sources_to_requested_top_n(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    captured = {}

    def make_fake(n):
        async def fake_retrieve(rq, top_k, qfilter, need_body=False):
            captured["top_k"] = top_k
            return [
                SourceArticle(id=i, title=f"t{i}", url=f"u{i}", published_date="2025-03-01",
                              summary="s", body="b" * 4000, score=0.9)
                for i in range(n)
            ]
        return fake_retrieve

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "body_rescue", fake_rescue)
    monkeypatch.setattr(main, "retrieve_and_rerank", make_fake(10))

    turn = _run(chat_module._prepare_turn("top 10 ipo deals in 2025", []))
    assert captured["top_k"] == 10
    assert len(turn.sources) == 10
    assert "max 10" in turn.system

    monkeypatch.setattr(main, "retrieve_and_rerank", make_fake(chat_module.config.TOP_K))
    turn = _run(chat_module._prepare_turn("who invested in Ola Electric?", []))
    assert captured["top_k"] == chat_module.config.TOP_K
    assert len(turn.sources) == chat_module.config.TOP_K


def test_prepare_turn_budgets_body_excerpts_across_sources(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [
            SourceArticle(id=i, title=f"t{i}", url=f"u{i}", published_date="2025-03-01",
                          summary="s", body="x" * 50000, score=0.9)
            for i in range(20)
        ]

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    turn = _run(chat_module._prepare_turn("top 20 ipo deals in 2025", []))
    assert turn.answer.count("x" * 20000) >= 20
    assert "x" * 20001 not in turn.answer


def test_answer_with_dataviz_keeps_first_answer_when_nudge_fails(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)
        raise chat_module.LLMUnavailableError()

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_budget_disabled(monkeypatch)

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 2
    assert result.content == "No chart here [1]."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5



def test_failed_nudge_retry_is_charged_to_the_turn_settle(monkeypatch):
    """A nudge that burned its retries is a real billed call; settling only the successful one under-counts the cap when the provider is flaky."""
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)
        raise chat_module.LLMUnavailableError(attempts=3)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

    spend = chat_module._FailedCallSpend()
    result = _run(
        chat_module._answer_with_dataviz(
            "show me a chart of top 5 deals", "PROMPT", [], spend=spend
        )
    )
    assert len(calls) == 2
    assert result.content == "No chart here [1]."
    assert spend.usd == pytest.approx(3 * 0.02)


def test_failed_ranking_nudge_retry_is_charged_to_the_turn_settle(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="This list cannot be generated because exact amounts are not available [1].",
                prompt_tokens=10,
                completion_tokens=5,
            )
        raise chat_module.LLMUnavailableError(attempts=2)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

    spend = chat_module._FailedCallSpend()
    result = _run(
        chat_module._answer_ranked(
            "top 10 ipo deals in 2025", "PROMPT", [], spend=spend
        )
    )
    assert len(calls) == 2
    assert "cannot be generated" in result.content
    assert spend.usd == pytest.approx(2 * 0.02)


def test_zero_attempt_nudge_failure_is_not_charged(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)
        raise chat_module.LLMUnavailableError(attempts=0)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_budget_disabled(monkeypatch)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

    spend = chat_module._FailedCallSpend()
    _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", [], spend=spend))
    assert len(calls) == 2
    assert spend.usd == 0.0


def test_run_turn_settles_failed_nudge_retries_with_the_turn(monkeypatch):
    generate_calls = []
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_generate(client, prompt, model, system_prompt=None):
        generate_calls.append(prompt)
        if len(generate_calls) == 1:
            # A chart ask whose answer refuses, so both nudges fire and fail after being billed.
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=100_000, completion_tokens=0
            )
        raise chat_module.LLMUnavailableError(attempts=3 if len(generate_calls) == 2 else 2)

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    answer, _sources, _note, _pt, _ct, cost = _run(
        chat_module._run_turn("show me a chart of top 10 ipo deals in 2025", [])
    )

    assert len(generate_calls) == 3
    assert answer.startswith("I cannot generate a ranked list")
    assert budget.writes == [("settle", 200_000)]
    assert budget.counter == 200_000
    assert budget.holds == {}
    assert cost == pytest.approx(0.20)


def test_run_turn_no_usage_still_settles_positive_reserve_estimate(monkeypatch):
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_generate(client, prompt, model, system_prompt=None):
        # Zero tokens means cost() is 0.0 = "usage unknown", never "free".
        return chat_module.LLMResult(content="A delivered answer [1].", prompt_tokens=0, completion_tokens=0)

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    _answer, _sources, _note, _pt, _ct, cost = _run(chat_module._run_turn("Who invested in fintech?", []))

    assert budget.writes == [("settle", 20_000)]
    assert budget.counter == 20_000
    assert cost == pytest.approx(0.02)
    assert cost == pytest.approx(budget.writes[0][1] / 1_000_000)


def test_run_turn_no_usage_with_failed_nudge_settles_reserve_plus_nudge(monkeypatch):
    """Ordering: the reserve fallback must be computed BEFORE spend.usd is added.
    Adding it first would make the total already positive, the `<= 0` fallback
    would never fire, and the answer call's reserve would silently go uncharged."""
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
    generate_calls = []

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_generate(client, prompt, model, system_prompt=None):
        generate_calls.append(prompt)
        if len(generate_calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=0, completion_tokens=0)
        raise chat_module.LLMUnavailableError(attempts=3)

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    _answer, _sources, _note, _pt, _ct, cost = _run(
        chat_module._run_turn("show me a chart of top 5 deals", [])
    )

    assert len(generate_calls) == 2
    assert budget.writes == [("settle", 80_000)]
    assert budget.counter == 80_000
    assert cost == pytest.approx(0.08)
    assert cost == pytest.approx(budget.writes[0][1] / 1_000_000)


def test_json_loads_malformed_returns_empty(caplog):
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads("{not json") == []
        assert chat_module.json_loads(None) == []
        assert chat_module.json_loads("") == []
        assert chat_module.json_loads("[]") == []
    assert "chat.json_loads: malformed stored JSON" in caplog.text
    assert "{not json" in caplog.text


def test_json_loads_logs_unexpected_shape(caplog):
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads('{"sources": 1}') == []
        assert chat_module.json_loads('[1, {"a": 1}]') == [{"a": 1}]
    assert "expected a JSON list" in caplog.text
    assert "dropped 1 non-object item(s)" in caplog.text


def test_json_loads_non_str_payload_degrades_to_empty(caplog):
    with caplog.at_level(logging.ERROR, logger="chat"):
        assert chat_module.json_loads(123, row_id=5) == []
        assert chat_module.json_loads({"a": 1}) == []
    assert "payload must be str/bytes, got int" in caplog.text
    assert "123" in caplog.text
    assert "row_id=5" in caplog.text
    assert "row_id=unknown" in caplog.text
    assert "got dict" in caplog.text


def test_json_loads_logs_row_id_for_shape_problems(caplog):
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads('{"sources": 1}', row_id=99) == []
        assert chat_module.json_loads("[1]", row_id=99) == []
    assert "expected a JSON list" in caplog.text
    assert "dropped 1 non-object item(s)" in caplog.text
    assert "row_id=99" in caplog.text


def test_row_to_message_passes_row_id_to_json_loads(caplog):
    row = {
        "id": 7,
        "role": "user",
        "content": "hello",
        "sources": "{bad json",
        "created_at": 123.0,
        "prompt_tokens": None,
        "completion_tokens": "12",
        "cost": None,
        "latency_ms": "0.5",
        "aborted": 0,
    }
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module._row_to_message(row).sources == []
    assert "row_id=7" in caplog.text


def test_log_preview_is_bounded_for_large_payloads():
    budget = chat_module._JSON_LOG_PREVIEW + 100  # clip marker + omitted-count suffix

    huge_list = [{"id": i, "body": "x" * 50_000} for i in range(5_000)]
    list_preview = chat_module._log_preview(huge_list)
    assert len(list_preview) <= budget
    assert "x" * 1000 not in list_preview
    assert list_preview.endswith("(showing 3 of 5000 item(s))")

    bytes_preview = chat_module._log_preview(b"y" * 5_000_000)
    assert len(bytes_preview) <= budget
    assert "y" * 1000 not in bytes_preview

    dict_preview = chat_module._log_preview({f"key-{i}": "z" * 20_000 for i in range(2_000)})
    assert len(dict_preview) <= budget
    assert dict_preview.endswith("(showing 3 of 2000 item(s))")

    long_str = "s" * 1_000_000
    assert chat_module._log_preview(long_str) == "s" * chat_module._JSON_LOG_PREVIEW + "...(truncated)"

    assert chat_module._log_preview([{"a": 1}]) == "[{'a': 1}]"
    assert chat_module._log_preview("{not json") == "{not json"


class _LazyItemsDict(dict):
    """Counts the members a consumer walks, which is what separates a streaming preview from a materialising one."""

    def __init__(self, source):
        super().__init__(source)
        self.yielded = 0

    def items(self):
        for pair in super().items():
            self.yielded += 1
            yield pair


class _ReprSpy:
    rendered: ClassVar[list[str]] = []

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        _ReprSpy.rendered.append(self.label)
        return f"<spy {self.label}>"


def test_log_preview_does_not_materialise_a_huge_mapping():
    """Asserted on members walked, not on preview length: a repr-then-truncate preview renders everything first."""
    mapping = _LazyItemsDict({f"key-{i}": i for i in range(500_000)})

    preview = chat_module._log_preview(mapping)

    assert mapping.yielded <= chat_module._JSON_PREVIEW_ITEMS
    assert len(preview) <= chat_module._JSON_LOG_PREVIEW + 100


def test_log_preview_never_renders_dropped_or_over_deep_values():
    """Asserted on the spy, not on length: a repr-then-truncate preview renders both dropped and over-deep members."""
    _ReprSpy.rendered = []
    dropped = _ReprSpy("dropped-member")
    over_deep = _ReprSpy("over-deep-member")

    wide = chat_module._log_preview([{"kept": 1}] * chat_module._JSON_PREVIEW_ITEMS + [{"dropped": dropped}])
    nested = chat_module._log_preview([[[over_deep, "x" * 2_000_000]]])

    assert _ReprSpy.rendered == []
    assert "dropped-member" not in wide
    assert "over-deep-member" not in nested
    assert "x" * 1000 not in nested
    assert "<list len=2 omitted>" in nested  # the depth cap stubbed the container
    assert len(nested) <= chat_module._JSON_LOG_PREVIEW + 100


def test_json_loads_degrades_on_pathological_nesting(caplog):
    """Deep nesting raises RecursionError, not ValueError, so it must degrade to [] rather than escape as a 500."""
    payload = "[" * 100_000 + "]" * 100_000
    with pytest.raises(RecursionError):  # guard: the payload really hits the limit
        json.loads(payload)

    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads(payload, row_id=12) == []

    assert "chat.json_loads: malformed stored JSON" in caplog.text
    assert "RecursionError" in caplog.text
    assert "row_id=12" in caplog.text


def test_json_loads_empty_payload_is_not_corruption(caplog):
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads("") == []
        assert chat_module.json_loads(b"") == []
        assert chat_module.json_loads(bytearray(b"")) == []
        assert chat_module.json_loads(None) == []

    assert "chat.json_loads" not in caplog.text
def test_answer_with_dataviz_skips_nudge_when_budget_exhausted(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]]}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=10.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "No chart here [1]."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def _async(fn):
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class _FakeBudgetRedis:
    """Real in-memory implementation of _BUDGET_LUA, so a test exercises the module rather than a stub: the counter is micro-USD, live holds sit in a hash with an expiry zset, the cap measures counter PLUS live holds, and every mode sweeps lapsed holds first. Counter writes are recorded in `writes` as (mode, amount_micros)."""

    def __init__(self, counter_micros=0):
        self.counter = int(counter_micros)
        self.holds: dict[str, int] = {}
        self.expiry: dict[str, int] = {}
        self.writes: list[tuple[str, int]] = []

    def register_script(self, lua):
        store = self

        class _Script:
            async def __call__(self, keys=None, args=None, client=None):
                return store._run(list(args or []))

        return _Script()

    def _sweep(self, now):
        for rid, when in list(self.expiry.items()):
            if when <= now:
                self.expiry.pop(rid, None)
                self.holds.pop(rid, None)

    def _run(self, args):
        mode, now, hold_ttl, _counter_ttl, budget, amount, new_id = args[:7]
        now, hold_ttl = int(now), max(1, int(hold_ttl))
        budget, amount = int(budget), int(amount)
        self._sweep(now)
        if mode == "reserve":
            total = self.counter + sum(self.holds.values())
            if budget > 0 and total + amount > budget:
                return 1
            self.holds[new_id] = amount
            self.expiry[new_id] = now + hold_ttl
            return 0
        if mode == "settle":
            for rid in args[7:]:
                self.holds.pop(rid, None)
                self.expiry.pop(rid, None)
            self.counter = max(0, self.counter + amount)
            self.writes.append(("settle", amount))
            return 0
        if mode == "release":
            for rid in args[7:]:
                self.holds.pop(rid, None)
                self.expiry.pop(rid, None)
            return 0
        return -1


def _route_real_budget(monkeypatch, budget_usd, counter_micros=0):
    """Drops the cached script handle so the module registers against the fake rather than reusing the real one."""
    fake = _FakeBudgetRedis(counter_micros)
    monkeypatch.setattr(cost_budget_module, "_BUDGET_SCRIPT", None)
    monkeypatch.setattr(cost_budget_module, "_client", lambda: fake)
    monkeypatch.setattr(cost_budget_module.config, "LLM_DAILY_BUDGET_USD", budget_usd)
    monkeypatch.setattr(cost_budget_module, "_budget_reached_warned", False)
    return fake


def _pin_cost_accounting(monkeypatch, budget_usd, spend_usd):
    """Pricing pinned to $1 per 1M prompt tokens so the cost arithmetic in these tests reads directly."""
    monkeypatch.setattr(cost_budget_module.config, "INR_PER_USD", 100.0)
    monkeypatch.setattr(cost_budget_module.config, "LLM_PRICE_INPUT_PER_1M", 1.0)
    monkeypatch.setattr(cost_budget_module.config, "LLM_PRICE_OUTPUT_PER_1M", 0.0)
    return _route_real_budget(monkeypatch, budget_usd, round(spend_usd * 1_000_000))


def _pin_budget_disabled(monkeypatch):

    async def reserve(estimate_usd: float = 0.0) -> str:
        return ""

    async def settle(ids, actual_usd: float) -> None:
        return None

    async def release(ids) -> None:
        return None

    monkeypatch.setattr(chat_module, "reserve", reserve)
    monkeypatch.setattr(chat_module, "settle", settle)
    monkeypatch.setattr(chat_module, "release", release)


class _TimingOutCompletions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **_kwargs):
        self._owner.calls += 1
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://llm.test/v1/chat/completions"))


class _TimingOutLLM:
    """A provider that never answers, so the real retry loop runs to exhaustion; `calls` counts requests actually sent, each billed."""

    def __init__(self):
        self.calls = 0
        self.chat = type("_Chat", (), {"completions": _TimingOutCompletions(self)})()


def _pin_llm_outage(monkeypatch, retries, reserve_usd):
    monkeypatch.setattr(chat_module.config, "LLM_MAX_RETRIES", retries)
    monkeypatch.setattr(chat_module.config, "LLM_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", reserve_usd)
    client = _TimingOutLLM()
    monkeypatch.setattr(chat_module, "state_llm", lambda: client)
    return client


def test_answer_with_dataviz_skips_nudge_when_turn_spend_exhausts_budget(monkeypatch):
    """The first call's cost reaches the daily counter only at the turn's end, so
    the guard must count the live hold: the retry is skipped, first answer kept."""
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=1_000_000, completion_tokens=0)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
    budget.holds["h-inflight"] = 1_000_000
    budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "No chart here [1]."
    assert result.prompt_tokens == 1_000_000


def test_answer_with_dataviz_nudges_when_turn_spend_still_within_budget(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=1_000_000, completion_tokens=0)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 2
    assert "dataviz" in result.content


def test_row_to_message_coerces_legacy_fields():
    row = {
        "id": 1,
        "role": "user",
        "content": "hello",
        "sources": "{bad json",
        "created_at": 123.0,
        "prompt_tokens": None,
        "completion_tokens": "12",
        "cost": None,
        "latency_ms": "0.5",
        "aborted": 0,
    }
    msg = chat_module._row_to_message(row)
    assert msg.sources == []
    assert msg.prompt_tokens == 0
    assert msg.completion_tokens == 12
    assert msg.cost == 0.0
    assert msg.latency_ms == 0.5


def test_smalltalk_reply_empty_and_long_fallthrough():
    assert chat_module._smalltalk_reply("") is None
    assert chat_module._smalltalk_reply("   ") is None
    assert chat_module._smalltalk_reply("hi " * 13) is None
    assert chat_module._smalltalk_reply("who invested in Ola Electric and also in Zepto and also in Meesho?") is None


def test_dataviz_helpers_edge_branches():
    assert chat_module._as_float("abc") is None  # non-numeric string
    assert chat_module._as_float(True) is None  # bool is not a number
    assert chat_module._as_float("1,200") == 1200.0
    assert chat_module._as_float(None) is None

    assert chat_module._missing_cell(None) is True
    assert chat_module._missing_cell("N/A") is True
    assert chat_module._missing_cell("  value not stated ") is True
    assert chat_module._missing_cell("—") is True
    assert chat_module._missing_cell(0) is False
    assert chat_module._missing_cell("present") is False

    assert chat_module._valid_value_column([["x", ""], ["y", "not stated"]], 1) is False
    assert chat_module._valid_value_column([["x", 1.0]], 1) is True
    assert chat_module._valid_value_column([["x", "abc"]], 1) is False

    assert chat_module._has_label_content([["x"]], ["A"], 0) is True
    assert chat_module._has_label_content([["", 1.0], ["", 2.0]], ["A", "B"], 1) is False
    assert chat_module._has_label_content([["x", 1.0], ["", 2.0]], ["A", "B"], 1) is True

    assert chat_module._first_numeric_column([]) is None
    assert chat_module._first_numeric_column([[]]) is None
    assert chat_module._first_numeric_column([["x", 3.0]]) == 1


def test_as_float_rejects_bool_before_int_coercion():
    """bool is a subclass of int, so the isinstance(v, bool) guard must precede the int/float branch."""
    assert chat_module._as_float(True) is None
    assert chat_module._as_float(False) is None
    assert chat_module._as_float("true") is None
    assert chat_module._as_float("false") is None

    assert chat_module._as_float(0) == 0.0
    assert chat_module._as_float(1) == 1.0
    assert chat_module._as_float(-3) == -3.0
    assert chat_module._as_float(2.5) == 2.5
    assert chat_module._as_float("0") == 0.0
    assert chat_module._as_float("-1.5") == -1.5

    assert chat_module._as_float("abc") is None
    assert chat_module._as_float(None) is None
    assert chat_module._as_float([]) is None
    assert chat_module._as_float({}) is None

    # The parse_dataviz callers must not treat a boolean column as numeric either.
    assert chat_module._valid_value_column([["x", True]], 1) is False
    assert chat_module._valid_value_column([["x", 1.0], ["y", False]], 1) is False
    assert chat_module._first_numeric_column([["Funded", True], ["Not", False]]) is None
    assert chat_module._first_numeric_column([["Flag", True, 5.0], ["Amount", False, 6.0]]) == 2

    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Deal", "Flag"], "rows": [["A", true], ["B", false]], "view": "bar"}\n```'
    ) is None


def test_parse_dataviz_rejection_paths():
    assert chat_module.parse_dataviz("```dataviz\n[1, 2, 3]\n```") is None
    assert chat_module.parse_dataviz("```dataviz\n\"just a string\"\n```") is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A", 1], "rows": [["x", 1.0]]}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": [], "rows": []}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": "A", "rows": [["x"]]}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A"], "rows": [1]}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A"], "rows": []}\n```') is None


def test_sanitize_dataviz_empty_and_no_fence():
    assert chat_module._sanitize_dataviz("") == ""
    assert chat_module._sanitize_dataviz("plain prose [1].") == "plain prose [1]."


def test_dataviz_nudge_pins_requested_view():
    nudge = chat_module._dataviz_nudge("show me a pie chart of top deals")
    assert "pie" in nudge
    assert "exact type of data block" in nudge
    generic = chat_module._dataviz_nudge("show me a chart of top deals")
    assert "exact type of data block" not in generic


def test_parse_dataviz_with_view_invalid():
    assert chat_module._parse_dataviz_with_view("no block here", "table") is None
    assert chat_module._parse_dataviz_with_view("```dataviz\n{not json}\n```", "table") is None
    assert chat_module._parse_dataviz_with_view("```dataviz\n[1, 2]\n```", "table") is None
    out = chat_module._parse_dataviz_with_view(
        '```dataviz\n{"columns": ["A"], "rows": [["x"]]}\n```', "table"
    )
    assert out is not None
    assert out["view"] == "table"


def test_apply_requested_view_keeps_malformed_block():
    """A block that fails to re-parse is left verbatim, never dropped."""
    text = "Prose.\n\n```dataviz\n{not json}\n```"
    assert chat_module._apply_requested_view(text, "show me a pie chart") == text


def test_is_ranking_refusal():
    assert chat_module._is_ranking_refusal("A ranked list cannot be generated because values are missing.")
    assert chat_module._is_ranking_refusal(
        "I cannot provide a ranked list as the articles do not contain specific amounts."
    )
    assert chat_module._is_ranking_refusal("The data do not include specific amounts, so no ranking exists.")
    assert chat_module._is_ranking_refusal("") is False
    assert not chat_module._is_ranking_refusal("Here are the top deals: Zepto $1B [1].")


def test_answer_ranked_nudges_after_refusal(monkeypatch):
    calls = []
    systems = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        systems.append(system_prompt or "")
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=10, completion_tokens=5
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_budget_disabled(monkeypatch)

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", [], "SYSTEM"))
    assert len(calls) == 2
    # The nudge rides in the instruction channel, never the untrusted user message.
    assert chat_module._RANKING_NUDGE in systems[1]
    assert chat_module._RANKING_NUDGE not in calls[1]
    assert calls[1] == "PROMPT"
    assert systems[1].startswith("SYSTEM")
    assert result.content == "Top deal: Zepto [1]."
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 13


def test_answer_ranked_keeps_first_answer_when_nudge_fails(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list.", prompt_tokens=10, completion_tokens=5
            )
        raise chat_module.LLMUnavailableError()

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_budget_disabled(monkeypatch)

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 2
    assert result.content == "I cannot generate a ranked list."
    assert result.prompt_tokens == 10


def test_answer_ranked_skips_nudge_when_budget_exhausted(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=10, completion_tokens=5
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=10.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "I cannot generate a ranked list [1]."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_answer_ranked_skips_nudge_when_turn_spend_exhausts_budget(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=1_000_000, completion_tokens=0
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
    budget.holds["h-inflight"] = 1_000_000
    budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "I cannot generate a ranked list [1]."
    assert result.prompt_tokens == 1_000_000


def test_answer_ranked_nudges_when_turn_spend_still_within_budget(monkeypatch):
    calls = []
    systems = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        systems.append(system_prompt or "")
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=1_000_000, completion_tokens=0
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", [], "SYSTEM"))
    assert len(calls) == 2
    assert chat_module._RANKING_NUDGE in systems[1]
    assert chat_module._RANKING_NUDGE not in calls[1]
    assert result.content == "Top deal: Zepto [1]."


def test_answer_ranked_single_call_when_not_refusal(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        return chat_module.LLMResult(content="Top deal is Zepto [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "Top deal is Zepto [1]."


def test_prepare_turn_vague_followup_with_year_range(monkeypatch):
    from app import main
    from app.chat import MessageOut
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    intents = {
        "top ipo deals": ("top ipo deals", None, None, None, None),
        "make this into a table for 2024": ("make this into a table for 2024", None, None, None, None),
    }
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: intents[q])

    captured = {}

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        captured["rq"] = rq
        captured["top_k"] = top_k
        captured["qfilter"] = qfilter
        return [SourceArticle(id=1, title="t", url="u", published_date="2024-05-01",
                              summary="s", body="b", score=0.9)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    def msg(i, role, content):
        return MessageOut(id=i, role=role, content=content, sources=[], created_at=float(i),
                          prompt_tokens=0, completion_tokens=0, cost=0.0, latency_ms=0.0)

    history = [
        msg(1, "user", "top ipo deals"),
        msg(2, "assistant", "Here are the IPOs."),
        msg(3, "user", "make this into a table for 2024"),
    ]
    turn = _run(chat_module._prepare_turn("make this into a table for 2024", history))
    assert turn.needs_llm
    assert captured["rq"] == "ipo deals"  # inherited previous topic
    assert captured["top_k"] == 10  # previous turn's list size
    keys = {c.key for c in captured["qfilter"].must}
    assert "published_date" in keys  # 2024 range applied


def test_prepare_turn_calls_body_rescue_when_enabled(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", True)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    rescued = []

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-01-01",
                              summary="s", body="", score=0.9)]

    async def fake_rescue(q, articles):
        rescued.append(q)
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("who invested in Ola Electric?", []))
    assert turn.needs_llm
    assert rescued == ["who invested in Ola Electric?"]


def test_prepare_turn_no_sources_short_circuit(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", False)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-01-01",
                              summary="s", body="b", score=0.01)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("niche topic nobody wrote about", []))
    assert not turn.needs_llm
    assert "No sufficiently relevant articles" in turn.answer
    assert turn.sources == []
    assert turn.cost == 0.0


def test_prepare_turn_weak_fallback(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", True)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [SourceArticle(id=1, title="t", url="u", published_date="2025-01-01",
                              summary="s", body="b", score=0.2)]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("niche topic", []))
    assert not turn.needs_llm
    assert "closest" in turn.answer
    assert len(turn.sources) == 1
    assert turn.note is not None


def test_prepare_turn_faceted_low_score_surfaces(monkeypatch):
    """With a facet resolved the filter is the relevance signal, so the score gate is dropped and matches present normally."""
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", True)
    monkeypatch.setattr(
        main, "_effective_intent",
        lambda q, f, t: (q, "2025-06-01", "2025-06-30", "Venture Capital", None),
    )

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [
            SourceArticle(id=1, title="t", url="u", published_date="2025-06-10",
                          summary="s", body="b", score=0.03),
            SourceArticle(id=2, title="t2", url="u2", published_date="2025-06-12",
                          summary="s", body="b", score=0.02),
        ]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("funding news in jun 2025", []))
    assert turn.needs_llm
    assert len(turn.sources) == 2
    assert "No sufficiently relevant" not in turn.answer
    assert "closest" not in turn.answer
    assert turn.note is None


def test_prepare_turn_multiple_moderate_sources_not_weak(monkeypatch):
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", True)
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None, None, None))

    async def fake_retrieve(rq, top_k, qfilter, need_body=False):
        return [
            SourceArticle(id=i, title=f"t{i}", url=f"u{i}", published_date="2025-06-10",
                          summary="s", body="b", score=0.25)
            for i in range(1, 6)
        ]

    async def fake_rescue(q, articles):
        return articles

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(main, "body_rescue", fake_rescue)

    turn = _run(chat_module._prepare_turn("broad but moderate topic", []))
    assert turn.needs_llm
    assert len(turn.sources) == 5
    assert "No sufficiently relevant" not in turn.answer
    assert "closest" not in turn.answer
    assert turn.note is None


def test_run_turn_records_cost_and_finalizes(monkeypatch):
    calls = {}
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note="note", needs_llm=True)

    async def fake_answer_ranked(question, prompt, holds, system_prompt="", *, spend=None):
        calls["prompt"] = prompt
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
            prompt_tokens=10,
            completion_tokens=5,
        )

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)

    answer, sources, note, pt, ct, _cost = _run(chat_module._run_turn("top 10 ipo deals in 2025", []))
    assert calls["prompt"] == "PROMPT"
    expected_cost = chat_module.to_usd(
        chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5).cost()
    )
    assert len(budget.writes) == 1
    assert budget.writes[0][1] == round(expected_cost * 1_000_000)
    assert budget.counter == round(expected_cost * 1_000_000)
    assert budget.holds == {}
    assert "dataviz" not in answer
    assert "Prose [1]" in answer
    assert sources == [{"id": 1}]
    assert note == "note"
    assert (pt, ct) == (10, 5)


def test_run_turn_charges_every_attempt_of_a_failed_call_and_reraises(monkeypatch):
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
    llm = _pin_llm_outage(monkeypatch, retries=2, reserve_usd=0.02)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note=None, needs_llm=True)

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)

    with pytest.raises(chat_module.LLMUnavailableError) as excinfo:
        _run(chat_module._run_turn("who invested in Ola?", []))

    assert llm.calls == 3  # LLM_MAX_RETRIES + 1 requests, every one billed
    assert excinfo.value.attempts == 3
    assert budget.writes == [("settle", 60_000)]
    assert budget.counter == 60_000
    assert budget.holds == {}


def test_run_turn_releases_the_hold_when_no_request_was_sent(monkeypatch):
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
    llm = _pin_llm_outage(monkeypatch, retries=-1, reserve_usd=0.02)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note=None, needs_llm=True)

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)

    with pytest.raises(chat_module.LLMUnavailableError) as excinfo:
        _run(chat_module._run_turn("who invested in Ola?", []))

    assert llm.calls == 0
    assert excinfo.value.attempts == 0
    assert budget.writes == []
    assert budget.counter == 0
    assert budget.holds == {}


def test_run_turn_settles_summed_turn_cost_exactly_once(monkeypatch):
    generate_calls = []
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_generate(client, prompt, model, system_prompt=None):
        generate_calls.append(prompt)
        # A chart ask whose answer refuses, so both nudges fire and the turn is billed 3x.
        prompt_tokens = {1: 100_000, 2: 200_000, 3: 300_000}[len(generate_calls)]
        return chat_module.LLMResult(
            content="I cannot generate a ranked list [1].", prompt_tokens=prompt_tokens, completion_tokens=0
        )

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    _run(chat_module._run_turn("show me a chart of top 10 ipo deals in 2025", []))

    assert len(generate_calls) == 3
    # The three reserve calls touch only the holds hash, so they are not counter writes.
    assert [mode for mode, _ in budget.writes] == ["settle"]
    assert budget.writes[0][1] == 600_000
    assert budget.counter == 600_000
    assert budget.holds == {}


def test_api_json_turn_with_no_usage_report_is_still_charged(tmp_path, monkeypatch):
    """A provider sending no usage yields a TRUTHY LLMResult with cost() 0.0; a zero is unknown, not free."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_generate(client, prompt, model, system_prompt=None):
            return chat_module.LLMResult(content="A fully delivered answer [1].", prompt_tokens=0, completion_tokens=0)

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
        monkeypatch.setattr(chat_module, "state_llm", lambda: object())

        r = client.post(
            f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "Who invested in fintech?"}
        )

        assert r.status_code == 200
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
        assert r.json()["assistant"]["cost"] == pytest.approx(0.02)
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_run_turn_with_reported_usage_still_uses_the_real_cost(monkeypatch):
    """A provider that DOES report usage must be charged its real cost, or the no-usage test would pass with every turn charged the reserve."""
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_answer_ranked(question, prompt, holds, system_prompt="", *, spend=None):
        return chat_module.LLMResult(content="A fully delivered answer [1].", prompt_tokens=100_000, completion_tokens=0)

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)

    _answer, _sources, _note, _pt, _ct, cost = _run(chat_module._run_turn("Who invested in fintech?", []))

    assert budget.writes == [("settle", 100_000)]
    assert budget.counter == 100_000
    assert cost == pytest.approx(0.1)


def test_api_require_store_uninitialized_503(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        chat_module.store = None
        h = _auth_cookies(auth_store)
        assert client.get("/api/chat/sessions", cookies=h).status_code == 503
        assert client.post("/api/chat/sessions", cookies=h).status_code == 503
    finally:
        chat_module.store = chat_store
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_too_long_422(tmp_path):
    """The bound itself is covered in test_chat_content_bound.py."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        long_msg = "x" * (chat_module.MAX_CONTENT_LEN + 1)
        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": long_msg})
        assert r.status_code == 422
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_budget_exceeded_429(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def boom(question, history):
            raise chat_module.BudgetExceeded()

        monkeypatch.setattr(chat_module, "_run_turn", boom)
        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "top deals"})
        assert r.status_code == 429
        assert "Daily AI budget reached" in r.text
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_total_llm_outage_is_503_and_the_sse_path_reports_the_same(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        llm = _pin_llm_outage(monkeypatch, retries=2, reserve_usd=0.02)

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note=None, needs_llm=True)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)

        expected = {
            "error": "LLM temporarily unavailable",
            "detail": "The language model could not be reached; please retry shortly.",
        }

        json_sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        r = client.post(f"/api/chat/sessions/{json_sid}/messages", cookies=h, json={"content": "Who invested in fintech?"})
        assert r.status_code == 503
        assert r.json()["detail"] == expected
        assert client.get(f"/api/chat/sessions/{json_sid}", cookies=h).json()["messages"] == []

        sse_sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        body = _stream_body(client, h, sse_sid, "Who invested in fintech?")
        assert "event: error" in body
        assert "event: done" not in body
        assert json.loads(re.search(r"event: error\ndata: (.*)", body).group(1)) == expected
        assert client.get(f"/api/chat/sessions/{sse_sid}", cookies=h).json()["messages"] == []

        assert llm.calls == 6
        assert budget.writes == [("settle", 60_000), ("settle", 60_000)]
        assert budget.counter == 120_000
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def _stream_body(client, cookies, sid, content):
    url = f"/api/chat/sessions/{sid}/messages/stream"
    with client.stream("POST", url, cookies=cookies, json={"content": content}) as r:
        assert r.status_code == 200
        return "".join(r.iter_text())


def _fake_prepare_llm():
    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="prompt-text", sources=[], note=None, needs_llm=True)

    return fake_prepare


def test_api_stream_dataviz_nudge_replaces_answer(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "no block here [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        async def fake_generate(client, prompt, model, system_prompt=None):
            return chat_module.LLMResult(
                content='Prose.\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
                prompt_tokens=20,
                completion_tokens=8,
            )

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
        _pin_budget_disabled(monkeypatch)

        body = _stream_body(client, h, sid, "show me a chart of top deals")
        assert "dataviz" in body
        assert '"prompt_tokens":30' in body  # 10 streamed + 20 nudge
        assert '"completion_tokens":13' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_dataviz_nudge_failure_keeps_answer(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "streamed answer without a block [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        async def boom(*args, **kwargs):
            raise chat_module.LLMUnavailableError()

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", boom)
        _pin_budget_disabled(monkeypatch)

        body = _stream_body(client, h, sid, "show me a chart of top deals")
        assert "streamed answer without a block [1]." in body
        assert '"prompt_tokens":10' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_dataviz_nudge_skipped_when_budget_exhausted(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "streamed answer without a block [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        nudge_calls = []

        async def fake_generate(client, prompt, model, system_prompt=None):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(
                content='Prose.\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
                prompt_tokens=20,
                completion_tokens=8,
            )

        # The stream's hold is granted; the nudge's second hold is refused.
        budget_calls = {"n": 0}

        async def exhausted_after_first_reserve(estimate_usd=0.0):
            budget_calls["n"] += 1
            if budget_calls["n"] > 1:
                raise chat_module.BudgetExceeded()
            return "h-gate"

        async def settle(ids, actual_usd):
            return None

        async def release(ids):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
        monkeypatch.setattr(chat_module, "reserve", exhausted_after_first_reserve)
        monkeypatch.setattr(chat_module, "settle", settle)
        monkeypatch.setattr(chat_module, "release", release)

        body = _stream_body(client, h, sid, "show me a chart of top deals")
        assert nudge_calls == []
        assert "streamed answer without a block [1]." in body
        assert '"prompt_tokens":10' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_dataviz_nudge_skipped_when_turn_spend_exhausts_budget(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "streamed answer without a block [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=1_000_000, completion_tokens=0))

        nudge_calls = []

        async def fake_generate(client, prompt, model, system_prompt=None):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(
                content='Prose.\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
                prompt_tokens=20,
                completion_tokens=8,
            )

        async def noop(*args, **kwargs):
            return None

        budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
        budget.holds["h-inflight"] = 450_000
        budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

        body = _stream_body(client, h, sid, "show me a chart of top deals")
        assert nudge_calls == []
        assert "streamed answer without a block [1]." in body
        assert '"prompt_tokens":1000000' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_ranking_nudge_replaces_answer(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        async def fake_generate(client, prompt, model, system_prompt=None):
            return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
        _pin_budget_disabled(monkeypatch)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")
        assert "Top deal: Zepto [1]." in body
        assert '"prompt_tokens":30' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_ranking_nudge_failure_keeps_answer(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        async def boom(*args, **kwargs):
            raise chat_module.LLMUnavailableError()

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", boom)
        _pin_budget_disabled(monkeypatch)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")
        assert "I cannot generate a ranked list because amounts are missing." in body
        assert '"prompt_tokens":10' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_ranking_nudge_skipped_when_budget_exhausted(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        nudge_calls = []

        async def fake_generate(client, prompt, model, system_prompt=None):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

        # The stream's hold is granted; the ranking nudge's second hold is refused.
        budget_calls = {"n": 0}

        async def exhausted_after_first_reserve(estimate_usd=0.0):
            budget_calls["n"] += 1
            if budget_calls["n"] > 1:
                raise chat_module.BudgetExceeded()
            return "h-gate"

        async def settle(ids, actual_usd):
            return None

        async def release(ids):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
        monkeypatch.setattr(chat_module, "reserve", exhausted_after_first_reserve)
        monkeypatch.setattr(chat_module, "settle", settle)
        monkeypatch.setattr(chat_module, "release", release)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")
        assert nudge_calls == []
        assert "I cannot generate a ranked list because amounts are missing." in body
        assert '"prompt_tokens":10' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_ranking_nudge_skipped_when_turn_spend_exhausts_budget(tmp_path, monkeypatch):
    """As above for the ranking nudge: the stream's cost is recorded only at the
    turn's end, so the guard must count it."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=1_000_000, completion_tokens=0))

        nudge_calls = []

        async def fake_generate(client, prompt, model, system_prompt=None):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

        async def noop(*args, **kwargs):
            return None

        budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
        budget.holds["h-inflight"] = 450_000
        budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")
        assert nudge_calls == []
        assert "I cannot generate a ranked list because amounts are missing." in body
        assert '"prompt_tokens":1000000' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_records_summed_turn_cost_exactly_once(tmp_path, monkeypatch):
    """The only test driving the REAL cost_budget module through chat.py's whole turn path, so it catches a wrong-arity or wrong-semantics call."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        nudge_calls = []

        async def fake_generate(client, prompt, model, system_prompt=None):
            nudge_calls.append(prompt)
            prompt_tokens = 200_000 if len(nudge_calls) == 1 else 300_000
            return chat_module.LLMResult(
                content="I cannot be generated [1].", prompt_tokens=prompt_tokens, completion_tokens=0
            )

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

        body = _stream_body(client, h, sid, "show me a chart of top 10 ipo deals in 2025")

        assert len(nudge_calls) == 2
        # The three reserves only touched the holds hash, so they are not counter writes.
        assert [mode for mode, _ in budget.writes] == ["settle"]
        assert budget.writes[0][1] == 600_000
        assert budget.counter == 600_000
        assert budget.holds == {}
        assert '"prompt_tokens":600000' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_llm_unavailable_sse(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def boom(*args, **kwargs):
            raise chat_module.LLMUnavailableError()
            yield  # pragma: no cover - makes boom an async generator

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", boom)
        _pin_budget_disabled(monkeypatch)

        body = _stream_body(client, h, sid, "who invested in Ola Electric?")
        assert "event: error" in body
        assert "LLM temporarily unavailable" in body
        assert "event: done" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_generic_error_sse(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def boom(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes boom an async generator

        async def noop(*args, **kwargs):
            return None

        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", boom)
        _pin_budget_disabled(monkeypatch)

        body = _stream_body(client, h, sid, "who invested in Ola Electric?")
        assert "event: error" in body
        assert "Something went wrong" in body
        assert "event: done" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_retention_loop_purges_and_swallows_errors(monkeypatch, tmp_path):
    store = _store(tmp_path)
    chat_module.store = store
    try:
        a = _run(store.create_session(USER_A))
        _run(store._db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time() - 200 * 86400, a.id)))
        _run(store._db.commit())

        orig_purge = store.purge_expired
        attempts = []
        sleeps = []

        async def flaky_purge():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("db locked")
            return await orig_purge()

        def fake_sleep(delay):
            async def _sleep(d):
                sleeps.append(d)
                if len(sleeps) >= 2:
                    raise asyncio.CancelledError()
            return _sleep(delay)

        monkeypatch.setattr(store, "purge_expired", flaky_purge)
        monkeypatch.setattr(chat_module.asyncio, "sleep", fake_sleep)

        with pytest.raises(asyncio.CancelledError):
            _run(chat_module.retention_loop())
        assert len(attempts) == 2
        assert sleeps == [chat_module.config.CHAT_PURGE_INTERVAL_SECONDS] * 2
        assert _run(store.get_session(a.id, USER_A)) is None
    finally:
        chat_module.store = None
        _run(store.close())


def _disconnect_after(monkeypatch, n_deltas):
    """`streamed` in chat.py counts the deltas it has yielded, so `n_deltas` selects which side of the abort rule a test hits."""
    state = {"deltas": 0}

    # Patched onto the class, so the function is bound and receives `self`.
    async def is_disconnected(self):
        return state["deltas"] >= n_deltas

    monkeypatch.setattr(chat_module.Request, "is_disconnected", is_disconnected)

    real_sse = chat_module._sse

    def counting_sse(event, data):
        if event == "delta":
            state["deltas"] += 1
        return real_sse(event, data)

    monkeypatch.setattr(chat_module, "_sse", counting_sse)
    return state


def test_stream_abort_after_deltas_persists_the_turn(tmp_path, monkeypatch):
    """Once any delta has been streamed the turn is PERSISTED, never deleted:
    erasing it after the client already rendered the answer would leave the
    server's history and the client's screen disagreeing."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            for piece in ["Partial ", "answer ", "rest."]:
                yield piece
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        _disconnect_after(monkeypatch, 1)
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"]
        assistant = msgs[1]
        assert assistant["aborted"] is True
        assert "Partial" in assistant["content"]
        assert "[answer truncated]" in assistant["content"]
        assert "event: done" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_abort_before_any_delta_deletes_user_message(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "never seen by the client"
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        _disconnect_after(monkeypatch, 0)
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert msgs == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_mid_failure_with_gone_client_persists_aborted(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "half an answer"
            raise RuntimeError("provider dropped the connection")

        _disconnect_after(monkeypatch, 1)
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert "half an answer" in msgs[1]["content"]
        assert "[answer truncated]" in msgs[1]["content"]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def _delete_mid_turn_scenario(tmp_path, monkeypatch, delete_after_deltas):
    """Returns the raw body: the assertion is what the client sees, so an uncollectable body must fail by raising out of _stream_body."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        user_id = _run(auth_store.get_user_by_email(EMAIL_A)).id

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client_, prompt, model, usage_holder=None, system_prompt=None):
            if not delete_after_deltas:
                await chat_store.delete_session(sid, user_id)
            yield "first chunk "
            if delete_after_deltas:
                await chat_store.delete_session(sid, user_id)
            yield "second chunk"
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        return _stream_body(client, h, sid, "Who invested in fintech?")
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


@pytest.mark.parametrize("delete_after_deltas", [False, True], ids=["before_any_delta", "after_first_delta"])
def test_stream_conversation_deleted_mid_turn_closes_with_error_event(tmp_path, monkeypatch, delete_after_deltas):
    body = _delete_mid_turn_scenario(tmp_path, monkeypatch, delete_after_deltas)

    assert "event: start" in body
    assert "event: error" in body
    assert "event: done" not in body


def test_send_message_disconnect_rolls_back_without_assistant(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_run_turn(question, history):
            return "An answer the client never receives.", [], None, 10, 5, 0.0001

        # Patched onto the class, so the function is bound and receives `self`.
        async def always_gone(self):
            return True

        monkeypatch.setattr(chat_module, "_run_turn", fake_run_turn)
        monkeypatch.setattr(chat_module.Request, "is_disconnected", always_gone)

        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "top deals"})
        assert r.status_code >= 400
        assert client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_send_message_budget_unavailable_is_503_not_empty_answer(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def boom(question, history):
            raise chat_module.BudgetUnavailable("redis down")

        monkeypatch.setattr(chat_module, "_run_turn", boom)

        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "top deals"})
        assert r.status_code == 503
        assert "budget" in json.dumps(r.json()).lower()
        assert client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_budget_unavailable_is_error_event(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def boom(estimate_usd=0.0):
            raise chat_module.BudgetUnavailable("redis down")

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "reserve", boom)

        body = _stream_body(client, h, sid, "question")
        assert "event: error" in body
        assert "budget service unavailable" in body
        assert "event: done" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_nudge_unreachable_counter_degrades_without_billing(monkeypatch, caplog):
    calls = []

    async def fake_generate(client, prompt, model, system_prompt=None):
        calls.append(prompt)
        return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)

    async def unavailable(estimate_usd=0.0):
        raise chat_module.BudgetUnavailable("redis down")

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    monkeypatch.setattr(chat_module, "reserve", unavailable)

    with caplog.at_level(logging.WARNING, logger="chat"):
        result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))

    assert len(calls) == 1
    assert result.content == "No chart here [1]."
    assert "unavailable" in caplog.text.lower()


def _msg(role, content):
    return chat_module.MessageOut(id=1, role=role, content=content, created_at=0.0)


def test_trim_history_drops_oldest_until_within_budget():
    history = [_msg("user", "a" * 40), _msg("assistant", "b" * 40), _msg("user", "c" * 40)]

    kept = chat_module._trim_history(history, 80)
    assert [m.content for m in kept] == ["b" * 40, "c" * 40]

    kept = chat_module._trim_history(history, 40)
    assert [m.content for m in kept] == ["c" * 40]

    assert chat_module._trim_history(history, 10_000) == history


def test_trim_history_keeps_oversized_newest_message():
    history = [_msg("user", "a" * 100), _msg("assistant", "b" * 5000)]
    kept = chat_module._trim_history(history, 100)
    assert [m.content for m in kept] == ["b" * 5000]


def test_trim_history_zero_disables_the_cap():
    history = [_msg("user", "a" * 10_000), _msg("assistant", "b" * 10_000)]
    assert chat_module._trim_history(history, 0) == history
    assert chat_module._trim_history(history, -1) == history


def test_start_turn_applies_the_char_cap(tmp_path, monkeypatch):
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        for i in range(6):
            _run(store.append_message(sid, USER_A, "user" if i % 2 == 0 else "assistant", "x" * 5000))

        monkeypatch.setattr(chat_module.config, "CHAT_MAX_HISTORY_TURNS", 10)
        monkeypatch.setattr(chat_module.config, "CHAT_MAX_HISTORY_CHARS", 12_000)

        _user_msg, history, _session = _run(chat_module._start_turn(store, sid, USER_A, "next question"))

        assert sum(len(m.content) for m in history) <= 12_000 + len("next question")
        assert len(history) < 7
    finally:
        _run(store.close())


def test_dataviz_unclosed_fence_never_leaks_json():
    unclosed = 'Here is the data:\n\n```dataviz\n{"columns": ["A"], "rows": [[1, 2'
    for question in ("show me a chart of top deals", "who invested in Ola Electric?"):
        out = chat_module._finalize_answer(unclosed, question)
        assert '{"' not in out
        assert "```dataviz" not in out
        assert "Here is the data:" in out

    assert '{"' not in chat_module._sanitize_dataviz(unclosed)
    assert '{"' not in chat_module._append_nudge("Answer [1].", unclosed)


def test_dataviz_newline_optional_fence_is_handled():
    body = '{"columns": ["Deal", "Value"], "rows": [["Zepto", 1.0]], "value_column": 1}'
    no_newline = f"Prose [1].\n\n```dataviz{body}```"
    assert chat_module.parse_dataviz(f"```dataviz{body}```") is not None
    out = chat_module._finalize_answer(no_newline, "show me a table of top deals")
    assert "```dataviz" in out
    assert "Prose [1]." in out


def test_dataviz_valid_fence_survives_the_unclosed_rule():
    valid = (
        'Prose [1].\n\n```dataviz\n'
        '{"columns": ["Deal", "Value"], "rows": [["Zepto", 1.0]], "value_column": 1}\n```'
    )
    assert chat_module._sanitize_dataviz(valid) == valid
    # _finalize_answer may legitimately REWRITE a valid block (it pins the view), so
    # the invariant is that the block SURVIVED, not that the text is byte-identical.
    finalized = chat_module._finalize_answer(valid, "show me a table of top deals")
    assert finalized.startswith("Prose [1].")
    assert finalized.rstrip().endswith("```")
    assert chat_module.parse_dataviz(finalized) is not None


# The dataviz grammar lives in this ONE module, so the backend can execute the shipped
# frontend rules under node instead of re-typing them.
_DATAVIZ_CONTRACT_TS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "frontend"
    / "app"
    / "chat"
    / "datavizContract.ts"
)


def test_fence_src_matches_python_fence_pattern():
    raw = re.search(r"const FENCE_SRC = '([^']*)'", _DATAVIZ_CONTRACT_TS.read_text()).group(1)
    # Single-quoted TS, so \S is written \\S: unescape it the way JS would first.
    js_runtime = json.loads('"' + raw + '"')
    assert js_runtime == chat_module.DATAVIZ_FENCE_PATTERN


_FENCE_FIXTURES = {
    "closed_with_newline": 'Prose.\n```dataviz\n{"columns": ["A"], "rows": [["x", 1]], "value_column": 1}\n```',
    "closed_without_newline": 'Prose.\n```dataviz{"columns": ["A"], "rows": [["x", 1]], "value_column": 1}```',
    "unclosed": 'Prose.\n```dataviz\n{"columns": ["A"], "rows": [["x", 1',
    "language_tag": 'Prose.\n```dataviz  \n{"columns": ["A"], "rows": [["x", 1]], "value_column": 1}\n```',
}


def _node_strip_open_fence(fixtures):
    """Run the TSX's own regex + truncation rule under node. Both are read verbatim
    out of the frontend contract module, so this is the shipped rule, not a
    re-typed approximation."""
    tsx = _DATAVIZ_CONTRACT_TS.read_text()
    fence_src = re.search(r"const FENCE_SRC = '([^']*)'", tsx).group(1)
    strip = re.search(r"(function stripOpenFence\(md: string\): string \{.*?\n\})", tsx, re.DOTALL).group(1)
    # Drop the TS type annotation; the body is plain JS.
    js = f"""
// The captured group is the BODY of the TSX's own single-quoted literal, so
// re-wrapping it verbatim reproduces that literal exactly. It legitimately
// contains backticks and backslashes, which is why it must NOT be re-quoted
// or escaped -- doing so turns the leading backticks into a template literal.
const FENCE_SRC = '{fence_src}'
{strip.replace('md: string', 'md')}
const cases = {json.dumps(fixtures)}
const out = {{}}
for (const [k, v] of Object.entries(cases)) {{
  const re = new RegExp(FENCE_SRC, 'g')
  const m = re.exec(v)
  out[k] = {{
    matched: !!m,
    span: m ? [m.index, m.index + m[0].length] : null,
    captured: m ? m[1] : null,
    // What the frontend actually renders: splitContent hands the text left
    // around the (matched) blocks to stripOpenFence.
    rendered: stripOpenFence(m ? v.slice(0, m.index) + v.slice(m.index + m[0].length) : v),
  }}
}}
console.log(JSON.stringify(out))
"""
    proc = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_fence_parity_between_frontend_and_backend_behaviour():
    """Cross-language parity, checked BEHAVIOURALLY: `node -e` needs no
    node_modules, so the TSX's own FENCE_SRC and stripOpenFence run under node and
    their spans, captures and rendered text are compared with Python's. Rendered
    text matters because the `unclosed` fixture matches NOTHING on both sides."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH; cannot run the frontend fence rule")

    js = _node_strip_open_fence(_FENCE_FIXTURES)

    for name, text in _FENCE_FIXTURES.items():
        py_match = chat_module._DATAVIZ_FENCE_RE.search(text)
        py_matched = py_match is not None
        assert py_matched == js[name]["matched"], name
        if py_matched:
            assert list(py_match.span()) == js[name]["span"], name
            assert py_match.group(1) == js[name]["captured"], name

    unclosed = _FENCE_FIXTURES["unclosed"]
    js_rendered = js["unclosed"]["rendered"]
    py_rendered = chat_module._finalize_answer(unclosed, "show me a chart of top deals")
    assert '{"' not in js_rendered
    assert '{"' not in py_rendered
    assert js_rendered.strip().startswith("Prose.")
    assert py_rendered.strip().startswith("Prose.")
    assert js_rendered.split("Prose.")[1] == py_rendered.split("Prose.")[1]


def test_connect_migrates_legacy_db_without_aborted_column(tmp_path):
    db_path = tmp_path / "chat.db"
    _legacy_db(
        db_path,
        "CREATE TABLE messages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
        " role TEXT NOT NULL, content TEXT NOT NULL,"
        " sources TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL,"
        " prompt_tokens INTEGER NOT NULL DEFAULT 0,"
        " completion_tokens INTEGER NOT NULL DEFAULT 0,"
        " cost REAL NOT NULL DEFAULT 0,"
        " latency_ms REAL NOT NULL DEFAULT 0)",
    )
    store = ChatStore(str(db_path))
    _run(store.connect())
    try:
        cols = {c["name"] for c in _run(store._db.execute_fetchall("PRAGMA table_info(messages)"))}
        assert "aborted" in cols
        _run(store.close())
        _run(store.connect())
        cols2 = {c["name"] for c in _run(store._db.execute_fetchall("PRAGMA table_info(messages)"))}
        assert "aborted" in cols2
    finally:
        _run(store.close())


def test_aborted_flag_round_trips_through_the_store(tmp_path):
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _run(store.append_message(sid, USER_A, "user", "q"))
        aborted_msg = _run(
            store.append_message(sid, USER_A, "assistant", "partial [answer truncated]", aborted=True)
        )
        assert aborted_msg.aborted is True

        msgs, _total = _run(store.messages_page(sid, USER_A))
        assert [m.aborted for m in msgs] == [False, True]

        recent = _run(store.recent_turns(sid, USER_A, 5))
        assert [m.aborted for m in recent] == [False, True]
    finally:
        _run(store.close())


def test_stream_abort_in_gate_window_charges_the_hold(tmp_path, monkeypatch):
    """The provider bills the call by the time it emits a piece, so letting the gate hold lapse would charge the estimate anyway."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        entered = {"v": False}

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            # The call has been billed; the disconnect check runs next, with `streamed`
            # still False because the piece has not been yielded.
            entered["v"] = True
            yield "one piece nobody will see"

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

        async def is_disconnected(self):
            return entered["v"]

        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

        monkeypatch.setattr(chat_module.Request, "is_disconnected", is_disconnected)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert entered["v"] is True
        # The disconnect rule, not the catch-all: an error would emit an 'error' event.
        assert "event: error" not in body
        assert client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"] == []
        assert budget.holds == {}
        assert budget.writes == [("settle", 20_000)]  # the $0.02 estimate, charged once
        assert budget.counter == 20_000
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_mid_failure_after_deltas_charges_the_hold(tmp_path, monkeypatch):
    """usage_holder is empty because the response never completed, but the provider already billed the tokens on the wire."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "half an answer"
            raise RuntimeError("provider dropped the connection")

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "half an answer" in msgs[1]["content"]
        assert "[answer truncated]" in msgs[1]["content"]
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_failure_before_any_delta_charges_every_attempt(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        llm = _pin_llm_outage(monkeypatch, retries=2, reserve_usd=0.02)

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert "LLM temporarily unavailable" in body
        assert client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"] == []
        assert llm.calls == 3  # LLM_MAX_RETRIES + 1 requests, every one billed
        assert budget.writes == [("settle", 60_000)]
        assert budget.counter == 60_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_settle_failure_after_a_delivered_stream_does_not_rewrite_it(tmp_path, monkeypatch):
    """The settle is the last step, so failing closed could only destroy an already-delivered answer and prevents no spend either."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            for piece in ["The ", "complete ", "answer."]:
                yield piece
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        async def dead_settle(ids, actual_usd):
            raise chat_module.BudgetUnavailable("redis went down mid-turn")

        _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module, "settle", dead_settle)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert "event: done" in body
        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["content"] == "The complete answer."
        assert "[answer truncated]" not in msgs[1]["content"]
        assert msgs[1]["aborted"] is False
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_settle_failure_after_a_billed_call_keeps_the_answer(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_answer_ranked(question, prompt, holds, system_prompt="", *, spend=None):
            return chat_module.LLMResult(content="A billed answer [1].", prompt_tokens=50, completion_tokens=10)

        async def dead_settle(ids, actual_usd):
            raise chat_module.BudgetUnavailable("redis went down mid-turn")

        _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module, "settle", dead_settle)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)

        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "top deals"})

        assert r.status_code == 200
        assert r.json()["assistant"]["content"] == "A billed answer [1]."
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


class _DeadBudgetRedis:
    """A counter that cannot be reached, so a test can assert a turn never consults it -- which is what the cap opt-out has to mean."""

    def __init__(self):
        self.touched = 0

    def register_script(self, lua):
        self.touched += 1
        raise ConnectionError("redis is down")


def _pin_budget_disabled_with_dead_store(monkeypatch):
    """The cap off AND its counter unreachable -- the combination that must not 503 a chat for a deployment that opted out."""
    dead = _DeadBudgetRedis()
    monkeypatch.setattr(cost_budget_module, "_BUDGET_SCRIPT", None)
    monkeypatch.setattr(cost_budget_module, "_client", lambda: dead)
    monkeypatch.setattr(cost_budget_module.config, "LLM_DAILY_BUDGET_USD", 0.0)
    return dead


def test_disabled_cap_never_consults_the_store(tmp_path, monkeypatch):
    """An empty hold list with a non-zero amount deliberately does not return early, so settle really reaches the store."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_answer_ranked(question, prompt, holds, system_prompt="", *, spend=None):
            return chat_module.LLMResult(content="An unmetered answer [1].", prompt_tokens=50, completion_tokens=10)

        dead = _pin_budget_disabled_with_dead_store(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)

        r = client.post(f"/api/chat/sessions/{sid}/messages", cookies=h, json={"content": "top deals"})

        assert r.status_code == 200
        assert r.json()["assistant"]["content"] == "An unmetered answer [1]."
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert dead.touched == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_disabled_cap_never_consults_the_store_on_the_stream_path(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "A streamed, unmetered answer."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        dead = _pin_budget_disabled_with_dead_store(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert "event: done" in body
        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert msgs[1]["content"] == "A streamed, unmetered answer."
        assert dead.touched == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_disconnect_after_a_completed_stream_is_still_charged(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            # No dataviz block, so the chart-intent nudge gate re-checks the client after the stream has finished.
            yield "No chart here [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        _disconnect_after(monkeypatch, 1)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "show me a chart of top deals")

        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert "No chart here" in msgs[1]["content"]
        assert msgs[1]["cost"] == pytest.approx(0.1)
        assert budget.writes == [("settle", 100_000)]
        assert budget.counter == 100_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_disconnect_at_the_ranking_nudge_check_is_still_charged(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            # A refusal, so the ranking gate is the post-stream check that runs.
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        _disconnect_after(monkeypatch, 1)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")

        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert msgs[1]["cost"] == pytest.approx(0.1)
        assert budget.writes == [("settle", 100_000)]
        assert budget.counter == 100_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_fail_turn_after_deltas_charges_the_estimate_when_usage_is_unreported(tmp_path, monkeypatch):
    """fail_turn with `streamed` True would release the gate hold for an answer already on the wire and already billed."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            # A ranking refusal reaches the nudge gate after the stream; the nudge then raises into fail_turn.
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                # A TRUTHY LLMResult with ZERO tokens, whose cost 0.0 would make finish_holds release the hold.
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=0, completion_tokens=0))

        async def exploding_nudge(client, prompt, model):
            raise RuntimeError("the provider died on the retry")

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", exploding_nudge)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")

        assert "event: error" in body
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert "I cannot generate a ranked list" in msgs[1]["content"]
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_completed_turn_with_no_usage_report_is_still_charged(tmp_path, monkeypatch):
    """A zero means the cost is unknown, not free, so the held estimate is charged and stored."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "A fully delivered answer [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=0, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert "event: done" in body
        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()["messages"]
        assert msgs[1]["aborted"] is False
        assert msgs[1]["cost"] == pytest.approx(0.02)
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_completed_turn_with_reported_usage_still_uses_the_real_cost(tmp_path, monkeypatch):
    """A provider that DOES report usage must be charged its real cost, or the no-usage test would pass with every turn charged the reserve."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "A fully delivered answer [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        assert budget.writes == [("settle", 100_000)]
        assert budget.counter == 100_000
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def _seed_messages(store, sid, count, *, user_id=USER_A, sources_per_message=0, content_len=1):
    """Bypasses the turn pipeline, so a thread can exceed the read cap without an LLM."""
    for i in range(count):
        sources = (
            [{"id": j, "title": f"s{j}", "summary": "x" * 200, "score": 0.5} for j in range(sources_per_message)]
            if sources_per_message
            else None
        )
        _run(
            store.append_message(
                sid,
                user_id,
                "user" if i % 2 == 0 else "assistant",
                f"msg-{i:04d}-" + "c" * content_len,
                sources=sources,
            )
        )


def test_session_read_returns_only_the_most_recent_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 20)
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _seed_messages(store, sid, 200)

        msgs, total = _run(store.messages_page(sid, USER_A))

        assert total == 200
        assert len(msgs) == 20
        contents = [m.content for m in msgs]
        # Chronological (oldest first) AND the newest 20, not the oldest 20.
        assert contents == [f"msg-{i:04d}-c" for i in range(180, 200)]
        assert [m.id for m in msgs] == sorted(m.id for m in msgs)
    finally:
        _run(store.close())


def test_session_read_is_deterministic_when_timestamps_tie(tmp_path, monkeypatch):
    """`ORDER BY created_at DESC` alone leaves the tail undefined when rows share a timestamp; the `id` tiebreak is what makes the selection stable."""
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 5)
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _seed_messages(store, sid, 30)
        _run(store._db.execute("UPDATE messages SET created_at = 1000.0 WHERE session_id = ?", (sid,)))
        _run(store._db.commit())

        msgs, total = _run(store.messages_page(sid, USER_A))

        assert total == 30
        assert [m.content for m in msgs] == [f"msg-{i:04d}-c" for i in range(25, 30)]
        assert [m.id for m in msgs] == sorted(m.id for m in msgs)
    finally:
        _run(store.close())


def test_session_read_under_the_cap_is_returned_completely_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 50)
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _seed_messages(store, sid, 7, sources_per_message=2)

        msgs, total = _run(store.messages_page(sid, USER_A))

        assert total == 7
        assert total == len(msgs)
        assert [m.content for m in msgs] == [f"msg-{i:04d}-c" for i in range(7)]
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[1].sources == [
            {"id": j, "title": f"s{j}", "summary": "x" * 200, "score": 0.5} for j in range(2)
        ]
        assert [m.aborted for m in msgs] == [False] * 7
        assert [m.cost for m in msgs] == [0.0] * 7
    finally:
        _run(store.close())


def test_session_read_caps_sources_per_message(tmp_path, monkeypatch):
    """Each row deserialises its sources, so an unbounded list multiplies the response on top of the row cap."""
    monkeypatch.setattr(chat_module.config, "CHAT_MESSAGE_SOURCE_LIMIT", 3)
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _seed_messages(store, sid, 2, sources_per_message=25)

        msgs, _total = _run(store.messages_page(sid, USER_A))

        assert [len(m.sources) for m in msgs] == [3, 3]
        assert [s["id"] for s in msgs[1].sources] == [0, 1, 2]
    finally:
        _run(store.close())


def test_api_session_detail_flags_truncation_to_the_client(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 5)
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        _seed_messages(chat_store, sid, 40, user_id=_run(auth_store.get_user_by_email(EMAIL_A)).id)

        body = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()

        assert len(body["messages"]) == 5
        assert body["truncated"] is True
        assert body["total_messages"] == 40
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_session_detail_under_the_cap_is_not_flagged_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 200)
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_cookies(auth_store)
        sid = client.post("/api/chat/sessions", cookies=h).json()["id"]
        _seed_messages(chat_store, sid, 12, user_id=_run(auth_store.get_user_by_email(EMAIL_A)).id)

        body = client.get(f"/api/chat/sessions/{sid}", cookies=h).json()

        assert body["truncated"] is False
        assert body["total_messages"] == 12
        assert len(body["messages"]) == 12
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_session_read_response_size_is_bounded_by_the_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 20)
    monkeypatch.setattr(chat_module.config, "CHAT_MESSAGE_SOURCE_LIMIT", 5)
    store = _store(tmp_path)
    try:
        small = _run(store.create_session(USER_A)).id
        _seed_messages(store, small, 20, sources_per_message=5, content_len=200)
        big = _run(store.create_session(USER_A)).id
        _seed_messages(store, big, 400, sources_per_message=5, content_len=200)

        small_msgs, _ = _run(store.messages_page(small, USER_A))
        big_msgs, big_total = _run(store.messages_page(big, USER_A))

        assert big_total == 400
        assert len(big_msgs) == len(small_msgs) == 20
        assert sum(len(m.sources) for m in big_msgs) == 100
        big_bytes = len(json.dumps([m.model_dump() for m in big_msgs]))
        small_bytes = len(json.dumps([m.model_dump() for m in small_msgs]))
        # Same per-message shape, so the sizes differ only by id/offset digits.
        assert big_bytes < small_bytes * 2
    finally:
        _run(store.close())


def test_session_cap_does_not_touch_the_prompt_history_path(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_module.config, "CHAT_SESSION_MESSAGE_LIMIT", 2)
    monkeypatch.setattr(chat_module.config, "CHAT_MAX_HISTORY_TURNS", 5)
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _seed_messages(store, sid, 6)

        msgs, _total = _run(store.messages_page(sid, USER_A))
        recent = _run(store.recent_turns(sid, USER_A, 5))

        assert len(msgs) == 2
        assert len(recent) == 6
    finally:
        _run(store.close())


def test_session_cap_defaults_are_bounded_and_read_from_env(monkeypatch):
    """A plain importlib.reload would rebind app.config.config and every module that did
    `from app.config import config` would keep the old one, so this loads a STANDALONE
    copy. Nothing asserts against the live singleton: app/config.py calls load_dotenv()
    at import, so a deployment legitimately setting these knobs would fail the suite."""
    import importlib.util

    import dotenv

    from app import config as config_module

    src = pathlib.Path(config_module.__file__)
    # app/config.py does `from dotenv import load_dotenv`, so patch it before
    # exec_module; patching the CWD is not enough because load_dotenv searches
    # upward from the calling file, which is inside the repo.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.config

    monkeypatch.delenv("CHAT_SESSION_MESSAGE_LIMIT", raising=False)
    monkeypatch.delenv("CHAT_MESSAGE_SOURCE_LIMIT", raising=False)

    defaults = load("config_default_probe")
    assert defaults.CHAT_SESSION_MESSAGE_LIMIT == 200
    assert defaults.CHAT_MESSAGE_SOURCE_LIMIT == 20

    monkeypatch.setenv("CHAT_SESSION_MESSAGE_LIMIT", "25")
    monkeypatch.setenv("CHAT_MESSAGE_SOURCE_LIMIT", "4")
    overridden = load("config_env_override_probe")
    assert overridden.CHAT_SESSION_MESSAGE_LIMIT == 25
    assert overridden.CHAT_MESSAGE_SOURCE_LIMIT == 4

    example = (src.parent.parent / ".env.example").read_text()
    assert "CHAT_SESSION_MESSAGE_LIMIT=200" in example
    assert "CHAT_MESSAGE_SOURCE_LIMIT=20" in example


def _cancel_request():
    """`is_disconnected()` is always False, so nothing but the cancellation under test can reconcile the stored turn."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat/sessions/s/messages",
        "headers": [],
        "query_string": b"",
    }

    async def receive():
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    request = Request(scope, receive)
    request.state.user_id = USER_A
    return request


async def _cancel_while_consuming(agen, ready):
    """Returns True only if the CancelledError kept propagating; a turn that swallows it returns normally, a bug of its own."""
    async def consume():
        async for _ in agen:
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(ready.wait(), timeout=5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return True
    return False


def _turn_rows(store, sid, user_id=USER_A):
    rows, _total = _run(store.messages_page(sid, user_id))
    return [(m.role, m.content, m.aborted) for m in rows]


def _store_with_session(tmp_path):
    store = _store(tmp_path)
    chat_module.store = store
    sid = _run(store.create_session(USER_A)).id
    return store, sid


def _release_store(store):
    chat_module.store = None
    _run(store.close())


def test_stream_cancelled_during_retrieval_rolls_back_the_user_message(tmp_path, monkeypatch):
    """Retrieval never reaches an `await aborted()` checkpoint, and `except Exception` cannot see a CancelledError."""
    store, sid = _store_with_session(tmp_path)
    try:
        ready = asyncio.Event()

        async def stuck_retrieval(question, history):
            ready.set()
            await asyncio.sleep(3600)

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", stuck_retrieval)

        response = _run(
            chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), _cancel_request()
            )
        )
        assert _run(_cancel_while_consuming(response.body_iterator, ready)) is True
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def test_stream_cancelled_inside_the_provider_stream_rolls_back_the_user_message(tmp_path, monkeypatch):
    """The window the polling guard cannot cover: the turn is suspended in the provider call, not at a checkpoint."""
    store, sid = _store_with_session(tmp_path)
    try:
        ready = asyncio.Event()

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def stuck_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            ready.set()
            await asyncio.sleep(3600)
            yield "the client will never see this"

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", stuck_stream)

        response = _run(
            chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), _cancel_request()
            )
        )
        assert _run(_cancel_while_consuming(response.body_iterator, ready)) is True
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def test_stream_cancelled_after_deltas_persists_the_truncated_turn(tmp_path, monkeypatch):
    store, sid = _store_with_session(tmp_path)
    try:
        ready = asyncio.Event()

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def half_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "Half an "
            yield "answer."
            ready.set()
            await asyncio.sleep(3600)
            yield " never seen by the client"

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", half_stream)

        response = _run(
            chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), _cancel_request()
            )
        )
        assert _run(_cancel_while_consuming(response.body_iterator, ready)) is True

        rows = _turn_rows(store, sid)
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][2] is True
        assert rows[1][1].startswith("Half an answer.")
        assert "[answer truncated]" in rows[1][1]
        assert "never seen" not in rows[1][1]
    finally:
        _release_store(store)


def test_stream_cancelled_after_the_reply_is_stored_keeps_the_completed_turn(tmp_path, monkeypatch):
    """Rolling back after the reply is stored would orphan a reply the client is reading; the `persisted` flag is what stops it."""
    store, sid = _store_with_session(tmp_path)
    try:
        ready = asyncio.Event()

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="Hi there.", sources=[], note=None, needs_llm=False)

        async def stuck_auto_title(*args, **kwargs):
            ready.set()
            await asyncio.sleep(3600)

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "_auto_title", stuck_auto_title)

        response = _run(chat_module.send_message_stream(sid, chat_module.MessageIn(content="hello"), _cancel_request()))
        assert _run(_cancel_while_consuming(response.body_iterator, ready)) is True

        rows = _turn_rows(store, sid)
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][1] == "Hi there."
    finally:
        _release_store(store)


def test_json_turn_cancelled_mid_generation_rolls_back_the_user_message(tmp_path, monkeypatch):
    store, sid = _store_with_session(tmp_path)
    try:
        ready = asyncio.Event()

        async def stuck_turn(question, history):
            ready.set()
            await asyncio.sleep(3600)

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_run_turn", stuck_turn)

        async def run_it():
            task = asyncio.create_task(
                chat_module.send_message(sid, chat_module.MessageIn(content="what deals happened"), _cancel_request())
            )
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def test_json_turn_cancelled_after_the_reply_is_stored_keeps_the_completed_turn(tmp_path, monkeypatch):
    store, sid = _store_with_session(tmp_path)
    try:
        ready = asyncio.Event()

        async def quick_turn(question, history):
            return ("A full answer.", [], None, 0, 0, 0.0)

        async def stuck_auto_title(*args, **kwargs):
            ready.set()
            await asyncio.sleep(3600)

        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_run_turn", quick_turn)
        monkeypatch.setattr(chat_module, "_auto_title", stuck_auto_title)

        async def run_it():
            task = asyncio.create_task(
                chat_module.send_message(sid, chat_module.MessageIn(content="what deals happened"), _cancel_request())
            )
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True
        rows = _turn_rows(store, sid)
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][1] == "A full answer."
    finally:
        _release_store(store)


def _asgi_disconnect_after_deltas(tmp_path, monkeypatch, n_deltas, park_in_send=False):
    """`park_in_send` decides where the turn is when the disconnect lands: inside
    `stream_answer` by default, or inside Starlette's `send()` between deltas, where
    the cancellation reaches the CONSUMER and the generator is finalised with
    GeneratorExit. anyio delivers it as a LEVEL cancellation, re-raising at every
    following await, so an unshielded rollback is interrupted before it writes."""
    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    app = FastAPI()
    app.include_router(chat_module.router)
    chat_module.store = chat_store
    auth_module.store = auth_store
    session = _auth_cookies(auth_store)
    # A raw ASGI scope has no cookie jar, so write out the header a browser would attach.
    cookie_header = "; ".join(f"{name}={value}" for name, value in session.items())
    # The session must belong to the cookie's user, or the turn 404s and nothing streams.
    user_id = _run(auth_store.get_user_by_email(EMAIL_A)).id
    sid = _run(chat_store.create_session(user_id)).id
    state = {"drop": False}

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_stream(client, prompt, model, usage_holder=None, system_prompt=None):
        for piece in ["One ", "two ", "three."][:n_deltas]:
            await asyncio.sleep(0)
            yield piece
        state["drop"] = True
        await asyncio.sleep(3600)
        yield " never delivered"

    _pin_budget_disabled(monkeypatch)
    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

    body = json.dumps({"content": "what deals happened"}).encode()
    sent_body = False
    sent = {"deltas": 0, "bodies": []}

    async def send(message):
        if message["type"] == "http.response.body":
            sent["bodies"].append(message.get("body", b"")[:80])
        if message["type"] == "http.response.body" and b"event: delta" in message.get("body", b""):
            sent["deltas"] += 1
            if park_in_send and sent["deltas"] >= n_deltas:
                # The generator is suspended at its yield, so the cancellation lands
                # on the consumer and no generator handler runs.
                state["drop"] = True
                await asyncio.sleep(3600)

    async def receive():
        # FastAPI reads the body through this same callable, so it must come first;
        # the disconnect follows once the provider stream has parked.
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        while not state["drop"]:
            await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/api/chat/sessions/{sid}/messages/stream",
        "raw_path": f"/api/chat/sessions/{sid}/messages/stream".encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"cookie", cookie_header.encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    async def call_app():
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
        # Without this a 404 or an early error would make the rollback assertions pass for the wrong reason.
        assert any(b"event: start" in body for body in sent["bodies"]), sent["bodies"]
        assert sent["deltas"] == n_deltas, sent["bodies"]

    def run():
        """A missing cancellation rollback does not merely give the wrong rows: the
        abandoned commit leaves the aiosqlite connection unusable and a main-thread
        read blocks forever, wedging the suite instead of reporting."""
        outcome: dict = {}

        async def call_and_read():
            await call_app()
            rows, _total = await chat_store.messages_page(sid, user_id)
            return rows

        def worker():
            try:
                outcome["rows"] = asyncio.run(call_and_read())
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                outcome["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=20)
        if thread.is_alive():
            raise AssertionError(
                "the ASGI turn never finished: a cancelled turn with no rollback "
                "can stall the app task, and its store, rather than fail"
            )
        if "error" in outcome:
            raise outcome["error"]
        return [(m.role, m.content, m.aborted) for m in outcome["rows"]]

    return run, chat_store, auth_store, sid, user_id


def test_real_disconnect_before_any_delta_rolls_back_the_turn(tmp_path, monkeypatch):
    """An unshielded rollback is interrupted before the delete lands."""
    run, chat_store, auth_store, _sid, _user_id = _asgi_disconnect_after_deltas(tmp_path, monkeypatch, 0)
    try:
        assert run() == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())
        chat_module.store = None
        auth_module.store = None


def test_real_disconnect_after_deltas_persists_the_truncated_turn(tmp_path, monkeypatch):
    run, chat_store, auth_store, _sid, _user_id = _asgi_disconnect_after_deltas(tmp_path, monkeypatch, 1)
    try:
        rows = run()
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][2] is True
        assert "[answer truncated]" in rows[1][1]
        assert "never delivered" not in rows[1][1]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())
        chat_module.store = None
        auth_module.store = None


def test_stream_cancelled_after_the_gate_charges_the_billed_call(tmp_path, monkeypatch):
    """The provider billed the prompt when the request went out, so releasing would make that spend invisible to the cap."""
    store, sid = _store_with_session(tmp_path)
    try:
        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
        ready = asyncio.Event()

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def stuck_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            ready.set()
            await asyncio.sleep(3600)
            yield "never delivered"

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", stuck_stream)

        response = _run(
            chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), _cancel_request()
            )
        )
        assert _run(_cancel_while_consuming(response.body_iterator, ready)) is True

        assert budget.writes == [("settle", 20_000)]
        assert budget.holds == {}
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


@pytest.mark.parametrize("streaming", [False, True], ids=["json", "sse"])
def test_turn_cancelled_before_its_own_handlers_roll_back_the_user_message(
    tmp_path, monkeypatch, streaming
):
    """Both turn paths call `_start_turn` before their cancellation handlers exist, so a cancel in that window has no handler able to remove the user row."""
    store, sid = _store_with_session(tmp_path)
    try:
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        ready = asyncio.Event()
        real_recent = store.recent_turns

        async def stuck_recent_turns(session_id, user_id, max_turns):
            ready.set()
            await asyncio.sleep(3600)
            return await real_recent(session_id, user_id, max_turns)

        monkeypatch.setattr(store, "recent_turns", stuck_recent_turns)
        request = _cancel_request()
        if streaming:
            endpoint = lambda: chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), request
            )
        else:
            endpoint = lambda: chat_module.send_message(
                sid, chat_module.MessageIn(content="what deals happened"), request
            )

        async def run_it():
            task = asyncio.create_task(endpoint())
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def _park_at_first_assistant_commit(monkeypatch, store, parked):
    """aiosqlite resolves an independently cancellable future only after running the
    statement, so a cancel at a row's own COMMIT finds the write done while the append
    never returns -- and a local "did the append return" flag is still False, which is
    what used to delete the user message under a stored reply. Parked on
    `_append_authorized` for the first assistant append so a reconciliation's own
    write still lands."""
    ready = asyncio.Event()
    real_append = store._append_authorized

    async def slow_append(session, role, *args, **kwargs):
        result = await real_append(session, role, *args, **kwargs)
        if role == "assistant" and not parked["done"]:
            parked["done"] = True
            ready.set()
            await asyncio.sleep(3600)
        return result

    monkeypatch.setattr(store, "_append_authorized", slow_append)
    return ready


def test_json_turn_cancelled_at_the_reply_commit_keeps_the_completed_turn(tmp_path, monkeypatch):
    store, sid = _store_with_session(tmp_path)
    parked = {"done": False}
    try:
        _pin_budget_disabled(monkeypatch)

        async def quick_turn(question, history):
            return ("A full answer.", [], None, 0, 0, 0.0)

        monkeypatch.setattr(chat_module, "_run_turn", quick_turn)
        ready = _park_at_first_assistant_commit(monkeypatch, store, parked)

        async def run_it():
            task = asyncio.create_task(
                chat_module.send_message(sid, chat_module.MessageIn(content="q"), _cancel_request())
            )
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True
        rows = _turn_rows(store, sid)
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][1] == "A full answer."
    finally:
        _release_store(store)


def test_stream_cancelled_at_the_reply_commit_stores_no_second_row(tmp_path, monkeypatch):
    """A rollback trusting only a local flag would persist a SECOND truncated assistant row, so the client sees the answer twice."""
    store, sid = _store_with_session(tmp_path)
    parked = {"done": False}
    try:
        _pin_budget_disabled(monkeypatch)

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def short_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            yield "The whole answer."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=5, completion_tokens=2))

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", short_stream)
        ready = _park_at_first_assistant_commit(monkeypatch, store, parked)

        response = _run(
            chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="q"), _cancel_request()
            )
        )
        assert _run(_cancel_while_consuming(response.body_iterator, ready)) is True

        rows = _turn_rows(store, sid)
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][1] == "The whole answer."
        assert rows[1][2] is False
    finally:
        _release_store(store)


@pytest.mark.parametrize("streaming", [False, True], ids=["json", "sse"])
def test_turn_cancelled_at_the_user_insert_commit_rolls_back(tmp_path, monkeypatch, streaming):
    """The row exists but nothing knows its id, so `_start_turn` finds it instead."""
    store, sid = _store_with_session(tmp_path)
    try:
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        ready = asyncio.Event()
        real_append = store._append_authorized
        parked = {"done": False}

        async def slow_append(session, role, *args, **kwargs):
            result = await real_append(session, role, *args, **kwargs)
            if role == "user" and not parked["done"]:
                parked["done"] = True
                ready.set()
                await asyncio.sleep(3600)
            return result

        monkeypatch.setattr(store, "_append_authorized", slow_append)
        request = _cancel_request()
        if streaming:
            endpoint = lambda: chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), request
            )
        else:
            endpoint = lambda: chat_module.send_message(
                sid, chat_module.MessageIn(content="what deals happened"), request
            )

        async def run_it():
            task = asyncio.create_task(endpoint())
            await asyncio.wait_for(ready.wait(), timeout=5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)


def test_real_disconnect_while_starlette_is_sending_persists_the_turn(tmp_path, monkeypatch):
    """The generator is finalised with GeneratorExit, which no `except` in it can catch, so the response's background task must persist the truncated turn."""
    run, chat_store, auth_store, _sid, _user_id = _asgi_disconnect_after_deltas(
        tmp_path, monkeypatch, 1, park_in_send=True
    )
    try:
        rows = run()
        assert [role for role, _c, _a in rows] == ["user", "assistant"]
        assert rows[1][2] is True
        assert "[answer truncated]" in rows[1][1]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())
        chat_module.store = None
        auth_module.store = None


def test_cancelled_inside_the_body_iterator_finishes_its_own_rollback(tmp_path, monkeypatch):
    """The generator's own handler is the only thing that can finish the turn, and anyio re-raises at every await, so the delete lands only if that handler's write is shielded."""
    store, sid = _store_with_session(tmp_path)
    try:
        _pin_budget_disabled(monkeypatch)

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        state = {"parked": False}

        async def parked_stream(client, prompt, model, usage_holder=None, system_prompt=None):
            state["parked"] = True
            await asyncio.sleep(3600)
            yield "never delivered"

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", parked_stream)

        response = _run(
            chat_module.send_message_stream(
                sid, chat_module.MessageIn(content="what deals happened"), _cancel_request()
            )
        )
        body_iterator = response.body_iterator

        async def stream_response():
            async for _chunk in body_iterator:
                pass

        async def listen_for_disconnect():
            while not state["parked"]:
                await asyncio.sleep(0)
            return {"type": "http.disconnect"}

        async def drive():
            async with anyio.create_task_group() as task_group:

                async def wrap(func):
                    await func()
                    task_group.cancel_scope.cancel()

                task_group.start_soon(wrap, partial(stream_response))
                await wrap(partial(listen_for_disconnect))

        _run(asyncio.wait_for(drive(), timeout=10))
        assert _turn_rows(store, sid) == []
    finally:
        _release_store(store)
