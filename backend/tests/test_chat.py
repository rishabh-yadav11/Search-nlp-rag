"""Chat store and API tests: per-user session CRUD, ownership isolation,
retention purging, and the message-turn flow (retrieval + LLM stubbed)."""

import asyncio
import json
import logging
import pathlib
import re
import shutil
import sqlite3
import subprocess
import time
from typing import ClassVar

import pytest
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


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _stub_temporal_date_window(monkeypatch):
    """retrieve_by_date_window needs a live Qdrant client (state['qdrant']); chat
    unit tests stub retrieval at retrieve_and_rerank and do not stand up Qdrant,
    so neutralize the temporal fallback here. The fallback itself is exercised by
    the date-window retrieval path, not by these chat unit tests."""
    from app import main as _main

    async def _noop(*args, **kwargs):
        return []

    monkeypatch.setattr(_main, "retrieve_by_date_window", _noop)


def _store(tmp_path):
    s = ChatStore(str(tmp_path / "chat.db"))
    _run(s.connect())
    return s


def _auth_store(tmp_path):
    s = AuthStore(str(tmp_path / "auth.db"))
    _run(s.connect())
    return s


def _auth_headers(auth_store, email=EMAIL_A, role="user"):
    """Create/upgrade the account and return a valid Bearer header for it."""
    user = _run(auth_store.get_user_by_email(email))
    if user is None:
        user = _run(auth_store.create_user(email, "secret1", email.split("@")[0], role))
    elif user.role != role:
        _run(auth_store.update_user(user.id, None, role, None))
    token = _run(auth_store.issue_token(user.id, 7))
    return {"Authorization": f"Bearer {token}"}


def test_create_and_list_sessions(tmp_path):
    store = _store(tmp_path)
    try:
        a = _run(store.create_session(USER_A))
        b = _run(store.create_session(USER_A))
        listed = _run(store.list_sessions(USER_A))
        assert [s.id for s in listed] == [b.id, a.id]  # most recent first
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
            _run(store.messages(a.id, USER_B))
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
        msgs = _run(store.messages(a.id, USER_A))
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
        assert [t.content for t in turns] == ["q1", "a1", "q2", "a2"]  # oldest first, newest pair kept
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
    """Create a pre-token-tracking SQLite DB (the schema chat.py must migrate)."""
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
    """A DB created before token/cost tracking gets the missing columns added by
    connect() (ERROR PATH — legacy/malformed SQLite schema)."""
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
        assert rows[0]["prompt_tokens"] == 0  # migrated columns default to 0
        assert rows[0]["latency_ms"] == 0
    finally:
        _run(store.close())


def test_connect_adds_missing_latency_ms_only(tmp_path):
    """connect() also adds latency_ms on its own when only that column is
    missing from an otherwise current schema."""
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
    _run(store.close())  # already closed -> no-op
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


def test_global_stats_never_raises_on_error(tmp_path, monkeypatch):
    """global_stats degrades to an error payload instead of raising when the
    underlying query fails (ERROR PATH — DB/query failure)."""
    store = _store(tmp_path)
    try:
        async def boom(*args, **kwargs):
            raise RuntimeError("db gone")

        monkeypatch.setattr(store, "_fetchone", boom)
        monkeypatch.setattr(store, "_fetchall", boom)
        assert _run(store.global_stats()) == {"error": "chat analytics unavailable"}
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
        # A device-id header (X-User-Id) no longer bypasses auth.
        assert client.post("/api/chat/sessions", headers={"X-User-Id": USER_A}).status_code == 401
        assert client.post("/api/chat/sessions", headers={"Authorization": "Bearer garbage"}).status_code == 401
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_create_and_list(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        created = client.post("/api/chat/sessions", headers=h).json()
        assert created["id"]
        listed = client.get("/api/chat/sessions", headers=h).json()
        assert [s["id"] for s in listed] == [created["id"]]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_get_rename_delete_flow(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        h_b = _auth_headers(auth_store, email=EMAIL_B)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        detail = client.get(f"/api/chat/sessions/{sid}", headers=h).json()
        assert detail["messages"] == []

        renamed = client.patch(f"/api/chat/sessions/{sid}", headers=h, json={"content": "Renamed"}).json()
        assert renamed["title"] == "Renamed"

        # Other accounts cannot read this conversation.
        assert client.get(f"/api/chat/sessions/{sid}", headers=h_b).status_code == 404

        assert client.delete(f"/api/chat/sessions/{sid}", headers=h).status_code == 200
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).status_code == 404
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_runs_turn(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_turn(question, history):
            assert question == "Who invested in fintech?"
            assert [m.role for m in history] == ["user"]  # prior turn context included
            return "A fintech investor is [1].", [{"id": 1, "title": "Fintech funding"}], None, 120, 45, 0.0012

        monkeypatch.setattr(chat_module, "_run_turn", fake_turn)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "Who invested in fintech?"})
        assert r.status_code == 200
        body = r.json()
        assert body["user"]["content"] == "Who invested in fintech?"
        assert body["assistant"]["content"] == "A fintech investor is [1]."
        assert body["assistant"]["sources"][0]["title"] == "Fintech funding"
        assert body["assistant"]["prompt_tokens"] == 120
        assert body["assistant"]["completion_tokens"] == 45
        assert body["assistant"]["cost"] == 0.0012

        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["title"] == "Who invested in fintech?"

        detail = client.get(f"/api/chat/sessions/{sid}", headers=h).json()
        assert len(detail["messages"]) == 2
        assert detail["messages"][1]["prompt_tokens"] == 120
        assert detail["messages"][1]["cost"] == 0.0012
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_usage_stats(tmp_path, monkeypatch):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        h_b = _auth_headers(auth_store, email=EMAIL_B)
        assert client.get("/api/chat/usage", headers=h).json() == {
            "sessions": 0, "messages": 0, "total_tokens": 0, "total_cost": 0.0
        }

        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_turn(question, history):
            return "answer", [], None, 100, 50, 0.0005

        monkeypatch.setattr(chat_module, "_run_turn", fake_turn)
        client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "query one"})
        client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "query two"})

        usage = client.get("/api/chat/usage", headers=h).json()
        assert usage["sessions"] == 1
        assert usage["messages"] == 4  # 2 user + 2 assistant
        assert usage["total_tokens"] == 300  # 2 * (100 + 50)
        assert abs(usage["total_cost"] - 0.001) < 1e-9

        # Other users see their own usage only.
        assert client.get("/api/chat/usage", headers=h_b).json()["total_tokens"] == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_rejects_empty(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]
        assert client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "   "}).status_code == 400
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
    """SSE stream for small talk emits a single done event with a canned reply."""
    from app import main

    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def boom(*args, **kwargs):
            raise AssertionError("retrieval should not run for small talk")

        monkeypatch.setattr(main, "retrieve_and_rerank", boom)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", headers=h, json={"content": "good morning"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())

        assert "event: start" in body
        assert "event: done" in body
        assert "Hello!" in body
        assert "event: error" not in body

        # Assistant message persisted.
        detail = client.get(f"/api/chat/sessions/{sid}", headers=h).json()
        assert len(detail["messages"]) == 2
        assert detail["messages"][1]["role"] == "assistant"
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_full_turn(tmp_path, monkeypatch):
    """SSE stream with a real LLM path emits deltas + a done event with usage."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(
                answer="prompt-text",
                sources=[{"id": 1, "title": "Src"}],
                note=None,
                needs_llm=True,
            )

        async def fake_stream(client, prompt, model, usage_holder=None):
            for piece in ["Hello ", "world", "!"]:
                yield piece
            if usage_holder is not None:
                # Real stream_answer fills the holder with an LLMResult.
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        _pin_budget_disabled(monkeypatch)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", headers=h, json={"content": "Who invested in fintech?"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())

        assert body.count("event: delta") == 3
        assert "Hello world!" in body
        assert "event: done" in body
        assert "prompt_tokens" in body

        detail = client.get(f"/api/chat/sessions/{sid}", headers=h).json()
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
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="prompt-text", sources=[{"id": 1}], note=None, needs_llm=True)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        # $10.00 already spent against a $2.00 cap: the first gate's reserve is
        # refused, so no billed call is ever started.
        _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=10.0)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", headers=h, json={"content": "question"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())

        assert "event: error" in body
        assert "Daily AI budget reached" in body
        assert "event: done" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_global_stats_aggregates(tmp_path):
    """global_stats returns cross-user counts, tokens, cost and top tables."""
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
    """/analytics/chat returns global chat stats through the app (admin-only)."""
    from fastapi.testclient import TestClient

    from app import main

    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    chat_module.store = chat_store
    auth_module.store = auth_store
    client = TestClient(main.app)
    try:
        admin_h = _auth_headers(auth_store, email="admin@example.com", role="admin")
        sid = client.post("/api/chat/sessions", headers=admin_h).json()["id"]
        client.post(f"/api/chat/sessions/{sid}/messages", headers=admin_h, json={"content": "hello"})

        # Regular users are denied analytics.
        user_h = _auth_headers(auth_store, email=EMAIL_A)
        assert client.get("/analytics/chat", headers=user_h).status_code == 403
        # Unauthenticated requests are rejected.
        assert client.get("/analytics/chat").status_code == 401

        res = client.get("/analytics/chat", headers=admin_h)
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


def test_prepare_turn_passes_intent_date_filter_to_retrieval(monkeypatch):
    """Chat must apply the auto date filter derived by _effective_intent,
    matching /search (regression: chat passed qfilter=None)."""
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
    """Empty (score-gated) sources must NOT carry a weak_results_note: the
    'No sufficiently relevant articles' answer already explains the miss, and a
    note saying 'Showing the closest 2020 matches' alongside zero results is a
    lie (regression: weak_results_note([]) returned a misleading string)."""
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
    """Non-empty weak sources (score above the ASK_MIN_SCORE gate but below the
    weak threshold) must still get the weak_results_note on the fallback turn."""
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
    """A vague follow-up ('make this into a table') has no standalone topic:
    retrieval must inherit the previous turn's query + date filter + top-N,
    otherwise the embedding on the bare follow-up finds nothing and the turn
    short-circuits to 'no relevant articles' before the LLM sees the history."""
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
    assert captured["top_k"] == 10  # previous turn's 'top ...' list size
    assert "make this into a table" in turn.answer  # current question in prompt
    assert "top ipo in 2025" in turn.answer  # history included for the LLM


def test_is_vague_followup_treats_prior_result_reference_as_vague():
    """A follow-up that references the previous result/answer with only generic
    words ('share the data in chart', 'share the last result data in chart') has
    no standalone topic and must be treated as vague so retrieval inherits the
    prior turn's query instead of searching the non-topical words."""
    from app.chat import _is_vague_followup

    assert _is_vague_followup("share the data in chart") is True
    assert _is_vague_followup("share the last result data in chart") is True
    assert _is_vague_followup("show that data as a chart") is True
    assert _is_vague_followup("give the previous answer in a table") is True
    # Regression: anaphoric 'this deals' + format words is a follow-up reference
    # to the prior result, not a new topic ('m&a deals in 2025' -> tabular/table).
    assert _is_vague_followup("share this deals in tabular format") is True
    assert _is_vague_followup("share this deals in table format") is True
    assert chat_module._requested_view("share this deals in tabular format") == "table"
    assert chat_module._requested_view("share this deals in table format") == "table"
    # A real topic must NOT be swallowed as vague.
    assert _is_vague_followup("m&a deals in 2025") is False
    assert _is_vague_followup("make a table of top 15 deals in 2024-25") is False
    # A new predication on the noun is a standalone topic, not a reference.
    assert _is_vague_followup("this table shows Q3 deals") is False
    assert _is_vague_followup("this week deals in fintech") is False
    # Trailing politeness does not make a reference topical.
    assert _is_vague_followup("share the last result thanks") is True
    assert _is_vague_followup("share the last result, thanks") is True


def test_previous_user_question_skips_chained_vague_followups():
    """When the immediately preceding turn is itself a vague follow-up, the prior
    topic lookup must skip past it and return the real preceding query (the IPO
    table turn), not the degenerate 'share the last result' question."""
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
    """'share the last result data in chart' must inherit the real preceding IPO
    query (not a degenerate earlier follow-up) so retrieval finds the same
    sources the IPO table was built from (regression for the 'No relevant
    articles' dead-end)."""
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
    """A standalone question (even one that asks for a table) must use its own
    retrieval topic, not the previous turn's."""
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
    """The names chat lazily imports from app.main must stay available after
    endpoint removals (regression: /ask removal dropped source_context)."""
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
    # inconsistent row widths
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A","B"], "rows": [["x", 1], ["y"]]}\n```') is None
    # value column is not numeric
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A","B"], "rows": [["x", "y"], ["z", "w"]]}\n```') is None


def test_parse_dataviz_infers_numeric_column():
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Deal", "Value"], "rows": [["Zepto", 1.0], ["MUFG", 4.4]]}\n```'
    )
    assert data is not None
    assert data["value_column"] == 1


def test_parse_dataviz_allows_missing_values():
    """A top-N table may include rows whose value isn't stated ("" or None) as
    long as at least one row has a number and no non-empty cell is non-numeric."""
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

    # A genuinely non-numeric cell still invalidates the block.
    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", "abc"], ["Groww", 1200]], "value_column": 1}\n```'
    ) is None
    # A value column with no numeric cell at all is invalid for a chart block...
    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", ""], ["Groww", "value not stated"]], "value_column": 1}\n```'
    ) is None
    # ...but a table block with no numeric column (value_column null) is valid
    # for a plain text table (e.g. every item's value is 'not stated').
    data = chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Status"], '
        '"rows": [["Wakefit", "not stated"], ["Groww", "not stated"]], "value_column": null, "view": "table"}\n```'
    )
    assert data is not None
    assert data["value_column"] is None
    # When a numeric column exists, a missing value_column still auto-detects it.
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

    # A bare ```dataviz``` tag in prose now matches the (newline-optional)
    # grammar, carries an empty body, fails to parse, and is stripped like any
    # other malformed block. It used to survive only because the old pattern
    # required a newline after the tag -- the bypass #255 removes. The frontend
    # already stripped it, so this is the two sides agreeing (#255).
    plain = "Just prose with a ```dataviz``` mention."
    out = chat_module._sanitize_dataviz(plain)
    assert "```dataviz" not in out
    # The trailing \s* of the fence grammar eats the space before "mention."
    assert out == "Just prose with a mention."


def test_parse_dataviz_rejects_all_empty_label_cells():
    """A table whose label column is blank on every row (e.g. the model emitted
    only values and no deal/company names) is useless and must be treated as
    malformed so the nudge retry rebuilds it."""
    empty_labels = (
        '```dataviz\n{"columns": ["Deal", "Value ($B)"], '
        '"rows": [["", 8.5], ["", 4.3], ["", 0.35]], "value_column": 1, "format": "$B"}\n```'
    )
    assert chat_module.parse_dataviz(empty_labels) is None
    out = chat_module._sanitize_dataviz("Prose [1].\n\n" + empty_labels)
    assert "```dataviz" not in out
    assert "Prose [1]" in out

    # A block with names present (even if a few rows are blank) stays valid.
    partly = (
        '```dataviz\n{"columns": ["Deal", "Value ($B)"], '
        '"rows": [["Reliance", 8.5], ["", 4.3], ["Zepto", 0.35]], "value_column": 1}\n```'
    )
    assert chat_module.parse_dataviz(partly) is not None


def test_parse_dataviz_accepts_numeric_identifier_labels():
    """A numeric identifier column (e.g. Year: 2024/2025) is valid identifying
    content for the label side of a table, so such a block must parse — it is
    not an all-empty-label malformed block. Checked both with an explicit
    value_column and on the auto-detect path."""
    # Explicit value_column: the numeric Year labels count as label content.
    explicit = (
        '```dataviz\n{"columns": ["Year", "Revenue"], '
        '"rows": [[2024, 100], [2025, 150]], "value_column": 1}\n```'
    )
    data = chat_module.parse_dataviz(explicit)
    assert data is not None
    assert data["value_column"] == 1
    assert data["rows"] == [[2024, 100], [2025, 150]]

    # Auto-detect: no value_column, so the first numeric column (Year here, since
    # both columns are numeric) is inferred; the block still parses successfully.
    auto = (
        '```dataviz\n{"columns": ["Year", "Revenue"], '
        '"rows": [[2024, 100], [2025, 150]]}\n```'
    )
    data = chat_module.parse_dataviz(auto)
    assert data is not None
    assert data["value_column"] == 0  # _first_numeric_column picks Year
    assert data["rows"] == [[2024, 100], [2025, 150]]


def test_finalize_answer_only_keeps_charts_on_explicit_request():
    """Charts must never appear unless the user explicitly asked for one
    (guards non-deterministic model emission of dataviz blocks)."""
    with_block = (
        "Prose [1].\n\n```dataviz\n"
        '{"columns": ["Company", "Value"], "rows": [["Wakefit", 1.0]], "value_column": 1}\n'
        "```"
    )
    # No chart ask -> the block is stripped, prose kept.
    out = chat_module._finalize_answer(with_block, "top 10 ipo deals in 2025")
    assert "dataviz" not in out
    assert "Prose [1]" in out
    # Explicit table ask -> block kept (and pinned).
    out = chat_module._finalize_answer(with_block, "make a table of top 10 ipo deals")
    assert "dataviz" in out
    assert chat_module.parse_dataviz(out)["view"] == "table"
    # Plain prose, no ask -> untouched.
    assert chat_module._finalize_answer("Just prose [1].", "top deals") == "Just prose [1]."


def test_chart_intent_regex():
    """Only an explicit chart/graph/plot/table request counts as chart intent;
    ranked/numeric questions without a visual ask must stay plain prose."""
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
    # Bare 'in table' is a common-noun phrase, not a view request.
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

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
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

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 2
    assert chat_module._dataviz_nudge("show me a chart of top 5 deals") in calls[1]
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 13
    assert "dataviz" in result.content


def test_answer_with_dataviz_single_call_when_block_present(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model):
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

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        return chat_module.LLMResult(content="Plain answer [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("who invested in Ola Electric?", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "Plain answer [1]."


def test_answer_with_dataviz_no_retry_for_ranked_question_without_chart_ask(monkeypatch):
    """A ranked-list question that does NOT ask for a visual must not nudge a
    dataviz block into the answer (regression: top-N used to auto-chart)."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        return chat_module.LLMResult(content="Top deal is Zepto [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1
    assert "dataviz" not in result.content


def test_effective_chat_k_dynamic():
    """The chat source count scales to the requested 'top N' (floored at TOP_K,
    capped at CHAT_MAX_SOURCES) instead of always being TOP_K."""
    assert chat_module._effective_chat_k("top 10 ipo deals in 2025") == 10
    assert chat_module._effective_chat_k("top ipo deals in 2025") == 10  # bare top -> list default
    assert chat_module._effective_chat_k("who invested in Ola Electric?") == chat_module.config.TOP_K
    assert chat_module._effective_chat_k("top 50 ipo deals") == chat_module.config.CHAT_MAX_SOURCES


def test_dataviz_nudge_row_cap_scales():
    assert "max 10 rows" in chat_module._dataviz_nudge("top 10 ipo deals in 2025")
    assert f"max {chat_module.config.TOP_K} rows" in chat_module._dataviz_nudge("who invested in Ola Electric?")


def test_chat_prompt_instructs_constructing_top_n_lists():
    """A 'top N' request must be answered by extracting and ranking the named
    items from the articles, not refused because no pre-made ranking exists
    (regression: 'top 10 ipo deals in 2025' was refused despite relevant data)."""
    assert "Build the list only from items the articles actually name" in chat_module.CHAT_PROMPT
    assert "Never refuse" in chat_module.CHAT_PROMPT
    assert "just because the articles lack exact values or a pre-made ranking" in chat_module.CHAT_PROMPT
    assert "always beats a refusal" in chat_module.CHAT_PROMPT
    # IPO questions must yield companies that went public, not M&A/stake deals,
    # and a list item with no stated value must still be included.
    assert "are COMPANIES that went public or filed for an IPO" in chat_module.CHAT_PROMPT
    assert "never private funding rounds, stake sales, or M&A" in chat_module.CHAT_PROMPT
    assert "write \"value not stated\"" in chat_module.CHAT_PROMPT
    # The dataviz table must include every listed item; missing values use "".
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
    # Generic chart ask, no specific view.
    assert chat_module._requested_view("show me a chart of top deals") is None
    # Not a chart request at all.
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

    # A table ask pins view to table and leaves the table rendered as-is.
    out = chat_module._apply_requested_view(text, "show me a table of deals")
    block = chat_module.parse_dataviz(out)
    assert block["view"] == "table"

    # A value-less table block (every item's value 'not stated') is accepted for
    # an explicit table ask and rendered as a plain text table (value_column null).
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

    # Generic chart ask leaves the block untouched.
    assert chat_module._apply_requested_view(text, "show me a chart of deals") == text
    # No block -> untouched.
    assert chat_module._apply_requested_view("Just prose [1].", "show me a pie chart") == "Just prose [1]."


def test_prepare_turn_scales_sources_to_requested_top_n(monkeypatch):
    """Chat must retrieve as many sources as the question asks for ('top 10 ...'
    -> top_k=10) and budget the body excerpts so the prompt stays bounded
    (regression: chat always retrieved TOP_K)."""
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
    assert "max 10" in turn.answer  # dataviz cap matches the requested N

    monkeypatch.setattr(main, "retrieve_and_rerank", make_fake(chat_module.config.TOP_K))
    turn = _run(chat_module._prepare_turn("who invested in Ola Electric?", []))
    assert captured["top_k"] == chat_module.config.TOP_K
    assert len(turn.sources) == chat_module.config.TOP_K


def test_prepare_turn_budgets_body_excerpts_across_sources(monkeypatch):
    """With more sources than body-budget / CHAT_BODY_CHAR_LIMIT, each source's
    body excerpt is trimmed so the total prompt stays bounded."""
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
                          summary="s", body="x" * 50000, score=0.9)  # 50K body each
            for i in range(20)
        ]

    monkeypatch.setattr(main, "retrieve_and_rerank", fake_retrieve)
    turn = _run(chat_module._prepare_turn("top 20 ipo deals in 2025", []))
    # 20 sources x 20K excerpt = 400K, the total body budget.
    assert turn.answer.count("x" * 20000) >= 20
    assert "x" * 20001 not in turn.answer  # no source exceeds its share


def test_answer_with_dataviz_keeps_first_answer_when_nudge_fails(monkeypatch):
    """A failed dataviz nudge retry must keep the first answer instead of
    erroring the turn."""
    calls = []

    async def fake_generate(client, prompt, model):
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


def test_json_loads_malformed_returns_empty(caplog):
    """Decode failures on a str/bytes payload still degrade to [] but must be
    logged, so corrupt stored rows are diagnosable instead of silently empty."""
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads("{not json") == []
        assert chat_module.json_loads(None) == []
        assert chat_module.json_loads("") == []
        assert chat_module.json_loads("[]") == []
    assert "chat.json_loads: malformed stored JSON" in caplog.text
    assert "{not json" in caplog.text


def test_json_loads_logs_unexpected_shape(caplog):
    """Corrupt payloads that decode but are not a list-of-objects are logged,
    not silently dropped (#175)."""
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads('{"sources": 1}') == []
        assert chat_module.json_loads('[1, {"a": 1}]') == [{"a": 1}]
    assert "expected a JSON list" in caplog.text
    assert "dropped 1 non-object item(s)" in caplog.text


def test_json_loads_non_str_payload_degrades_to_empty(caplog):
    """A non-str/bytes payload is a caller bug, but still must not break a
    history read: it degrades to [] and is logged at error level with the type,
    the row id and a bounded preview (#175)."""
    with caplog.at_level(logging.ERROR, logger="chat"):
        assert chat_module.json_loads(123, row_id=5) == []
        assert chat_module.json_loads({"a": 1}) == []
    assert "payload must be str/bytes, got int" in caplog.text
    assert "123" in caplog.text
    assert "row_id=5" in caplog.text
    assert "row_id=unknown" in caplog.text
    assert "got dict" in caplog.text


def test_json_loads_logs_row_id_for_shape_problems(caplog):
    """Every degraded path names the row, so the corrupt stored row can be found
    from the logs (#175)."""
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads('{"sources": 1}', row_id=99) == []
        assert chat_module.json_loads("[1]", row_id=99) == []
    assert "expected a JSON list" in caplog.text
    assert "dropped 1 non-object item(s)" in caplog.text
    assert "row_id=99" in caplog.text


def test_row_to_message_passes_row_id_to_json_loads(caplog):
    """The only caller must identify the row it read, otherwise the corruption
    warning cannot be traced back to a stored message (#175)."""
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
    """A huge non-str payload must not be repr'd in full on the error path: the
    render cost is capped by the preview limits, not the payload size (#175)."""
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

    # Small payloads are rendered verbatim, with no truncation or omission marker.
    assert chat_module._log_preview([{"a": 1}]) == "[{'a': 1}]"
    assert chat_module._log_preview("{not json") == "{not json"


class _LazyItemsDict(dict):
    """Mapping that counts how many of its members a consumer actually walks.

    Lazy on purpose: the count is what distinguishes a preview that streams the
    first few members from one that materialises the whole mapping first.
    """

    def __init__(self, source):
        super().__init__(source)
        self.yielded = 0

    def items(self):
        for pair in super().items():
            self.yielded += 1
            yield pair


class _ReprSpy:
    """Value that records whether anything ever rendered it."""

    rendered: ClassVar[list[str]] = []

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        _ReprSpy.rendered.append(self.label)
        return f"<spy {self.label}>"


def test_log_preview_does_not_materialise_a_huge_mapping():
    """A preview must pull only the members it renders: building the whole
    items() list first would cost O(size of the payload) on an error path, which
    is exactly what the preview limits are there to prevent (#175).

    A repr-then-truncate implementation walks every member, so it fails here.
    """
    mapping = _LazyItemsDict({f"key-{i}": i for i in range(500_000)})

    preview = chat_module._log_preview(mapping)

    assert mapping.yielded <= chat_module._JSON_PREVIEW_ITEMS
    assert len(preview) <= chat_module._JSON_LOG_PREVIEW + 100


def test_log_preview_never_renders_dropped_or_over_deep_values():
    """Members past the preview window and containers nested deeper than the
    depth cap must be stubbed out, not rendered: both are unbounded otherwise,
    and only the final clip would hide it (#175).

    A repr-then-truncate implementation renders both spies, so it fails here.
    """
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
    """Nesting deep enough to blow the decoder's stack raises RecursionError,
    which is not a ValueError: it must degrade to [] with the row id instead of
    escaping as a 500 out of a history read (#175)."""
    payload = "[" * 100_000 + "]" * 100_000
    with pytest.raises(RecursionError):  # guard: the payload really hits the limit
        json.loads(payload)

    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads(payload, row_id=12) == []

    assert "chat.json_loads: malformed stored JSON" in caplog.text
    assert "RecursionError" in caplog.text
    assert "row_id=12" in caplog.text


def test_json_loads_empty_payload_is_not_corruption(caplog):
    """An empty payload holds no data, in any str/bytes flavour: it degrades to
    [] silently rather than reporting b''/bytearray(b'') as corrupt (#175)."""
    with caplog.at_level(logging.WARNING, logger="chat"):
        assert chat_module.json_loads("") == []
        assert chat_module.json_loads(b"") == []
        assert chat_module.json_loads(bytearray(b"")) == []
        assert chat_module.json_loads(None) == []

    assert "chat.json_loads" not in caplog.text
def test_answer_with_dataviz_skips_nudge_when_budget_exhausted(monkeypatch):
    """Regression (#177): the nudge retry is a second billed call, so it must
    re-check the daily cap instead of relying on the outer caller's check. An
    exhausted budget skips the retry and keeps the first answer."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]]}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    # $10.00 already spent against a $2.00 cap: the guard's own reserve is refused.
    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=10.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 1  # retry never billed
    assert result.content == "No chart here [1]."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def _async(fn):
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


class _FakeBudgetRedis:
    """Fake Redis implementing the server-side contract of _BUDGET_LUA.

    It is a real (if in-memory) implementation of that script, not a stub that
    returns 0: the counter is integer micro-USD, holds live in a hash with an
    expiry zset, the cap is measured against counter PLUS every live hold, and
    every mode sweeps lapsed holds first. Without that, a test would prove the
    fake rather than the module (#255).

    Every counter write is recorded in `writes` as (mode, amount_micros) so a
    test can assert how many times a turn wrote the day total and for how much,
    without re-pinning the script's argument layout.
    """

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
            # The hold was never added to the counter, so the real cost is
            # recorded whole — deliberately allowed to exceed the estimate.
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
    """Point the REAL app.cost_budget at a _FakeBudgetRedis, leaving
    reserve/settle/release unpatched so the tests exercise the shipped module.

    The cached script handle is dropped so the fake client is the one the
    module registers against. Returns the fake, whose `writes` list shows every
    counter write a turn made."""
    fake = _FakeBudgetRedis(counter_micros)
    monkeypatch.setattr(cost_budget_module, "_BUDGET_SCRIPT", None)
    monkeypatch.setattr(cost_budget_module, "_client", lambda: fake)
    monkeypatch.setattr(cost_budget_module.config, "LLM_DAILY_BUDGET_USD", budget_usd)
    monkeypatch.setattr(cost_budget_module, "_budget_reached_warned", False)
    return fake


def _pin_cost_accounting(monkeypatch, budget_usd, spend_usd):
    """Pin pricing so the turn's cost arithmetic is deterministic ($1 per 1M
    prompt tokens, ₹100 == $1) and route the real budget module at a fake
    counter already holding `spend_usd` of today's spend.

    Returns the _FakeBudgetRedis driving the gate; `spend_usd` is in USD."""
    monkeypatch.setattr(cost_budget_module.config, "INR_PER_USD", 100.0)
    monkeypatch.setattr(cost_budget_module.config, "LLM_PRICE_INPUT_PER_1M", 1.0)
    monkeypatch.setattr(cost_budget_module.config, "LLM_PRICE_OUTPUT_PER_1M", 0.0)
    return _route_real_budget(monkeypatch, budget_usd, round(spend_usd * 1_000_000))


def _pin_budget_disabled(monkeypatch):
    """Pin the budget gate to a no-op that records nothing, for tests whose
    subject is not cost accounting (LLM output, streaming, persistence)."""

    async def reserve(estimate_usd: float = 0.0) -> str:
        return ""

    async def settle(ids, actual_usd: float) -> None:
        return None

    async def release(ids) -> None:
        return None

    monkeypatch.setattr(chat_module, "reserve", reserve)
    monkeypatch.setattr(chat_module, "settle", settle)
    monkeypatch.setattr(chat_module, "release", release)


def test_answer_with_dataviz_skips_nudge_when_turn_spend_exhausts_budget(monkeypatch):
    """Regression (#177, the ORDINARY single-turn case): the first call's cost
    only reaches the daily counter at the end of the turn, so the guard must
    count it. Here the recorded total is still under the cap but this turn's
    first call pushes the day past it, so the retry is skipped and the first
    answer is kept (no error surfaced)."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            # 1M prompt tokens == $1.00 with the pinned pricing.
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=1_000_000, completion_tokens=0)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    # $1.50 recorded, and this turn's first call still holds $1.00: the guard's
    # own reserve is measured against counter PLUS live hold = $2.50 > $2.00, so
    # the retry is refused. The first call's cost reaches the counter only at the
    # turn's settle, so counting the live hold is what closes that window.
    budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
    budget.holds["h-inflight"] = 1_000_000
    budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 1  # retry never billed
    assert result.content == "No chart here [1]."
    assert result.prompt_tokens == 1_000_000


def test_answer_with_dataviz_nudges_when_turn_spend_still_within_budget(monkeypatch):
    """Counterpart to the case above: with headroom left after this turn's
    first call, the guard must NOT block the retry."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=1_000_000, completion_tokens=0)
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
            prompt_tokens=20,
            completion_tokens=8,
        )

    # $0.00 recorded and no live hold: the guard's reserve fits under the cap.
    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))
    assert len(calls) == 2
    assert "dataviz" in result.content


def test_row_to_message_coerces_legacy_fields():
    """_row_to_message must tolerate malformed/legacy rows: bad sources JSON and
    NULL/string token/cost fields fall back to 0 defaults (ERROR PATH — malformed
    stored rows)."""
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
    """Line 436: empty queries and >12-word messages are NOT small talk."""
    assert chat_module._smalltalk_reply("") is None
    assert chat_module._smalltalk_reply("   ") is None
    assert chat_module._smalltalk_reply("hi " * 13) is None  # 13 words > 12
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

    # _valid_value_column: no present cell at all -> False; numeric cells -> True.
    assert chat_module._valid_value_column([["x", ""], ["y", "not stated"]], 1) is False
    assert chat_module._valid_value_column([["x", 1.0]], 1) is True
    assert chat_module._valid_value_column([["x", "abc"]], 1) is False

    # _has_label_content: no label columns -> True; all-empty labels -> False.
    assert chat_module._has_label_content([["x"]], ["A"], 0) is True
    assert chat_module._has_label_content([["", 1.0], ["", 2.0]], ["A", "B"], 1) is False
    assert chat_module._has_label_content([["x", 1.0], ["", 2.0]], ["A", "B"], 1) is True

    # _first_numeric_column: empty rows / empty first row short-circuit to None.
    assert chat_module._first_numeric_column([]) is None
    assert chat_module._first_numeric_column([[]]) is None
    assert chat_module._first_numeric_column([["x", 3.0]]) == 1


def test_as_float_rejects_bool_before_int_coercion():
    """Regression (#176): bool is a subclass of int, so an int-first coercion
    turns True/False into 1.0/0.0. _as_float must reject bools before the
    int/float branch, while genuine numbers (and numeric strings, including a
    legit 0/1 cell) still coerce and non-numeric input still returns None.

    The isinstance(v, bool) guard already existed when #176 was reported, so
    this test locks in behaviour that was already correct: it exists purely to
    fail if the guard is ever removed or reordered below the int/float branch.
    """
    assert chat_module._as_float(True) is None
    assert chat_module._as_float(False) is None
    assert chat_module._as_float("true") is None
    assert chat_module._as_float("false") is None

    # Genuine ints (0 and 1 included), floats and numeric strings still coerce.
    assert chat_module._as_float(0) == 0.0
    assert chat_module._as_float(1) == 1.0
    assert chat_module._as_float(-3) == -3.0
    assert chat_module._as_float(2.5) == 2.5
    assert chat_module._as_float("0") == 0.0
    assert chat_module._as_float("-1.5") == -1.5

    # Non-numeric input keeps returning None.
    assert chat_module._as_float("abc") is None
    assert chat_module._as_float(None) is None
    assert chat_module._as_float([]) is None
    assert chat_module._as_float({}) is None

    # Callers (_valid_value_column / _first_numeric_column, reached from
    # parse_dataviz) must not treat a boolean column as numeric either.
    assert chat_module._valid_value_column([["x", True]], 1) is False
    assert chat_module._valid_value_column([["x", 1.0], ["y", False]], 1) is False
    assert chat_module._first_numeric_column([["Funded", True], ["Not", False]]) is None
    assert chat_module._first_numeric_column([["Flag", True, 5.0], ["Amount", False, 6.0]]) == 2

    # A chart block whose only value column holds JSON true/false is malformed.
    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Deal", "Flag"], "rows": [["A", true], ["B", false]], "view": "bar"}\n```'
    ) is None


def test_parse_dataviz_rejection_paths():
    # line 561: data is not a dict
    assert chat_module.parse_dataviz("```dataviz\n[1, 2, 3]\n```") is None
    assert chat_module.parse_dataviz("```dataviz\n\"just a string\"\n```") is None
    # line 565: columns not a non-empty list of strings
    assert chat_module.parse_dataviz('```dataviz\n{"columns": ["A", 1], "rows": [["x", 1.0]]}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": [], "rows": []}\n```') is None
    assert chat_module.parse_dataviz('```dataviz\n{"columns": "A", "rows": [["x"]]}\n```') is None
    # line 567: rows not a non-empty list of lists
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
    # line 712: no fence
    assert chat_module._parse_dataviz_with_view("no block here", "table") is None
    # lines 715-716: invalid JSON
    assert chat_module._parse_dataviz_with_view("```dataviz\n{not json}\n```", "table") is None
    # line 718: data not a dict
    assert chat_module._parse_dataviz_with_view("```dataviz\n[1, 2]\n```", "table") is None
    # dict data gets the view applied
    out = chat_module._parse_dataviz_with_view(
        '```dataviz\n{"columns": ["A"], "rows": [["x"]]}\n```', "table"
    )
    assert out is not None
    assert out["view"] == "table"


def test_apply_requested_view_keeps_malformed_block():
    """Line 734: a block that fails to re-parse is left verbatim (never dropped)."""
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
    """A ranked-list answer that refuses must be re-asked once with the ranking
    nudge (lines 799-808)."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=10, completion_tokens=5
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    _pin_budget_disabled(monkeypatch)

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 2
    assert chat_module._RANKING_NUDGE in calls[1]
    assert result.content == "Top deal: Zepto [1]."
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 13


def test_answer_ranked_keeps_first_answer_when_nudge_fails(monkeypatch):
    """A failed ranking-nudge retry must keep the first answer (LLMUnavailableError
    guard at lines 803-804)."""
    calls = []

    async def fake_generate(client, prompt, model):
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
    """Regression (#177): the ranking-refusal nudge is a second billed call, so
    it must re-check the daily cap like the dataviz nudge does. An exhausted
    budget skips the retry and keeps the refusal (no error surfaced)."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=10, completion_tokens=5
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    # $10.00 already spent against a $2.00 cap: the guard's own reserve is refused.
    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=10.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1  # retry never billed
    assert result.content == "I cannot generate a ranked list [1]."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_answer_ranked_skips_nudge_when_turn_spend_exhausts_budget(monkeypatch):
    """Regression (#177, the ORDINARY single-turn case) for the ranking nudge:
    the first call's cost only reaches the daily counter at the end of the turn,
    so the guard must count it. The recorded total is still under the cap but
    this turn's first call pushes the day past it, so the refusal is kept."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            # 1M prompt tokens == $1.00 with the pinned pricing.
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=1_000_000, completion_tokens=0
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    # $1.50 recorded, and this turn's first call still holds $1.00: the guard's
    # own reserve is measured against counter PLUS live hold = $2.50 > $2.00, so
    # the ranking retry is refused and the refusal is kept.
    budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
    budget.holds["h-inflight"] = 1_000_000
    budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1  # retry never billed
    assert result.content == "I cannot generate a ranked list [1]."
    assert result.prompt_tokens == 1_000_000


def test_answer_ranked_nudges_when_turn_spend_still_within_budget(monkeypatch):
    """Counterpart to the case above: with headroom left after this turn's first
    call, the ranking-nudge guard must NOT block the retry."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:
            return chat_module.LLMResult(
                content="I cannot generate a ranked list [1].", prompt_tokens=1_000_000, completion_tokens=0
            )
        return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

    # $0.00 recorded and no live hold: the guard's reserve fits under the cap.
    _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=0.0)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 2
    assert chat_module._RANKING_NUDGE in calls[1]
    assert result.content == "Top deal: Zepto [1]."


def test_answer_ranked_single_call_when_not_refusal(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        return chat_module.LLMResult(content="Top deal is Zepto [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT", []))
    assert len(calls) == 1
    assert result.content == "Top deal is Zepto [1]."


def test_prepare_turn_vague_followup_with_year_range(monkeypatch):
    """A vague follow-up that adds a year window ('for 2024') keeps the previous
    turn's topic but pins the new date range (lines 856-859)."""
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
    """line 876: sources below ASK_MIN_SCORE yield a plain 'no articles' answer
    instead of an LLM call."""
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
    """line 879: weak-but-nonempty sources produce the honest fallback answer
    with a note, no LLM call."""
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
    """A query that resolves a category facet (e.g. 'funding news in jun 2025')
    must surface its on-topic matches even when the cross-encoder scores them
    below ASK_MIN_SCORE: the facet filter is the relevance signal, so the gate is
    dropped and the matches are presented normally (no weak disclaimer)."""
    from app import main
    from app.main import SourceArticle

    monkeypatch.setattr(chat_module, "_smalltalk_reply", lambda q: None)
    monkeypatch.setattr(chat_module.config, "ENABLE_BODY_RESCUE", False)
    monkeypatch.setattr(chat_module.config, "ENABLE_WEAK_FALLBACK", True)
    # dealtype resolved -> faceted path
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
    assert turn.needs_llm  # normal answer, not the weak/empty short-circuit
    assert len(turn.sources) == 2
    assert "No sufficiently relevant" not in turn.answer
    assert "closest" not in turn.answer
    assert turn.note is None


def test_prepare_turn_multiple_moderate_sources_not_weak(monkeypatch):
    """Regression (issue #196): a query whose several on-topic sources each
    score only modestly above the inclusion gate (but below TOP_WEAK_THRESHOLD)
    must still be answered, not refused as 'weakly related'. Retrieval with
    several relevant sources is genuinely sufficient."""
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
    assert turn.needs_llm  # answered, not the weak short-circuit
    assert len(turn.sources) == 5
    assert "No sufficiently relevant" not in turn.answer
    assert "closest" not in turn.answer
    assert turn.note is None


def test_run_turn_records_cost_and_finalizes(monkeypatch):
    """_run_turn holds budget before the billed call, calls the LLM, settles the
    hold with the real cost, and finalizes the answer (strips unrequested
    dataviz blocks)."""
    calls = {}
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note="note", needs_llm=True)

    async def fake_answer_ranked(question, prompt, holds):
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
    # The turn wrote the counter exactly once, for the real cost, and left no
    # hold behind: the gate holds before the call and settles after it.
    assert len(budget.writes) == 1
    assert budget.writes[0][1] == round(expected_cost * 1_000_000)
    assert budget.counter == round(expected_cost * 1_000_000)
    assert budget.holds == {}  # every hold discharged
    # No chart intent -> dataviz block stripped by _finalize_answer.
    assert "dataviz" not in answer
    assert "Prose [1]" in answer
    assert sources == [{"id": 1}]
    assert note == "note"
    assert (pt, ct) == (10, 5)


def test_run_turn_releases_hold_when_llm_unavailable(monkeypatch):
    """When the LLM cannot be reached nothing was billed, so the turn's hold is
    RELEASED rather than settled: an unbilled hold that stayed live would block
    later calls against spend that never happened."""
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note=None, needs_llm=True)

    async def boom(question, prompt, holds):
        raise chat_module.LLMUnavailableError()

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "_answer_ranked", boom)

    answer, _sources, _note, pt, ct, cost = _run(chat_module._run_turn("who invested in Ola?", []))

    assert budget.writes == []  # nothing billed -> nothing recorded
    assert budget.counter == 0
    assert budget.holds == {}  # the unbilled hold was released, not left eating cap
    assert (pt, ct, cost) == (0, 0, 0.0)
    assert "couldn't generate" in answer


def test_run_turn_settles_summed_turn_cost_exactly_once(monkeypatch):
    """The end-of-turn settle must be the ONLY counter write for the turn and
    must carry the SUMMED cost of every billed call in it.

    Each nudge takes its own hold, so a future change that ALSO recorded one
    again — a second settle, or settling the summed figure twice — would
    double-count the day and fail here instead of passing silently."""
    generate_calls = []
    budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

    async def fake_generate(client, prompt, model):
        generate_calls.append(prompt)
        # A chart ask whose answer refuses: the dataviz nudge (call 2) and then
        # the ranking nudge (call 3) all fire, so the turn is billed three
        # times and only their SUM may reach the counter.
        prompt_tokens = {1: 100_000, 2: 200_000, 3: 300_000}[len(generate_calls)]
        return chat_module.LLMResult(
            content="I cannot generate a ranked list [1].", prompt_tokens=prompt_tokens, completion_tokens=0
        )

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    _run(chat_module._run_turn("show me a chart of top 10 ipo deals in 2025", []))

    assert len(generate_calls) == 3
    # The COUNTER is written exactly once for the whole turn, carrying the
    # SUM of all three calls. The three reserve calls touch only the holds
    # hash, so they are not counter writes.
    assert [mode for mode, _ in budget.writes] == ["settle"]
    assert budget.writes[0][1] == 600_000
    assert budget.counter == 600_000
    assert budget.holds == {}  # gate hold + both nudge holds all discharged


def test_api_require_store_uninitialized_503(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        chat_module.store = None
        h = _auth_headers(auth_store)
        assert client.get("/api/chat/sessions", headers=h).status_code == 503
        assert client.post("/api/chat/sessions", headers=h).status_code == 503
    finally:
        chat_module.store = chat_store
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_too_long_400(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]
        long_msg = "x" * (chat_module.MAX_CONTENT_LEN + 1)
        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": long_msg})
        assert r.status_code == 400
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_budget_exceeded_429(tmp_path, monkeypatch):
    """send_message fails closed with 429 when the daily LLM budget is hit
    (ERROR PATH — daily budget)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def boom(question, history):
            raise chat_module.BudgetExceeded()

        monkeypatch.setattr(chat_module, "_run_turn", boom)
        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "top deals"})
        assert r.status_code == 429
        assert "Daily AI budget reached" in r.text
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_send_message_llm_unavailable_503(tmp_path, monkeypatch):
    """send_message returns 503 when the LLM cannot be reached after retries
    (ERROR PATH — LLM retry exhaustion)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def boom(question, history):
            raise chat_module.LLMUnavailableError()

        monkeypatch.setattr(chat_module, "_run_turn", boom)
        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "top deals"})
        assert r.status_code == 503
        assert "LLM temporarily unavailable" in r.text
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def _stream_body(client, headers, sid, content):
    url = f"/api/chat/sessions/{sid}/messages/stream"
    with client.stream("POST", url, headers=headers, json={"content": content}) as r:
        assert r.status_code == 200
        return "".join(r.iter_text())


def _fake_prepare_llm():
    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="prompt-text", sources=[], note=None, needs_llm=True)

    return fake_prepare


def test_api_stream_dataviz_nudge_replaces_answer(tmp_path, monkeypatch):
    """Streaming: an explicit chart ask without a block re-asks once and swaps in
    the nudge answer, summing token usage (lines 1173-1176)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "no block here [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        async def fake_generate(client, prompt, model):
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
    """Streaming: a failed dataviz nudge retry keeps the streamed answer instead
    of erroring the turn (lines 1171-1172)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
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
        assert '"prompt_tokens":10' in body  # unchanged
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_dataviz_nudge_skipped_when_budget_exhausted(tmp_path, monkeypatch):
    """Streaming regression (#177): a dataviz nudge retry is a second billed
    call that re-checks the cap; once the budget is gone the retry is skipped
    and the already-streamed answer is served."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "streamed answer without a block [1]."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        nudge_calls = []

        async def fake_generate(client, prompt, model):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(
                content='Prose.\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
                prompt_tokens=20,
                completion_tokens=8,
            )

        # The stream's own hold is granted; the nudge's second hold is refused,
        # so the retry never starts.
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
        assert nudge_calls == []  # retry never billed
        assert "streamed answer without a block [1]." in body
        assert '"prompt_tokens":10' in body  # unchanged
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_dataviz_nudge_skipped_when_turn_spend_exhausts_budget(tmp_path, monkeypatch):
    """Streaming regression (#177, the ORDINARY single-turn case): the streamed
    call's cost is recorded only at the end of the turn, so the guard must count
    it. The recorded total is under the cap but the stream alone pushes the day
    past it, so the retry is skipped and the streamed answer is served (no
    error event)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "streamed answer without a block [1]."
            if usage_holder is not None:
                # 1M prompt tokens == $1.00 with the pinned pricing.
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=1_000_000, completion_tokens=0))

        nudge_calls = []

        async def fake_generate(client, prompt, model):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(
                content='Prose.\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
                prompt_tokens=20,
                completion_tokens=8,
            )

        async def noop(*args, **kwargs):
            return None

        # $1.50 recorded + $1.00 streamed = $2.50 > $2.00 cap, while the
        # outer pre-turn check still passes ($1.50 < $2.00).
        budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
        # A call already in flight in THIS turn is holding $0.45. The stream's
        # own gate still fits (1.50 + 0.45 + 0.05 = $2.00), but the nudge's
        # second reserve does not (1.50 + 0.45 + 0.05 + 0.05 = $2.05 > cap), so
        # the retry is refused while the already-streamed answer is served.
        budget.holds["h-inflight"] = 450_000
        budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

        body = _stream_body(client, h, sid, "show me a chart of top deals")
        assert nudge_calls == []  # retry never billed
        assert "streamed answer without a block [1]." in body
        assert '"prompt_tokens":1000000' in body  # unchanged
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_ranking_nudge_replaces_answer(tmp_path, monkeypatch):
    """Streaming: a ranked-list refusal is re-asked once with the ranking nudge
    (lines 1185-1188)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        async def fake_generate(client, prompt, model):
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
    """Streaming: a failed ranking-nudge retry keeps the streamed refusal
    (lines 1183-1184)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
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
    """Streaming regression (#177): the ranking nudge is a second billed call,
    so it re-checks the cap; once the budget is gone the retry is skipped and
    the already-streamed refusal is served (no error event)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5))

        nudge_calls = []

        async def fake_generate(client, prompt, model):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

        # The stream's own hold is granted; the ranking nudge's second hold is
        # refused, so the retry never starts.
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
        assert nudge_calls == []  # retry never billed
        assert "I cannot generate a ranked list because amounts are missing." in body
        assert '"prompt_tokens":10' in body  # unchanged
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_ranking_nudge_skipped_when_turn_spend_exhausts_budget(tmp_path, monkeypatch):
    """Streaming regression (#177, the ORDINARY single-turn case) for the ranking
    nudge: the stream's cost is recorded only at the end of the turn, so the
    guard must count it. The recorded total is under the cap but the stream alone
    pushes the day past it, so the retry is skipped and the streamed refusal is
    served (no error event)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                # 1M prompt tokens == $1.00 with the pinned pricing.
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=1_000_000, completion_tokens=0))

        nudge_calls = []

        async def fake_generate(client, prompt, model):
            nudge_calls.append(prompt)
            return chat_module.LLMResult(content="Top deal: Zepto [1].", prompt_tokens=20, completion_tokens=8)

        async def noop(*args, **kwargs):
            return None

        # $1.50 recorded + $1.00 streamed = $2.50 > $2.00 cap, while the
        # outer pre-turn check still passes ($1.50 < $2.00).
        budget = _pin_cost_accounting(monkeypatch, budget_usd=2.0, spend_usd=1.5)
        # As above: the stream's gate fits, the ranking nudge's second reserve
        # does not, so the refusal is served instead of an error event.
        budget.holds["h-inflight"] = 450_000
        budget.expiry["h-inflight"] = cost_budget_module._now_ts() + 900
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")
        assert nudge_calls == []  # retry never billed
        assert "I cannot generate a ranked list because amounts are missing." in body
        assert '"prompt_tokens":1000000' in body  # unchanged
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_records_summed_turn_cost_exactly_once(tmp_path, monkeypatch):
    """Streaming counterpart of the non-streaming case, and the one test that
    drives the REAL cost_budget module through chat.py's whole turn path:
    real reserve -> billed stream -> real settle, against a contract-level fake
    store. It is what catches a wrong-arity or wrong-semantics call in chat.py
    (settling the wrong amount, or reserving and never settling).

    The stream plus the dataviz and ranking nudges are ONE turn: the day counter
    is written exactly once, carrying all three calls' summed cost."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        nudge_calls = []

        async def fake_generate(client, prompt, model):
            nudge_calls.append(prompt)
            # 200K for the dataviz nudge, 300K for the ranking nudge.
            prompt_tokens = 200_000 if len(nudge_calls) == 1 else 300_000
            return chat_module.LLMResult(
                content="I cannot be generated [1].", prompt_tokens=prompt_tokens, completion_tokens=0
            )

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)
        monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

        body = _stream_body(client, h, sid, "show me a chart of top 10 ipo deals in 2025")

        assert len(nudge_calls) == 2  # both nudges billed
        # The COUNTER is written once for the turn; the three reserves only
        # touched the holds hash, so they are not counter writes.
        assert [mode for mode, _ in budget.writes] == ["settle"]
        # 100K streamed + 200K + 300K = 600K tokens @ $1/1M == 600_000 micro-USD.
        assert budget.writes[0][1] == 600_000
        assert budget.counter == 600_000
        assert budget.holds == {}  # stream hold + both nudge holds discharged
        assert '"prompt_tokens":600000' in body
        assert "event: done" in body
        assert "event: error" not in body
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_llm_unavailable_sse(tmp_path, monkeypatch):
    """Streaming: an LLMUnavailableError mid-stream yields an error SSE event
    (line 1213) instead of a done event (ERROR PATH — LLM retry exhaustion)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

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
    """Streaming: an unexpected exception yields a generic error SSE event
    (lines 1216-1218), never a 500 to the client."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

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
    """retention_loop purges expired conversations on each tick and swallows
    per-tick errors (lines 1225-1233)."""
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
        assert len(attempts) == 2  # first raised, second succeeded
        assert sleeps == [chat_module.config.CHAT_PURGE_INTERVAL_SECONDS] * 2
        assert _run(store.get_session(a.id, USER_A)) is None  # purged on 2nd tick
    finally:
        chat_module.store = None
        _run(store.close())


# ---------------------------------------------------------------------------
# Issue #255: the abort/persist rule, char-bounded history, and the shared
# dataviz fence grammar.
# ---------------------------------------------------------------------------


def _disconnect_after(monkeypatch, n_deltas):
    """Make the client disappear once `n_deltas` delta events have been emitted.

    `streamed` in chat.py counts exactly the deltas it has yielded, so a client
    that drops after N deltas is the "deltas already sent" case; dropping before
    any is the clean-rollback case. Returns a counter of how many deltas the
    turn actually emitted, so a test can assert which side of the rule it hit.
    """
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
    """The rule: once any delta has been streamed, the turn is PERSISTED, never
    deleted.

    The old code erased the user message whenever the client disconnected, even
    after the client had already rendered the answer, so the server's history
    and the client's screen disagreed (#255). Here the client drops after the
    first delta, and the store must still hold the user message plus an
    assistant message flagged aborted."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            for piece in ["Partial ", "answer ", "rest."]:
                yield piece
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        _disconnect_after(monkeypatch, 1)
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = _run(chat_store.messages(sid, USER_A)) if False else client.get(
            f"/api/chat/sessions/{sid}", headers=h
        ).json()["messages"]
        roles = [m["role"] for m in msgs]
        # The user message survives: the client already saw its answer.
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
    """The other half of the same rule: with nothing streamed yet there is
    nothing the client saw, so the dangling user message is rolled back and no
    assistant row is written."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "never seen by the client"
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        # Disconnects immediately: the stream is never entered.
        _disconnect_after(monkeypatch, 0)
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert msgs == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_mid_failure_with_gone_client_persists_aborted(tmp_path, monkeypatch):
    """A mid-stream failure AFTER deltas were sent must persist the truncated
    turn even when the client is already gone — the bytes were on the wire, so
    deleting the turn would recreate the history divergence (#255). The row is
    flagged aborted."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "half an answer"
            raise RuntimeError("provider dropped the connection")

        # Gone as soon as the first delta is out, which is also when the
        # failure hits.
        _disconnect_after(monkeypatch, 1)
        _pin_budget_disabled(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert "half an answer" in msgs[1]["content"]
        assert "[answer truncated]" in msgs[1]["content"]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_send_message_disconnect_rolls_back_without_assistant(tmp_path, monkeypatch):
    """The non-stream path never checked the client at all. A JSON client
    receives nothing until the turn is persisted, so a disconnect is a clean
    rollback: no assistant message, and the user message is removed."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_run_turn(question, history):
            return "An answer the client never receives.", [], None, 10, 5, 0.0001

        # Patched onto the class, so the function is bound and receives `self`.
        async def always_gone(self):
            return True

        monkeypatch.setattr(chat_module, "_run_turn", fake_run_turn)
        monkeypatch.setattr(chat_module.Request, "is_disconnected", always_gone)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "top deals"})
        # Not a success, and certainly not a 200 TurnOut.
        assert r.status_code >= 400
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_send_message_budget_unavailable_is_503_not_empty_answer(tmp_path, monkeypatch):
    """A BudgetUnavailable at the FIRST gate must fail closed with an explicit
    503. Treating an unreadable counter as "budget fine" is exactly the
    fail-open hole #255 removes, and returning the retrieval fallback answer
    instead would hide it behind a plausible 200."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def boom(question, history):
            raise chat_module.BudgetUnavailable("redis down")

        monkeypatch.setattr(chat_module, "_run_turn", boom)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "top deals"})
        assert r.status_code == 503
        assert "budget" in json.dumps(r.json()).lower()
        # The dangling user message is rolled back, not left behind.
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_api_stream_budget_unavailable_is_error_event(tmp_path, monkeypatch):
    """Same rule on the SSE path: an unreadable counter at the first gate is an
    explicit error event, never a silently served answer."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

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
    """The OPTIONAL nudge path may degrade to the answer already produced, but it
    must admit no further spend: an unreachable counter skips the billed call
    and is logged."""
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        return chat_module.LLMResult(content="No chart here [1].", prompt_tokens=10, completion_tokens=5)

    async def unavailable(estimate_usd=0.0):
        raise chat_module.BudgetUnavailable("redis down")

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())
    monkeypatch.setattr(chat_module, "reserve", unavailable)

    with caplog.at_level(logging.WARNING, logger="chat"):
        result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT", []))

    assert len(calls) == 1  # the billed retry never started
    assert result.content == "No chart here [1]."
    assert "unavailable" in caplog.text.lower()


def _msg(role, content):
    return chat_module.MessageOut(id=1, role=role, content=content, created_at=0.0)


def test_trim_history_drops_oldest_until_within_budget():
    """Oldest-first, keeping the newest messages that fit, never splitting one."""
    history = [_msg("user", "a" * 40), _msg("assistant", "b" * 40), _msg("user", "c" * 40)]

    kept = chat_module._trim_history(history, 80)
    assert [m.content for m in kept] == ["b" * 40, "c" * 40]

    kept = chat_module._trim_history(history, 40)
    assert [m.content for m in kept] == ["c" * 40]

    # Fits entirely -> untouched.
    assert chat_module._trim_history(history, 10_000) == history


def test_trim_history_keeps_oversized_newest_message():
    """A single message larger than the cap is kept on its own: dropping it
    would leave the prompt with no context at all."""
    history = [_msg("user", "a" * 100), _msg("assistant", "b" * 5000)]
    kept = chat_module._trim_history(history, 100)
    assert [m.content for m in kept] == ["b" * 5000]


def test_trim_history_zero_disables_the_cap():
    history = [_msg("user", "a" * 10_000), _msg("assistant", "b" * 10_000)]
    assert chat_module._trim_history(history, 0) == history
    assert chat_module._trim_history(history, -1) == history


def test_start_turn_applies_the_char_cap(tmp_path, monkeypatch):
    """_start_turn bounds history by BOTH knobs: turns in the query, characters
    in what actually reaches the prompt."""
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        for i in range(6):
            _run(store.append_message(sid, USER_A, "user" if i % 2 == 0 else "assistant", "x" * 5000))

        monkeypatch.setattr(chat_module.config, "CHAT_MAX_HISTORY_TURNS", 10)
        monkeypatch.setattr(chat_module.config, "CHAT_MAX_HISTORY_CHARS", 12_000)

        _user_msg, history = _run(chat_module._start_turn(store, sid, USER_A, "next question"))

        # 6 x 5000 = 30000 chars of history, trimmed to at most 12000 + the
        # newly appended question. The cap really bound the prompt.
        assert sum(len(m.content) for m in history) <= 12_000 + len("next question")
        assert len(history) < 7  # some were dropped
    finally:
        _run(store.close())


def test_dataviz_unclosed_fence_never_leaks_json():
    """A fence the model never finished must not render its raw JSON. Both the
    chart-intent and the plain branch truncate from the marker to the end."""
    unclosed = 'Here is the data:\n\n```dataviz\n{"columns": ["A"], "rows": [[1, 2'
    for question in ("show me a chart of top deals", "who invested in Ola Electric?"):
        out = chat_module._finalize_answer(unclosed, question)
        assert '{"' not in out
        assert "```dataviz" not in out
        assert "Here is the data:" in out  # the prose survives

    assert '{"' not in chat_module._sanitize_dataviz(unclosed)
    assert '{"' not in chat_module._append_nudge("Answer [1].", unclosed)


def test_dataviz_newline_optional_fence_is_handled():
    """A fence written without a newline after the tag is the same grammar the
    frontend uses. The old backend pattern required a newline, so such a block
    was left in the stored answer while the UI had already stripped it (#255)."""
    body = '{"columns": ["Deal", "Value"], "rows": [["Zepto", 1.0]], "value_column": 1}'
    no_newline = f"Prose [1].\n\n```dataviz{body}```"
    assert chat_module.parse_dataviz(f"```dataviz{body}```") is not None
    out = chat_module._finalize_answer(no_newline, "show me a table of top deals")
    assert "```dataviz" in out  # recognised and preserved, not left as raw text
    assert "Prose [1]." in out


def test_dataviz_valid_fence_survives_the_unclosed_rule():
    """The unclosed-fence pass must not truncate at a marker inside a VALID
    closed fence, or every chart would lose its block."""
    valid = (
        'Prose [1].\n\n```dataviz\n'
        '{"columns": ["Deal", "Value"], "rows": [["Zepto", 1.0]], "value_column": 1}\n```'
    )
    assert chat_module._sanitize_dataviz(valid) == valid
    # _finalize_answer may legitimately REWRITE a valid block (it pins the view
    # the question asked for), so the invariant under test is that the block
    # SURVIVED the unclosed pass -- not that the text is byte-identical.
    finalized = chat_module._finalize_answer(valid, "show me a table of top deals")
    assert finalized.startswith("Prose [1].")
    assert finalized.rstrip().endswith("```")
    assert chat_module.parse_dataviz(finalized) is not None


def test_fence_src_matches_python_fence_pattern():
    """The frontend and the backend must use the SAME grammar string, so a fence
    the UI renders is the fence the server finalized (#255)."""
    tsx = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "app" / "chat" / "DataViz.tsx"
    raw = re.search(r"const FENCE_SRC = '([^']*)'", tsx.read_text()).group(1)
    # The literal is single-quoted TS, where \S is written \\S; unescape it the
    # way JS would before comparing to the Python source string.
    js_runtime = json.loads('"' + raw + '"')
    assert js_runtime == chat_module.DATAVIZ_FENCE_PATTERN


_FENCE_FIXTURES = {
    "closed_with_newline": 'Prose.\n```dataviz\n{"columns": ["A"], "rows": [["x", 1]], "value_column": 1}\n```',
    "closed_without_newline": 'Prose.\n```dataviz{"columns": ["A"], "rows": [["x", 1]], "value_column": 1}```',
    "unclosed": 'Prose.\n```dataviz\n{"columns": ["A"], "rows": [["x", 1',
    "language_tag": 'Prose.\n```dataviz  \n{"columns": ["A"], "rows": [["x", 1]], "value_column": 1}\n```',
}


def _node_strip_open_fence(fixtures):
    """Run the TSX's own regex + truncation rule under node.

    `stripOpenFence` and `FENCE_SRC` are copied verbatim out of DataViz.tsx so
    this exercises the shipped frontend rule, not a re-typed approximation."""
    tsx = (pathlib.Path(__file__).resolve().parents[2] / "frontend" / "app" / "chat" / "DataViz.tsx").read_text()
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
    """Cross-language parity, checked BEHAVIOURALLY.

    The repo has no JS test runner, but `node -e` needs no node_modules, so the
    TSX's own FENCE_SRC and stripOpenFence are executed under node and their
    match spans, captures and resulting text are compared with Python's.

    Comparing rendered text rather than just match/null matters: the `unclosed`
    fixture matches NOTHING on both sides, so a match-only comparison would
    assert nothing about the acceptance item that raw JSON must not leak."""
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

    # The unclosed case: neither side renders the JSON, both keep the prose.
    unclosed = _FENCE_FIXTURES["unclosed"]
    js_rendered = js["unclosed"]["rendered"]
    py_rendered = chat_module._finalize_answer(unclosed, "show me a chart of top deals")
    assert '{"' not in js_rendered
    assert '{"' not in py_rendered
    assert js_rendered.strip().startswith("Prose.")
    assert py_rendered.strip().startswith("Prose.")
    assert js_rendered.split("Prose.")[1] == py_rendered.split("Prose.")[1]


def test_connect_migrates_legacy_db_without_aborted_column(tmp_path):
    """A DB created before the abort flag existed is migrated by connect(), so
    the abort rule works on conversations that predate it."""
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
        # Reconnecting is idempotent: the column already exists.
        _run(store.close())
        _run(store.connect())
        cols2 = {c["name"] for c in _run(store._db.execute_fetchall("PRAGMA table_info(messages)"))}
        assert "aborted" in cols2
    finally:
        _run(store.close())


def test_aborted_flag_round_trips_through_the_store(tmp_path):
    """aborted is PERSISTED, not just carried in memory: a read back from SQLite
    reports it, and an ordinary message defaults to False."""
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _run(store.append_message(sid, USER_A, "user", "q"))
        aborted_msg = _run(
            store.append_message(sid, USER_A, "assistant", "partial [answer truncated]", aborted=True)
        )
        assert aborted_msg.aborted is True

        msgs = _run(store.messages(sid, USER_A))
        assert [m.aborted for m in msgs] == [False, True]

        recent = _run(store.recent_turns(sid, USER_A, 5))
        assert [m.aborted for m in recent] == [False, True]
    finally:
        _run(store.close())


def test_stream_abort_in_gate_window_charges_the_hold(tmp_path, monkeypatch):
    """A disconnect between the gate reserve and the first delta must CHARGE
    the gate hold, not release it (#255).

    The loop's disconnect check runs before `streamed = True`, so the rollback
    branch is taken and the user message is deleted: nothing reached the
    client, so that is right. What was wrong was the MONEY. The provider has
    already billed the call by the time it emits a piece, and releasing the
    hold is the one option that guarantees free spend -- letting a hold lapse
    is precisely the mechanism that CHARGES a crashed call, so the alternative
    to releasing is never "no cost".

    This reverses a previously test-pinned behaviour that asserted the hold was
    released. Its reasoning weighed only that a live hold would sit in the
    store for the full COST_RESERVATION_TTL_SECONDS, billing the day against "a
    call whose cost was never recorded" -- true, and precisely why the call is
    not free. The hold is settled at the estimate it was taken at, which is the
    same figure the sweeper would have charged had it been left to lapse.

    stream_answer yields one piece, then the client is reported gone at the
    in-loop check — which runs BEFORE `streamed = True`."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        entered = {"v": False}

        async def fake_stream(client, prompt, model, usage_holder=None):
            # The LLM call has now happened and been billed; the loop's
            # disconnect check runs immediately after, with `streamed` still
            # False because the piece has not been yielded yet.
            entered["v"] = True
            yield "one piece nobody will see"

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

        # Connected for every check up to and including the budget gate, gone
        # from the first check that follows the billed call.
        async def is_disconnected(self):
            return entered["v"]

        # Pinned so the assertion below reads as "charged at the estimate this
        # call reserved" rather than tracking an unrelated config default.
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

        monkeypatch.setattr(chat_module.Request, "is_disconnected", is_disconnected)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        # The turn really did reach the billed call, so the gate hold existed.
        assert entered["v"] is True
        # It aborted on the disconnect rule, NOT via the catch-all error
        # handler: an error would emit an 'error' event instead.
        assert "event: error" not in body
        # Rollback branch: nothing reached the client, so the user message is gone.
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"] == []
        # ...and the gate hold is NOT refunded: the call was made and billed, so
        # the money is settled, not handed back.
        assert budget.holds == {}
        assert budget.writes == [("settle", 20_000)]  # the $0.02 estimate, charged once
        assert budget.counter == 20_000
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_mid_failure_after_deltas_charges_the_hold(tmp_path, monkeypatch):
    """A mid-stream failure AFTER deltas were sent is a billed call, not a free
    one (#255).

    `stream_answer` only reports usage once the whole response has arrived, so
    here usage_holder is empty -- but the provider generated and billed the
    tokens that were already on the wire. The turn used to release the gate
    hold on the strength of that empty usage_holder, which is a different
    question from the one that matters. The turn is still persisted with its
    truncation marker (the ONE rule); it is simply also charged."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "half an answer"
            raise RuntimeError("provider dropped the connection")

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        _stream_body(client, h, sid, "Who invested in fintech?")

        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        # The ONE rule still holds: deltas were on the wire, so the turn persists.
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "half an answer" in msgs[1]["content"]
        assert "[answer truncated]" in msgs[1]["content"]
        # ...and the billed call is charged at the estimate it held, not refunded.
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_stream_failure_before_any_delta_releases_the_hold(tmp_path, monkeypatch):
    """The other side of the gate-window rule (#255): a turn that made NO
    billed call releases its hold, so the fix above cannot charge for calls
    that never happened.

    The LLM fails before emitting anything, so the provider billed nothing;
    the hold is dropped rather than settled, and the day total is untouched."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            raise chat_module.LLMUnavailableError()
            yield  # pragma: no cover -- an async generator that never runs

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert "LLM temporarily unavailable" in body
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"] == []
        # Nothing was billed, so nothing is charged: the hold is released, and
        # it is released rather than left eating budget.
        assert budget.holds == {}
        assert budget.writes == []
        assert budget.counter == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_settle_failure_after_a_delivered_stream_does_not_rewrite_it(tmp_path, monkeypatch):
    """A Redis blip while recording the cost of a turn whose answer is ALREADY
    on the wire must not rewrite that answer (#255).

    The settle is the last thing the completed path does, after the final
    disconnect check, so a BudgetUnavailable here used to reach the handler,
    persist the turn through fail_turn() as `<answer>\n\n[answer truncated]`
    with aborted=True, and emit `error` instead of `done` -- while the client
    held the complete answer. That is the exact client/server divergence this
    issue removed, reintroduced through the accounting path, and it is also a
    regression: the pre-fix cost recording was best-effort and never raised.

    Failing closed here prevents no spend either -- the hold stays live and the
    sweep charges it -- so it could only destroy a delivered answer."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
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

        # The turn completed normally: the failure was in the accounting, not
        # in the answer the client already received.
        assert "event: done" in body
        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["content"] == "The complete answer."
        assert "[answer truncated]" not in msgs[1]["content"]
        assert msgs[1]["aborted"] is False
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_settle_failure_after_a_billed_call_keeps_the_answer(tmp_path, monkeypatch):
    """The same defect on the non-stream path: a 503 plus deletion of the user
    message for an answer the LLM had already produced and been billed for."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_answer_ranked(question, prompt, holds):
            return chat_module.LLMResult(content="A billed answer [1].", prompt_tokens=50, completion_tokens=10)

        async def dead_settle(ids, actual_usd):
            raise chat_module.BudgetUnavailable("redis went down mid-turn")

        _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module, "settle", dead_settle)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "top deals"})

        assert r.status_code == 200
        assert r.json()["assistant"]["content"] == "A billed answer [1]."
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


class _DeadBudgetRedis:
    """A spend counter that cannot be reached at all.

    Every attempt to run a script on it raises, so a test can assert that a
    turn never consults the store -- which is what the documented cap opt-out
    (`LLM_DAILY_BUDGET_USD <= 0`, "a deliberate opt-out for deployments that
    meter spend elsewhere") has to mean if it is to mean anything (#255)."""

    def __init__(self):
        self.touched = 0

    def register_script(self, lua):
        self.touched += 1
        raise ConnectionError("redis is down")


def _pin_budget_disabled_with_dead_store(monkeypatch):
    """The cap switched off AND its counter unreachable -- the exact
    combination that used to 503 a chat for a deployment that opted out."""
    dead = _DeadBudgetRedis()
    monkeypatch.setattr(cost_budget_module, "_BUDGET_SCRIPT", None)
    monkeypatch.setattr(cost_budget_module, "_client", lambda: dead)
    monkeypatch.setattr(cost_budget_module.config, "LLM_DAILY_BUDGET_USD", 0.0)
    return dead


def test_disabled_cap_never_consults_the_store(tmp_path, monkeypatch):
    """With the cap switched off, a dead spend counter must not affect chat at
    all (#255). `reserve()` returns "" without touching the store in that mode,
    but the turn's settle used to run anyway -- and with an empty hold list and
    a non-zero amount it deliberately does not return early, so a Redis outage
    503'd the turn and deleted the user message for a deployment that never
    intended to consult a counter, AFTER the LLM call had been made and
    billed."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_answer_ranked(question, prompt, holds):
            return chat_module.LLMResult(content="An unmetered answer [1].", prompt_tokens=50, completion_tokens=10)

        dead = _pin_budget_disabled_with_dead_store(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "top deals"})

        assert r.status_code == 200
        assert r.json()["assistant"]["content"] == "An unmetered answer [1]."
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        # Opting out means opting out: not one command reached the counter.
        assert dead.touched == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_disabled_cap_never_consults_the_store_on_the_stream_path(tmp_path, monkeypatch):
    """The same opt-out on the SSE path: `finish_holds` discharged a
    `charged_usd > 0` turn against an empty hold list, so a dead counter turned
    a completed answer into an `error` event (#255)."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "A streamed, unmetered answer."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=50, completion_tokens=10))

        dead = _pin_budget_disabled_with_dead_store(monkeypatch)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        assert "event: done" in body
        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert msgs[1]["content"] == "A streamed, unmetered answer."
        assert dead.touched == 0
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_disconnect_after_a_completed_stream_is_still_charged(tmp_path, monkeypatch):
    """A disconnect at a POST-stream check must be charged for the finished
    call (#255).

    Both nudge gates re-check the client after the stream has completed and its
    usage is known, so a client that drops there produced a fully billed turn.
    Those two checks passed tokens but no cost, which sent the abort down the
    persist branch with cost 0.0: the turn was stored as though it were free
    and its hold RELEASED, so the money was handed back instead of recorded."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_prepare(question, history):
            return chat_module.PreparedTurn(answer="PROMPT", sources=[], note=None, needs_llm=True)

        async def fake_stream(client, prompt, model, usage_holder=None):
            # No dataviz block, so the chart-intent nudge gate re-checks the
            # client after the stream has finished.
            yield "No chart here [1]."
            if usage_holder is not None:
                # 100k prompt tokens at the pinned $1 / 1M is exactly $0.10.
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        # Gone as soon as the single delta is out: connected through the
        # stream, disconnected at the post-stream check that follows it.
        _disconnect_after(monkeypatch, 1)
        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "show me a chart of top deals")

        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        # Deltas reached the client, so the turn is persisted and flagged.
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert "No chart here" in msgs[1]["content"]
        # The finished call is billed, exactly once, and not released.
        assert msgs[1]["cost"] == pytest.approx(0.1)
        assert budget.writes == [("settle", 100_000)]
        assert budget.counter == 100_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_disconnect_at_the_ranking_nudge_check_is_still_charged(tmp_path, monkeypatch):
    """The SECOND post-stream disconnect check, which is a different call site
    from the dataviz one and needs its own coverage (#255).

    A ranked-list question whose streamed answer is a refusal reaches the
    ranking nudge gate after the stream has completed. A client that drops
    there produced a fully billed answer, and it must be charged for it rather
    than stored at cost 0.0 with its hold released."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            # A refusal, so the RANKING nudge gate is the post-stream check that
            # runs -- the dataviz gate is skipped because there is no chart ask.
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=100_000, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        # Gone as soon as the single delta is out: disconnected at the
        # post-stream check that follows it.
        _disconnect_after(monkeypatch, 1)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")

        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert msgs[1]["cost"] == pytest.approx(0.1)
        assert budget.writes == [("settle", 100_000)]
        assert budget.counter == 100_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_fail_turn_after_deltas_settles_the_actual_cost(tmp_path, monkeypatch):
    """A failure raised OUTSIDE the stream loop must still CHARGE the turn
    (#255) -- this is the `fail_turn` sibling of the mid-stream-failure path.

    A ranked-list question whose answer is a refusal reaches the ranking nudge
    gate after the stream completed; a nudge that raises a non-LLM error lands
    in `fail_turn` with `streamed` True. Passing a flat 0.0 there would release
    the gate hold for an answer already on the wire and already billed.

    `stream_answer` fills usage_holder whenever a stream completes, so this
    branch always has the real cost; the assertion pins that it is charged and
    not released. (The `else` arm on that line is defence in depth for a
    backend that reports no usage at all, and is deliberately not claimed as
    covered here -- it cannot be reached through `stream_answer`.)"""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            # A ranking refusal, so the turn reaches the ranking nudge gate
            # after the stream; the nudge below then raises, which lands the
            # turn in fail_turn with its answer already delivered.
            yield "I cannot generate a ranked list because amounts are missing."
            if usage_holder is not None:
                # Exactly what stream_answer does when a provider sends no
                # usage chunk: a TRUTHY LLMResult carrying ZERO tokens. Its
                # cost is 0.0, so anything testing `usage` for truthiness
                # would settle 0.0 -- and finish_holds(0.0) RELEASES the hold,
                # making a delivered, billed call free.
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
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        # Deltas reached the client, so the ONE rule persists the turn.
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["aborted"] is True
        assert "I cannot generate a ranked list" in msgs[1]["content"]
        # ...and the billed call is CHARGED at the estimate it held, because a
        # zero-token usage report means the cost is unknown, not zero.
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_completed_turn_with_no_usage_report_is_still_charged(tmp_path, monkeypatch):
    """The SUCCESS path must charge too, and this is where the trap hides (#255).

    `stream_answer` fills usage_holder whenever a stream completes, but a
    provider that sends no usage chunk produces an LLMResult with zero tokens.
    `LLMResult.cost()` is then 0.0, so the turn's cost computed to zero --
    and `finish_holds(0.0)` RELEASES the hold. For a provider that never
    reports usage that makes the cap silently inert: every turn is free, and
    the answer was delivered and billed the whole time.

    A zero is not evidence that nothing was spent, it is evidence that the cost
    is unknown, so the estimate the gate held is charged instead. The stored
    message cost is the same figure, so the budget and the reported cost
    cannot disagree."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "A fully delivered answer [1]."
            if usage_holder is not None:
                # Exactly what stream_answer does with no usage chunk: a truthy
                # result carrying zero tokens.
                usage_holder.append(chat_module.LLMResult(content="", prompt_tokens=0, completion_tokens=0))

        budget = _pin_cost_accounting(monkeypatch, budget_usd=10.0, spend_usd=0.0)
        monkeypatch.setattr(chat_module.config, "LLM_CALL_RESERVE_USD", 0.02)
        monkeypatch.setattr(chat_module, "_prepare_turn", _fake_prepare_llm())
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

        body = _stream_body(client, h, sid, "Who invested in fintech?")

        # An ordinary, complete turn: no disconnect, no failure.
        assert "event: done" in body
        assert "event: error" not in body
        msgs = client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"]
        assert msgs[1]["aborted"] is False
        # Charged, not released, and the stored cost matches what was charged.
        assert msgs[1]["cost"] == pytest.approx(0.02)
        assert budget.writes == [("settle", 20_000)]
        assert budget.counter == 20_000
        assert budget.holds == {}
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_completed_turn_with_reported_usage_still_uses_the_real_cost(tmp_path, monkeypatch):
    """The other side of that rule: a provider that DOES report usage must be
    charged its real cost, not the estimate. Without this the previous test
    would also pass if every turn were blindly charged the reserve."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def fake_stream(client, prompt, model, usage_holder=None):
            yield "A fully delivered answer [1]."
            if usage_holder is not None:
                # 100k prompt tokens at the pinned $1 / 1M is exactly $0.10,
                # five times the $0.02 estimate -- the two must not be confused.
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
