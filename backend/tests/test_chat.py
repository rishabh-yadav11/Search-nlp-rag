"""Chat store and API tests: per-user session CRUD, ownership isolation,
retention purging, and the message-turn flow (retrieval + LLM stubbed)."""

import asyncio
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import chat as chat_module
from app.chat import ChatStore, _smalltalk_reply

USER_A = "user-a-device-id-0001"
USER_B = "user-b-device-id-0002"


def _run(coro):
    return asyncio.run(coro)


def _store(tmp_path):
    s = ChatStore(str(tmp_path / "chat.db"))
    _run(s.connect())
    return s


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


def _make_client(tmp_path):
    store = _store(tmp_path)
    app = FastAPI()
    app.include_router(chat_module.router)
    chat_module.store = store
    client = TestClient(app)
    return client, store


def test_api_requires_user_header(tmp_path):
    client, store = _make_client(tmp_path)
    try:
        assert client.post("/api/chat/sessions").status_code == 400
        assert client.post("/api/chat/sessions", headers={"X-User-Id": "short"}).status_code == 400
    finally:
        _run(store.close())


def test_api_create_and_list(tmp_path):
    client, store = _make_client(tmp_path)
    try:
        h = {"X-User-Id": USER_A}
        created = client.post("/api/chat/sessions", headers=h).json()
        assert created["id"]
        listed = client.get("/api/chat/sessions", headers=h).json()
        assert [s["id"] for s in listed] == [created["id"]]
    finally:
        _run(store.close())


def test_api_get_rename_delete_flow(tmp_path):
    client, store = _make_client(tmp_path)
    try:
        h = {"X-User-Id": USER_A}
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        detail = client.get(f"/api/chat/sessions/{sid}", headers=h).json()
        assert detail["messages"] == []

        renamed = client.patch(f"/api/chat/sessions/{sid}", headers=h, json={"content": "Renamed"}).json()
        assert renamed["title"] == "Renamed"

        assert client.get(f"/api/chat/sessions/{sid}", headers={"X-User-Id": USER_B}).status_code == 404

        assert client.delete(f"/api/chat/sessions/{sid}", headers=h).status_code == 200
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).status_code == 404
    finally:
        _run(store.close())


def test_api_send_message_runs_turn(tmp_path, monkeypatch):
    client, store = _make_client(tmp_path)
    try:
        h = {"X-User-Id": USER_A}
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
        _run(store.close())


def test_api_usage_stats(tmp_path, monkeypatch):
    client, store = _make_client(tmp_path)
    try:
        h = {"X-User-Id": USER_A}
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
        assert client.get("/api/chat/usage", headers={"X-User-Id": USER_B}).json()["total_tokens"] == 0
    finally:
        _run(store.close())


def test_api_send_message_rejects_empty(tmp_path):
    client, store = _make_client(tmp_path)
    try:
        h = {"X-User-Id": USER_A}
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]
        assert client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": "   "}).status_code == 400
    finally:
        _run(store.close())


def test_smalltalk_returns_canned_reply():
    for greeting in ["hi", "hello", "hey", "good morning", "Good Afternoon!", "namaste", "how are you?", "thanks", "thank you", "bye", "who are you?"]:
        reply = _smalltalk_reply(greeting, [])
        assert reply is not None, greeting
        assert "ASK VCCircle" in reply or "archive" in reply

def test_smalltalk_ignores_real_queries():
    for q in ["who invested in Ola Electric?", "top 10 fintech deals 2025", "Ola Electric IPO", "what is the latest funding news", "how many deals did Sequoia do last year?"]:
        assert _smalltalk_reply(q, []) is None, q

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