"""Chat store and API tests: per-user session CRUD, ownership isolation,
retention purging, and the message-turn flow (retrieval + LLM stubbed)."""

import asyncio
import sqlite3
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import auth as auth_module
from app import chat as chat_module
from app.auth import AuthStore
from app.chat import ChatStore, _smalltalk_reply

USER_A = "user-a-device-id-0001"
USER_B = "user-b-device-id-0002"
EMAIL_A = "user-a@example.com"
EMAIL_B = "user-b@example.com"


def _run(coro):
    return asyncio.run(coro)


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


def test_api_stream_smalltalk_short_circuits(tmp_path):
    """SSE stream for small talk emits a single done event with a canned reply."""
    from app import main

    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = client.post("/api/chat/sessions", headers=h).json()["id"]

        async def boom(*args, **kwargs):
            raise AssertionError("retrieval should not run for small talk")

        monkeypatch_global = pytest.MonkeyPatch()
        monkeypatch_global.setattr(main, "retrieve_and_rerank", boom)

        with client.stream("POST", f"/api/chat/sessions/{sid}/messages/stream", headers=h, json={"content": "good morning"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())
        monkeypatch_global.undo()

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
                usage_holder.append(type("U", (), {"prompt_tokens": 50, "completion_tokens": 10})())

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "stream_answer", fake_stream)

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

        async def boom(*args, **kwargs):
            raise chat_module.BudgetExceeded()

        monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
        monkeypatch.setattr(chat_module, "assert_within_budget", boom)

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

    plain = "Just prose with a ```dataviz``` mention."
    assert chat_module._sanitize_dataviz(plain) == plain


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
    ]:
        assert chat_module._CHART_INTENT_RE.search(q), q
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

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT"))
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

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT"))
    assert len(calls) == 1
    assert result.prompt_tokens == 10


def test_answer_with_dataviz_no_retry_for_non_numeric_question(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        return chat_module.LLMResult(content="Plain answer [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_with_dataviz("who invested in Ola Electric?", "PROMPT"))
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

    result = _run(chat_module._answer_with_dataviz("top 10 ipo deals in 2025", "PROMPT"))
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

    result = _run(chat_module._answer_with_dataviz("show me a chart of top 5 deals", "PROMPT"))
    assert len(calls) == 2
    assert result.content == "No chart here [1]."
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_json_loads_malformed_returns_empty():
    assert chat_module.json_loads("{not json") == []
    assert chat_module.json_loads(None) == []
    assert chat_module.json_loads("") == []
    assert chat_module.json_loads('[]') == []
    assert chat_module.json_loads(123) == []  # TypeError branch


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

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT"))
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

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT"))
    assert len(calls) == 2
    assert result.content == "I cannot generate a ranked list."
    assert result.prompt_tokens == 10


def test_answer_ranked_single_call_when_not_refusal(monkeypatch):
    calls = []

    async def fake_generate(client, prompt, model):
        calls.append(prompt)
        return chat_module.LLMResult(content="Top deal is Zepto [1].", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "state_llm", lambda: object())

    result = _run(chat_module._answer_ranked("top 10 ipo deals in 2025", "PROMPT"))
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


def test_run_turn_records_cost_and_finalizes(monkeypatch):
    """Lines 917-920: _run_turn checks the budget, calls the LLM, records cost,
    and finalizes the answer (strips unrequested dataviz blocks)."""
    calls = {}

    async def fake_prepare(question, history):
        return chat_module.PreparedTurn(answer="PROMPT", sources=[{"id": 1}], note="note", needs_llm=True)

    async def fake_budget():
        calls["budget"] = True

    async def fake_answer_ranked(question, prompt):
        calls["prompt"] = prompt
        return chat_module.LLMResult(
            content='Prose [1].\n\n```dataviz\n{"columns": ["A", "B"], "rows": [["x", 1.0]], "value_column": 1}\n```',
            prompt_tokens=10,
            completion_tokens=5,
        )

    async def fake_record_cost(cost):
        calls["cost"] = cost

    monkeypatch.setattr(chat_module, "_prepare_turn", fake_prepare)
    monkeypatch.setattr(chat_module, "assert_within_budget", fake_budget)
    monkeypatch.setattr(chat_module, "_answer_ranked", fake_answer_ranked)
    monkeypatch.setattr(chat_module, "record_cost", fake_record_cost)

    answer, sources, note, pt, ct, _cost = _run(chat_module._run_turn("top 10 ipo deals in 2025", []))
    assert calls["budget"] is True
    assert calls["prompt"] == "PROMPT"
    expected_cost = chat_module.LLMResult(content="", prompt_tokens=10, completion_tokens=5).cost()
    assert calls["cost"] == pytest.approx(expected_cost)
    # No chart intent -> dataviz block stripped by _finalize_answer.
    assert "dataviz" not in answer
    assert "Prose [1]" in answer
    assert sources == [{"id": 1}]
    assert note == "note"
    assert (pt, ct) == (10, 5)


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
        monkeypatch.setattr(chat_module, "assert_within_budget", noop)
        monkeypatch.setattr(chat_module, "record_cost", noop)

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
        monkeypatch.setattr(chat_module, "assert_within_budget", noop)
        monkeypatch.setattr(chat_module, "record_cost", noop)

        body = _stream_body(client, h, sid, "show me a chart of top deals")
        assert "streamed answer without a block [1]." in body
        assert '"prompt_tokens":10' in body  # unchanged
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
        monkeypatch.setattr(chat_module, "assert_within_budget", noop)
        monkeypatch.setattr(chat_module, "record_cost", noop)

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
        monkeypatch.setattr(chat_module, "assert_within_budget", noop)
        monkeypatch.setattr(chat_module, "record_cost", noop)

        body = _stream_body(client, h, sid, "top 10 ipo deals in 2025")
        assert "I cannot generate a ranked list because amounts are missing." in body
        assert '"prompt_tokens":10' in body
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
        monkeypatch.setattr(chat_module, "assert_within_budget", noop)
        monkeypatch.setattr(chat_module, "record_cost", noop)

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
        monkeypatch.setattr(chat_module, "assert_within_budget", noop)
        monkeypatch.setattr(chat_module, "record_cost", noop)

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
