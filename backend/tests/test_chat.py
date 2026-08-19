"""Chat store and API tests: per-user session CRUD, ownership isolation,
retention purging, and the message-turn flow (retrieval + LLM stubbed)."""

import asyncio
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
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: ("q", "2024-01-01", "2024-12-31"))

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
    # A value column with no numeric cell at all is invalid.
    assert chat_module.parse_dataviz(
        '```dataviz\n{"columns": ["Company", "Value"], '
        '"rows": [["Wakefit", ""], ["Groww", "value not stated"]], "value_column": 1}\n```'
    ) is None


def test_sanitize_dataviz_keeps_valid_strips_malformed():
    valid = "Prose [1].\n\n```dataviz\n{\"columns\": [\"Deal\", \"Value\"], \"rows\": [[\"Zepto\", 1.0]]}\n```"
    assert chat_module._sanitize_dataviz(valid) == valid

    bad = "Prose [1].\n\n```dataviz\n{not json}\n```"
    out = chat_module._sanitize_dataviz(bad)
    assert "```dataviz" not in out
    assert "Prose [1]" in out

    plain = "Just prose with a ```dataviz``` mention."
    assert chat_module._sanitize_dataviz(plain) == plain


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
    assert "build the ranked list from the specific items the articles actually name" in chat_module.CHAT_PROMPT
    assert "Do NOT refuse a top-N list just because the articles lack a pre-made ranking" in chat_module.CHAT_PROMPT
    assert "never respond with only \"the articles do not provide a ranking\"" in chat_module.CHAT_PROMPT
    # IPO questions must yield companies that went public, not M&A/stake deals,
    # and a list item with no stated value must still be included.
    assert "the list items are the COMPANIES that went public or filed for an IPO" in chat_module.CHAT_PROMPT
    assert "Do not substitute M&A or PE-VC deals when the question asks for IPOs" in chat_module.CHAT_PROMPT
    assert "write \"value not stated\"" in chat_module.CHAT_PROMPT
    # The dataviz table must include every listed item; missing values use "".
    assert "EVERY item you list in your answer must appear as a row" in chat_module.CHAT_PROMPT
    assert "never drop the row" in chat_module.CHAT_PROMPT


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
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None))

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
    assert "max 10 rows" in turn.answer  # dataviz cap matches the requested N

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
    monkeypatch.setattr(main, "_effective_intent", lambda q, f, t: (q, None, None))

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
