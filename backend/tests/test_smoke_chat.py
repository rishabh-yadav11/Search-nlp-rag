"""Smoke tests for the chat API: session create + message turns (JSON and SSE).

The LLM is the FakeLLMClient, so answers are deterministic. A chat turn needs a
strong retrieval hit, which the seed corpus + fake reranker (logit 1.0 ->
~0.73) provide.
"""

from __future__ import annotations

import itertools
import os

_PASSWORD = "Password1"
_EMAIL_COUNTER = itertools.count()
_QUESTION = "what happened with Ola Electric funding news"


def _email() -> str:
    return f"chat-{next(_EMAIL_COUNTER)}-{os.getpid()}@example.test"


def _authed(app_client):
    app_client.cookies.clear()
    email = _email()
    app_client.post("/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "Chat"})
    r = app_client.post("/api/auth/login", json={"email": email, "password": _PASSWORD})
    assert r.status_code == 200, r.text
    return email


def test_create_session_json_turn(app_client) -> None:
    _authed(app_client)
    sess = app_client.post("/api/chat/sessions")
    assert sess.status_code == 200, sess.text
    session_id = sess.json()["id"]
    assert session_id

    turn = app_client.post(f"/api/chat/sessions/{session_id}/messages", json={"content": _QUESTION})
    assert turn.status_code == 200, turn.text[:500]
    body = turn.json()
    assert body["user"]["role"] == "user"
    assert body["assistant"]["role"] == "assistant"
    answer = body["assistant"]["content"]
    assert isinstance(answer, str) and answer.strip()
    # The fake's canonical answer was routed to storage end to end.
    assert "Ola Electric" in answer or "PhonePe" in answer
    # Sources were attached and persisted.
    assert isinstance(body["assistant"].get("sources"), list)
    assert len(body["assistant"].get("sources", [])) >= 1


def test_chat_requires_auth(app_client) -> None:
    app_client.cookies.clear()
    assert app_client.post("/api/chat/sessions").status_code == 401


def test_sse_stream_returns_done_event(app_client) -> None:
    _authed(app_client)
    session_id = app_client.post("/api/chat/sessions").json()["id"]
    with app_client.stream(
        "POST",
        f"/api/chat/sessions/{session_id}/messages/stream",
        json={"content": _QUESTION},
    ) as resp:
        assert resp.status_code == 200
        chunks = b"".join(resp.iter_bytes()).decode("utf-8", "replace")
    assert "event: start" in chunks
    assert "event: done" in chunks
    assert "Ola Electric" in chunks or "PhonePe" in chunks
