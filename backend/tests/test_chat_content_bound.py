"""The ``MessageIn.content`` length bound (#350).

The bound is declared on the request model rather than in one route's helper, so
every route that accepts chat text is covered by a single declaration: the two
message routes, the SSE stream, and the session rename, plus any route added
later. A bound that lives in a helper is a bound the next route forgets to call.

Oversized content is REJECTED, never truncated. A silently shortened question is
answered as though it were the whole one, which is worse than a refusal: the
user gets a confident answer to something they never asked.
"""

import pytest
from _support import run_sync as _run
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import auth as auth_module
from app import chat as chat_module
from app.auth import AuthStore
from app.chat import ChatStore, MessageIn

EMAIL_A = "user-a@example.com"


def _store(tmp_path):
    s = ChatStore(str(tmp_path / "chat.db"))
    _run(s.connect())
    return s


def _auth_store(tmp_path):
    s = AuthStore(str(tmp_path / "auth.db"))
    _run(s.connect())
    return s


def _auth_headers(auth_store, email=EMAIL_A, role="user"):
    user = _run(auth_store.get_user_by_email(email))
    if user is None:
        user = _run(auth_store.create_user(email, "secret1", email.split("@")[0], role))
    elif user.role != role:
        _run(auth_store.update_user(user.id, None, role, None))
    token = _run(auth_store.issue_token(user.id, 7))
    return {"Authorization": f"Bearer {token}"}


def _make_client(tmp_path):
    chat_store = _store(tmp_path)
    auth_store = _auth_store(tmp_path)
    app = FastAPI()
    app.include_router(chat_module.router)
    chat_module.store = chat_store
    auth_module.store = auth_store
    return TestClient(app), chat_store, auth_store


def _new_session(client, headers):
    return client.post("/api/chat/sessions", headers=headers).json()["id"]


def _too_long():
    return "x" * (chat_module.MAX_CONTENT_LEN + 1)


def _at_the_bound():
    return "x" * chat_module.MAX_CONTENT_LEN


def test_bound_is_enforced_by_the_model_itself():
    """The guarantee is a property of ``MessageIn``, so it holds for every route
    that takes the model -- including one that has no validation helper at all,
    and one written after this fix."""
    assert MessageIn(content=_at_the_bound()).content == _at_the_bound()
    with pytest.raises(ValidationError):
        MessageIn(content=_too_long())


def test_message_at_the_bound_is_accepted(tmp_path, monkeypatch):
    """The bound is inclusive: a question of exactly MAX_CONTENT_LEN chars is a
    real question and must be answered, not rejected."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = _new_session(client, h)

        async def fake_turn(question, history):
            return "An answer.", [], None, 10, 5, 0.0

        monkeypatch.setattr(chat_module, "_run_turn", fake_turn)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": _at_the_bound()})
        assert r.status_code == 200
        # Stored whole: an accepted message is not quietly clipped on the way in.
        assert r.json()["user"]["content"] == _at_the_bound()
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_oversized_message_is_rejected_and_nothing_is_stored(tmp_path):
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = _new_session(client, h)

        r = client.post(f"/api/chat/sessions/{sid}/messages", headers=h, json={"content": _too_long()})
        assert r.status_code == 422

        # The refusal names the field and the limit, so the client can tell the
        # user what to shorten instead of showing a bare "unprocessable".
        err = r.json()["detail"][0]
        assert err["loc"] == ["body", "content"]
        assert str(chat_module.MAX_CONTENT_LEN) in err["msg"]

        # Rejected, not truncated: no row at all, and certainly not a silently
        # shortened copy of the oversized text.
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_oversized_stream_is_rejected_before_the_stream_opens(tmp_path):
    """The SSE route takes the same model, so an oversized message is refused
    with a plain 422 rather than a stream that fails mid-flight."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = _new_session(client, h)

        r = client.post(f"/api/chat/sessions/{sid}/messages/stream", headers=h, json={"content": _too_long()})
        assert r.status_code == 422
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["messages"] == []
    finally:
        _run(auth_store.close())
        _run(chat_store.close())


def test_oversized_rename_is_rejected_and_leaves_the_title_alone(tmp_path):
    """The rename route writes client text through the same model and used to
    have no length check of its own: the store clipped the title to 200 chars,
    so a rename came back 200 with a title the user never typed. It is now
    refused, and the stored title is untouched."""
    client, chat_store, auth_store = _make_client(tmp_path)
    try:
        h = _auth_headers(auth_store)
        sid = _new_session(client, h)

        assert client.patch(f"/api/chat/sessions/{sid}", headers=h, json={"content": "Renamed"}).json()["title"] == "Renamed"

        r = client.patch(f"/api/chat/sessions/{sid}", headers=h, json={"content": _too_long()})
        assert r.status_code == 422
        assert r.json()["detail"][0]["loc"] == ["body", "content"]
        assert client.get(f"/api/chat/sessions/{sid}", headers=h).json()["title"] == "Renamed"
    finally:
        _run(auth_store.close())
        _run(chat_store.close())
