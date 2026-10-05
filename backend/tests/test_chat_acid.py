"""Atomicity tests for the chat store's SQLite writes.

Each test asserts the observable wrong outcome -- a lost row, a table stuck half
migrated, a file another worker cannot write -- so a regression fails here rather
than in production. Faults are injected with real SQLite (a trigger that refuses
a statement), never by stubbing the store's own methods.
"""

import asyncio
import sqlite3

import pytest
from _support import run_sync as _run
from fastapi import HTTPException

from app import chat as chat_module
from app.chat import ChatStore

USER_A = "user-a-device-id-0001"


def _store(tmp_path):
    s = ChatStore(str(tmp_path / "chat.db"))
    _run(s.connect())
    return s


def _legacy_db(path, messages_schema):
    """Create a chat DB with a caller-chosen `messages` schema, holding one
    session and one message, so `connect()` has something to migrate."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE sessions ("
        " id TEXT PRIMARY KEY, user_id TEXT NOT NULL,"
        " title TEXT NOT NULL DEFAULT 'New chat',"
        " created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.execute(messages_schema)
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, created_at, updated_at)"
        " VALUES ('s1', ?, 'Legacy', 0, 0)",
        (USER_A,),
    )
    conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES ('s1', 'user', 'hello', 0)")
    conn.commit()
    conn.close()


def _refuse(store_path, sql):
    """Install a trigger that makes one statement fail with a constraint error,
    leaving everything before it in the same transaction pending -- the shape of
    a full disk or an interrupted delete."""
    conn = sqlite3.connect(str(store_path))
    conn.execute(sql)
    conn.commit()
    conn.close()


_REFUSE_SESSION_DELETE = (
    "CREATE TRIGGER refuse_session_delete BEFORE DELETE ON sessions"
    " BEGIN SELECT RAISE(ABORT, 'session delete refused'); END"
)
_REFUSE_SESSION_UPDATE = (
    "CREATE TRIGGER refuse_session_update BEFORE UPDATE ON sessions"
    " BEGIN SELECT RAISE(ABORT, 'session update refused'); END"
)


def test_connect_finishes_a_half_migrated_messages_table(tmp_path):
    """A crash between the migration's ALTERs used to brick the table forever.

    The migration keyed the later columns off `prompt_tokens`, so a database that
    already had it was read as complete: `cost` was never added, no boot could
    ever add it, and every read path then failed on `no such column: cost` with
    nothing to repair itself. The state below is exactly one crash into that
    sequence -- the first two ALTERs landed, the third did not.
    """
    db_path = tmp_path / "chat.db"
    _legacy_db(
        db_path,
        "CREATE TABLE messages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
        " role TEXT NOT NULL, content TEXT NOT NULL,"
        " sources TEXT NOT NULL DEFAULT '[]', created_at REAL NOT NULL,"
        " prompt_tokens INTEGER NOT NULL DEFAULT 0,"
        " completion_tokens INTEGER NOT NULL DEFAULT 0)",
    )
    store = ChatStore(str(db_path))
    _run(store.connect())
    try:
        cols = {c["name"] for c in _run(store._db.execute_fetchall("PRAGMA table_info(messages)"))}
        assert {"prompt_tokens", "completion_tokens", "cost", "latency_ms", "aborted"} <= cols
        # The read paths that raised `no such column: cost` now work, and the
        # pre-migration row survived with the added columns defaulted.
        assert [s.id for s in _run(store.list_sessions(USER_A))] == ["s1"]
        msgs, total = _run(store.messages_page("s1", USER_A))
        assert total == 1
        assert (msgs[0].cost, msgs[0].prompt_tokens, msgs[0].aborted) == (0.0, 0, False)
        stats = _run(store.stats(USER_A))
        assert (stats.messages, stats.total_cost) == (1, 0.0)
    finally:
        _run(store.close())


def test_a_failed_delete_session_keeps_the_conversation(tmp_path):
    """A delete that fails after it has started must leave nothing behind.

    The delete used to remove the messages, then the session, then commit, with
    no rollback anywhere. A failure between those statements returned a 500 and
    left the message removal pending on the shared connection, where the NEXT
    unrelated write published it -- the user came back to a conversation that
    still existed and had lost its whole history after a delete that had
    reported failure.
    """
    db_path = tmp_path / "chat.db"
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        _run(store.append_message(sid, USER_A, "user", "first question"))
        _run(store.append_message(sid, USER_A, "assistant", "first answer"))
        _refuse(db_path, _REFUSE_SESSION_DELETE)

        with pytest.raises(sqlite3.IntegrityError):
            _run(store.delete_session(sid, USER_A))

        # An unrelated write is exactly what used to publish the abandoned delete.
        _run(store.create_session(USER_A))

        assert _run(store.get_session(sid, USER_A)) is not None
        msgs, total = _run(store.messages_page(sid, USER_A))
        assert total == 2
        assert [m.content for m in msgs] == ["first question", "first answer"]
    finally:
        _run(store.close())


def test_a_failed_append_leaves_no_write_transaction_behind(tmp_path):
    """An append that fails between its two statements must not keep the file
    locked.

    A cancellation or a crash between the INSERT and the COMMIT used to leave
    the shared connection holding the WAL write lock with its write pending: the
    other three workers failed every write with `database is locked` until this
    worker died, and any later commit published the abandoned row.
    """
    db_path = tmp_path / "chat.db"
    store = _store(tmp_path)
    other = ChatStore(str(db_path))
    _run(other.connect())
    try:
        sid = _run(store.create_session(USER_A)).id
        _refuse(db_path, _REFUSE_SESSION_UPDATE)

        with pytest.raises(sqlite3.IntegrityError):
            _run(store.append_message(sid, USER_A, "user", "half an append"))

        # A second worker must be able to write: the lock is not still held.
        assert _run(other.create_session(USER_A)).id
        # And this worker's own connection is back to a usable state.
        assert _run(store.create_session(USER_A)).id
        # The failed append stored nothing, half or whole.
        _msgs, total = _run(store.messages_page(sid, USER_A))
        assert total == 0
    finally:
        _run(other.close())
        _run(store.close())


def test_append_to_a_deleted_conversation_404s_and_writes_nothing(tmp_path):
    """The guarded INSERT's 404 is raised from inside the unit of work, so it has
    to reach the caller intact AND roll the unit back."""
    db_path = tmp_path / "chat.db"
    store = _store(tmp_path)
    try:
        sid = _run(store.create_session(USER_A)).id
        session = _run(store.get_session(sid, USER_A))
        _run(store.delete_session(sid, USER_A))

        with pytest.raises(HTTPException) as exc:
            _run(store._append_authorized(session, "user", "too late"))
        assert exc.value.status_code == 404

        # Nothing written, nothing left half-open: another worker can still write.
        other = ChatStore(str(db_path))
        _run(other.connect())
        try:
            assert _run(other.create_session(USER_A)).id
        finally:
            _run(other.close())
    finally:
        _run(store.close())


def _cancel_request():
    """A Request whose `receive()` never returns, so `is_disconnected()` is always
    False and only the cancellation under test can reconcile the stored turn."""
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


def test_a_cancelled_turn_drops_only_the_row_it_wrote(tmp_path, monkeypatch):
    """Two tabs can ask the same question at the same time.

    A turn cancelled inside the append's own COMMIT never learns the id of the
    row it wrote, so it finds the row by what it holds. Matching on the newest
    user message carrying that text found the OTHER turn's row instead whenever
    this one committed first and a second tab committed later -- leaving that tab
    with a reply and no question, and rebasing the conversation.
    """
    store = _store(tmp_path)
    chat_module.store = store
    try:
        sid = _run(store.create_session(USER_A)).id
        session = _run(store.get_session(sid, USER_A))
        question = "what deals happened"
        ready = asyncio.Event()
        parked = {"done": False}
        real_append = store._append_authorized
        other_turn: dict = {}

        async def slow_append(session_, role, *args, **kwargs):
            result = await real_append(session_, role, *args, **kwargs)
            if role == "user" and not parked["done"]:
                parked["done"] = True
                ready.set()
                await asyncio.sleep(3600)
            return result

        monkeypatch.setattr(store, "_append_authorized", slow_append)

        async def run_it():
            task = asyncio.create_task(
                chat_module.send_message(sid, chat_module.MessageIn(content=question), _cancel_request())
            )
            await asyncio.wait_for(ready.wait(), timeout=5)
            # A second tab posts the identical text while the first is parked, so
            # its user row is the newest one carrying this question.
            other_turn["msg"] = await store._append_authorized(session, "user", question)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        assert _run(run_it()) is True

        rows = _run(store.recent_turns(sid, USER_A, 5))
        # The cancelled turn reconciled itself and left the other tab's row alone.
        assert [(m.role, m.id) for m in rows] == [("user", other_turn["msg"].id)]

        # The reconciliation ran under a cancellation and still left the file
        # writable for another worker.
        other = ChatStore(str(tmp_path / "chat.db"))
        _run(other.connect())
        try:
            assert _run(other.create_session(USER_A)).id
        finally:
            _run(other.close())
    finally:
        chat_module.store = None
        _run(store.close())