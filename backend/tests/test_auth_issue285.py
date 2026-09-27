"""Issue #285: a password change must not be able to half-happen.

``change_password`` used to run three writes that only mean anything together
-- store the new hash, revoke the old tokens, mint a replacement -- each
committing on its own. A worker killed between any two of them (gunicorn's
``--timeout 120`` does exactly that) left the durable state inside the change;
the sharpest version of that harm is a new password standing next to every
previously issued token, still valid.

Each test below kills the change at one interruption point and then reads the
database back through a brand-new connection, which is the only view that
reflects what a process surviving the kill would actually see.
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.auth import AuthStore

OLD_PW = "secret12"
NEW_PW = "secret21"

_REAL_AIO_SQLITE = auth.aiosqlite


class _SimulatedKill(BaseException):
    """Stands in for a worker being killed mid-request.

    A ``BaseException`` on purpose: that is what asyncio cancellation is, and
    a SIGKILL -- which runs no handler at all -- leaves the database in exactly
    the state an uncommitted transaction leaves it in.
    """


# The writes of the change, in the order the store performs them, plus the
# commit that makes them durable. A kill is scheduled at one of these steps.
_STEP_REVOKE = "revoke tokens"
_STEP_HASH = "write new hash"
_STEP_MINT = "mint replacement token"
_STEP_COMMIT = "commit"

_SQL_STEP = (
    ("DELETE FROM auth_tokens", _STEP_REVOKE),
    ("UPDATE users SET password_hash", _STEP_HASH),
    ("INSERT INTO auth_tokens", _STEP_MINT),
)


class _KillSwitch:
    """Records the step the change reaches and kills the worker at one.

    A step is recorded as reached whether or not it completed, so the last
    entry is the one the kill interrupted.
    """

    def __init__(self, kill_at: int):
        self.kill_at = kill_at
        self.steps: list[str] = []

    def enter(self, step: str) -> None:
        self.steps.append(step)
        if len(self.steps) == self.kill_at:
            raise _SimulatedKill(step)


def _watch(conn, kill: _KillSwitch):
    """Make ``conn`` report the change's step before each write, and kill there."""
    real_execute, real_commit = conn.execute, conn.commit

    async def execute(sql, *args, **kwargs):
        text = " ".join(sql.split())
        for prefix, step in _SQL_STEP:
            if text.startswith(prefix):
                kill.enter(step)
                break
        return await real_execute(sql, *args, **kwargs)

    async def commit():
        kill.enter(_STEP_COMMIT)
        return await real_commit()

    conn.execute = execute
    conn.commit = commit
    return conn


class _KillSwitchAioSqlite:
    """Stands in for the ``aiosqlite`` module, wrapping only ``connect()``."""

    def __init__(self, kill: _KillSwitch):
        self._kill = kill

    def connect(self, *args, **kwargs):
        return self._open(_REAL_AIO_SQLITE.connect(*args, **kwargs))

    async def _open(self, coro):
        return _watch(await coro, self._kill)

    def __getattr__(self, name):
        return getattr(_REAL_AIO_SQLITE, name)


def _arm(monkeypatch, store: AuthStore, kill_at: int) -> _KillSwitch:
    """Kill the worker just as the change reaches step ``kill_at`` (1-based).

    The store's already-open connection is watched as well as every connection
    it opens later, so the test does not presuppose which one the change runs
    on -- only that the worker can die between two of its writes.
    """
    kill = _KillSwitch(kill_at)
    _watch(store._db, kill)
    monkeypatch.setattr(auth, "aiosqlite", _KillSwitchAioSqlite(kill))
    return kill


def _seeded(tmp_path):
    """A connected store, one user, two live tokens, and the db path."""
    db_path = str(tmp_path / "auth.db")
    store = AuthStore(db_path)
    asyncio.run(store.connect())
    user = asyncio.run(store.create_user("a@x.co", OLD_PW, "A", "user"))
    tokens = [asyncio.run(store.issue_token(user.id, 7)) for _ in range(2)]
    return store, db_path, user, tokens


def _assert_no_partial_change(db_path: str, user_id: str, old_tokens: list[str], new_password: str) -> None:
    """Assert the committed state, read through an independent connection.

    Two things have to hold.

    The harm #285 describes must be impossible: a password that has already
    changed must never sit behind a token minted before the change. And the
    user must not be locked out -- at least one of the two passwords they know
    still authenticates.

    And because every test that calls this kills the worker *before* the
    commit, the state must be exactly the pre-change one: an uncommitted
    transaction is invisible to everyone, so none of the three writes may have
    landed.
    """
    reader = AuthStore(db_path)
    asyncio.run(reader.connect())
    try:
        stored = asyncio.run(reader.get_user(user_id))
        assert stored is not None, "the user row disappeared"
        new_works = auth.verify_password(new_password, stored.password_hash)
        old_works = auth.verify_password(OLD_PW, stored.password_hash)
        still_live = [t for t in old_tokens if asyncio.run(reader.user_for_token(t)) is not None]
        assert not (new_works and still_live), (
            f"the password change was durable while {len(still_live)} pre-existing "
            "token(s) still authenticate"
        )
        assert old_works or new_works, "the user can no longer log in with either password"
        assert old_works and not new_works, "a password write became visible without its commit"
        assert still_live == old_tokens, f"tokens were partially revoked: {len(still_live)} of {len(old_tokens)} left"
        assert asyncio.run(reader.active_token_count(user_id)) == len(old_tokens)
    finally:
        asyncio.run(reader.close())


def _change(store: AuthStore, user_id: str) -> str:
    """Run the change the way the endpoint does: hash first, then the store."""
    return asyncio.run(store.change_password(user_id, auth.hash_password(NEW_PW), 7))


def _unarm(monkeypatch):
    monkeypatch.setattr(auth, "aiosqlite", _REAL_AIO_SQLITE)


# --- an interruption at each point between the three writes ---


def test_kill_at_the_second_write_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=2)  # the worker dies at the second write
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        assert len(kill.steps) == 2 and set(kill.steps) == {_STEP_REVOKE, _STEP_HASH}, kill.steps
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_kill_at_the_third_write_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=3)  # the worker dies at the third write
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        assert set(kill.steps) == {_STEP_REVOKE, _STEP_HASH, _STEP_MINT} and len(kill.steps) == 3, kill.steps
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_kill_before_commit_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=4)  # all three writes done, commit not yet
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        assert kill.steps[-1] == _STEP_COMMIT and len(kill.steps) == 4, kill.steps
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_killed_change_leaves_the_shared_connection_writable(tmp_path, monkeypatch):
    """The store's own connection must not be left holding a write transaction.

    A change that borrowed the shared connection and died inside its
    transaction would leave that connection mid-write for every later request
    on this worker -- each of them then waits out the 5 s busy timeout and
    fails. So: after a killed change the shared connection still writes.
    """
    store, _db_path, user, _tokens = _seeded(tmp_path)
    _arm(monkeypatch, store, kill_at=3)
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        fresh = asyncio.run(store.issue_token(user.id, 7))
        assert asyncio.run(store.user_for_token(fresh)) is not None
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_endpoint_killed_mid_change_keeps_the_old_session_working(tmp_path, monkeypatch):
    """The whole request, not just the store call: a kill leaves the old
    session usable and cannot be reported to the caller as a success."""
    store, db_path, user, tokens = _seeded(tmp_path)
    auth.store = store
    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app, raise_server_exceptions=False)
    _arm(monkeypatch, store, kill_at=3)
    try:
        request = SimpleNamespace(state=SimpleNamespace(user=user))
        body = auth.ChangePasswordIn(current_password=OLD_PW, new_password=NEW_PW)
        with pytest.raises(_SimulatedKill):
            asyncio.run(auth.change_password(body=body, request=request, _=None))

        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
        # and the account is still reachable for real: the old token works and
        # a fresh login with the old password succeeds.
        assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens[0]}"}).status_code == 200
        login = client.post("/api/auth/login", json={"email": "a@x.co", "password": OLD_PW})
        assert login.status_code == 200, login.text
        assert client.post("/api/auth/login", json={"email": "a@x.co", "password": NEW_PW}).status_code == 401
    finally:
        auth.store = None
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_change_password_commits_the_hash_the_revocation_and_the_token_together(tmp_path):
    """The success path, unchanged: one unit, and only its result is visible."""
    store, db_path, user, tokens = _seeded(tmp_path)
    try:
        new_token = _change(store, user.id)
        assert asyncio.run(store.user_for_token(new_token)) is not None
        for dead in tokens:
            assert asyncio.run(store.user_for_token(dead)) is None
        assert asyncio.run(store.active_token_count(user.id)) == 1
        stored = asyncio.run(store.get_user(user.id))
        assert auth.verify_password(NEW_PW, stored.password_hash)
        assert not auth.verify_password(OLD_PW, stored.password_hash)
        # visible to an independent connection, i.e. genuinely committed
        reader = AuthStore(db_path)
        asyncio.run(reader.connect())
        try:
            assert asyncio.run(reader.user_for_token(new_token)) is not None
        finally:
            asyncio.run(reader.close())
    finally:
        asyncio.run(store.close())
