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
import contextlib
import sqlite3
from types import SimpleNamespace

import pytest
from conftest import auth_cookie
from fastapi import FastAPI
from fastapi.responses import Response
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

    async def enter(self, step: str) -> None:
        self.steps.append(step)
        if len(self.steps) == self.kill_at:
            raise _SimulatedKill(step)


class _PauseAt:
    """Holds the change open at one step until the test lets it go.

    Event-driven rather than a sleep, so the test does not depend on how long
    the surrounding statements take. Every other step passes straight through.
    """

    def __init__(self, step: str):
        self.step = step
        self.reached = asyncio.Event()
        self.resume = asyncio.Event()

    async def enter(self, step: str) -> None:
        if step != self.step or self.resume.is_set():
            return
        self.reached.set()
        await self.resume.wait()

    def release(self) -> None:
        self.resume.set()


def _watch(conn, controller):
    """Make ``conn`` report the change's step before each write, and act there."""
    real_execute, real_commit = conn.execute, conn.commit

    async def execute(sql, *args, **kwargs):
        text = " ".join(sql.split())
        for prefix, step in _SQL_STEP:
            if text.startswith(prefix):
                await controller.enter(step)
                break
        return await real_execute(sql, *args, **kwargs)

    async def commit():
        await controller.enter(_STEP_COMMIT)
        return await real_commit()

    conn.execute = execute
    conn.commit = commit
    return conn


class _WatchedAioSqlite:
    """Stands in for the ``aiosqlite`` module, wrapping only ``connect()``."""

    def __init__(self, controller):
        self._controller = controller

    def connect(self, *args, **kwargs):
        return self._open(_REAL_AIO_SQLITE.connect(*args, **kwargs))

    async def _open(self, coro):
        return _watch(await coro, self._controller)

    def __getattr__(self, name):
        return getattr(_REAL_AIO_SQLITE, name)


def _attach(monkeypatch, store: AuthStore, controller) -> None:
    """Watch the store's open connection and every one it opens after this.

    Both are watched so a test does not presuppose which connection the change
    runs on -- only where in its sequence it can be interrupted or held.
    """
    _watch(store._db, controller)
    monkeypatch.setattr(auth, "aiosqlite", _WatchedAioSqlite(controller))


def _arm(monkeypatch, store: AuthStore, kill_at: int) -> _KillSwitch:
    """Kill the worker just as the change reaches step ``kill_at`` (1-based)."""
    kill = _KillSwitch(kill_at)
    _attach(monkeypatch, store, kill)
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


def test_kill_at_the_first_write_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=1)  # the worker dies before any write lands
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        reached = list(kill.steps)
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
        assert reached == [_STEP_REVOKE], reached
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_kill_at_the_second_write_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=2)  # the worker dies at the second write
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        reached = list(kill.steps)
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
        assert len(reached) == 2 and set(reached) == {_STEP_REVOKE, _STEP_HASH}, reached
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_kill_at_the_third_write_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=3)  # the worker dies at the third write
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        reached = list(kill.steps)
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
        assert set(reached) == {_STEP_REVOKE, _STEP_HASH, _STEP_MINT} and len(reached) == 3, reached
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_kill_before_commit_changes_nothing(tmp_path, monkeypatch):
    store, db_path, user, tokens = _seeded(tmp_path)
    kill = _arm(monkeypatch, store, kill_at=4)  # all three writes done, commit not yet
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        reached = list(kill.steps)
        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
        assert reached[-1] == _STEP_COMMIT and len(reached) == 4, reached
    finally:
        _unarm(monkeypatch)
        asyncio.run(store.close())


def test_killed_change_does_not_lock_out_other_connections(tmp_path, monkeypatch):
    """A change that died inside a transaction must not leave the file locked.

    If the writes ran on the shared connection and it was abandoned mid
    transaction, that connection still holds SQLite's write lock, and every
    other connection -- each later request in this worker, and every other
    gunicorn worker -- waits out the 5 s busy timeout and then fails. A
    connection never blocks on a lock it holds itself, so this has to be
    probed from a *second* connection: create a user on one and see it through.
    """
    store, db_path, user, _tokens = _seeded(tmp_path)
    _arm(monkeypatch, store, kill_at=3)
    other = AuthStore(db_path)
    try:
        with pytest.raises(_SimulatedKill):
            _change(store, user.id)
        asyncio.run(other.connect())
        created = asyncio.run(other.create_user("b@x.co", OLD_PW, "B", "user"))
        assert created.email == "b@x.co"
    finally:
        _unarm(monkeypatch)
        asyncio.run(other.close())
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
        response = Response()
        body = auth.ChangePasswordIn(current_password=OLD_PW, new_password=NEW_PW)
        with pytest.raises(_SimulatedKill):
            asyncio.run(
                auth.change_password(body=body, request=request, response=response, _=None)
            )

        _assert_no_partial_change(db_path, user.id, tokens, NEW_PW)
        # and the account is still reachable for real: the pre-existing
        # session cookie still authenticates, and a fresh login with the old
        # password succeeds.
        assert client.get("/api/auth/me", cookies=auth_cookie(tokens[0])).status_code == 200
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


def test_a_concurrent_login_cannot_publish_a_half_finished_change(tmp_path, monkeypatch):
    """A password change must be invisible to every other writer.

    The change runs on its own connection precisely so ordinary traffic cannot
    land inside it. This store's shared connection serves every request in the
    worker, so on a shared-connection design a login committing between the
    hash write and the revocation publishes a half-finished change -- measured
    by hand: a separate connection then saw a new password next to two
    still-live old tokens, which is the harm #285 describes, produced by
    normal traffic rather than a crash. The docstring claims the dedicated
    connection prevents that; this is what holds it to the claim.
    """

    async def main():
        db_path = str(tmp_path / "auth.db")
        store = AuthStore(db_path)
        await store.connect()
        user = await store.create_user("a@x.co", OLD_PW, "A", "user")
        tokens = [await store.issue_token(user.id, 7) for _ in range(2)]
        # Opened before the change starts, so its own schema writes are not
        # competing for the write lock the change is about to take.
        reader = AuthStore(db_path)
        await reader.connect()
        # Keep the losing writer's wait short: what is asserted is that it
        # cannot get in, not how long it waits for the lock.
        await store._db.execute("PRAGMA busy_timeout=200")
        pause = _PauseAt(_STEP_HASH)
        _attach(monkeypatch, store, pause)
        change = None
        try:
            change = asyncio.create_task(store.change_password(user.id, auth.hash_password(NEW_PW), 7))
            await asyncio.wait_for(pause.reached.wait(), 10)
            # An unrelated request writes through the store's own connection.
            try:
                await store.issue_token(user.id, 7)
            except sqlite3.OperationalError:
                pass  # refused the write lock, one of the two acceptable outcomes
            stored = await reader.get_user(user.id)
            assert not auth.verify_password(NEW_PW, stored.password_hash), (
                "an unrelated write published the new password before the change committed"
            )
            assert all([await reader.user_for_token(t) is not None for t in tokens]), (
                "an unrelated write published part of the revocation before the change committed"
            )
            pause.release()
            token = await asyncio.wait_for(change, 10)
            # Once it does commit, the whole change is visible at once.
            stored = await reader.get_user(user.id)
            assert auth.verify_password(NEW_PW, stored.password_hash)
            assert all([await reader.user_for_token(t) is None for t in tokens])
            assert await reader.user_for_token(token) is not None
        finally:
            # Always let the change finish before anything is closed. A
            # mutation that trips an assertion above leaves this task parked
            # on the pause, so it has to be released and awaited first --
            # closing the store out from under it would report "Cannot
            # operate on a closed database" as the failure and bury the
            # assertion that actually caught the mutation.
            pause.release()
            if change is not None:
                # Bounded, so a future mutation that wedges this task fails
                # the test instead of hanging the suite. Timing out here is
                # suppressed because this path is only ever load-bearing when
                # an assertion above has already fired, and the failure worth
                # reporting is that assertion, not the cleanup.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.gather(change, return_exceptions=True), 10)
            _unarm(monkeypatch)
            await reader.close()
            await store.close()

    asyncio.run(main())
