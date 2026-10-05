"""What the auth store commits must match what it reports.

Every case here is a defect whose committed state is wrong from OUTSIDE the
store: a login answered 200 with a cookie that cannot authenticate, an
interrupted login leaving a session nobody holds (and evicting one somebody
does), a password change reported as done against an account that is already
gone. So every assertion reads the table or goes through the API; none of them
check that a particular statement ran, that a particular helper was reached, or
which connection carried the write.

The store may run a unit on its shared connection or on a short-lived
dedicated one, so a fault is injected into both and each test states only the
outcome.
"""

import asyncio
import sqlite3
import time

import pytest
from conftest import auth_cookie

from app import auth
from app.auth import AuthStore
from app.config import config

_REAL_AIO_SQLITE = auth.aiosqlite

# The first eviction of the cap, which runs after the row the caller is about to
# be handed has been written. Matched against the SQL with its whitespace
# collapsed.
_EVICT_SQL = "DELETE FROM auth_tokens"


class _Killed(BaseException):
    """Stands in for a worker being killed mid-request.

    A BaseException, because to SQLite a kill and a cancellation are the same
    event: nothing after them gets to run.
    """


class _Fault:
    """Raise on the first statement whose SQL starts with ``prefix``."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.tripped = False

    def attach(self, conn):
        real_execute = conn.execute

        async def execute(sql, *args, **kwargs):
            if not self.tripped and " ".join(sql.split()).startswith(self.prefix):
                self.tripped = True
                raise _Killed(self.prefix)
            return await real_execute(sql, *args, **kwargs)

        conn.execute = execute
        return conn


class _PauseOn:
    """Hold the first statement whose SQL starts with ``prefix`` until the test
    lets it through, so the state a second connection can see in the middle of a
    write is observable.

    Event-driven rather than a sleep, so nothing here depends on how long the
    surrounding statements take.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.reached = asyncio.Event()
        self.resume = asyncio.Event()

    def attach(self, conn):
        real_execute = conn.execute

        async def execute(sql, *args, **kwargs):
            if not self.reached.is_set() and " ".join(sql.split()).startswith(self.prefix):
                self.reached.set()
                await self.resume.wait()
            return await real_execute(sql, *args, **kwargs)

        conn.execute = execute
        return conn


class _WatchedAioSqlite:
    """Stands in for the ``aiosqlite`` module, wrapping only ``connect()`` so a
    fault reaches every connection the store opens after this point."""

    def __init__(self, fault):
        self._fault = fault

    def connect(self, *args, **kwargs):
        return self._open(_REAL_AIO_SQLITE.connect(*args, **kwargs))

    async def _open(self, coro):
        return self._fault.attach(await coro)

    def __getattr__(self, name):
        return getattr(_REAL_AIO_SQLITE, name)


def _store(tmp_path) -> AuthStore:
    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    return s


def _user(store, email: str = "a@b.co"):
    return asyncio.run(store.create_user(email, "secret12", "A", "user"))


def _token_rows(store, user_id: str) -> list:
    return asyncio.run(store._fetchall("SELECT * FROM auth_tokens WHERE user_id = ?", (user_id,)))


def _app(store):
    """The real router over a real store, with server exceptions turned into the
    500 a caller would really see."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    auth.store = store
    app = FastAPI()
    app.include_router(auth.router)
    return TestClient(app, raise_server_exceptions=False)


def _auth_cookies(response) -> list[str]:
    prefix = f"{config.AUTH_COOKIE_NAME}="
    return [h for h in response.headers.get_list("set-cookie") if h.startswith(prefix)]


# --- A1: a password change against an account that no longer exists ---


def test_change_password_on_a_deleted_account_stores_no_session(tmp_path):
    """The user the change is for can be gone by the time the write runs. A
    zero-row UPDATE is not a password change: it must fail the unit rather than
    mint a session for an account nothing references."""
    store = _store(tmp_path)
    try:
        user = _user(store)
        old = asyncio.run(store.issue_token(user.id, 7))
        asyncio.run(store.delete_user(user.id))

        with pytest.raises(ValueError):
            asyncio.run(store.change_password(user.id, auth.hash_password("newpass12"), 7))

        assert asyncio.run(store.user_for_token(old)) is None, "the account deletion left a session live"
        assert _token_rows(store, user.id) == [], (
            "a password change for a deleted account stored a session row nobody holds"
        )
    finally:
        asyncio.run(store.close())


def test_change_password_does_not_answer_200_with_a_cookie_that_cannot_authenticate(tmp_path, monkeypatch):
    """The whole request, in the shape that produced it: the route reads the
    user, spends a bcrypt round on the old password, and only then calls the
    store -- so an admin DELETE landing in that window must not come back to the
    user as a successful password change."""
    store = _store(tmp_path)
    try:
        user = _user(store)
        session = asyncio.run(store.issue_token(user.id, 7))
        client = _app(store)
        real_change = AuthStore.change_password

        async def deleted_first(self, user_id, new_password_hash, ttl_days):
            await self.delete_user(user_id)  # the admin DELETE that lands in the window
            return await real_change(self, user_id, new_password_hash, ttl_days)

        monkeypatch.setattr(AuthStore, "change_password", deleted_first)

        response = client.post(
            "/api/auth/change-password",
            json={"current_password": "secret12", "new_password": "newpass12"},
            cookies=auth_cookie(session),
        )
        assert response.status_code >= 400, response.text
        assert _auth_cookies(response) == [], (
            "the request was answered with a session cookie for an account that is gone"
        )
        assert _token_rows(store, user.id) == [], "the failed change left an orphan session row behind"
    finally:
        auth.store = None
        asyncio.run(store.close())


# --- A2: minting a session and honouring the cap are one unit ---


def test_an_interrupted_login_stores_no_session_and_leaves_the_live_ones_alone(tmp_path, monkeypatch):
    """Both halves or neither. Publishing the new row on its own left a user
    over the cap by exactly one with nothing left to notice, and evicting on
    their own is what costs a real session its place."""
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 2)
    store = _store(tmp_path)
    try:
        user = _user(store)
        live = [asyncio.run(store.issue_token(user.id, 7)) for _ in range(2)]

        fault = _Fault(_EVICT_SQL)
        fault.attach(store._db)
        monkeypatch.setattr(auth, "aiosqlite", _WatchedAioSqlite(fault))
        try:
            with pytest.raises(_Killed):
                asyncio.run(store.issue_token(user.id, 7))
            assert fault.tripped, "the eviction never ran, so this proved nothing"
        finally:
            monkeypatch.setattr(auth, "aiosqlite", _REAL_AIO_SQLITE)

        assert [t for t in live if asyncio.run(store.user_for_token(t)) is not None] == live, (
            "the interrupted login evicted a session and then failed"
        )
        assert asyncio.run(store.active_token_count(user.id)) == 2, (
            "the interrupted login stored a session nobody was handed"
        )

        # Still usable: a failed unit rolled back rather than leaving the write
        # lock held, and the next login brings the user back under the cap.
        again = asyncio.run(store.issue_token(user.id, 7))
        assert asyncio.run(store.user_for_token(again)) is not None
        assert asyncio.run(store.active_token_count(user.id)) == 2
    finally:
        asyncio.run(store.close())


def test_a_login_is_invisible_to_other_connections_until_it_commits(tmp_path, monkeypatch):
    """A session that only becomes real at the commit: anything that can see it
    earlier is handing out a token the table does not have, and a user over the
    cap the login was supposed to hold down."""
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 1)
    store = _store(tmp_path)
    # Opened before the pause, so the reader's own writes are not paused too.
    reader = AuthStore(store._path)
    asyncio.run(reader.connect())
    try:
        user = _user(store)
        asyncio.run(store.issue_token(user.id, 7))  # the session already at the cap
        pause = _PauseOn(_EVICT_SQL)
        pause.attach(store._db)
        monkeypatch.setattr(auth, "aiosqlite", _WatchedAioSqlite(pause))

        async def scenario():
            login = asyncio.create_task(store.issue_token(user.id, 7))
            try:
                await asyncio.wait_for(pause.reached.wait(), 10)
                # The mint has run and the eviction is not committed: nothing
                # outside this unit may see the new session yet.
                assert await reader.active_token_count(user.id) == 1, (
                    "the new session was readable before the login committed"
                )
                assert not login.done(), "issue_token returned before its work was durable"
            finally:
                pause.resume.set()
            token = await asyncio.wait_for(login, 10)
            assert await reader.user_for_token(token) is not None, "the committed session is not usable"
            assert await reader.active_token_count(user.id) == 1

        asyncio.run(scenario())
    finally:
        monkeypatch.setattr(auth, "aiosqlite", _REAL_AIO_SQLITE)
        asyncio.run(reader.close())
        asyncio.run(store.close())


# --- A3: eviction order must not come from the wall clock ---


def test_a_backwards_clock_step_does_not_evict_the_token_being_issued(tmp_path, monkeypatch):
    """The clock is not monotonic -- a host resumes from suspend, an NTP
    correction lands, a clock is set by hand -- and a row minted after such a
    step is the newest by insertion but the oldest by ``created_at``. Evicting
    it hands the caller a token that is already gone, and the cookie carrying it
    fails on the very next request with nothing to explain it."""
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 1)
    store = _store(tmp_path)
    try:
        user = _user(store)
        first = asyncio.run(store.issue_token(user.id, 7))

        stepped_back = time.time() - 86400.0
        monkeypatch.setattr(auth, "_now", lambda: stepped_back)
        second = asyncio.run(store.issue_token(user.id, 7))

        assert asyncio.run(store.user_for_token(second)) is not None, (
            "the login evicted the session it had just issued"
        )
        assert asyncio.run(store.user_for_token(first)) is None, (
            "the cap must still be enforced, and it must evict the older session"
        )
        assert asyncio.run(store.active_token_count(user.id)) == 1
    finally:
        asyncio.run(store.close())


# --- A4: the configured service token, and an honest count ---


def test_revoke_configured_service_token_counts_exactly_what_it_changed(tmp_path):
    """The number is how an operator tells a real revocation from a no-op, so
    all three outcomes are pinned: a value with no row yet (the cold start the
    tombstone exists for), a live one, and one already dead."""
    store = _store(tmp_path)
    try:
        # Cold start: never presented, so there is no row to revoke.
        assert asyncio.run(store.revoke_configured_service_token("svc-cold", {"chat:use"})) == 1
        assert asyncio.run(store.service_token_for("svc-cold")) is None
        # Re-seeding must not hand a killed credential back.
        asyncio.run(store.ensure_bootstrap_service_token("svc-cold", {"chat:use"}, 3600))
        assert asyncio.run(store.service_token_for("svc-cold")) is None, (
            "seeding resurrected a revoked configured value"
        )
        # And a second kill of the same value is a no-op, not another change.
        assert asyncio.run(store.revoke_configured_service_token("svc-cold", {"chat:use"})) == 0

        # Live: seeded by a request and authenticating right now.
        asyncio.run(store.ensure_bootstrap_service_token("svc-live", {"chat:use"}, 3600))
        assert asyncio.run(store.service_token_for("svc-live")) is not None
        seeded = asyncio.run(
            store._fetchone(
                "SELECT created_at, expires_at FROM auth_service_tokens WHERE token_hash = ?",
                (auth.hash_token("svc-live"),),
            )
        )
        assert asyncio.run(store.revoke_configured_service_token("svc-live", {"chat:use"})) == 1
        assert asyncio.run(store.service_token_for("svc-live")) is None, "a live credential survived the kill"
        # Only the revocation is written: a kill may make a credential deader,
        # never younger.
        killed = asyncio.run(
            store._fetchone(
                "SELECT created_at, expires_at, revoked_at FROM auth_service_tokens WHERE token_hash = ?",
                (auth.hash_token("svc-live"),),
            )
        )
        assert killed["created_at"] == seeded["created_at"]
        assert killed["expires_at"] == seeded["expires_at"]
        assert killed["revoked_at"] is not None
        assert asyncio.run(store.revoke_configured_service_token("svc-live", {"chat:use"})) == 0
    finally:
        asyncio.run(store.close())


# --- A5: a cancelled write must not take the file's write lock with it ---


def test_a_cancelled_signup_does_not_lock_out_other_connections(tmp_path, monkeypatch):
    """``create_user`` rolls back on ``BaseException``, not just ``Exception``: a
    request cancelled between its INSERT and its commit has to roll back too, or
    the shared connection keeps holding the write lock and every other
    connection waits out its busy timeout and fails."""
    store = _store(tmp_path)
    other = AuthStore(store._path)
    asyncio.run(other.connect())
    # What is asserted is the refusal, not how long the loser waited for it.
    asyncio.run(other._db.execute("PRAGMA busy_timeout=200"))
    try:

        async def cancelled():
            raise asyncio.CancelledError()

        monkeypatch.setattr(store._db, "commit", cancelled)

        async def scenario():
            with pytest.raises(asyncio.CancelledError):
                await store.create_user("a@b.co", "secret12", "A", "user")
            assert await store.get_user_by_email("a@b.co") is None, "the cancelled signup became durable"
            # The reason the rollback exists: a second connection can still write.
            created = await other.create_user("b@b.co", "secret12", "B", "user")
            assert created.email == "b@b.co"

        asyncio.run(scenario())
    finally:
        asyncio.run(other.close())
        asyncio.run(store.close())


def test_a_failed_delete_user_does_not_lock_out_other_connections(tmp_path):
    """A delete that fails mid-way must not keep the file's write lock.

    The user row and its tokens used to be removed by two statements sharing one
    commit on the shared connection, with no rollback on any path. A failure
    between them left that connection holding the WAL write lock, so the other
    three workers failed every write with ``database is locked`` until this
    worker exited -- one bad request taking the whole deployment's writes down.
    """
    store = _store(tmp_path)
    other = AuthStore(store._path)
    asyncio.run(other.connect())
    asyncio.run(other._db.execute("PRAGMA busy_timeout=200"))
    try:
        user = _user(store)
        asyncio.run(store.issue_token(user.id, 7))

        conn = sqlite3.connect(store._path)
        conn.execute(
            "CREATE TRIGGER refuse_token_delete BEFORE DELETE ON auth_tokens"
            " BEGIN SELECT RAISE(ABORT, 'refused'); END;"
        )
        conn.commit()
        conn.close()

        async def scenario():
            with pytest.raises(sqlite3.IntegrityError):
                await store.delete_user(user.id)
            # The reason the rollback exists: a second connection can still write.
            created = await other.create_user("b@b.co", "secret12", "B", "user")
            assert created.email == "b@b.co"
            # And the account that could not be deleted is still there.
            assert await store.get_user(user.id) is not None
            assert await other.get_user_by_email("b@b.co") is not None

        asyncio.run(scenario())
    finally:
        asyncio.run(other.close())
        asyncio.run(store.close())