import asyncio
import re
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import auth
from app.auth import AuthStore, DuplicateEmailError, StoredUser, bootstrap_admin


def _headers(*pairs) -> dict:
    return {k: v for k, v in pairs}


def _req(headers: dict, state=None) -> SimpleNamespace:
    return SimpleNamespace(headers=headers, client=None, state=state or SimpleNamespace())


@pytest.fixture
def store(tmp_path) -> AuthStore:
    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    yield s
    asyncio.run(s.close())


def test_password_hashes_never_plaintext(store):
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    assert user.password_hash != "secret12"
    assert re.match(r"^\$2[ab]\$", user.password_hash)  # bcrypt format
    assert auth.verify_password("secret12", user.password_hash)
    assert not auth.verify_password("wrong12", user.password_hash)
    # hash-only on disk: raw password must not appear in the db or its WAL
    for path in (store._path, store._path + "-wal"):
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except FileNotFoundError:
            continue
        assert b"secret12" not in blob


def test_validate_email_rejects_and_normalizes():
    assert auth.validate_email("  A@B.Co ") == "a@b.co"
    for bad in ("", "not-an-email", "a@b", "a b@c.co", "@c.co", "a@" + "x" * 300 + ".co"):
        with pytest.raises(HTTPException) as e:
            auth.validate_email(bad)
        assert e.value.status_code == 422


def test_validate_password_rules():
    auth.validate_password("password1")  # ok
    for bad in ("", "short1", "nodigits", "12345678", "alllower1" * 20):
        with pytest.raises(HTTPException) as e:
            auth.validate_password(bad)
        assert e.value.status_code == 422


def test_validate_name_rules():
    assert auth.validate_name("  Alice  ") == "Alice"
    with pytest.raises(HTTPException) as e:
        auth.validate_name("n" * 61)
    assert e.value.status_code == 422
    with pytest.raises(HTTPException) as e:
        auth.validate_name("bad\x00name")
    assert e.value.status_code == 422


def test_tokens_hashed_stored_revoked_expire(store):
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    raw = asyncio.run(store.issue_token(user.id, 7))
    # raw token never persisted anywhere (WAL sidecar included)
    for path in (store._path, store._path + "-wal"):
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except FileNotFoundError:
            continue
        assert raw.encode() not in blob
    # the stored token is the SHA-256 hash
    row = asyncio.run(store._fetchone(
        "SELECT token_hash FROM auth_tokens WHERE user_id = ?", (user.id,)))
    assert row is not None and row["token_hash"] == auth.hash_token(raw)

    resolved = asyncio.run(store.user_for_token(raw))
    assert resolved is not None and resolved.id == user.id

    # expiry: back-date the token
    asyncio.run(store._db.execute(
        "UPDATE auth_tokens SET expires_at = ? WHERE token_hash = ?",
        (auth._now() - 1, auth.hash_token(raw)),
    ))
    asyncio.run(store._db.commit())
    assert asyncio.run(store.user_for_token(raw)) is None

    # revoke
    raw2 = asyncio.run(store.issue_token(user.id, 7))
    asyncio.run(store.revoke_token(raw2))
    assert asyncio.run(store.user_for_token(raw2)) is None


def test_disabled_user_token_rejected(store):
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    raw = asyncio.run(store.issue_token(user.id, 7))
    asyncio.run(store.update_user(user.id, None, None, False))
    assert asyncio.run(store.user_for_token(raw)) is None


def test_multiple_tokens_and_revoke_all(store):
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    t1 = asyncio.run(store.issue_token(user.id, 7))
    t2 = asyncio.run(store.issue_token(user.id, 7))
    asyncio.run(store.revoke_all_tokens(user.id))
    assert asyncio.run(store.user_for_token(t1)) is None
    assert asyncio.run(store.user_for_token(t2)) is None


def test_email_unique_and_case_insensitive(store):
    asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    dup = asyncio.run(store.get_user_by_email("A@B.CO"))
    assert dup is not None
    assert dup.email == "a@b.co"


def test_bootstrap_admin_created_once(store, monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "adminpass1")
    monkeypatch.setattr(auth, "store", store)
    asyncio.run(bootstrap_admin())
    admin = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert admin is not None and admin.role == "admin" and admin.is_active
    # never overwrites an existing account (password stays verifiable)
    asyncio.run(store.set_password(admin.id, auth.hash_password("newpass1")))
    asyncio.run(bootstrap_admin())
    again = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert auth.verify_password("newpass1", again.password_hash)


def test_bootstrap_admin_disabled_without_env(store, monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "")
    monkeypatch.setattr(auth, "store", store)
    asyncio.run(bootstrap_admin())
    assert asyncio.run(store.list_users()) == []


def test_role_permissions_matrix():
    perms = auth.ROLE_PERMISSIONS
    assert "chat:use" in perms["user"] and "analytics:read" not in perms["user"]
    assert {"chat:use", "analytics:read", "users:read", "users:manage"} <= perms["admin"]


@pytest.mark.parametrize(
    "role,perm,expected",
    [
        ("user", "chat:use", None),
        ("user", "analytics:read", 403),
        ("admin", "analytics:read", None),
        ("admin", "users:manage", None),
        ("admin", "chat:use", None),
    ],
)
def test_require_permission(role, perm, expected):
    async def check():
        checker = auth.require_permission(perm)
        req = _req({}, state=SimpleNamespace(user=SimpleNamespace(role=role)))
        return await checker(req)

    if expected is None:
        asyncio.run(check())
    else:
        with pytest.raises(HTTPException) as e:
            asyncio.run(check())
        assert e.value.status_code == expected


def test_require_permission_without_auth_is_401():
    async def check():
        checker = auth.require_permission("chat:use")
        return await checker(_req({}, state=SimpleNamespace()))

    with pytest.raises(HTTPException) as e:
        asyncio.run(check())
    assert e.value.status_code == 401


def test_require_auth_accepts_bearer_and_rejects_missing(store, monkeypatch):
    monkeypatch.setattr(auth, "store", store)
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    raw = asyncio.run(store.issue_token(user.id, 7))

    async def with_bearer():
        req = _req({"authorization": f"Bearer {raw}"})
        await auth.require_auth(req)
        return req.state.user_id

    assert asyncio.run(with_bearer()) == user.id

    async def expect_401(headers: dict):
        await auth.require_auth(_req(headers))

    for bad_headers in ({}, {"authorization": "Bearer garbage"}):
        with pytest.raises(HTTPException) as e:
            asyncio.run(expect_401(bad_headers))
        assert e.value.status_code == 401


def test_service_token_acts_as_admin(monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-tok-123")
    req = _req({"x-service-token": "svc-tok-123"})
    asyncio.run(auth.require_auth(req))
    assert req.state.user_id == auth.SERVICE_USER_ID
    assert req.state.user.role == "admin"
    # wrong service token falls through to 401 (no user token)
    async def wrong():
        await auth.require_auth(_req({"x-service-token": "nope"}))
    with pytest.raises(HTTPException) as e:
        asyncio.run(wrong())
    assert e.value.status_code == 401


def test_rate_limit_429_and_reset(monkeypatch):
    calls = {"n": 0}

    class _FakeRedis:
        async def set(self, key, value, nx=False, ex=None):
            # First-hit window establishment; for the test we just need it to not
            # raise so the subsequent incr drives the count.
            return True

        async def incr(self, key):
            calls["n"] += 1
            return calls["n"]

        async def expire(self, key, ttl):
            return True

    monkeypatch.setattr(auth, "_rate_client", _FakeRedis())
    monkeypatch.setattr(auth, "_client_ip", lambda r: "1.2.3.4")
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 2)

    async def attempt():
        await auth._check_rate_limit(_req({}), "login", auth.config.AUTH_LOGIN_RATE_PER_MIN)

    asyncio.run(attempt())
    asyncio.run(attempt())
    with pytest.raises(HTTPException) as e:
        asyncio.run(attempt())
    assert e.value.status_code == 429
    assert "Retry-After" in e.value.headers


def test_rate_limit_disabled_when_zero(monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)
    asyncio.run(auth._check_rate_limit(_req({}), "login", 0))  # must not raise


def test_last_admin_protected(store, monkeypatch):
    monkeypatch.setattr(auth, "store", store)
    admin = asyncio.run(store.create_user("admin@x.co", "adminpass1", "Admin", "admin"))
    assert asyncio.run(store.count_admins()) == 1
    with pytest.raises(HTTPException) as e:
        asyncio.run(_patch_guard(admin.id, "user"))
    assert e.value.status_code == 400
    # adding a second admin then demoting the first is fine
    second = asyncio.run(store.create_user("a2@x.co", "adminpass1", "A2", "admin"))
    asyncio.run(store.update_user(admin.id, None, "user", None))
    assert asyncio.run(store.get_user(admin.id)).role == "user"
    asyncio.run(store.update_user(second.id, None, None, False))
    assert not asyncio.run(store.get_user(second.id)).is_active


async def _patch_guard(user_id: str, role: str):
    s = auth._require_auth_store()
    target = await s.get_user(user_id)
    if target.role == "admin" and await s.count_admins() <= 1:
        raise HTTPException(status_code=400, detail="cannot demote or deactivate the last admin")
    await s.update_user(user_id, None, role, None)


# --- HTTP-level endpoint tests ---


def _auth_app(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    auth.store = s
    app = FastAPI()
    app.include_router(auth.router)
    return TestClient(app), s


def test_signup_login_me_flow(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        r = client.post("/api/auth/signup", json={"email": "  New@Example.com ", "password": "secret12", "name": "Alice"})
        assert r.status_code == 200
        data = r.json()
        assert data["token"]
        assert data["user"]["email"] == "new@example.com"
        assert data["user"]["role"] == "user"

        # me with the issued token
        assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {data['token']}"}).json()["email"] == "new@example.com"

        # login returns a fresh token
        login = client.post("/api/auth/login", json={"email": "new@example.com", "password": "secret12"})
        assert login.status_code == 200
        assert login.json()["user"]["email"] == "new@example.com"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_validation_errors(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        cases = [
            {"email": "not-an-email", "password": "secret12"},
            {"email": "", "password": "secret12"},
            {"email": "a@b.co", "password": "short1"},
            {"email": "a@b.co", "password": "alllower"},
            {"email": "a@b.co", "password": "12345678"},
            {"email": "a@b.co", "password": "secret12", "name": "n" * 61},
        ]
        for body in cases:
            assert client.post("/api/auth/signup", json=body).status_code == 422, body
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_duplicate_email_409(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        assert client.post("/api/auth/signup", json={"email": "dup@x.co", "password": "secret12"}).status_code == 200
        assert client.post("/api/auth/signup", json={"email": "DUP@x.co", "password": "secret12"}).status_code == 409
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_login_invalid_credentials_identical_401(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        client.post("/api/auth/signup", json={"email": "a@x.co", "password": "secret12"})
        bad1 = client.post("/api/auth/login", json={"email": "a@x.co", "password": "wrong12"})
        bad2 = client.post("/api/auth/login", json={"email": "nobody@x.co", "password": "secret12"})
        assert bad1.status_code == bad2.status_code == 401
        assert bad1.json() == bad2.json()  # identical message: no account enumeration
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_me_requires_auth_and_logout_revokes(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        assert client.get("/api/auth/me").status_code == 401
        token = client.post("/api/auth/signup", json={"email": "a@x.co", "password": "secret12"}).json()["token"]
        h = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/auth/me", headers=h).status_code == 200
        assert client.post("/api/auth/logout", headers=h).json() == {"ok": True}
        assert client.get("/api/auth/me", headers=h).status_code == 401  # token revoked
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_change_password_invalidates_other_tokens(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        token = client.post("/api/auth/signup", json={"email": "a@x.co", "password": "secret12"}).json()["token"]
        h = {"Authorization": f"Bearer {token}"}

        wrong = client.post("/api/auth/change-password", headers=h, json={"current_password": "nope12", "new_password": "secret21"})
        assert wrong.status_code == 400

        ok = client.post("/api/auth/change-password", headers=h, json={"current_password": "secret12", "new_password": "secret21"})
        assert ok.status_code == 200
        new_token = ok.json()["token"]
        # old token was revoked, the new one works
        assert client.get("/api/auth/me", headers=h).status_code == 401
        assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_token}"}).status_code == 200
        # and the new password logs in
        assert client.post("/api/auth/login", json={"email": "a@x.co", "password": "secret21"}).status_code == 200
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_concurrent_create_duplicate_race_no_poison(tmp_path):
    """Two separate connections racing to INSERT the same UNIQUE email must yield
    exactly one success and one DuplicateEmailError. Each attempt gets its own
    AuthStore/connection: a single SQLite connection must never be shared across
    concurrent coroutines (that was the source of the flakiness)."""
    db_path = str(tmp_path / "auth.db")

    async def attempt(name: str):
        s = AuthStore(db_path)
        await s.connect()
        try:
            return await s.create_user("race@x.co", "secret12", name, "user")
        finally:
            await s.close()

    async def main():
        # Prime the schema once on a throwaway connection.
        prime = AuthStore(db_path)
        await prime.connect()
        await prime.close()

        results = await asyncio.gather(attempt("A"), attempt("B"), return_exceptions=True)
        ok = [r for r in results if isinstance(r, StoredUser)]
        dup = [r for r in results if isinstance(r, DuplicateEmailError)]
        assert len(ok) == 1 and len(dup) == 1
        # the failed insert must not have poisoned either connection (use a fresh one)
        after = AuthStore(db_path)
        await after.connect()
        try:
            u = await after.create_user("after@x.co", "secret12", "C", "user")
            token = await after.issue_token(u.id, 7)
            assert (await after.user_for_token(token)).id == u.id
        finally:
            await after.close()
            auth.store = None

    asyncio.run(main())


def test_signup_duplicate_race_returns_409(tmp_path):
    """Two concurrent duplicate signups must yield exactly one 200 and one 409.

    The race is exercised as two coroutines on a single event loop (via an
    ASGI transport) rather than two OS threads sharing one TestClient — the
    latter is flaky because the store's SQLite connection is bound to the
    thread it was opened on, so a cross-thread request can error out
    intermittently. Coroutines keep the connection on one loop while still
    racing the INSERTs. The store is opened/closed inside the same coroutine so
    its aiosqlite connection stays bound to the loop that runs the requests."""
    import asyncio

    import httpx
    from fastapi import FastAPI

    async def main():
        s = AuthStore(str(tmp_path / "auth.db"))
        await s.connect()
        auth.store = s
        try:
            app = FastAPI()
            app.include_router(auth.router)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                results = await asyncio.gather(
                    client.post("/api/auth/signup", json={"email": "same@x.co", "password": "secret12"}),
                    client.post("/api/auth/signup", json={"email": "same@x.co", "password": "secret12"}),
                )
                assert sorted(r.status_code for r in results) == [200, 409]
        finally:
            await s.close()
            auth.store = None

    asyncio.run(main())


def test_bootstrap_admin_survives_duplicate_and_lock_races(tmp_path, monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "adminpass1")

    async def main():
        store = AuthStore(str(tmp_path / "auth.db"))
        await store.connect()
        auth.store = store
        try:
            # another worker already created it -> DuplicateEmailError path is safe
            original = store.create_user

            async def racy_create(email, password, name, role):
                await original(email, password, name, role)
                raise DuplicateEmailError(email)

            monkeypatch.setattr(store, "create_user", racy_create)
            await auth.bootstrap_admin()  # must not raise

            # lock contention -> retries then gives up gracefully instead of failing
            async def locked_create(email, password, name, role):
                raise sqlite3.OperationalError("database is locked")

            monkeypatch.setattr(store, "create_user", locked_create)
            await auth.bootstrap_admin()  # must not raise
        finally:
            await store.close()
            auth.store = None

    asyncio.run(main())


def test_admin_user_management_rbac(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        user_token = client.post("/api/auth/signup", json={"email": "user@x.co", "password": "secret12"}).json()["token"]
        admin_token = client.post("/api/auth/signup", json={"email": "boss@x.co", "password": "secret12", "name": "Boss"}).json()["token"]
        asyncio.run(s.update_user(asyncio.run(s.get_user_by_email("boss@x.co")).id, None, "admin", None))

        uh = {"Authorization": f"Bearer {user_token}"}
        ah = {"Authorization": f"Bearer {admin_token}"}

        # regular users cannot read or manage users
        assert client.get("/api/auth/users", headers=uh).status_code == 403
        assert client.patch("/api/auth/users/some-id", headers=uh, json={"role": "user"}).status_code == 403
        # admins can list; non-existent id -> 404
        listing = client.get("/api/auth/users", headers=ah)
        assert listing.status_code == 200 and len(listing.json()) == 2
        assert client.get("/api/auth/users/nope", headers=ah).status_code == 404
        # invalid role -> 422
        uid = listing.json()[0]["id"]
        assert client.patch(f"/api/auth/users/{uid}", headers=ah, json={"role": "superuser"}).status_code == 422
        # promote the user
        user_id = asyncio.run(s.get_user_by_email("user@x.co")).id
        assert client.patch(f"/api/auth/users/{user_id}", headers=ah, json={"role": "admin"}).status_code == 200
        # revoke all tokens
        assert client.post(f"/api/auth/users/{user_id}/tokens/revoke", headers=ah).json() == {"ok": True}
        assert client.get("/api/auth/me", headers=uh).status_code == 401
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_verify_password_malformed_hash_returns_false():
    """A malformed stored password hash raises ValueError inside bcrypt, which
    verify_password must swallow as a plain False (ERROR PATH — malformed stored
    password hash)."""
    assert auth.verify_password("secret12", "not-a-bcrypt-hash") is False
    assert auth.verify_password("secret12", "") is False


def test_create_user_generic_error_rolls_back_and_raises(store, monkeypatch):
    """A non-integrity INSERT failure must roll back so the connection never
    holds an open write transaction, then re-raise (ERROR PATH — SQLite write
    failure)."""
    calls = {"rollback": 0}
    orig_execute = store._db.execute

    async def fake_execute(query, params=()):
        if query.startswith("INSERT INTO users"):
            raise RuntimeError("disk full")
        return await orig_execute(query, params)

    async def fake_rollback():
        calls["rollback"] += 1

    monkeypatch.setattr(store._db, "execute", fake_execute)
    monkeypatch.setattr(store._db, "rollback", fake_rollback)

    with pytest.raises(RuntimeError):
        asyncio.run(store.create_user("x@b.co", "secret12", "X", "user"))
    assert calls["rollback"] == 1


def test_update_user_no_op_and_name_update(store, monkeypatch):
    """update_user with no fields issues no SQL; a name update persists (lines
    284-285, 293)."""
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    execs = []
    orig_execute = store._db.execute

    async def fake_execute(query, params=()):
        execs.append(query)
        return await orig_execute(query, params)

    monkeypatch.setattr(store._db, "execute", fake_execute)
    asyncio.run(store.update_user(user.id, None, None, None))
    assert execs == []  # empty update is a true no-op

    asyncio.run(store.update_user(user.id, "New Name", None, None))
    assert len(execs) == 1 and execs[0].startswith("UPDATE users SET name = ?")
    assert asyncio.run(store.get_user(user.id)).name == "New Name"


def test_delete_user_removes_tokens_and_user(store):
    """delete_user removes the user and cascades its tokens (lines 299-301)."""
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    token = asyncio.run(store.issue_token(user.id, 7))
    assert asyncio.run(store.user_for_token(token)) is not None
    asyncio.run(store.delete_user(user.id))
    assert asyncio.run(store.get_user(user.id)) is None
    assert asyncio.run(store.user_for_token(token)) is None


def test_issue_token_error_rolls_back_and_raises(store, monkeypatch):
    """A token INSERT failure must roll back before re-raising (ERROR PATH —
    SQLite write failure)."""
    calls = {"rollback": 0}
    orig_execute = store._db.execute

    async def fake_execute(query, params=()):
        if query.startswith("INSERT INTO auth_tokens"):
            raise RuntimeError("db gone")
        return await orig_execute(query, params)

    async def fake_rollback():
        calls["rollback"] += 1

    monkeypatch.setattr(store._db, "execute", fake_execute)
    monkeypatch.setattr(store._db, "rollback", fake_rollback)

    with pytest.raises(RuntimeError):
        asyncio.run(store.issue_token("u1", 7))
    assert calls["rollback"] == 1


def test_require_auth_store_uninitialized_503(monkeypatch):
    monkeypatch.setattr(auth, "store", None)
    with pytest.raises(HTTPException) as e:
        auth._require_auth_store()
    assert e.value.status_code == 503


def test_client_ip_x_forwarded_for_and_fallback():
    """_client_ip honors X-Forwarded-For only behind a trusted proxy, else the
    real socket peer, then 'unknown'."""
    # Default (no trusted proxy): XFF is ignored, the socket peer is authoritative.
    req = _req({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
    req.client = SimpleNamespace(host="1.2.3.4")
    assert auth._client_ip(req) == "1.2.3.4"
    # Behind a trusted proxy: first XFF hop wins.
    auth.config.AUTH_TRUST_X_FORWARDED_FOR = True
    try:
        req = _req({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
        assert auth._client_ip(req) == "203.0.113.9"
        req = _req({"x-forwarded-for": " 203.0.113.9 "})
        assert auth._client_ip(req) == "203.0.113.9"
    finally:
        auth.config.AUTH_TRUST_X_FORWARDED_FOR = False
    # socket peer fallback
    req = _req({})
    req.client = SimpleNamespace(host="1.2.3.4")
    assert auth._client_ip(req) == "1.2.3.4"
    # no peer at all -> "unknown"
    assert auth._client_ip(_req({})) == "unknown"


def _promote_to_admin(client, s, email):
    uid = asyncio.run(s.get_user_by_email(email)).id
    asyncio.run(s.update_user(uid, None, "admin", None))
    return uid


def test_get_user_endpoint(tmp_path):
    """GET /api/auth/users/{id} returns the requested user (line 548)."""
    client, s = _auth_app(tmp_path)
    try:
        token = client.post("/api/auth/signup", json={"email": "boss@x.co", "password": "secret12"}).json()["token"]
        uid = _promote_to_admin(client, s, "boss@x.co")
        ah = {"Authorization": f"Bearer {token}"}
        r = client.get(f"/api/auth/users/{uid}", headers=ah)
        assert r.status_code == 200
        assert r.json()["email"] == "boss@x.co"
        assert r.json()["role"] == "admin"
        assert client.get("/api/auth/users/nope", headers=ah).status_code == 404
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_patch_user_last_admin_guard_endpoint(tmp_path):
    """PATCH cannot demote or deactivate the last active admin (line 566) —
    ERROR PATH — self-lockout protection."""
    client, s = _auth_app(tmp_path)
    try:
        token = client.post("/api/auth/signup", json={"email": "boss@x.co", "password": "secret12"}).json()["token"]
        uid = _promote_to_admin(client, s, "boss@x.co")
        ah = {"Authorization": f"Bearer {token}"}
        assert client.patch(f"/api/auth/users/{uid}", headers=ah, json={"role": "user"}).status_code == 400
        assert client.patch(f"/api/auth/users/{uid}", headers=ah, json={"is_active": False}).status_code == 400
        # once a second admin exists the guard releases
        asyncio.run(s.create_user("other@x.co", "secret12", "O", "admin"))
        assert client.patch(f"/api/auth/users/{uid}", headers=ah, json={"role": "user"}).status_code == 200
        assert asyncio.run(s.get_user(uid)).role == "user"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_delete_user_last_admin_guard_endpoint(tmp_path):
    """DELETE cannot remove the last active admin (lines 580-585) — ERROR PATH —
    self-lockout protection."""
    client, s = _auth_app(tmp_path)
    try:
        token = client.post("/api/auth/signup", json={"email": "boss@x.co", "password": "secret12"}).json()["token"]
        uid = _promote_to_admin(client, s, "boss@x.co")
        ah = {"Authorization": f"Bearer {token}"}
        assert client.delete(f"/api/auth/users/{uid}", headers=ah).status_code == 400
        # a second admin releases the guard
        asyncio.run(s.create_user("other@x.co", "secret12", "O", "admin"))
        assert client.delete(f"/api/auth/users/{uid}", headers=ah).status_code == 200
        assert asyncio.run(s.get_user(uid)) is None
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_bootstrap_admin_gives_up_after_write_lock_retries(store, monkeypatch):
    """bootstrap_admin retries 5 times on a write lock, sleeping between attempts,
    then gives up gracefully instead of failing startup (lines 620-624)."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "adminpass1")
    monkeypatch.setattr(auth, "store", store)

    calls = {"create": 0, "sleep": []}

    async def locked_create(email, password, name, role):
        calls["create"] += 1
        raise sqlite3.OperationalError("database is locked")

    async def fake_sleep(seconds):
        calls["sleep"].append(seconds)

    monkeypatch.setattr(store, "create_user", locked_create)
    monkeypatch.setattr(auth.asyncio, "sleep", fake_sleep)

    asyncio.run(auth.bootstrap_admin())  # must not raise
    assert calls["create"] == 5  # 5 attempts before giving up
    assert calls["sleep"] == [1, 1, 1, 1]  # slept between attempts 0-3
