"""Tests for issue #251: the auth rate limiter must not fail open, must bound
distributed attempts per account, must not reintroduce an account-existence
oracle, active tokens must be capped and the eviction must be a real revocation,
and the service token must be scoped, expiring and rotatable.

Kept in its own module (rather than appended to test_auth.py) because the
behaviour under test is entirely about module-level limiter state and the
service-token table, and these cases each want to drive that state directly.
"""

import asyncio
import socket
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import auth
from app.auth import AuthStore


def _req(headers: dict, ip: str | None = "9.9.9.9") -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=ip) if ip else None,
        state=SimpleNamespace(),
    )


@pytest.fixture
def store(tmp_path) -> AuthStore:
    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    yield s
    asyncio.run(s.close())


def _dead_redis_url() -> str:
    """A Redis URL that is genuinely unreachable.

    Bind a socket to port 0, read the port the OS gave it, then close it: that
    port had nothing listening on it a moment ago and, with nothing else on the
    box racing for it, nothing is listening now. This is a real TCP connect
    failure, not a stubbed client, so the fallback path is exercised through the
    actual redis client and its actual exception.
    """
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    return f"redis://127.0.0.1:{port}/0"


# --- 1. the limiter must not fail open when Redis is unreachable ---


def test_redis_down_login_is_still_rate_limited_not_wide_open(store, monkeypatch):
    """With the limiter's Redis genuinely unreachable, login past the limit is
    REJECTED. Before the fix the same request was admitted: the limiter's
    except branch logged and returned, so a Redis outage was an unlimited
    credential-stuffing window."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth, "config", auth.config)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 3)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    # Point the limiter at a port with nothing on it and drop any cached client.
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    seen = []
    for _ in range(3):
        r = client.post("/api/auth/login", json={"email": "a@b.co", "password": "secret12"})
        seen.append(r.status_code)
    over = client.post("/api/auth/login", json={"email": "a@b.co", "password": "secret12"})

    assert seen == [401, 401, 401], "the first attempts are just bad credentials"
    assert over.status_code == 429, f"Redis is unreachable; the limiter must still bite, got {seen + [over.status_code]}"
    assert "Retry-After" in over.headers


def test_redis_down_signup_is_still_rate_limited(store, monkeypatch):
    """Same for signup, which is the mass-account-creation surface."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SIGNUP_RATE_PER_MIN", 2)
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    body = {"email": "a@b.co", "password": "secret12", "name": "A"}
    assert client.post("/api/auth/signup", json=body).status_code == 200
    assert client.post("/api/auth/signup", json=body).status_code == 200
    assert client.post("/api/auth/signup", json=body).status_code == 429


def test_redis_down_counter_actually_counts_upwards(monkeypatch):
    """Unit level: the fallback returns a monotonically rising count, so the
    comparison in _consume_counter rejects rather than admits."""
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)
    auth.reset_local_rate_limits()
    counts = [auth._local_rate_hit("k", 60) for _ in range(4)]
    assert counts == [1, 2, 3, 4]
    assert sum(1 for n in counts if n <= 2) == 2
    auth.reset_local_rate_limits()


def test_redis_down_check_rate_limit_rejects_past_the_limit(monkeypatch):
    """The same thing one layer up: _check_rate_limit itself, with a dead Redis,
    must raise 429 rather than return quietly."""
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    async def attempt():
        await auth._check_rate_limit(_req({}), "login", 2)

    asyncio.run(attempt())
    asyncio.run(attempt())
    with pytest.raises(HTTPException) as e:
        asyncio.run(attempt())
    assert e.value.status_code == 429

def test_local_fallback_limiter_is_bounded(monkeypatch):
    """An attacker who can force the fallback can also mint unlimited distinct
    keys; the dict must not grow with them."""
    monkeypatch.setattr(auth, "_LOCAL_RATE_MAX_KEYS", 100)
    auth.reset_local_rate_limits()
    for i in range(1000):
        auth._local_rate_hit(f"key-{i}", 60)
    assert len(auth._local_rate_counters) <= 100
    auth.reset_local_rate_limits()


def test_fail_closed_surfaces_still_answer_503(monkeypatch):
    """The public surface keeps its 503 posture: this change did not silently
    downgrade /search to in-process limiting."""
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    async def call():
        await auth._consume_counter("k", 5, 60, action="search", fail_closed=True)

    with pytest.raises(HTTPException) as e:
        asyncio.run(call())
    assert e.value.status_code == 503


# --- 2. per-account throttle ---


def test_per_account_limit_bounds_ip_rotation(store, monkeypatch):
    """Rotating source addresses must not buy an attacker a fresh bucket."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)  # per-IP OFF
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 3)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    codes = []
    for i in range(6):
        # a brand-new source address on every single attempt
        r = client.post(
            "/api/auth/login",
            json={"email": "victim@example.com", "password": "secret12"},
            headers={"X-Forwarded-For": f"10.0.0.{i}"},
        )
        codes.append(r.status_code)
    assert codes[:3] == [401, 401, 401], "per-IP limit is disabled, so only the account limit can act"
    assert codes[3:] == [429, 429, 429], f"rotating IPs must not reset the account bucket: {codes}"


def test_per_account_limit_is_per_account(store, monkeypatch):
    """The account bucket must not become a global one: a second address is
    unaffected by the first address exhausting its own bucket."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    async def attempt(email: str):
        await auth._check_account_rate_limit(_req({}), "login", 2, email)

    async def scenario():
        allowed, blocked = 0, 0
        for _ in range(3):
            try:
                await attempt("one@b.co")
            except HTTPException as e:
                assert e.status_code == 429
                blocked += 1
            else:
                allowed += 1
        # A different account is untouched by the first one's exhausted bucket.
        await attempt("two@b.co")
        return allowed, blocked

    assert asyncio.run(scenario()) == (2, 1)


def test_per_account_limit_normalises_the_address(store, monkeypatch):
    """"A@B.co" and "a@b.co" are the same account and must share one bucket,
    or the throttle is trivially sidestepped by changing case."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    async def scenario():
        await auth._check_account_rate_limit(_req({}), "login", 2, "a@b.co")
        await auth._check_account_rate_limit(_req({}), "login", 2, "a@b.co")
        with pytest.raises(HTTPException) as e:
            await auth._check_account_rate_limit(_req({}), "login", 2, "A@B.CO")
        assert e.value.status_code == 429

    asyncio.run(scenario())


# --- 3. the per-account path must not leak account existence (#276) ---


def test_per_account_limit_is_identical_for_known_and_unknown_addresses(store, monkeypatch):
    """A registered and an unregistered address must be indistinguishable
    through the limiter: same status, same body, same headers, at every attempt
    including the one that trips the limit. #276 removed the existence oracle
    from the credential check; the throttle must not put one back."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 2)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    known = asyncio.run(store.create_user("known@b.co", "secret12", "K", "user"))

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    def probe(email: str) -> list[tuple[int, str, str]]:
        out = []
        for _ in range(4):
            r = client.post("/api/auth/login", json={"email": email, "password": "wrongpw1"})
            out.append((r.status_code, r.text, r.headers.get("retry-after", "")))
        return out

    unknown_trace = probe("nosuch@b.co")
    # Reset the shared counter so the registered address starts from the same
    # state, then probe it identically.
    auth.reset_local_rate_limits()
    known_trace = probe(known.email)

    assert unknown_trace == known_trace, (
        "the per-account limiter distinguishes a registered address from an "
        f"unregistered one: {unknown_trace} vs {known_trace}"
    )
    assert known_trace[-1][0] == 429, "the limit really was reached for both"


def test_per_account_limit_does_no_account_lookup(store, monkeypatch):
    """The limit must be computable from the submitted address alone. A lookup
    keyed on existence is the oracle, so assert none happens."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    looked_up: list[str] = []
    original = store.get_user_by_email

    async def spy(email):
        looked_up.append(email)
        return await original(email)

    monkeypatch.setattr(store, "get_user_by_email", spy)

    async def scenario():
        with pytest.raises(HTTPException):
            # Over the limit: rejected before any account is consulted.
            for _ in range(4):
                await auth._check_account_rate_limit(_req({}), "login", 2, "a@b.co")

    asyncio.run(scenario())
    assert looked_up == []


# --- 4. bounded active tokens ---


def test_active_token_cap_revokes_the_oldest_token(store, monkeypatch):
    """Passing the cap must make the oldest token genuinely unusable, not merely
    absent from a listing: the check is a 401 from an authenticated route."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 3)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    tokens = [asyncio.run(store.issue_token(user.id, 7)) for _ in range(6)]
    assert asyncio.run(store.active_token_count(user.id)) == 3

    def usable(token: str) -> int:
        return client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code

    # The three oldest were revoked; the newest three still work.
    assert [usable(t) for t in tokens[:3]] == [401, 401, 401]
    assert [usable(t) for t in tokens[3:]] == [200, 200, 200]

    # And a revoked token stays dead in the store itself, not just on the route.
    assert asyncio.run(store.user_for_token(tokens[0])) is None


def test_token_cap_does_not_evict_other_users(store, monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 2)
    a = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    b = asyncio.run(store.create_user("b@b.co", "secret12", "B", "user"))
    ta = [asyncio.run(store.issue_token(a.id, 7)) for _ in range(5)]
    tb = [asyncio.run(store.issue_token(b.id, 7)) for _ in range(1)]
    assert asyncio.run(store.user_for_token(tb[0])) is not None
    assert asyncio.run(store.active_token_count(b.id)) == 1
    assert asyncio.run(store.active_token_count(a.id)) == 2
    assert asyncio.run(store.user_for_token(ta[-1])) is not None


def test_token_cap_does_not_evict_expired_rows(store, monkeypatch):
    """An already-dead row must not be chosen as the victim in preference to a
    live one; the purge loop owns dead rows, the cap owns live ones."""
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 1)
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    dead = asyncio.run(store.issue_token(user.id, 0))  # expires immediately
    live = asyncio.run(store.issue_token(user.id, 7))
    assert asyncio.run(store.user_for_token(live)) is not None
    # The dead row is still on disk, untouched: the cap only ever counts and
    # revokes unexpired tokens.
    assert asyncio.run(store.user_for_token(dead)) is None


def test_token_cap_zero_disables(store, monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 0)
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    tokens = [asyncio.run(store.issue_token(user.id, 7)) for _ in range(12)]
    assert all(asyncio.run(store.user_for_token(t)) is not None for t in tokens)


def test_login_stays_under_the_token_cap(store, monkeypatch):
    """The cap is enforced on the login path, not only when the store is called
    directly."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 2)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    client.post("/api/auth/signup", json={"email": "a@b.co", "password": "secret12", "name": "A"})

    tokens = [
        client.post("/api/auth/login", json={"email": "a@b.co", "password": "secret12"}).json()["token"]
        for _ in range(5)
    ]
    user = asyncio.run(store.get_user_by_email("a@b.co"))
    assert asyncio.run(store.active_token_count(user.id)) == 2
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens[0]}"}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens[-1]}"}).status_code == 200


# --- 5. the service token ---


def test_service_token_honours_its_expiry(store, monkeypatch):
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 1)

    async def fresh():
        req = _req({"x-service-token": "svc-abc"})
        await auth.require_auth(req)
        return req.state.user_id

    assert asyncio.run(fresh()) == auth.SERVICE_USER_ID

    # Let the seeded record's expiry pass; the env value is unchanged.
    async def expire():
        row = await store._fetchone("SELECT token_hash FROM auth_service_tokens")
        await store._db.execute("UPDATE auth_service_tokens SET expires_at = ?", (time.time() - 1,))
        await store._db.commit()
        assert row is not None

    asyncio.run(expire())

    async def after():
        await auth.require_auth(_req({"x-service-token": "svc-abc"}))

    with pytest.raises(HTTPException) as e:
        asyncio.run(after())
    assert e.value.status_code == 401, "an expired service token must not authenticate"


def test_service_token_restart_does_not_extend_its_life(store, monkeypatch):
    """Seeding is INSERT OR IGNORE, so a worker restart must not push the
    expiry out -- otherwise the expiry would be theatre."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 100)

    async def scenario():
        await store.ensure_bootstrap_service_token("svc-abc", {"chat:use"}, 100)
        first = await store.service_token_for("svc-abc")
        await store.ensure_bootstrap_service_token("svc-abc", {"chat:use"}, 100)
        second = await store.service_token_for("svc-abc")
        return first.expires_at, second.expires_at

    first, second = asyncio.run(scenario())
    assert first == second


def test_service_token_is_scoped(store, monkeypatch):
    """The shipped default scope is chat:use; the admin routes must refuse it
    even though its role is admin."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    assert client.get("/api/auth/users", headers={"X-Service-Token": "svc-abc"}).status_code == 403
    # Minting is a users:manage action, so a scoped machine credential cannot
    # mint itself a successor.
    assert client.post("/api/auth/service-tokens", headers={"X-Service-Token": "svc-abc"}).status_code == 403


def test_service_token_is_rotatable(store, monkeypatch):
    """An admin can mint a fresh token and revoke the old ones."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))
    hdr = {"Authorization": f"Bearer {admin_token}"}

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    # The env bootstrap token, minted and then rotated away.
    asyncio.run(store.ensure_bootstrap_service_token("old-svc", {"chat:use"}, 3600))
    assert client.get("/api/auth/me", headers={"X-Service-Token": "old-svc"}).status_code == 200

    minted = client.post("/api/auth/service-tokens", headers=hdr)
    assert minted.status_code == 200, minted.text
    new_token = minted.json()["token"]
    assert new_token
    assert minted.json()["scope"] == ["chat:use"]
    assert new_token != "old-svc", "rotation mints a fresh value, not the old one"

    revoked = client.post("/api/auth/service-tokens/revoke", headers=hdr)
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] == 2

    # Both are dead now, and the new one was never stored in plaintext.
    assert client.get("/api/auth/me", headers={"X-Service-Token": "old-svc"}).status_code == 401
    assert client.get("/api/auth/me", headers={"X-Service-Token": new_token}).status_code == 401
    assert asyncio.run(store.service_token_for(new_token)) is None


def test_service_token_logout_actually_revokes(store, monkeypatch):
    """Logout used to answer ok and leave the machine credential working."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    hdr = {"X-Service-Token": "svc-abc"}

    assert client.get("/api/auth/me", headers=hdr).status_code == 200
    assert client.post("/api/auth/logout", headers=hdr).status_code == 200
    assert client.get("/api/auth/me", headers=hdr).status_code == 401
    assert client.post("/api/auth/logout", headers=hdr).status_code == 401


def test_service_token_plaintext_is_never_stored(store, monkeypatch):
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    raw, _ = asyncio.run(store.issue_service_token({"chat:use"}, 3600))
    for path in (store._path, store._path + "-wal"):
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except FileNotFoundError:
            continue
        assert raw.encode() not in blob


def test_service_token_unknown_scope_names_are_dropped(monkeypatch):
    """A typo in the .env must narrow the token and warn, not silently produce
    a token whose scope is not what the .env says."""
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use", "chat:typpo", "not:apermission"))
    assert auth._service_token_scope() == frozenset({"chat:use"})
