"""The auth rate limiter must not fail open, must bound distributed attempts per
account, must not reintroduce an account-existence oracle, active tokens must be
capped and the eviction must be a real revocation, and the service token must be
scoped, expiring and rotatable.

Kept in its own module because the behaviour under test is entirely about
module-level limiter state and the service-token table.
"""

import asyncio
import socket
import time
from types import SimpleNamespace

import pytest
from conftest import auth_cookie, session_cookie_value
from fastapi import HTTPException

from app import auth
from app.auth import AuthStore


def _req(headers: dict, ip: str | None = "9.9.9.9") -> SimpleNamespace:
    """``cookies`` and ``method`` are real attributes, not conveniences: the
    credential is a session cookie and ``require_auth`` runs the same-origin
    guard, which is scoped to unsafe methods and so reads ``method``."""
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=ip) if ip else None,
        state=SimpleNamespace(),
        cookies={},
        method="GET",
    )


@pytest.fixture
def store(tmp_path) -> AuthStore:
    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    yield s
    asyncio.run(s.close())


def _dead_redis_url() -> str:
    """Bind a socket to port 0, read the OS-assigned port, then close it: nothing
    is listening there, so this is a real TCP connect failure through the actual
    redis client and its actual exception, not a stub."""
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
    REJECTED: an except branch that logs and returns would make a Redis outage
    an unlimited credential-stuffing window."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth, "config", auth.config)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 3)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
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
    """The fallback returns a monotonically rising count, so the comparison in
    _consume_counter rejects rather than admits."""
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)
    auth.reset_local_rate_limits()
    counts = [auth._local_rate_hit("k", 60) for _ in range(4)]
    assert counts == [1, 2, 3, 4]
    assert sum(1 for n in counts if n <= 2) == 2
    auth.reset_local_rate_limits()


def test_redis_down_check_rate_limit_rejects_past_the_limit(monkeypatch):
    """_check_rate_limit itself, with a dead Redis, must raise 429 rather than
    return quietly."""
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
    """An attacker who can force the fallback can mint unlimited distinct keys;
    the dict must not grow with them."""
    monkeypatch.setattr(auth, "_LOCAL_RATE_MAX_KEYS", 100)
    auth.reset_local_rate_limits()
    for i in range(1000):
        auth._local_rate_hit(f"key-{i}", 60)
    assert len(auth._local_rate_counters) <= 100
    auth.reset_local_rate_limits()


def test_fail_closed_surfaces_still_answer_503(monkeypatch):
    """The public surface keeps its 503 posture: /search is not downgraded to
    in-process limiting."""
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
    # TestClient's peer is "testclient", not an IP, so the shipped auto rule
    # (trust X-Forwarded-For only from a loopback peer) would discard the header
    # and every request below would land in one bucket. This mirrors the
    # deployed topology, where nginx forwards from 127.0.0.1.
    monkeypatch.setattr(auth.config, "AUTH_TRUST_X_FORWARDED_FOR", True)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    # Prove the rotation is real before relying on it.
    def ip_seen_by_the_limiter(xff: str) -> str:
        return auth._client_ip(
            auth.Request(
                {"type": "http", "headers": [(b"x-forwarded-for", xff.encode())], "client": ("testclient", 50000)}
            )
        )

    assert len({ip_seen_by_the_limiter(f"10.0.0.{i}") for i in range(6)}) == 6, (
        "the header is not being honoured, so this test would not be rotating IPs at all"
    )

    codes = [
        client.post(
            "/api/auth/login",
            json={"email": "victim@example.com", "password": "secret12"},
            headers={"X-Forwarded-For": f"10.0.0.{i}"},  # a new address every attempt
        ).status_code
        for i in range(6)
    ]
    assert codes[:3] == [401, 401, 401], "per-IP limit is disabled, so only the account limit can act"
    assert codes[3:] == [429, 429, 429], f"rotating IPs must not reset the account bucket: {codes}"


def test_a_correct_password_is_never_rate_limited(store, monkeypatch):
    """A bucket fed by every attempt, and consulted BEFORE the credential check,
    lets an anonymous caller deny a known address access indefinitely: send the
    limit's worth of wrong passwords from rotating addresses and the real owner
    is refused with 429 no matter what they type."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)  # per-IP OFF
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 3)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "AUTH_TRUST_X_FORWARDED_FOR", True)
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    client.post("/api/auth/signup", json={"email": "owner@corp.example", "password": "secret12", "name": "O"})

    for i in range(10):
        r = client.post(
            "/api/auth/login",
            json={"email": "owner@corp.example", "password": "wrongpw1"},
            headers={"X-Forwarded-For": f"10.0.0.{i}"},
        )
        assert r.status_code in (401, 429)

    owner = client.post(
        "/api/auth/login",
        json={"email": "owner@corp.example", "password": "secret12"},
        headers={"X-Forwarded-For": "203.0.113.9"},
    )
    assert owner.status_code == 200, f"the throttle locked the owner out: {owner.status_code} {owner.text}"
    assert owner.json()["user"]["email"] == "owner@corp.example"


def test_a_correct_password_does_not_consume_the_budget(store, monkeypatch):
    """Successful logins must not count, or a user signing in from home and work
    walks themselves into their own 429."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", 2)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "AUTH_TRUST_X_FORWARDED_FOR", True)
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 100)
    monkeypatch.setattr(auth.config, "REDIS_URL", _dead_redis_url())
    monkeypatch.setattr(auth, "_rate_client", None)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    client.post("/api/auth/signup", json={"email": "a@b.co", "password": "secret12", "name": "A"})

    codes = [
        client.post(
            "/api/auth/login",
            json={"email": "a@b.co", "password": "secret12"},
            headers={"X-Forwarded-For": f"10.0.1.{i}"},
        ).status_code
        for i in range(6)
    ]
    assert codes == [200] * 6, f"correct logins must never be throttled: {codes}"


def test_per_account_bucket_survives_a_key_flood(monkeypatch):
    """Flooding the fallback with fresh keys must not let an attacker discard the
    very bucket throttling them. Because the dict is kept least-recently-used
    and a hit re-inserts its key at the tail, the attacked bucket is the last
    one eviction would reach; dropping by insertion order instead would discard
    it and hand the attacker a fresh count."""
    monkeypatch.setattr(auth, "_LOCAL_RATE_MAX_KEYS", 50)
    auth.reset_local_rate_limits()
    victim = "auth:rl:acct:login:victim@example.com"

    seen = []
    for i in range(500):
        auth._local_rate_hit(f"auth:rl:login:10.0.0.{i}", 600)
        seen.append(auth._local_rate_hit(victim, 600))

    assert len(auth._local_rate_counters) <= 50
    assert seen == list(range(1, 501)), "the per-account bucket was reset by the key flood"
    assert auth._local_rate_hit(victim, 600) == 501
    auth.reset_local_rate_limits()


def test_dead_service_tokens_are_purged_but_the_configured_tombstone_is_kept(store, monkeypatch):
    """Rotation must not grow the table without bound -- but reaping the
    configured value's row would let the next request re-seed it and hand a
    revoked or expired credential straight back, so that one row is kept."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-configured")

    async def scenario():
        await store.ensure_bootstrap_service_token("svc-configured", {"chat:use"}, 3600)
        minted = [(await store.issue_service_token({"chat:use"}, 3600))[0] for _ in range(3)]
        await store.revoke_service_token(minted[0])
        await store._db.execute(
            "UPDATE auth_service_tokens SET expires_at = ? WHERE token_hash != ?",
            (time.time() - 1, auth.hash_token("svc-configured")),
        )
        await store._db.commit()

        async def remaining():
            rows = await store._fetchall("SELECT token_hash FROM auth_service_tokens")
            return {r["token_hash"] for r in rows}

        before = await remaining()
        purged = await store.purge_dead_service_tokens(
            keep_hash=auth.hash_token("svc-configured")
        )
        return purged, before, await remaining()

    purged, before, after = asyncio.run(scenario())
    assert purged == 3
    assert len(before) == 4
    assert after == {auth.hash_token("svc-configured")}


def test_reaping_the_tombstone_would_resurrect_it(store, monkeypatch):
    """Why the exclusion exists: with the row deleted, the configured value
    re-seeds with a full fresh lifetime."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-configured")

    async def scenario():
        await store.ensure_bootstrap_service_token("svc-configured", {"chat:use"}, 3600)
        await store.revoke_service_token("svc-configured")
        assert await store.service_token_for("svc-configured") is None
        await store.purge_dead_service_tokens()
        assert await store.service_token_for("svc-configured") is None
        req = _req({"x-service-token": "svc-configured"})
        await auth.require_auth(req)
        return req.state.user_id

    assert asyncio.run(scenario()) == auth.SERVICE_USER_ID, (
        "deleting the tombstone let the revoked configured token back in"
    )


def test_reaping_keeps_a_dead_configured_token_dead(store, monkeypatch):
    """The exclusion must hold for a configured token that is ALREADY dead: the
    row survives as a tombstone and presenting it is still a 401. Without the
    exclusion in the SQL the row is reaped and INSERT OR IGNORE silently hands
    the credential a fresh lifetime, permanently."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-configured")

    async def scenario():
        await store.ensure_bootstrap_service_token("svc-configured", {"chat:use"}, 3600)
        await store.revoke_service_token("svc-configured")

        purged = await store.purge_dead_service_tokens(
            keep_hash=auth.hash_token("svc-configured")
        )
        row = await store._fetchone(
            "SELECT token_hash, revoked_at FROM auth_service_tokens WHERE token_hash = ?",
            (auth.hash_token("svc-configured"),),
        )
        # Nothing was reaped, and the tombstone is still there.
        assert purged == 0
        assert row is not None and row["revoked_at"] is not None

        # Presenting it is still a 401, not a silently revived credential.
        with pytest.raises(HTTPException) as e:
            await auth.require_auth(_req({"x-service-token": "svc-configured"}))
        assert e.value.status_code == 401
        return True

    assert asyncio.run(scenario())


def test_token_purge_loop_reaps_both_token_tables(store, monkeypatch):
    """The background reaper must actually drive both purges, and must pass the
    configured value's hash through as the tombstone to keep."""
    calls = []

    async def record_user_purge():
        calls.append("user")
        return 0

    async def record_service_purge(keep_hash=""):
        calls.append(("service", keep_hash))
        return 0

    class StopLoop(Exception):
        pass

    async def stop_after_one_iteration(_interval):
        raise StopLoop

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_TOKEN_PURGE_INTERVAL_SECONDS", 60)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-configured")
    monkeypatch.setattr(store, "purge_expired_tokens", record_user_purge)
    monkeypatch.setattr(store, "purge_dead_service_tokens", record_service_purge)
    monkeypatch.setattr(auth.asyncio, "sleep", stop_after_one_iteration)

    async def run():
        try:
            await auth.token_purge_loop()
        except StopLoop:
            pass

    asyncio.run(run())
    assert calls == ["user", ("service", auth.hash_token("svc-configured"))]


def test_per_account_limit_is_per_account(store, monkeypatch):
    """A second account must be unaffected by the first one exhausting its
    bucket."""
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


# --- 3. the per-account path must not leak account existence ---


def test_per_account_limit_is_identical_for_known_and_unknown_addresses(store, monkeypatch):
    """A registered and an unregistered address must be indistinguishable
    through the limiter: same status, same body, same headers, at every attempt
    including the one that trips the limit."""
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
    # Reset the shared counter so the registered address starts from the same state.
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
            for _ in range(4):
                await auth._check_account_rate_limit(_req({}), "login", 2, "a@b.co")

    asyncio.run(scenario())
    assert looked_up == []


# --- 4. bounded active tokens ---


def test_active_token_cap_revokes_the_oldest_token(store, monkeypatch):
    """Passing the cap must make the oldest token genuinely unusable, not merely
    absent from a listing."""
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
        return client.get("/api/auth/me", cookies=auth_cookie(token)).status_code

    assert [usable(t) for t in tokens[:3]] == [401, 401, 401]
    assert [usable(t) for t in tokens[3:]] == [200, 200, 200]

    # A revoked token stays dead in the store itself, not just on the route.
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
    """The cap must only ever count and revoke UNEXPIRED rows. Two live tokens
    against a cap of one, so "nothing was evicted" and "the wrong row was
    evicted" cannot look identical from the outside."""
    monkeypatch.setattr(auth.config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 1)
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    dead = asyncio.run(store.issue_token(user.id, 0))  # expires immediately
    oldest_live = asyncio.run(store.issue_token(user.id, 7))
    newest_live = asyncio.run(store.issue_token(user.id, 7))

    assert asyncio.run(store.user_for_token(oldest_live)) is None
    assert asyncio.run(store.user_for_token(newest_live)) is not None

    # The oldest LIVE token went, not the dead row: asserted against the table,
    # because user_for_token returns None for any expired row.
    async def stored_hashes():
        rows = await store._fetchall("SELECT token_hash FROM auth_tokens WHERE user_id = ?", (user.id,))
        return {r["token_hash"] for r in rows}

    assert auth.hash_token(dead) in asyncio.run(stored_hashes()), "the cap must not touch expired rows"
    assert auth.hash_token(newest_live) in asyncio.run(stored_hashes())
    assert auth.hash_token(oldest_live) not in asyncio.run(stored_hashes())


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
        session_cookie_value(client.post("/api/auth/login", json={"email": "a@b.co", "password": "secret12"}))
        for _ in range(5)
    ]
    user = asyncio.run(store.get_user_by_email("a@b.co"))
    assert asyncio.run(store.active_token_count(user.id)) == 2
    assert client.get("/api/auth/me", cookies=auth_cookie(tokens[0])).status_code == 401
    assert client.get("/api/auth/me", cookies=auth_cookie(tokens[-1])).status_code == 200


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


def _chat_route_permissions() -> set[str]:
    """Read from the router rather than hardcoded, so this test tracks the real
    dependency: a literal would keep passing if chat.py's permission changed."""
    from app.chat import router

    found = set()
    for dep in router.dependencies:
        closure = getattr(dep.dependency, "__closure__", None) or ()
        for cell in closure:
            if isinstance(cell.cell_contents, str):
                found.add(cell.cell_contents)
    assert found, "could not read a permission off the chat router; update this test deliberately"
    return found


def test_service_token_default_scope_covers_the_eval_scripts(store, monkeypatch):
    """The one in-repo consumer, backend/scripts/eval_runner.py, sends the raw
    AUTH_SERVICE_TOKEN and only ever calls /api/chat, so the shipped default
    scope must COVER what app/chat.py's router requires. Fails if either side
    drifts."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-from-env")
    # Pin the shipped default explicitly, not whatever load_dotenv() loaded.
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))

    required = _chat_route_permissions()
    scope = auth._service_token_scope()
    assert required, "app/chat.py's router declares no permission to check"
    assert scope, "the default service-token scope is empty; nothing would work"
    assert required <= scope, (
        f"the default scope {sorted(scope)} does not cover what app/chat.py "
        f"requires {sorted(required)}; eval_runner.py would get 403 on every chat turn"
    )

    async def scenario():
        req = _req({"x-service-token": "svc-from-env"})
        await auth.require_auth(req)
        for permission in required:
            await auth.require_permission(permission)(req)
        return req.state.user_id

    assert asyncio.run(scenario()) == auth.SERVICE_USER_ID


def test_default_scope_would_not_cover_a_permission_chat_now_requires(store, monkeypatch):
    """The negative control: the scope check really is an intersection with
    what the routes require, so a route demanding more than the default scope
    is genuinely refused. Without this the test above would also pass if scope
    were ignored entirely."""
    from fastapi import HTTPException as _HTTPException

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-from-env")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))

    async def scenario():
        req = _req({"x-service-token": "svc-from-env"})
        await auth.require_auth(req)
        with pytest.raises(_HTTPException) as e:
            await auth.require_permission("users:manage")(req)
        return e.value.status_code

    assert asyncio.run(scenario()) == 403


def test_rate_limiter_pins_its_redis_db(monkeypatch):
    """The shipped REDIS_URL points at DB 0, which deploys flush; an unpinned
    limiter kept its security counters there and a flush reset every bucket."""
    captured = {}

    def fake_from_url(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(auth.config, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(auth.config, "AUTH_RATE_LIMIT_REDIS_DB", 3)
    monkeypatch.setattr(auth, "_rate_client", None)
    monkeypatch.setattr(auth.aioredis, "from_url", fake_from_url)

    assert auth._rate_redis() == "client"
    assert captured.get("db") == 3, "the limiter's db is not pinned; it will follow REDIS_URL"


def test_rate_limiter_db_default_is_not_the_flushed_cache_db():
    """The shipped default must not be DB 0, or pinning it would be pointless."""
    # Read the getenv default from the source, not the ambient config.
    import inspect
    import re as _re

    from app import config as config_module

    source = inspect.getsource(config_module.Config)
    match = _re.search(r'AUTH_RATE_LIMIT_REDIS_DB\s*=\s*int\(os\.getenv\([^,]+,\s*"(\d+)"\)', source)
    assert match, "AUTH_RATE_LIMIT_REDIS_DB must have a numeric default"
    assert int(match.group(1)) != 0, "the rate limiter must not default to the deploy-flushed cache DB"


def test_revoked_service_token_is_not_resurrected_by_re_seeding(store, monkeypatch):
    """Revocation must stick even though the value is still in the environment.
    The seed on a store miss is INSERT OR IGNORE, so it cannot overwrite a
    revoked row's tombstone; if it ever could, every revoked configured token
    would come back to life with a full fresh lifetime."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    async def scenario():
        first = _req({"x-service-token": "svc-abc"})
        await auth.require_auth(first)
        await store.revoke_service_token("svc-abc")
        # Presented again, with the environment still naming the same value.
        for _ in range(3):
            second = _req({"x-service-token": "svc-abc"})
            try:
                await auth.require_auth(second)
            except HTTPException as e:
                assert e.status_code == 401
            else:
                raise AssertionError("a revoked service token came back to life")
        row = await store._fetchone("SELECT revoked_at FROM auth_service_tokens")
        return row is not None and row["revoked_at"] is not None

    assert asyncio.run(scenario())


def test_cold_start_tombstone_survives_the_reaper(store, monkeypatch):
    """The reaper must not collect the tombstone written by a cold-start kill:
    it is a dead row, so the reaper's predicate selects it, and deleting it
    would let the next request re-seed the configured value live again."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "cold-svc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    killed = client.post(
        "/api/auth/service-tokens/revoke",
        json={"token": "cold-svc"},
        cookies=auth_cookie(admin_token),
    )
    assert killed.json()["revoked"] == 1

    # Drive the REAL reaper: a re-implementation of its keep_hash here would
    # keep passing even if the loop stopped passing the exclusion.
    class StopLoop(Exception):
        pass

    async def stop_after_one_iteration(_interval):
        raise StopLoop

    monkeypatch.setattr(auth.config, "AUTH_TOKEN_PURGE_INTERVAL_SECONDS", 60)
    monkeypatch.setattr(auth.asyncio, "sleep", stop_after_one_iteration)

    async def run_reaper():
        try:
            await auth.token_purge_loop()
        except StopLoop:
            pass

    asyncio.run(run_reaper())

    row = asyncio.run(
        store._fetchone("SELECT revoked_at FROM auth_service_tokens WHERE token_hash = ?", (auth.hash_token("cold-svc"),))
    )
    assert row is not None and row["revoked_at"] is not None, "the reaper collected the configured value's tombstone"
    assert client.get("/api/auth/me", headers={"X-Service-Token": "cold-svc"}).status_code == 401


def test_service_token_restart_does_not_extend_its_life(store, monkeypatch):
    """Seeding is INSERT OR IGNORE, so a worker restart must not push the expiry
    out -- otherwise the expiry would be theatre."""
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
    assert client.post("/api/auth/service-tokens", headers={"X-Service-Token": "svc-abc"}).status_code == 403


def test_service_token_rotation_in_the_safe_order(store, monkeypatch):
    """Rotation must work in the order an operator actually reaches for: mint
    the replacement, move consumers onto it, then retire the old one -- with no
    window in which no credential works."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))
    hdr = auth_cookie(admin_token)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    # The credential in service today.
    asyncio.run(store.ensure_bootstrap_service_token("old-svc", {"chat:use"}, 3600))
    assert client.get("/api/auth/me", headers={"X-Service-Token": "old-svc"}).status_code == 200

    # 1. Mint the replacement; both work, so nothing has an outage.
    minted = client.post("/api/auth/service-tokens", cookies=hdr)
    assert minted.status_code == 200, minted.text
    new_token = minted.json()["token"]
    assert new_token
    assert minted.json()["scope"] == ["chat:use"]
    assert new_token != "old-svc", "rotation mints a fresh value, not the old one"
    assert client.get("/api/auth/me", headers={"X-Service-Token": new_token}).status_code == 200

    # 2. Retire the old one BY VALUE, which must not take the new one with it.
    retired = client.post("/api/auth/service-tokens/revoke", json={"token": "old-svc"}, cookies=hdr)
    assert retired.status_code == 200, retired.text
    assert retired.json()["revoked"] == 1
    assert client.get("/api/auth/me", headers={"X-Service-Token": "old-svc"}).status_code == 401
    assert client.get("/api/auth/me", headers={"X-Service-Token": new_token}).status_code == 200



def test_kill_switch_works_before_the_configured_token_is_ever_used(store, monkeypatch):
    """Seeding is lazy, so a configured value that has never been presented has
    no row to UPDATE: a plain revoke reports 0 rows and the next request seeds
    the credential live with a full fresh lifetime. Both forms must tombstone
    it instead."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "cold-svc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))
    hdr = auth_cookie(admin_token)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    async def row_count():
        return len(await store._fetchall("SELECT token_hash FROM auth_service_tokens"))

    # Never presented: no row exists, so there is nothing for an UPDATE to hit.
    assert asyncio.run(row_count()) == 0

    killed = client.post("/api/auth/service-tokens/revoke", json={"token": "cold-svc"}, cookies=hdr)
    assert killed.status_code == 200, killed.text
    assert killed.json()["revoked"] == 1, "a cold configured token must be killable, not a silent no-op"
    assert asyncio.run(row_count()) == 1, "a revoked tombstone must exist to block re-seeding"

    for _ in range(3):
        assert client.get("/api/auth/me", headers={"X-Service-Token": "cold-svc"}).status_code == 401
    assert asyncio.run(store.service_token_for("cold-svc")) is None


def test_kill_everything_also_kills_an_unused_configured_token(store, monkeypatch):
    """The no-body containment call must reach the configured value too, from a
    cold start -- that is the form documented for a suspected leak."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "cold-svc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    killed = client.post("/api/auth/service-tokens/revoke", cookies=auth_cookie(admin_token))
    assert killed.status_code == 200, killed.text
    assert killed.json()["revoked"] == 1
    assert client.get("/api/auth/me", headers={"X-Service-Token": "cold-svc"}).status_code == 401


def test_revoke_all_service_tokens_kills_every_one(store, monkeypatch):
    """The no-body form is the suspected-leak response: everything dies, the
    replacement included. That is the deliberate difference between the two."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))
    hdr = auth_cookie(admin_token)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    asyncio.run(store.ensure_bootstrap_service_token("old-svc", {"chat:use"}, 3600))
    new_token = client.post("/api/auth/service-tokens", cookies=hdr).json()["token"]

    # A POST with NO body must mean revoke-everything, not a 422 for a missing body.
    revoked = client.post("/api/auth/service-tokens/revoke", cookies=hdr)
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["revoked"] == 2
    assert client.get("/api/auth/me", headers={"X-Service-Token": "old-svc"}).status_code == 401
    assert client.get("/api/auth/me", headers={"X-Service-Token": new_token}).status_code == 401
    assert asyncio.run(store.service_token_for(new_token)) is None


def test_revoke_reports_what_it_actually_revoked(store, monkeypatch):
    """``revoked`` is the operator's only confirmation that a retirement took,
    so revoking something unknown or already dead must say 0, not a reassuring
    1."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    admin = asyncio.run(store.create_user("admin@b.co", "secret12", "A", "admin"))
    admin_token = asyncio.run(store.issue_token(admin.id, 7))
    hdr = auth_cookie(admin_token)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)

    minted = client.post("/api/auth/service-tokens", cookies=hdr).json()["token"]
    first = client.post("/api/auth/service-tokens/revoke", json={"token": minted}, cookies=hdr)
    assert first.json()["revoked"] == 1
    # Same token again, and one that never existed: nothing changed.
    again = client.post("/api/auth/service-tokens/revoke", json={"token": minted}, cookies=hdr)
    assert again.json()["revoked"] == 0
    never = client.post("/api/auth/service-tokens/revoke", json={"token": "never-existed"}, cookies=hdr)
    assert never.json()["revoked"] == 0


def test_service_token_logout_actually_revokes(store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    app = FastAPI()
    app.include_router(auth.router)
    client = TestClient(app)
    # A machine credential is a HEADER, and deliberately still one: a browser
    # cannot be made to attach this.
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


def test_service_token_ttl_comes_from_config_and_never_from_nowhere(monkeypatch):
    """A non-positive configuration must fall back to the default rather than
    meaning 'never expires', or an operator setting 0 quietly reinstates a
    permanent admin credential."""
    # No assertion on the ambient config value: load_dotenv() populates it from
    # whatever backend/.env on the host says.
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 60)
    assert auth._service_token_ttl_seconds() == 60
    for non_positive in (0, -1, -86400):
        monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", non_positive)
        assert auth._service_token_ttl_seconds() == 86400, (
            f"AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS={non_positive} must not disable the expiry"
        )


def test_seeded_service_token_row_uses_the_configured_ttl(store, monkeypatch):
    """The TTL must reach the stored row at seed time, not merely exist in
    config: assert the row's own expiry is the configured distance away."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-abc")
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_SCOPE", ("chat:use",))
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 3600)

    async def scenario():
        await auth.require_auth(_req({"x-service-token": "svc-abc"}))
        return await store._fetchone("SELECT created_at, expires_at FROM auth_service_tokens")

    row = asyncio.run(scenario())
    # +/- a second of slack for the float round-trip through SQLite.
    assert 3590 < row["expires_at"] - row["created_at"] <= 3601
