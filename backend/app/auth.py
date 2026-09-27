"""Token + RBAC authentication for the API.

Signup issues no token and always answers the same thing (see the endpoint);
login issues opaque bearer tokens (hashed with SHA-256 in storage, expiring
after AUTH_TOKEN_TTL_DAYS, individually revocable, multiple per user).
A role-based access-control layer maps roles to permissions; endpoints assert
the permission they need via ``require_permission``. A bootstrap admin account
is seeded from AUTH_ADMIN_EMAIL / AUTH_ADMIN_PASSWORD at startup.

Roles:
- ``admin`` — everything (chat, analytics, user management)
- ``user``  — chat only (the public-signup default)

Machine clients (eval scripts) may authenticate with the AUTH_SERVICE_TOKEN
header, which is a SCOPED, EXPIRING credential rather than an unconditional
admin bypass: it carries an explicit permission set, stops working after
AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS, and can be revoked or rotated. All inputs
are validated server-side.
"""

import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

import aiosqlite
import bcrypt
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.config import config

logger = logging.getLogger("auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Module-level store; set by main.lifespan (and by tests).
store: "AuthStore | None" = None

VALID_ROLES = ("admin", "user")

# The only role public self-service signup can ever grant. Not configurable:
# see the signup docstring.
SIGNUP_ROLE = "user"

# Role -> permissions. Single source of truth for access control; add a
# resource-scoped permission here and assert it on the route that needs it.
ROLE_PERMISSIONS: ClassVar[dict[str, set[str]]] = {
    "admin": {"chat:use", "analytics:read", "users:read", "users:manage"},
    "user": {"chat:use"},
}

# Id used as the user_id for service-token requests (eval scripts, ops).
SERVICE_USER_ID = "service-token"

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
# bcrypt silently truncates its input at 72 bytes; cap there so validation
# matches what bcrypt actually hashes (otherwise two distinct long passwords
# can collide on their shared 72-byte prefix).
_BCRYPT_MAX_BYTES = 72


class SignupIn(BaseModel):
    email: str = ""
    password: str = ""
    name: str = ""


class LoginIn(BaseModel):
    email: str = ""
    password: str = ""


class ChangePasswordIn(BaseModel):
    current_password: str = ""
    new_password: str = ""


class UserPatchIn(BaseModel):
    name: str | None = None
    role: str | None = None
    is_active: bool | None = None


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: float

    @classmethod
    def from_user(cls, u: "StoredUser") -> "UserOut":
        return cls(
            id=u.id,
            email=u.email,
            name=u.name,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at,
        )


class AuthOut(BaseModel):
    token: str
    user: UserOut


class SignupOut(BaseModel):
    """Response of a successful-looking ``POST /api/auth/signup``.

    Deliberately carries neither a token nor a user record: whether the
    address was free or already registered, the response is this one fixed
    message, so the endpoint cannot be used to confirm that an address has
    an account here. Callers follow up with ``POST /api/auth/login``.
    """

    message: str


# The single response body every accepted signup gets. It must state the two
# possible outcomes without favouring one: this app has no confirmation-email
# flow, so a returning user who re-submits a registered address gets no mail
# and no error -- the message is their recovery route ("just sign in with
# your existing password"), and it is deliberately the same string a brand new
# address receives.
SIGNUP_ACCEPTED_MESSAGE = (
    "If this email is not already registered, your account is ready. "
    "Sign in with your email and password to continue; if you already have "
    "an account, sign in with your existing password."
)


@dataclass
class StoredUser:
    id: str
    email: str
    password_hash: str
    name: str
    role: str
    is_active: bool
    created_at: float


@dataclass
class StoredServiceToken:
    """A machine credential (X-Service-Token) as stored.

    ``scope`` is the explicit set of permissions the token may exercise --
    it is NOT the full admin permission set, and it is enforced per route by
    ``require_permission``. ``expires_at`` bounds the credential's life, so a
    leaked service token stops working on its own instead of forever.
    """

    token_hash: str
    scope: frozenset[str]
    created_at: float
    expires_at: float


class ServiceTokenOut(BaseModel):
    """Response of a service-token mint. The plaintext value is returned
    exactly once, at mint time; only its SHA-256 is ever stored."""

    token: str
    scope: list[str]
    expires_at: float


class ServiceTokenRevokeIn(BaseModel):
    """Which service token to retire. Omit ``token`` to revoke all of them."""

    token: str = ""


def validate_email(email: str) -> str:
    """Normalize + validate an email address, raising 422 on any violation."""
    email = (email or "").strip().lower()
    if not email or len(email) > config.AUTH_MAX_EMAIL_LEN or not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="invalid email address")
    return email


def validate_password(password: str) -> str:
    """Validate a password (length + letter/digit), raising 422 on violation."""
    if not password or len(password) < config.AUTH_PASSWORD_MIN_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"password must be at least {config.AUTH_PASSWORD_MIN_LEN} characters",
        )
    if len(password.encode("utf-8")) > _BCRYPT_MAX_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"password too long (max {_BCRYPT_MAX_BYTES} bytes)",
        )
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise HTTPException(status_code=422, detail="password must contain a letter and a digit")
    return password


def validate_name(name: str) -> str:
    """Trim + validate an optional display name, raising 422 on violation."""
    name = (name or "").strip()
    if len(name) > config.AUTH_MAX_NAME_LEN:
        raise HTTPException(status_code=422, detail=f"name too long (max {config.AUTH_MAX_NAME_LEN} chars)")
    if any(ord(c) < 32 for c in name):
        raise HTTPException(status_code=422, detail="name contains invalid characters")
    return name


def _password_bytes(password: str) -> bytes:
    """Normalize a password to exactly the bytes bcrypt will hash, so the
    set and verify paths agree (bcrypt truncates at 72 bytes)."""
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_password_bytes(password), hashed.encode("utf-8"))
    except ValueError:
        return False


# A bcrypt hash of a per-process random secret, computed once at import by the
# same ``hash_password`` -- and therefore the same ``gensalt()`` cost factor --
# that produced every stored hash. Logging in against an address that has no
# account verifies the supplied password against this constant, so that path
# costs the same wall-clock time as a wrong-password login. Without it the
# missing short-circuit is a remote account-existence oracle even though both
# paths return the identical 401 body. It must never be a cheaper hash: the
# cost factor is the whole point, so it is derived rather than hard-coded. The
# secret is discarded immediately and is never a valid password for anyone.
_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


class DuplicateEmailError(Exception):
    """Raised when an INSERT hits the users.email UNIQUE constraint (e.g. two
    gunicorn workers bootstrapping the same admin concurrently)."""


class AuthStore:
    """SQLite-backed user + token store (WAL mode, same concurrency discipline
    as the chat store). Token values are never stored plaintext."""

    def __init__(self, path: str):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id)"
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_service_tokens (
                token_hash TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                revoked_at REAL
            )
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _fetchone(self, query: str, params: tuple = ()):
        rows = await self._db.execute_fetchall(query, params)
        return rows[0] if rows else None

    async def _fetchall(self, query: str, params: tuple = ()):
        return await self._db.execute_fetchall(query, params)

    @staticmethod
    def _to_user(row) -> StoredUser:
        return StoredUser(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"],
            name=row["name"] or "",
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=float(row["created_at"]),
        )

    async def create_user(self, email: str, password: str, name: str, role: str) -> StoredUser:
        password_hash = await asyncio.to_thread(hash_password, password)
        user = StoredUser(
            id=uuid.uuid4().hex,
            email=email,
            password_hash=password_hash,
            name=name,
            role=role,
            is_active=True,
            created_at=_now(),
        )
        try:
            await self._db.execute(
                "INSERT INTO users (id, email, password_hash, name, role, is_active, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user.id, user.email, user.password_hash, user.name, user.role, 1, user.created_at),
            )
            await self._db.commit()
        except sqlite3.IntegrityError:
            # Duplicate email under concurrency (e.g. concurrent worker
            # bootstrap). Roll back so the failed statement never leaves this
            # connection holding an open write transaction (which would poison
            # the whole DB with "database is locked").
            await self._db.rollback()
            raise DuplicateEmailError(email) from None
        except Exception:
            await self._db.rollback()
            raise
        return user

    async def get_user_by_email(self, email: str) -> StoredUser | None:
        row = await self._fetchone("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,))
        return self._to_user(row) if row else None

    async def get_user(self, user_id: str) -> StoredUser | None:
        row = await self._fetchone("SELECT * FROM users WHERE id = ?", (user_id,))
        return self._to_user(row) if row else None

    async def list_users(self) -> list[UserOut]:
        rows = await self._fetchall("SELECT * FROM users ORDER BY created_at DESC")
        return [UserOut.from_user(self._to_user(r)) for r in rows]

    async def update_user(
        self,
        user_id: str,
        name: str | None,
        role: str | None,
        is_active: bool | None,
        guard_last_admin: bool = False,
    ) -> int:
        # Whitelist the columns that may be set so a future caller can never inject
        # a user-controlled column name into the SQL via f-string interpolation.
        allowed: dict[str, object] = {"name": name, "role": role, "is_active": is_active}
        sets, params = [], []
        for col, val in allowed.items():
            if val is None:
                continue
            sets.append(f"{col} = ?")
            if col == "is_active":
                params.append(1 if val else 0)
            else:
                params.append(val)
        if not sets:
            return 1
        params.append(user_id)
        where = "id = ?"
        if guard_last_admin:
            # Atomic guard: block only when this update would remove the last
            # remaining admin (demote to user or deactivate). A single statement
            # keeps the check and the write free of a count-then-set race.
            where += (
                " AND NOT (role = 'admin'"
                " AND (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1"
                " AND (? OR ?))"
            )
            params.append(1 if role == "user" else 0)
            params.append(1 if is_active is False else 0)
        cur = await self._db.execute(
            f"UPDATE users SET {', '.join(sets)} WHERE {where}", tuple(params)
        )
        await self._db.commit()
        return cur.rowcount

    async def delete_user(self, user_id: str, guard_last_admin: bool = False) -> int:
        where = "id = ?"
        params: tuple = (user_id,)
        if guard_last_admin:
            where += (
                " AND NOT (role = 'admin'"
                " AND (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1)"
            )
        cur = await self._db.execute(f"DELETE FROM users WHERE {where}", params)
        n = cur.rowcount
        if n:
            await self._db.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
        await self._db.commit()
        return n

    async def set_password(self, user_id: str, password_hash: str) -> None:
        await self._db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        await self._db.commit()

    async def count_admins(self) -> int:
        row = await self._fetchone("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
        return int(row["n"]) if row else 0

    async def issue_token(self, user_id: str, ttl_days: int) -> str:
        """Mint a bearer token and return it in plaintext (only its SHA-256 is
        stored). Enforces the per-user active-token cap, so a caller cannot
        grow the table by logging in repeatedly -- see ``_enforce_token_cap``.
        """
        raw = secrets.token_urlsafe(32)
        created = _now()
        expires = created + ttl_days * 86400
        try:
            await self._db.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (hash_token(raw), user_id, created, expires),
            )
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise
        await self._enforce_token_cap(user_id)
        return raw

    async def _enforce_token_cap(self, user_id: str) -> int:
        """Keep at most ``config.AUTH_MAX_ACTIVE_TOKENS_PER_USER`` unexpired
        tokens per user, revoking the oldest surplus. Returns how many were
        revoked.

        The surplus rows are DELETED, not just hidden: ``user_for_token``
        resolves a token by looking its hash up in this very table, so a
        deleted row means the credential no longer authenticates (401) the
        moment it is evicted. Leaving the row in place and merely not listing
        it would keep a live credential alive, which is the opposite of what a
        cap is for.

        Only unexpired rows count toward the cap and are candidates for
        eviction, so already-dead rows (the purge loop's job) neither occupy a
        slot nor get churned by this.
        """
        cap = int(getattr(config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 0))
        if cap <= 0:
            return 0
        now = _now()
        # rowid breaks ties between rows minted in the same clock tick so the
        # eviction order is deterministic rather than storage-dependent.
        rows = await self._fetchall(
            "SELECT token_hash FROM auth_tokens"
            " WHERE user_id = ? AND expires_at >= ?"
            " ORDER BY created_at ASC, rowid ASC",
            (user_id, now),
        )
        surplus = len(rows) - cap
        if surplus <= 0:
            return 0
        victims = [r["token_hash"] for r in rows[:surplus]]
        for token_hash in victims:
            await self._db.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (token_hash,))
        await self._db.commit()
        logger.info("auth: revoked %d token(s) over the per-user active-token cap", len(victims))
        return len(victims)

    async def active_token_count(self, user_id: str) -> int:
        """Number of the user's unexpired tokens. Exposed for the admin surface
        and for tests; not used to make an access decision."""
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM auth_tokens WHERE user_id = ? AND expires_at >= ?",
            (user_id, _now()),
        )
        return int(row["n"]) if row else 0

    # --- service tokens (machine credentials, X-Service-Token) ---

    async def issue_service_token(self, scope: set[str], ttl_seconds: float) -> tuple[str, StoredServiceToken]:
        """Mint a scoped, expiring machine credential. Returns the plaintext
        alongside its stored record; only the hash is persisted, so the caller
        gets exactly one chance to keep it."""
        raw = secrets.token_urlsafe(32)
        created = _now()
        record = StoredServiceToken(
            token_hash=hash_token(raw),
            scope=frozenset(scope),
            created_at=created,
            expires_at=created + ttl_seconds,
        )
        await self._db.execute(
            "INSERT INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
            " VALUES (?, ?, ?, ?, NULL)",
            (record.token_hash, ",".join(sorted(record.scope)), record.created_at, record.expires_at),
        )
        await self._db.commit()
        return raw, record

    async def ensure_bootstrap_service_token(self, raw: str, scope: set[str], ttl_seconds: float) -> None:
        """Seed the record for the env-configured ``AUTH_SERVICE_TOKEN``.

        INSERT OR IGNORE, and deliberately so: re-running this on every worker
        restart must NOT push the expiry out, or the credential would be
        eternal in practice and the expiry would be theatre. Rotating means
        revoking the row (or changing the env value, which hashes differently
        and seeds a new row) so the next call seeds a fresh lifetime.
        """
        # One timestamp for both columns: reading the clock twice would make
        # created_at and expires_at disagree by however long the two calls
        # straddle, which is not a lifetime anyone can reason about.
        created = _now()
        await self._db.execute(
            "INSERT OR IGNORE INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
            " VALUES (?, ?, ?, ?, NULL)",
            (hash_token(raw), ",".join(sorted(scope)), created, created + ttl_seconds),
        )
        await self._db.commit()

    async def service_token_for(self, raw: str) -> StoredServiceToken | None:
        """Resolve a machine credential, or None when it is unknown, revoked or
        expired. A revoked or expired token is a hard no: there is no
        'but the env still says so' path back in."""
        row = await self._fetchone(
            "SELECT token_hash, scope, created_at, expires_at FROM auth_service_tokens"
            " WHERE token_hash = ? AND revoked_at IS NULL",
            (hash_token(raw),),
        )
        if row is None or float(row["expires_at"]) < _now():
            return None
        return StoredServiceToken(
            token_hash=row["token_hash"],
            scope=frozenset(p for p in (row["scope"] or "").split(",") if p),
            created_at=float(row["created_at"]),
            expires_at=float(row["expires_at"]),
        )

    async def revoke_service_token(self, raw: str) -> None:
        await self._db.execute(
            "UPDATE auth_service_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_now(), hash_token(raw)),
        )
        await self._db.commit()

    async def revoke_all_service_tokens(self) -> int:
        cur = await self._db.execute(
            "UPDATE auth_service_tokens SET revoked_at = ? WHERE revoked_at IS NULL", (_now(),)
        )
        await self._db.commit()
        return cur.rowcount

    async def user_for_token(self, raw_token: str) -> StoredUser | None:
        """Resolve a raw bearer token to an active user, or None when the token
        is unknown, expired, or the account is disabled."""
        row = await self._fetchone(
            "SELECT * FROM auth_tokens WHERE token_hash = ?", (hash_token(raw_token),)
        )
        if row is None or float(row["expires_at"]) < _now():
            return None
        user = await self.get_user(row["user_id"])
        if user is None or not user.is_active:
            return None
        return user

    async def revoke_token(self, raw_token: str) -> None:
        await self._db.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (hash_token(raw_token),))
        await self._db.commit()

    async def revoke_all_tokens(self, user_id: str) -> None:
        await self._db.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
        await self._db.commit()

    async def purge_expired_tokens(self) -> int:
        """Delete rows whose expiry has passed. Parameterized to avoid SQL
        injection; returns the number of rows removed. Run periodically so the
        table can't grow without bound as tokens expire."""
        now = _now()
        cur = await self._db.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (now,))
        await self._db.commit()
        return cur.rowcount


def _require_auth_store() -> AuthStore:
    if store is None:
        raise HTTPException(status_code=503, detail="auth store not initialized")
    return store


# --- rate limiting (Redis-backed, per-IP) ---

_rate_client = None


def _rate_redis() -> aioredis.Redis:
    global _rate_client
    if _rate_client is None:
        _rate_client = aioredis.from_url(
            config.REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )
    return _rate_client


async def close_rate_redis() -> None:
    """Close the lazily-created auth rate-limit Redis client. Registered as a
    shutdown hook so the connection isn't leaked on worker exit."""
    global _rate_client
    if _rate_client is not None:
        await _rate_client.aclose()
        _rate_client = None


async def token_purge_loop() -> None:
    """Background task: purge expired auth_tokens rows. Never raises. Disabled
    when AUTH_TOKEN_PURGE_INTERVAL_SECONDS is 0 (e.g. tests)."""
    interval = config.AUTH_TOKEN_PURGE_INTERVAL_SECONDS
    while interval > 0:
        try:
            if store is not None:
                n = await store.purge_expired_tokens()
                if n:
                    logger.info("auth: purged %d expired token(s)", n)
        except Exception:
            logger.exception("auth token purge failed")
        await asyncio.sleep(interval)


def _peer_is_local_proxy(peer: str | None) -> bool:
    """True when the socket peer is a reverse proxy on this same host, i.e. a
    loopback address. A peer that is not a real IP (a test ASGI transport, for
    instance) is not loopback, so it never widens the trust."""
    if not peer:
        return False
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


def _trust_forwarded_for(peer: str | None) -> bool:
    """Whether X-Forwarded-For may be trusted for this request's peer.

    ``config.AUTH_TRUST_X_FORWARDED_FOR`` forces the answer when set to true or
    false. The shipped default is None ("auto"), which trusts the header only
    for a loopback peer: the reference deployment (setup.sh) always runs behind
    nginx forwarding from 127.0.0.1, so its clients get correct per-IP rate
    limiting out of the box, while a client connecting straight to the API
    port is its own non-loopback peer and cannot forge a header to escape its
    own bucket.
    """
    configured = config.AUTH_TRUST_X_FORWARDED_FOR
    if configured is not None:
        return configured
    return _peer_is_local_proxy(peer)


def _client_ip(request: Request) -> str:
    """Client IP. The X-Forwarded-For header is only honored when the request
    came through a trusted reverse proxy (see ``_trust_forwarded_for``), so a
    raw client cannot spoof its IP (e.g. for bypassing rate limits). Otherwise
    the socket peer wins.

    nginx ``$proxy_add_x_forwarded_for`` APPENDS ``$remote_addr`` (the real
    peer) to any client-supplied X-Forwarded-For list, so the *rightmost* hop
    is the one added by the trusted proxy while the leftmost is attacker
    controlled; take the rightmost."""
    peer = request.client.host if request.client else None
    if _trust_forwarded_for(peer):
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return peer or "unknown"


async def _check_rate_limit(
    request: Request,
    action: str,
    limit_per_min: int,
    *,
    key_prefix: str = "auth:rl",
    window_seconds: int | None = None,
    fail_closed: bool = False,
) -> None:
    """Enforce a per-IP rate limit with Redis INCR+EXPIRE.

    ``fail_closed`` selects what happens when the limiter's Redis is
    unreachable. The auth endpoints keep the default (False) and fall back to
    a bounded in-process limiter -- see ``_consume_counter``; the public search
    surface passes ``fail_closed=True`` and is answered 503 instead, because an
    unrated request against ``/search`` or ``/analytics/click`` is precisely
    the full-corpus scraping and analytics-poisoning vector these limits exist
    to close -- there, no answer is not an acceptable fallback.

    ``window_seconds`` defaults to the auth window so the existing auth call
    sites keep their current 60s window; the public limits pass their own.
    """
    if limit_per_min <= 0:
        return
    window = config.AUTH_RATE_WINDOW_SECONDS if window_seconds is None else window_seconds
    key = f"{key_prefix}:{action}:{_client_ip(request)}"
    await _consume_counter(key, limit_per_min, window, action=action, fail_closed=fail_closed)


async def _check_account_rate_limit(
    request: Request,
    action: str,
    limit_per_min: int,
    account: str,
) -> None:
    """Enforce a rate limit keyed on the submitted account rather than the
    source address, so rotating IPs cannot buy an attacker a fresh bucket.

    The key is the address folded to lower case and stripped, so one account
    cannot be handed a second bucket by changing case or padding -- the
    invariant is enforced here rather than left to the caller. Nothing about
    whether that address has an account here feeds into the decision: the
    counter is incremented and compared for a registered and an unregistered
    address alike, and no lookup is performed to compute it, so the counter's
    state, the 429 it yields and the work done to reach that answer are
    identical either way. A per-account limit that consulted the users table
    would reintroduce exactly the account-existence oracle that the dummy-hash
    login path exists to close.
    """
    key_account = (account or "").strip().lower()
    if limit_per_min <= 0 or not key_account:
        return
    await _consume_counter(
        f"auth:rl:acct:{action}:{key_account}",
        limit_per_min,
        config.AUTH_RATE_WINDOW_SECONDS,
        action=action,
        fail_closed=False,
    )


async def _consume_counter(
    key: str,
    limit: int,
    window: int,
    *,
    action: str,
    fail_closed: bool,
) -> None:
    """Count one hit against ``key`` and reject past ``limit``.

    Redis is the shared counter, so the limit is global across every worker
    process. When it is unreachable the choice is between three behaviours and
    the two obvious ones are both wrong on their own:

    * admit the request (the historical behaviour here) makes a Redis outage
      -- which an attacker can often induce, and which lasts exactly as long
      as they want -- into an unlimited credential-stuffing window on the
      login and signup endpoints. That is the hole this function exists to
      close.
    * reject every request turns the same outage into a total login outage.
      Credential stuffing is an attacker's problem; a Redis blip locking every
      legitimate user out of their own account is the defender's, and it hands
      a denial-of-service to whoever can disturb Redis without helping anyone
      guess a password.

    So the fallback is a third option: a bounded in-process limiter with the
    SAME limit. The degraded posture is "single-process limiting" rather than
    "no limiting" or "no service". It stays bounded under an unbounded key
    flood (see ``_local_rate_hit``), and it is not a hidden fail-open: past the
    limit the caller still gets 429, exactly as it would from Redis.

    The per-worker weakening is inherent to an in-process fallback and is
    accepted deliberately: a gunicorn deployment multiplies the effective limit
    by its worker count during a Redis outage, which is still a finite bound
    rather than none.
    """
    try:
        rc = _rate_redis()
        # Establish the sliding window atomically on the first hit: SET NX EX sets
        # the value to 0 with the window TTL only if the key did not already
        # exist, so the key always has a TTL. A later crash can never leave a
        # counter with no expiry (which would block the IP forever under the old
        # INCR + separate EXPIRE). Subsequent hits just increment.
        await rc.set(key, 0, nx=True, ex=window)
        n = await rc.incr(key)
    except Exception:
        if fail_closed:
            logger.exception("rate limiter unavailable for %s; failing closed", action)
            raise HTTPException(
                status_code=503,
                detail="Rate limiter unavailable",
                headers={"Retry-After": str(window)},
            ) from None
        logger.warning("rate limiter unavailable for %s; using the in-process fallback limiter", action, exc_info=True)
        n = _local_rate_hit(key, window)
    if n > limit:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please try again shortly.",
            headers={"Retry-After": str(window)},
        )


# --- in-process fallback limiter ---

# Guarded by a plain lock rather than asyncio primitives: the critical section
# is a dict lookup and two integer comparisons, it must also be safe against
# the shutdown-time calls that may arrive off-loop, and it never awaits.
_local_rate_lock = threading.Lock()
# key -> (count, window_expiry)
_local_rate_counters: dict[str, tuple[int, float]] = {}

# Hard bound on the fallback's memory. An attacker who can make us fail over to
# the fallback can also mint unlimited distinct keys (one per source address,
# one per submitted address), so an unbounded dict would be a memory-exhaustion
# DoS in place of the rate-limit DoS it replaced.
#
# What is dropped when the cap is hit matters, because the buckets this dict
# holds are not all equal. An attacker flooding it with fresh source addresses
# must not be able to use the pressure to discard the per-ACCOUNT bucket they
# are actually being throttled by. So the dict is kept in least-recently-used
# order (every hit re-inserts its key at the tail) and eviction drops from the
# head, which makes an actively-attacked bucket the last thing to go rather
# than the first. Windows that have already closed are reclaimed first,
# before any live bucket is touched -- they are worth nothing to anyone.
_LOCAL_RATE_MAX_KEYS = 20_000


def _local_rate_hit(key: str, window: int) -> int:
    """Count one hit against ``key`` in memory and return the running count.

    Fixed window, matching the Redis path: the first hit opens a window of
    ``window`` seconds and the count resets when it closes.
    """
    now = time.monotonic()
    with _local_rate_lock:
        count, expiry = _local_rate_counters.get(key, (0, now + window))
        if now >= expiry:
            count, expiry = 0, now + window
        count += 1
        # Re-insert rather than update: deleting first moves the key to the
        # tail, which is what makes this least-recently-used ordered.
        _local_rate_counters.pop(key, None)
        _local_rate_counters[key] = (count, expiry)
        if len(_local_rate_counters) > _LOCAL_RATE_MAX_KEYS:
            _prune_local_rate_counters(now)
        return count


def _prune_local_rate_counters(now: float) -> None:
    """Bring the fallback dict back under ``_LOCAL_RATE_MAX_KEYS``. Call with
    the lock held."""
    for key in [k for k, (_, expiry) in _local_rate_counters.items() if expiry <= now]:
        del _local_rate_counters[key]
    over = len(_local_rate_counters) - _LOCAL_RATE_MAX_KEYS
    if over <= 0:
        return
    # Head first == least recently used first. A live bucket is only ever
    # dropped when the attacker is generating keys faster than the windows
    # close, and then only in recency order.
    for stale in list(_local_rate_counters)[:over]:
        del _local_rate_counters[stale]


def reset_local_rate_limits() -> None:
    """Forget every in-process counter. Used by tests, which share one process
    (and therefore one fallback limiter) across every case; nothing in the
    service calls it."""
    with _local_rate_lock:
        _local_rate_counters.clear()


def public_rate_limit(
    action: str,
    limit_attr: str,
    *,
    fail_closed: bool = True,
) -> Callable[[Request], Awaitable[None]]:
    """Build the FastAPI dependency that rate-limits one public endpoint.

    ``limit_attr`` names a ``config`` attribute (e.g.
    ``"PUBLIC_SEARCH_RATE_PER_MIN"``) and is resolved per request rather than
    captured at import time, so the limit stays tunable -- and overridable in a
    test -- without rebuilding the app.

    ``fail_closed`` defaults to True because every current caller is on the
    public search surface. ``/ready`` is the one deliberate exception: it is
    polled by load balancers and orchestrators, and this service treats a Redis
    outage as a degraded-but-serving state (the HybridCache falls back to an
    in-process cache), so failing its limiter closed would pull healthy nodes
    out of rotation for a dependency the service does not require to be ready.
    It still counts and still answers 429 -- only a broken limiter store is
    tolerated there.
    """

    async def dependency(request: Request) -> None:
        await _check_rate_limit(
            request,
            action,
            int(getattr(config, limit_attr)),
            key_prefix="public:rl",
            window_seconds=config.PUBLIC_RATE_WINDOW_SECONDS,
            fail_closed=fail_closed,
        )

    return dependency


# --- dependencies ---


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


def _service_user() -> StoredUser:
    return StoredUser(
        id=SERVICE_USER_ID,
        email="service@internal",
        password_hash="",
        name="Service",
        role="admin",
        is_active=True,
        created_at=0.0,
    )


def _service_token_ttl_seconds() -> float:
    """Lifetime of a freshly minted service token. A non-positive configured
    value falls back to the shipped default instead of meaning 'never
    expires' -- an immortal machine admin credential is the hole, so there is
    deliberately no way to configure one back into existence."""
    configured = int(getattr(config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 0) or 0)
    return float(configured if configured > 0 else 86400)


def _service_token_scope() -> frozenset[str]:
    """Permissions a service token gets, from ``config.AUTH_SERVICE_TOKEN_SCOPE``.

    Unknown permission names are dropped with a warning rather than passed
    through: a typo in the operator's .env then yields a token that can do
    less than intended and is logged, instead of a token whose scope silently
    does not match what anyone reading the .env believes.
    """
    known = {p for role in ROLE_PERMISSIONS.values() for p in role}
    requested = tuple(getattr(config, "AUTH_SERVICE_TOKEN_SCOPE", ()) or ())
    scope = {p for p in requested if p in known}
    if len(scope) != len(set(requested)):
        logger.warning("AUTH_SERVICE_TOKEN_SCOPE names unknown permissions; ignoring them")
    return frozenset(scope)


async def _resolve_service_token(raw: str) -> StoredServiceToken | None:
    """Resolve a machine credential, seeding the env-configured one on first
    use. Returns None when the credential is unknown, revoked or expired.

    Every service token -- the one in the environment and any minted through
    the admin surface -- is resolved from the table by its hash. The
    environment value is only the SEED: it creates the record once, and from
    then on the stored row is the sole authority on the token's life, so
    revoking or expiring it actually takes effect.
    """
    s = _require_auth_store()
    record = await s.service_token_for(raw)
    if record is not None:
        return record
    if config.AUTH_SERVICE_TOKEN and secrets.compare_digest(raw, config.AUTH_SERVICE_TOKEN):
        await s.ensure_bootstrap_service_token(raw, set(_service_token_scope()), _service_token_ttl_seconds())
        return await s.service_token_for(raw)
    return None


async def require_auth(request: Request) -> None:
    """Validate the request's credentials and stash the user on request.state.
    Accepts ``Authorization: Bearer <token>`` (user tokens) or
    ``X-Service-Token`` (a scoped, expiring machine credential).

    A service token that is revoked or past its expiry is NOT honoured: it
    falls through to the bearer path and ends as a 401, rather than being
    granted admin because the environment still mentions it.
    """
    service = request.headers.get("x-service-token")
    if service:
        record = await _resolve_service_token(service)
        if record is not None:
            request.state.user = _service_user()
            request.state.user_id = SERVICE_USER_ID
            # The scope narrows this below the role's full permission set; a
            # token minted with an empty scope can authenticate but reach
            # nothing.
            request.state.scope = record.scope
            request.state.service_token = service
            return
        logger.warning("auth: a service token was presented but is unknown, revoked or expired")
    token = _token_from_request(request)
    if token is None:
        raise HTTPException(status_code=401, detail="authentication required")
    user = await _require_auth_store().user_for_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    request.state.user = user
    request.state.user_id = user.id


def require_permission(permission: str):
    """Dependency factory: require ``permission`` (see ROLE_PERMISSIONS).

    A service-token request is additionally checked against that token's own
    scope: its role is admin so it can reach admin routes at all, but the scope
    is what decides which of them. Human users carry no scope and are decided
    by their role alone, as before.
    """

    async def checker(request: Request) -> None:
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        if permission not in ROLE_PERMISSIONS.get(user.role, ()):
            raise HTTPException(status_code=403, detail="forbidden")
        scope = getattr(request.state, "scope", None)
        if scope is not None and permission not in scope:
            raise HTTPException(status_code=403, detail="service token is not scoped for this permission")

    return checker


# --- endpoints ---


@router.post("/signup", response_model=SignupOut)
async def signup(body: SignupIn, request: Request):
    """Register an account (public). Validated server-side: email format,
    password strength, name limits.

    The response is one fixed 200 ``{"message": ...}`` whether or not the
    address was already registered, so an unauthenticated caller cannot use
    this endpoint to learn which addresses have accounts here. It therefore
    carries no token: a token present only for fresh addresses would be the
    oracle all over again, and minting one for an existing account would hand
    an anonymous caller someone else's session. Callers follow up with
    ``POST /api/auth/login``.

    The role is hardcoded to 'user': public signups always land with the least
    privilege. There is deliberately no configuration knob here — a role that
    can be flipped by an env var (or a request field) turns a config mistake
    into a full account compromise. Privilege is granted only by an
    authenticated admin via PATCH /api/auth/users/{id}."""
    await _check_rate_limit(request, "signup", config.AUTH_SIGNUP_RATE_PER_MIN)
    email = validate_email(body.email)
    password = validate_password(body.password)
    name = validate_name(body.name)
    s = _require_auth_store()
    # No pre-flight "does this address exist" lookup: create_user hashes the
    # password and attempts the INSERT either way, so a duplicate costs the
    # same wall-clock time as a fresh registration. A pre-check would skip
    # that bcrypt work and leak existence through timing, exactly as login
    # did. The users.email UNIQUE constraint is the single source of truth.
    try:
        await s.create_user(email, password, name, role=SIGNUP_ROLE)
    except DuplicateEmailError:
        # Already registered (including losing a concurrent-creation race).
        # Swallow it into the same success-shaped answer a fresh address gets
        # and leave the stored row untouched: no re-hash, no rename, no
        # re-activation, no duplicate, no session.
        logger.info("signup for an already-registered address: reported as accepted")
    return SignupOut(message=SIGNUP_ACCEPTED_MESSAGE)


@router.post("/login", response_model=AuthOut)
async def login(body: LoginIn, request: Request):
    """Exchange email+password for a bearer token.

    An unknown address and a known address with a wrong password are
    indistinguishable to the caller: the identical 401 status and body, and
    the identical full bcrypt verify cost, because the unknown-address path
    verifies the supplied password against a fixed dummy hash at the same
    cost factor instead of skipping the check. A deactivated account is
    verified the same way, so it is not distinguishable either.
    The per-account rate limit below is held to the same rule: it is keyed on the
    submitted address alone, so a registered and an unregistered address reach
    the same counter, the same 429 and the same amount of work.
    """
    await _check_rate_limit(request, "login", config.AUTH_LOGIN_RATE_PER_MIN)
    email = validate_email(body.email)
    # Second, IP-independent bucket. Placed after normalisation (so
    # "A@B.co" and "a@b.co" share one bucket) and before any account lookup,
    # which is what keeps it from being an existence oracle.
    await _check_account_rate_limit(
        request, "login", config.AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN, email
    )
    s = _require_auth_store()
    user = await s.get_user_by_email(email)
    # Always pay the bcrypt cost, even with no account to compare against:
    # `user is not None and await ... verify_password(...)` short-circuits, and
    # that skipped ~100ms was a remote account-existence oracle.
    password_ok = await asyncio.to_thread(
        verify_password,
        body.password,
        user.password_hash if user is not None else _DUMMY_PASSWORD_HASH,
    )
    if not password_ok or user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="invalid email or password")
    token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)
    return AuthOut(token=token, user=UserOut.from_user(user))


@router.get("/me", response_model=UserOut)
async def me(request: Request, _: None = Depends(require_auth)):
    return UserOut.from_user(request.state.user)


@router.post("/logout")
async def logout(request: Request, _: None = Depends(require_auth)):
    """Revoke the credential this request authenticated with.

    For a service token that revocation is real: the token is marked revoked,
    so it stops working immediately. It used to be a no-op here -- logout only
    ever looked at a bearer token, so a machine credential answered ``ok`` and
    then went on working for as long as the process did."""
    service = getattr(request.state, "service_token", None)
    if service is not None:
        await _require_auth_store().revoke_service_token(service)
        return {"ok": True}
    token = _token_from_request(request)
    if token is not None:
        await _require_auth_store().revoke_token(token)
    return {"ok": True}


@router.post("/change-password")
async def change_password(body: ChangePasswordIn, request: Request, _: None = Depends(require_auth)):
    """Change the current user's password after verifying the old one. Invalidates
    every other token the user holds (the current session stays signed in)."""
    user = request.state.user
    s = _require_auth_store()
    stored = await s.get_user(user.id)
    if stored is None or not await asyncio.to_thread(
        verify_password, body.current_password, stored.password_hash
    ):
        raise HTTPException(status_code=400, detail="current password is incorrect")
    new_password = validate_password(body.new_password)
    await s.set_password(user.id, await asyncio.to_thread(hash_password, new_password))
    await s.revoke_all_tokens(user.id)
    token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)
    return AuthOut(token=token, user=UserOut.from_user(stored))


# --- admin user management (users:manage) ---


async def _get_user_or_404(s: AuthStore, user_id: str) -> StoredUser:
    user = await s.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user


@router.get("/users", response_model=list[UserOut])
async def list_users(
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:read")),
):
    return await _require_auth_store().list_users()


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(
    user_id: str,
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:read")),
):
    user = await _get_user_or_404(_require_auth_store(), user_id)
    return UserOut.from_user(user)


@router.patch("/users/{user_id}", response_model=UserOut)
async def patch_user(
    user_id: str,
    body: UserPatchIn,
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Update a user's name/role/is_active. Protects the last active admin from
    demotion or deactivation."""
    s = _require_auth_store()
    target = await _get_user_or_404(s, user_id)
    if body.role is not None and body.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail="invalid role")
    is_demote = target.role == "admin" and (body.role == "user" or body.is_active is False)
    name = validate_name(body.name) if body.name is not None else None
    rowcount = await s.update_user(user_id, name, body.role, body.is_active, guard_last_admin=is_demote)
    if is_demote and rowcount == 0:
        raise HTTPException(status_code=400, detail="cannot demote or deactivate the last admin")
    return UserOut.from_user(await _get_user_or_404(s, user_id))


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Permanently remove a user and revoke all their tokens."""
    s = _require_auth_store()
    target = await _get_user_or_404(s, user_id)
    n = await s.delete_user(user_id, guard_last_admin=(target.role == "admin"))
    if target.role == "admin" and n == 0:
        raise HTTPException(status_code=400, detail="cannot delete the last admin")
    return {"ok": True}


@router.post("/users/{user_id}/tokens/revoke")
async def revoke_user_tokens(
    user_id: str,
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Revoke every token a user holds (forces re-login)."""
    await _require_auth_store().revoke_all_tokens(user_id)
    return {"ok": True}


@router.post("/service-tokens", response_model=ServiceTokenOut)
async def mint_service_token(
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Mint a scoped, expiring machine credential (rotation).

    The plaintext value is in the response and nowhere else -- only its
    SHA-256 is stored -- so this is the only chance to record it. It is scoped
    to ``config.AUTH_SERVICE_TOKEN_SCOPE`` and expires after
    ``AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS``; neither can be widened per request.

    Requires ``users:manage``, which a service token scoped to the shipped
    default (``chat:use``) does not hold, so a leaked machine credential cannot
    mint itself a successor.
    """
    raw, record = await _require_auth_store().issue_service_token(
        set(_service_token_scope()), _service_token_ttl_seconds()
    )
    return ServiceTokenOut(token=raw, scope=sorted(record.scope), expires_at=record.expires_at)


@router.post("/service-tokens/revoke")
async def revoke_service_tokens(
    body: ServiceTokenRevokeIn,
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Revoke a service token: the other half of rotation.

    Pass the token to retire and only that one dies, which is what makes
    rotation safe to perform in the order an operator naturally reaches for --
    mint the replacement, move consumers onto it, *then* kill the old one,
    without a window in which no credential works. With no ``token`` in the
    body every live service token is revoked at once; that is the right move
    for a suspected leak, and the wrong one for a planned rotation because it
    would take down the replacement minted moments earlier.
    """
    s = _require_auth_store()
    if body.token:
        await s.revoke_service_token(body.token)
        return {"revoked": 1}
    return {"revoked": await s.revoke_all_service_tokens()}



async def bootstrap_admin() -> None:
    """Seed the bootstrap admin from config (once, at startup). Never overwrites
    an existing account's password. Safe under concurrent worker startups: the
    duplicate / write-lock races are handled instead of failing startup (which
    would restart-loop the worker)."""
    email = (config.AUTH_ADMIN_EMAIL or "").strip().lower()
    password = config.AUTH_ADMIN_PASSWORD or ""
    if not email or not password:
        return
    s = _require_auth_store()
    for attempt in range(5):
        if await s.get_user_by_email(email) is not None:
            return
        try:
            await s.create_user(email, password, "Administrator", role="admin")
            logger.info("bootstrapped admin account %s", email)
            return
        except DuplicateEmailError:
            logger.info("bootstrap admin %s already exists (concurrent worker)", email)
            return
        except sqlite3.OperationalError:
            if attempt == 4:
                logger.error("bootstrap admin %s could not be created (write lock)", email)
                return
            await asyncio.sleep(1)
