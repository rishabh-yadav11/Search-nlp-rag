"""Session-cookie + RBAC authentication for the API.

The session token is never returned in a body nor read from a header: a
script-readable token is one XSS bug from a durable account takeover, and a
header path stays reachable from ``fetch()``.

Because the browser attaches that cookie automatically, every unsafe
cookie-authenticated request is forgeable by a page the user visits, so
``enforce_same_origin`` guards them from inside ``require_auth`` -- coverage
cannot be forgotten on a new route.

``AUTH_SERVICE_TOKEN`` is a scoped, expiring machine credential rather than an
unconditional admin bypass.
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
from urllib.parse import urlsplit

import aiosqlite
import bcrypt
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.config import config

logger = logging.getLogger("auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

store: "AuthStore | None" = None

VALID_ROLES = ("admin", "user")

SIGNUP_ROLE = "user"

ROLE_PERMISSIONS: ClassVar[dict[str, set[str]]] = {
    "admin": {"chat:use", "analytics:read", "users:read", "users:manage"},
    "user": {"chat:use"},
}

SERVICE_USER_ID = "service-token"

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

# Prefixed onto every current hash to mark it as bcrypt over the SHA-256 pre-image;
# an unmarked hash is pre-migration ``bcrypt(raw[:72])``, which ``verify_password``
# still accepts.
_PASSWORD_SCHEME = "$bcrypt-sha256$"

# Only the width a pre-migration hash was built from; nothing is truncated any more.
_LEGACY_BCRYPT_MAX_BYTES = 72


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
    """Carries the user record and deliberately NOT the session token: publishing
    it in the body would hand every XSS on the site a durable account takeover."""

    user: UserOut


class SignupOut(BaseModel):
    """Carries neither a token nor a user record, so the fixed answer cannot be used
    to confirm that an address already has an account here."""

    message: str


# Must state both possible outcomes without favouring one: there is no
# confirmation-email flow, so for a returning user this message is the recovery route.
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
    """``scope`` is the explicit set of permissions this token may exercise, NOT the
    full admin set, and ``expires_at`` bounds its life so a leak stops working."""

    token_hash: str
    scope: frozenset[str]
    created_at: float
    expires_at: float


class ServiceTokenOut(BaseModel):
    """The plaintext value is returned exactly once, at mint time; only its SHA-256 is stored."""

    token: str
    scope: list[str]
    expires_at: float


class ServiceTokenRevokeIn(BaseModel):
    token: str = ""


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not email or len(email) > config.AUTH_MAX_EMAIL_LEN or not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="invalid email address")
    return email


def _has_letter_and_digit(password: str) -> bool:
    return bool(re.search(r"[A-Za-z]", password)) and bool(re.search(r"\d", password))


def _password_rejection(password: str) -> str | None:
    """Why this password is refused, or None when it is accepted: the single source of
    truth every path that can set a password must reach.

    Both rules are judged on the WHOLE value, because that is what authenticates:
    ``hash_password`` hands bcrypt a fixed-width SHA-256 pre-image, so no byte is dropped.

    There is deliberately NO maximum: bcrypt's 72-byte cut stopped ``bootstrap_admin``
    from seeding an admin whose own passphrase ``change_password`` then refused.
    """
    if len(password) < config.AUTH_PASSWORD_MIN_LEN:
        return f"password must be at least {config.AUTH_PASSWORD_MIN_LEN} characters"
    if not _has_letter_and_digit(password):
        return "password must contain a letter and a digit"
    return None


def validate_password(password: str) -> str:
    """Raise 422 unless the policy accepts the value; returns it unchanged, so a caller
    can never store something other than the exact value that was validated."""
    reason = _password_rejection(password)
    if reason:
        raise HTTPException(status_code=422, detail=reason)
    return password


def _validator_rejection(validator, value: str) -> str | None:
    """``validate_*`` signal failure by raising; the bootstrap path needs the reason as
    text, so it uses this instead."""
    try:
        validator(value)
    except HTTPException as exc:
        return str(exc.detail)
    return None


def validate_name(name: str) -> str:
    name = (name or "").strip()
    if len(name) > config.AUTH_MAX_NAME_LEN:
        raise HTTPException(status_code=422, detail=f"name too long (max {config.AUTH_MAX_NAME_LEN} chars)")
    if any(ord(c) < 32 for c in name):
        raise HTTPException(status_code=422, detail="name contains invalid characters")
    return name


def _prehash(password: str) -> bytes:
    """The fixed-width pre-image bcrypt is handed: SHA-256 rather than a second bcrypt,
    which would only move the truncation window somewhere else."""
    return hashlib.sha256(password.encode("utf-8")).digest()


def hash_password(password: str) -> str:
    digest = bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("utf-8")
    return _PASSWORD_SCHEME + digest


def needs_rehash(hashed: str) -> bool:
    return not hashed.startswith(_PASSWORD_SCHEME)


def verify_password(password: str, hashed: str) -> bool:
    """Check a password against a stored hash of EITHER scheme.

    The scheme is read from the STORED hash and only from it: trying both, or
    dispatching on what the caller submitted, would leave the shorter of the two as
    a working alternative for whichever row an attacker targeted.
    """
    if hashed.startswith(_PASSWORD_SCHEME):
        stored, preimage = hashed[len(_PASSWORD_SCHEME):], _prehash(password)
    else:
        stored = hashed
        preimage = password.encode("utf-8")[:_LEGACY_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(preimage, stored.encode("utf-8"))
    except ValueError:
        # A corrupt row, an empty placeholder or a truncated write: the same answer as a
        # wrong password, and never an exception out of an auth path.
        return False


# A per-process random secret hashed once at import, so a login against an address with
# no account costs the same wall-clock time as a wrong-password login; without it the
# missing short-circuit is a remote account-existence oracle behind the identical 401.
_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(presented: str, expected: str) -> bool:
    """Constant-time equality of two credential strings.

    ``secrets.compare_digest`` raises ``TypeError`` on a non-ASCII ``str`` and an
    ``X-Service-Token`` header is attacker-controlled bytes; ``surrogateescape`` keeps the
    encode total, so nothing a server can decode into the header raises here.
    """
    return secrets.compare_digest(
        presented.encode("utf-8", "surrogateescape"), expected.encode("utf-8", "surrogateescape")
    )


def _now() -> float:
    return time.time()


class DuplicateEmailError(Exception):
    """Raised when an INSERT hits the users.email UNIQUE constraint, e.g. two gunicorn
    workers bootstrapping the same admin concurrently."""


class AuthStore:
    """SQLite-backed user + token store (WAL mode). Token values are never stored plaintext."""

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
            # Duplicate email under concurrency (e.g. concurrent worker bootstrap).
            # Roll back so the failed statement never leaves this connection holding
            # an open write transaction, which would poison the DB with "database is locked".
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
        # Whitelisted so a caller can never inject a user-controlled column name into the
        # f-string SQL.
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
            # One statement, so the check and the write are free of a count-then-set race.
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
        """Overwrite one user's stored hash on its own. A caller that also has to invalidate
        credentials must not use this: a password write that commits without the matching
        revocation leaves every previously issued token alive. ``change_password`` is atomic."""
        await self._db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        await self._db.commit()

    async def upgrade_password_hash(self, user_id: str, observed_hash: str, new_hash: str) -> int:
        """Replace a pre-migration hash, but only while the row still holds the hash just
        verified. Returns rows changed.

        That compare-and-swap is what makes the opportunistic upgrade safe from a login: a
        concurrent ``change_password``, or a second worker upgrading the same account, must
        not be clobbered with a hash derived from a credential its owner already replaced.
        Losing that race is the correct outcome, so the rowcount is returned for the caller
        to log rather than retried -- a retry reopens the same race.
        """
        cur = await self._db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ? AND password_hash = ?",
            (new_hash, user_id, observed_hash),
        )
        await self._db.commit()
        return cur.rowcount

    async def count_legacy_passwords(self) -> int:
        """How many accounts still hold a pre-migration hash.

        An upper bound, not an exact one: a row records nothing about the original length, so
        an account whose password fitted inside bcrypt's window is counted too. ``substr``
        with a bound parameter, not ``NOT LIKE``, whose ``%`` and ``_`` are wildcards.
        """
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM users WHERE substr(password_hash, 1, ?) <> ?",
            (len(_PASSWORD_SCHEME), _PASSWORD_SCHEME),
        )
        return int(row["n"]) if row else 0

    async def count_admins(self) -> int:
        row = await self._fetchone("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
        return int(row["n"]) if row else 0

    async def issue_token(self, user_id: str, ttl_days: int) -> str:
        """Mint a bearer token, returned in plaintext; only its SHA-256 is stored."""
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
        """Keep at most ``config.AUTH_MAX_ACTIVE_TOKENS_PER_USER`` unexpired tokens per user,
        revoking the oldest surplus. Returns how many were revoked.

        The surplus rows are DELETED, not just hidden: ``user_for_token`` resolves a token by
        looking its hash up in this very table, so leaving the row would keep a live
        credential alive, which is the opposite of what a cap is for.
        """
        cap = int(getattr(config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 0))
        if cap <= 0:
            return 0
        now = _now()
        # rowid breaks ties between rows minted in the same clock tick, making the
        # eviction order deterministic rather than storage-dependent.
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
        """Number of the user's unexpired tokens; not used to make an access decision."""
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM auth_tokens WHERE user_id = ? AND expires_at >= ?",
            (user_id, _now()),
        )
        return int(row["n"]) if row else 0


    async def issue_service_token(self, scope: set[str], ttl_seconds: float) -> tuple[str, StoredServiceToken]:
        """Mint a scoped, expiring machine credential. Only the hash is persisted, so the
        caller gets exactly one chance to keep the plaintext."""
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

        INSERT OR IGNORE, and deliberately so: re-running this on every worker restart
        must NOT push the expiry out, or the credential would be eternal in practice.
        Rotating means revoking the row (or changing the env value, which hashes
        differently) so the next call seeds a fresh lifetime.
        """
        # One clock read for both columns, so created_at and expires_at cannot disagree.
        created = _now()
        await self._db.execute(
            "INSERT OR IGNORE INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
            " VALUES (?, ?, ?, ?, NULL)",
            (hash_token(raw), ",".join(sorted(scope)), created, created + ttl_seconds),
        )
        await self._db.commit()

    async def service_token_for(self, raw: str) -> StoredServiceToken | None:
        """Resolve a machine credential, or None when it is unknown, revoked or expired.
        A revoked or expired token is a hard no: there is no 'but the env still says so'
        path back in."""
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

    async def revoke_service_token(self, raw: str) -> int:
        """Soft-revoke one machine credential. Marks the row rather than deleting it: the
        row is the tombstone that stops a revoked configured value being re-seeded with a
        fresh lifetime. Returns how many rows it actually revoked.
        """
        cur = await self._db.execute(
            "UPDATE auth_service_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_now(), hash_token(raw)),
        )
        await self._db.commit()
        return cur.rowcount

    async def revoke_configured_service_token(self, raw: str, scope: set[str]) -> int:
        """Kill the env-configured value, including from a cold start. Returns the rows changed.

        Seeding is lazy, so a configured value that has never been presented has no row:
        the plain UPDATE then matches nothing, reports 0, and the next request seeds the
        value live with a fresh full lifetime -- "revoke this now" would silently do
        nothing exactly when an operator is containing a leak of a credential not yet in
        use. So when the value has no row, INSERT a tombstone: a row that exists and is
        already revoked. Seeding is INSERT OR IGNORE, so this can only ever make a
        credential deader, never younger.
        """
        now = _now()
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (hash_token(raw), ",".join(sorted(scope)), now, now, now),
        )
        tombstoned = cur.rowcount
        await self._db.commit()
        return tombstoned + await self.revoke_service_token(raw)

    async def revoke_all_service_tokens(self) -> int:
        cur = await self._db.execute(
            "UPDATE auth_service_tokens SET revoked_at = ? WHERE revoked_at IS NULL", (_now(),)
        )
        await self._db.commit()
        return cur.rowcount

    async def purge_dead_service_tokens(self, keep_hash: str = "") -> int:
        """Delete service-token rows that can no longer authenticate anything.

        ``keep_hash`` is excluded, and it must be: that row is the tombstone which stops a
        revoked or expired configured ``AUTH_SERVICE_TOKEN`` from being re-seeded with a
        fresh lifetime, and delete it and the reaper would hand the credential straight back.
        """
        now = _now()
        cur = await self._db.execute(
            "DELETE FROM auth_service_tokens"
            " WHERE (revoked_at IS NOT NULL OR expires_at < ?)"
            " AND (? = '' OR token_hash != ?)",
            (now, keep_hash, keep_hash),
        )
        await self._db.commit()
        return cur.rowcount

    async def user_for_token(self, raw_token: str) -> StoredUser | None:
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

    async def change_password(self, user_id: str, new_password_hash: str, ttl_days: int) -> str:
        """Store a new password, revoke every existing token and mint a replacement as ONE
        durable unit. Returns the replacement token.

        These three writes only mean anything together, so they run inside a single
        ``BEGIN IMMEDIATE`` .. ``COMMIT`` on a short-lived dedicated connection: it cannot
        be held on the shared one, where another coroutine's ``commit()`` landing mid-way
        would publish a half-finished change early and its ``rollback()`` would throw this
        one away. WAL lets it work alongside the shared connection, and ``isolation_level
        =None`` keeps this ``BEGIN`` as the connection's only transaction.

        The statements are ALSO ordered revoke -> set hash -> mint: defence in depth, not
        the guarantee. An interruption after either write leaves no live tokens either way.
        """
        if self._db is None:
            raise RuntimeError("auth store is not connected")
        db = await aiosqlite.connect(self._path, isolation_level=None)
        try:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
            await db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
            raw = secrets.token_urlsafe(32)
            created = _now()
            await db.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (hash_token(raw), user_id, created, created + ttl_days * 86400),
            )
            await db.commit()
        except BaseException:
            # BaseException, not Exception: a request cancelled out from under this
            # coroutine must not abandon an open write transaction, which is what leaves
            # the file locked against every other connection.
            await db.rollback()
            raise
        finally:
            await db.close()
        return raw

    async def purge_expired_tokens(self) -> int:
        """Delete rows whose expiry has passed, so the table cannot grow without bound."""
        now = _now()
        cur = await self._db.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (now,))
        await self._db.commit()
        return cur.rowcount


def _require_auth_store() -> AuthStore:
    if store is None:
        raise HTTPException(status_code=503, detail="auth store not initialized")
    return store


_rate_client = None


def _rate_redis() -> aioredis.Redis:
    global _rate_client
    if _rate_client is None:
        # Pinned so the counters never land in DB 0, which is the query cache and is
        # flushed during deploys -- a flush there would reset every bucket.
        _rate_client = aioredis.from_url(
            config.REDIS_URL,
            db=config.AUTH_RATE_LIMIT_REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _rate_client


async def close_rate_redis() -> None:
    """Close the lazily-created rate-limit Redis client; a shutdown hook, so the connection
    isn't leaked on worker exit."""
    global _rate_client
    if _rate_client is not None:
        await _rate_client.aclose()
        _rate_client = None


async def token_purge_loop() -> None:
    """Background task purging dead token rows. Never raises. Disabled when
    AUTH_TOKEN_PURGE_INTERVAL_SECONDS is 0 (e.g. tests)."""
    interval = config.AUTH_TOKEN_PURGE_INTERVAL_SECONDS
    while interval > 0:
        try:
            if store is not None:
                n = await store.purge_expired_tokens()
                if n:
                    logger.info("auth: purged %d expired token(s)", n)
                # The configured value's row is kept as a tombstone; see
                # purge_dead_service_tokens.
                keep = hash_token(config.AUTH_SERVICE_TOKEN) if config.AUTH_SERVICE_TOKEN else ""
                m = await store.purge_dead_service_tokens(keep_hash=keep)
                if m:
                    logger.info("auth: purged %d dead service token(s)", m)
        except Exception:
            logger.exception("auth token purge failed")
        await asyncio.sleep(interval)


def _peer_is_local_proxy(peer: str | None) -> bool:
    """True when the socket peer is a loopback reverse proxy on this same host. A peer
    that is not a real IP (a test ASGI transport) is not loopback, so never widens trust."""
    if not peer:
        return False
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


def _trust_forwarded_for(peer: str | None) -> bool:
    """Whether X-Forwarded-For may be trusted for this request's peer.

    ``config.AUTH_TRUST_X_FORWARDED_FOR`` forces the answer; the shipped default is None
    ("auto"), trusting the header only for a loopback peer. The reference deployment
    always runs behind nginx forwarding from 127.0.0.1, while a client connecting
    straight to the API port is its own non-loopback peer and cannot forge a header to
    escape its own bucket.
    """
    configured = config.AUTH_TRUST_X_FORWARDED_FOR
    if configured is not None:
        return configured
    return _peer_is_local_proxy(peer)


def _client_ip(request: Request) -> str:
    """Client IP, from X-Forwarded-For only when the request came through a trusted
    reverse proxy, so a raw client cannot spoof its IP to escape a rate-limit bucket.

    nginx ``$proxy_add_x_forwarded_for`` APPENDS the real peer to any client-supplied
    list, so the *rightmost* hop is the trusted one while the leftmost is attacker
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
    subject: str | None = None,
) -> None:
    """Enforce a rate limit with Redis INCR+EXPIRE.

    ``fail_closed`` selects what happens when the limiter's Redis is unreachable. The auth
    endpoints keep the default (False) and fall back to a bounded in-process limiter; the
    public search surface passes ``fail_closed=True`` and is answered 503 instead, because
    an unrated request against ``/search`` or ``/analytics/click`` is precisely the
    scraping and analytics-poisoning vector these limits close.

    ``subject`` overrides the bucket identity from the client IP to a caller supplied one,
    typically an authenticated user id: a per-IP bucket cannot bound one account behind a
    shared NAT or proxy address, so an endpoint that mints per-account state is limited on
    both axes. It changes ONLY which string is counted, so the two axes cannot drift.
    """
    if limit_per_min <= 0:
        return
    window = config.AUTH_RATE_WINDOW_SECONDS if window_seconds is None else window_seconds
    key = f"{key_prefix}:{action}:{subject or _client_ip(request)}"
    await _consume_counter(key, limit_per_min, window, action=action, fail_closed=fail_closed)


async def _check_account_rate_limit(
    request: Request,
    action: str,
    limit_per_min: int,
    account: str,
) -> None:
    """Enforce a rate limit keyed on the submitted account rather than the source address,
    so rotating IPs cannot buy an attacker a fresh bucket. The key is the address folded
    to lower case and stripped, so one account cannot be handed a second bucket by
    changing case or padding.

    THIS HELPER PERFORMS NO LOOKUP, and must not: it computes the same key and the same
    counter update whether or not the address has an account, which is what keeps it from
    being an account-existence oracle. Existence enters only in the caller's decision to
    call it -- login counts failed credential checks, and the only thing that skips the
    increment is supplying the correct password.
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

    Redis is the shared counter, so the limit is global across every worker process. When
    it is unreachable both obvious behaviours are wrong: admitting the request turns an
    attacker-inducible Redis outage into an unlimited credential-stuffing window, while
    rejecting everything hands a DoS to whoever can disturb Redis without helping anyone
    guess a password.

    So the fallback is a third option: a bounded in-process limiter with the SAME limit.
    The degraded posture is "single-process limiting" rather than "no limiting" or "no
    service"; the per-worker weakening is inherent to it and a gunicorn deployment
    multiplies the effective limit by its worker count, which is still a finite bound.
    """
    try:
        rc = _rate_redis()
        # Establish the sliding window atomically on the first hit: SET NX EX sets the
        # value to 0 with the window TTL only if the key did not already exist, so the key
        # always has a TTL. A later crash can never leave a counter with no expiry, which
        # under the old INCR + separate EXPIRE would block the IP forever.
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

# A plain lock rather than an asyncio primitive: the critical section never awaits, and
# it must also be safe against shutdown-time calls that may arrive off-loop.
_local_rate_lock = threading.Lock()
# key -> (count, window_expiry)
_local_rate_counters: dict[str, tuple[int, float]] = {}

# Hard bound on the fallback's memory: an attacker who can force the fallback can also
# mint unlimited distinct keys, so an unbounded dict would be a memory-exhaustion DoS in
# place of the rate-limit DoS it replaced.
#
# What is dropped when the cap is hit matters: a flood of fresh source addresses must not
# be able to use the pressure to discard the per-ACCOUNT bucket being throttled. So the
# dict stays in least-recently-used order (every hit re-inserts at the tail) and eviction
# drops from the head. Closed windows are reclaimed before any live bucket.
_LOCAL_RATE_MAX_KEYS = 20_000


def _local_rate_hit(key: str, window: int) -> int:
    """Count one hit against ``key`` in memory and return the running count. Fixed window,
    matching the Redis path."""
    now = time.monotonic()
    with _local_rate_lock:
        count, expiry = _local_rate_counters.get(key, (0, now + window))
        if now >= expiry:
            count, expiry = 0, now + window
        count += 1
        # Re-insert rather than update: deleting first moves the key to the tail, which is
        # what makes this least-recently-used ordered.
        _local_rate_counters.pop(key, None)
        _local_rate_counters[key] = (count, expiry)
        if len(_local_rate_counters) > _LOCAL_RATE_MAX_KEYS:
            _prune_local_rate_counters(now)
        return count


def _prune_local_rate_counters(now: float) -> None:
    """Bring the fallback dict back under ``_LOCAL_RATE_MAX_KEYS``. Call with the lock held."""
    for key in [k for k, (_, expiry) in _local_rate_counters.items() if expiry <= now]:
        del _local_rate_counters[key]
    over = len(_local_rate_counters) - _LOCAL_RATE_MAX_KEYS
    if over <= 0:
        return
    for stale in list(_local_rate_counters)[:over]:
        del _local_rate_counters[stale]


def reset_local_rate_limits() -> None:
    """Forget every in-process counter. Used by tests, which share one process (and
    therefore one fallback limiter) across every case."""
    with _local_rate_lock:
        _local_rate_counters.clear()


def public_rate_limit(
    action: str,
    limit_attr: str,
    *,
    fail_closed: bool = True,
) -> Callable[[Request], Awaitable[None]]:
    """Build the FastAPI dependency that rate-limits one public endpoint.

    ``limit_attr`` names a ``config`` attribute (e.g. ``"PUBLIC_SEARCH_RATE_PER_MIN"``)
    resolved per request rather than captured at import, so the limit stays tunable --
    and overridable in a test -- without rebuilding the app.

    ``fail_closed`` defaults to True. ``/ready`` is the one deliberate exception: it is
    polled by load balancers, and this service treats a Redis outage as
    degraded-but-serving, so failing its limiter closed would pull healthy nodes out of
    rotation. It still counts and still answers 429 -- only a broken store is tolerated.
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


def user_rate_limit(
    action: str,
    limit_attr: str,
    *,
    fail_closed: bool = True,
) -> Callable[[Request], Awaitable[None]]:
    """Build the FastAPI dependency that rate-limits one endpoint per ACCOUNT.

    Same counter, window, Redis path and in-process fallback as ``public_rate_limit``, so
    neither axis can drift into weaker enforcement; only the bucket identity and key
    prefix differ, and ``user:rl`` cannot collide with the per-IP bucket.

    Compose this WITH ``public_rate_limit`` on an endpoint that mints per-account
    persistent state: a per-IP bucket alone cannot bound one account behind a shared
    address, and a per-account bucket alone cannot bound one rotating addresses.
    """

    async def dependency(request: Request, _auth: None = Depends(require_auth)) -> None:
        await _check_rate_limit(
            request,
            action,
            int(getattr(config, limit_attr)),
            key_prefix="user:rl",
            window_seconds=config.PUBLIC_RATE_WINDOW_SECONDS,
            fail_closed=fail_closed,
            subject=getattr(request.state, "user_id", None) or _client_ip(request),
        )

    return dependency


def _token_from_request(request: Request) -> str | None:
    """The session token on this request, from the auth cookie alone.

    There is deliberately no ``Authorization: Bearer`` branch: a header credential is one
    the app must hand to script, and that path stays reachable from ``fetch()``.
    """
    return request.cookies.get(config.AUTH_COOKIE_NAME) or None


def _host_only(authority: str) -> str:
    """Lowercase an authority and drop its port, IPv6 literals included.

    ``host``, ``host:port``, ``[::1]:8001`` and a bare ``::1`` all reduce to their host. A
    split on the FIRST colon would turn ``[::1]:8001`` into ``[`` and let any bracketed
    address match any other, so brackets are peeled before the port. The bare form matters
    too: ``urlsplit(...).hostname`` returns an IPv6 address with its brackets already
    removed and feeds it back through here.
    """
    host = authority.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end]
        return host
    if host.count(":") > 1:
        # More than one colon and no brackets: a bare IPv6 literal. It cannot carry a port
        # (RFC 3986 requires brackets), so splitting would leave the empty string.
        return host
    return host.partition(":")[0]


def _origin_host(origin: str) -> str | None:
    """The host an ``Origin`` header names, or None when it names none.

    ``urlsplit(...).hostname`` already lowercases, drops the port and is IPv6-safe, but it
    returns None for the literal ``Origin: null`` a sandboxed iframe sends, so the caller
    rejects that rather than treating "no host" as "same host".
    """
    try:
        return urlsplit(origin.strip()).hostname
    except ValueError:
        return None


def _request_host(request: Request) -> str | None:
    """The host this request was addressed to, from the ``Host`` header alone.

    ``X-Forwarded-Host`` is deliberately NOT consulted: it is not a Fetch-spec forbidden
    header name, so a page can set it, and the shipped nginx config neither overwrites nor
    strips it. ``Host`` is the right basis because ``TrustedHostMiddleware`` already
    constrains it to this deployment's allow-list.
    """
    host = request.headers.get("host")
    return host.strip() if host and host.strip() else None


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


async def enforce_same_origin(request: Request) -> None:
    """Reject a cross-site unsafe request: 403 when the browser says it is cross-site, or
    when the request's own headers disagree about the host.

    The browser attaches the auth cookie whether or not the page means to send it, so a
    hostile page can make an authenticated state-changing request that rides the user's
    session. Two signals are checked, and neither covers the other's gap:

    - ``Sec-Fetch-Site: cross-site`` is set by the browser and cannot be set by script,
      so on its own it is decisive for every current browser. It says nothing about a
      client that omits it (curl, an eval script), which is the gap the second closes.
    - ``Origin`` is compared against the request's own ``Host``. This is a
      self-consistency check, not an allow-list, and it catches nothing when the client
      omits ``Origin`` too. It is only as trustworthy as the ``Host`` it is compared to,
      which is why ``_request_host`` refuses to let a client-settable
      ``X-Forwarded-Host`` pick that answer. Ports are stripped because the app sits
      behind TLS termination and cannot trust ``request.url.scheme``.

    ``CORS_ORIGINS`` is deliberately NOT used here: it is a localhost-only dev default
    that does not contain the production host, so an allow-list built from it would 403
    every real request.

    The honest limitation: a request with NEITHER header is allowed through. Every browser
    sends both on an unsafe method, so their absence means a non-browser client, which has
    no ambient cookie to ride in the first place.
    """
    if request.method.upper() not in _UNSAFE_METHODS:
        return
    if (request.headers.get("sec-fetch-site") or "").strip().lower() == "cross-site":
        raise HTTPException(status_code=403, detail="cross-site request rejected")
    origin = request.headers.get("origin")
    if origin is None:
        return
    origin_host = _origin_host(origin)
    if not origin_host:
        # The literal "null" from a sandboxed iframe or privacy browser names no host, so
        # it can never be shown to be ours.
        raise HTTPException(status_code=403, detail="cross-site request rejected")
    expected = _request_host(request)
    if expected is None or _host_only(origin_host) != _host_only(expected):
        raise HTTPException(status_code=403, detail="cross-site request rejected")


def _set_session_cookie(response: Response, token: str) -> None:
    """Attach the session cookie. No ``domain``: host-only keeps it working on a bare-IP
    deployment, where any Domain would have to name an address the operator may not control."""
    response.set_cookie(
        config.AUTH_COOKIE_NAME,
        token,
        max_age=config.AUTH_COOKIE_MAX_AGE_SECONDS,
        path=config.AUTH_COOKIE_PATH,
        httponly=True,
        secure=config.AUTH_COOKIE_SECURE,
        samesite=config.AUTH_COOKIE_SAMESITE,
    )


def _clear_session_cookie(response: Response) -> None:
    """Expire the session cookie. Every attribute must match ``_set_session_cookie``
    exactly: a browser treats a deletion whose name or path differs from the original as a
    different cookie, and the live one would then be left in place."""
    response.delete_cookie(
        config.AUTH_COOKIE_NAME,
        path=config.AUTH_COOKIE_PATH,
        httponly=True,
        secure=config.AUTH_COOKIE_SECURE,
        samesite=config.AUTH_COOKIE_SAMESITE,
    )


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
    """Lifetime of a freshly minted service token. A non-positive configured value falls
    back to the shipped default instead of meaning 'never expires' -- an immortal machine
    admin credential is the hole, so there is no way to configure one back into existence."""
    configured = int(getattr(config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 0) or 0)
    return float(configured if configured > 0 else 86400)


def _service_token_scope() -> frozenset[str]:
    """Permissions a service token gets, from ``config.AUTH_SERVICE_TOKEN_SCOPE``.

    Unknown names are dropped with a warning rather than passed through: a typo in the
    operator's .env then yields a token that can do less than intended and is logged,
    instead of a scope that silently does not match the .env.
    """
    known = {p for role in ROLE_PERMISSIONS.values() for p in role}
    requested = tuple(getattr(config, "AUTH_SERVICE_TOKEN_SCOPE", ()) or ())
    scope = {p for p in requested if p in known}
    if len(scope) != len(set(requested)):
        logger.warning("AUTH_SERVICE_TOKEN_SCOPE names unknown permissions; ignoring them")
    return frozenset(scope)


async def _resolve_service_token(raw: str) -> StoredServiceToken | None:
    """Resolve a machine credential, seeding the env-configured one on first use. Returns
    None when it is unknown, revoked or expired.

    The environment value is only the SEED: every service token, minted or configured, is
    resolved from the table by its hash, so the stored row is the sole authority on the
    token's life and revoking or expiring it actually takes effect.
    """
    s = _require_auth_store()
    record = await s.service_token_for(raw)
    if record is not None:
        return record
    if config.AUTH_SERVICE_TOKEN and tokens_match(raw, config.AUTH_SERVICE_TOKEN):
        await s.ensure_bootstrap_service_token(raw, set(_service_token_scope()), _service_token_ttl_seconds())
        return await s.service_token_for(raw)
    return None


async def require_auth(request: Request) -> None:
    """Validate the request's credentials and stash the user on request.state.

    A service token that is revoked or past its expiry is NOT honoured: it falls through to
    the cookie path and ends as a 401, rather than being granted admin because the
    environment still mentions it.
    """
    # Guard FIRST, so a cross-site request is refused whichever credential it presents.
    # Inside this dependency rather than on each route: a new cookie-authenticated
    # endpoint then cannot forget the guard.
    await enforce_same_origin(request)
    service = request.headers.get("x-service-token")
    if service:
        record = await _resolve_service_token(service)
        if record is not None:
            request.state.user = _service_user()
            request.state.user_id = SERVICE_USER_ID
            # The scope narrows this below the role's full permission set; a token minted
            # with an empty scope can authenticate but reach nothing.
            request.state.scope = record.scope
            request.state.service_token = service
            return
        # Debug, not warning: this sits on an unauthenticated, attacker-controlled path,
        # so at warning level any anonymous request with a junk header writes a log line.
        logger.debug("auth: a presented service token did not resolve")
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

    A service-token request is additionally checked against that token's own scope: its
    role is admin so it can reach admin routes at all, but the scope is what decides which
    of them. Human users carry no scope and are decided by their role alone.
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


@router.post("/signup", response_model=SignupOut)
async def signup(body: SignupIn, request: Request):
    """Register an account (public). Validated server-side: email format, password
    strength, name limits.

    The response is one fixed 200 ``{"message": ...}`` whether or not the address was
    already registered, and sets no cookie: a session present only for fresh addresses
    would be the oracle all over again, and minting one for an existing account would hand
    an anonymous caller someone else's session.

    The role is hardcoded to 'user'. There is deliberately no configuration knob here -- a
    role flippable by an env var (or a request field) turns a config mistake into a full
    account compromise. Privilege is granted only by an authenticated admin.
    """
    await _check_rate_limit(request, "signup", config.AUTH_SIGNUP_RATE_PER_MIN)
    email = validate_email(body.email)
    password = validate_password(body.password)
    name = validate_name(body.name)
    s = _require_auth_store()
    # No pre-flight "does this address exist" lookup: create_user hashes the password and
    # attempts the INSERT either way, so a duplicate costs the same wall-clock time as a
    # fresh registration. A pre-check would skip that bcrypt work and leak existence
    # through timing, exactly as login did.
    try:
        await s.create_user(email, password, name, role=SIGNUP_ROLE)
    except DuplicateEmailError:
        # Already registered (including losing a concurrent-creation race): swallowed into
        # the same success-shaped answer a fresh address gets, leaving the stored row
        # untouched -- no re-hash, no rename, no re-activation, no session.
        logger.info("signup for an already-registered address: reported as accepted")
    # No cookie on purpose (see the docstring). Login-CSRF is covered by the same-origin
    # guard on /login instead.
    return SignupOut(message=SIGNUP_ACCEPTED_MESSAGE)


@router.post("/login", response_model=AuthOut)
async def login(
    body: LoginIn,
    request: Request,
    response: Response,
    _: None = Depends(enforce_same_origin),
):
    """Exchange email+password for a session cookie.

    Guarded by ``enforce_same_origin`` even though it is unauthenticated: without it a
    hostile page could force a login with the *attacker's* credentials, so the victim's
    subsequent authenticated actions would post into the attacker's account ("login CSRF").

    An unknown address, a known address with a wrong password and a deactivated account are
    all indistinguishable: the identical 401 body and the identical full bcrypt verify cost,
    because the unknown-address path verifies against a fixed dummy hash rather than
    skipping the check.

    The per-account rate limit holds to the same rule -- keyed on the submitted address
    alone -- and counts FAILED attempts only, applied after the credential check. Counting
    every attempt and gating first made the throttle an account-lockout weapon: twenty
    anonymous wrong-password requests from twenty source addresses would lock the real owner
    out indefinitely without ever guessing a password.

    It deliberately does NOT bound attacker COST: the check runs after the bcrypt verify,
    so being refused is free. The per-IP limiter is the control that bounds attacker cost;
    anything making this counter do that job has to consult it BEFORE the verify, which
    reinstates both the lockout primitive and the existence timing oracle.

    A successful login also REWRITES the stored hash if it predates the pre-image scheme.
    That is a write on the read-looking path, and it is deliberate: the plaintext exists in
    this request and nowhere else, so this is the only moment a pre-migration credential can
    be re-expressed without the account owner doing anything but logging in as they already
    do. It revokes nothing and can never fail the login (see ``_upgrade_password_hash``).
    """
    await _check_rate_limit(request, "login", config.AUTH_LOGIN_RATE_PER_MIN)
    email = validate_email(body.email)
    s = _require_auth_store()
    user = await s.get_user_by_email(email)
    # Always pay the bcrypt cost, even with no account to compare against: short-circuiting
    # the verify skipped ~100ms and was a remote account-existence oracle.
    password_ok = await asyncio.to_thread(
        verify_password,
        body.password,
        user.password_hash if user is not None else _DUMMY_PASSWORD_HASH,
    )
    if password_ok and user is not None and user.is_active:
        # A real user with a real password is never counted, never gated, never
        # rate-limited. Only failures consume the account's budget.
        if needs_rehash(user.password_hash):
            # Opportunistic migration: the plaintext exists only in this request, so this
            # is the one moment the pre-migration hash can be re-expressed.
            await _upgrade_password_hash(s, user, body.password)
        token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)
        # Delivered ONLY as the HttpOnly cookie, never in the body, so there is nothing for
        # script on the page -- including XSS-injected script -- to read and exfiltrate.
        _set_session_cookie(response, token)
        return AuthOut(user=UserOut.from_user(user))

    # Failed. Counted against the address, keyed on the submitted string alone (no lookup
    # feeds this), so a registered and an unregistered address are indistinguishable here
    # too. Raises 429 past the limit.
    await _check_account_rate_limit(
        request, "login", config.AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN, email
    )
    raise HTTPException(status_code=401, detail="invalid email or password")


async def _upgrade_password_hash(s: AuthStore, user: StoredUser, password: str) -> None:
    """Re-store ``user``'s just-verified password in the current scheme.

    Only ever called after ``verify_password`` accepted ``password``, so the credential
    cannot change meaning here: the owner keeps logging in with the same value and every
    token already issued stays valid. Nothing is revoked, because this rewrites how one
    secret is stored, not which secret it is -- the opposite of ``change_password``.

    Never raises. A failure here is logged and dropped: the row is untouched and its
    pre-migration hash still authenticates, so a housekeeping write can never turn into an
    authentication outage. The user id is logged rather than the address so this line is not
    a record of who has successfully authenticated.
    """
    try:
        new_hash = await asyncio.to_thread(hash_password, password)
        changed = await s.upgrade_password_hash(user.id, user.password_hash, new_hash)
    except Exception:
        logger.warning(
            "could not upgrade the stored password hash for user %s; the existing hash still "
            "authenticates and the upgrade is retried on the next login",
            user.id,
            exc_info=True,
        )
        return
    if changed:
        logger.info("upgraded the stored password hash for user %s to the current scheme", user.id)


@router.get("/me", response_model=UserOut)
async def me(request: Request, _: None = Depends(require_auth)):
    return UserOut.from_user(request.state.user)


@router.post("/logout")
async def logout(request: Request, response: Response, _: None = Depends(require_auth)):
    """Revoke the credential this request authenticated with, and expire the cookie.

    For a user session the token is read from the cookie, which is the only place it can now
    be -- re-parsing a header here would leave the stored token live and turn logout into a
    silent no-op.
    """
    service = getattr(request.state, "service_token", None)
    if service is not None:
        await _require_auth_store().revoke_service_token(service)
    else:
        token = _token_from_request(request)
        if token is not None:
            await _require_auth_store().revoke_token(token)
    _clear_session_cookie(response)
    return {"ok": True}


@router.post("/change-password")
async def change_password(
    body: ChangePasswordIn,
    request: Request,
    response: Response,
    _: None = Depends(require_auth),
):
    """Change the current user's password after verifying the old one. Revokes every other
    token the user holds and re-issues the session cookie with a fresh token, so this
    session stays signed in and no other one does.

    The new password is judged by the same policy ``signup`` and ``bootstrap_admin`` use
    (see ``_password_rejection``), and stored the same way, so an admin seeded with a
    passphrase of any length can re-apply that very passphrase here.

    "Invalidates" holds even if the worker is killed mid-request: the hash write, the
    revocation and the replacement token commit together or not at all (see
    ``AuthStore.change_password``).
    """
    user = request.state.user
    s = _require_auth_store()
    stored = await s.get_user(user.id)
    if stored is None or not await asyncio.to_thread(
        verify_password, body.current_password, stored.password_hash
    ):
        raise HTTPException(status_code=400, detail="current password is incorrect")
    new_password = validate_password(body.new_password)
    # Hash first, then take the write lock: bcrypt is the slow part and has no business
    # being spent holding it.
    new_hash = await asyncio.to_thread(hash_password, new_password)
    # ONE transaction for the hash write, the revocation and the replacement token. Do not
    # split this back into set_password / revoke_all_tokens / issue_token: that shape leaves
    # a durable new password valid alongside still-authenticating pre-existing tokens
    # whenever the worker dies between them.
    token = await s.change_password(user.id, new_hash, config.AUTH_TOKEN_TTL_DAYS)
    # Revocation killed every token this user held, including the one in the cookie, so the
    # cookie has to be re-issued or the session dies on the very next request.
    _set_session_cookie(response, token)
    return AuthOut(user=UserOut.from_user(stored))


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
    """Update a user's name/role/is_active. Protects the last active admin from demotion or
    deactivation."""
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
    await _require_auth_store().revoke_all_tokens(user_id)
    return {"ok": True}


@router.post("/service-tokens", response_model=ServiceTokenOut)
async def mint_service_token(
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Mint a scoped, expiring machine credential (rotation). Neither the scope nor the
    lifetime can be widened per request.

    Requires ``users:manage``, which a service token scoped to the shipped default
    (``chat:use``) does not hold, so a leaked machine credential cannot mint itself a
    successor.
    """
    raw, record = await _require_auth_store().issue_service_token(
        set(_service_token_scope()), _service_token_ttl_seconds()
    )
    return ServiceTokenOut(token=raw, scope=sorted(record.scope), expires_at=record.expires_at)


@router.post("/service-tokens/revoke")
async def revoke_service_tokens(
    request: Request,
    body: ServiceTokenRevokeIn | None = None,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Revoke a service token: the other half of rotation.

    Pass the token to retire and only that one dies, which is what makes rotation safe in
    the order an operator naturally reaches for -- mint the replacement, move consumers
    onto it, *then* kill the old one, without a window in which no credential works. Posting
    no body at all revokes every live service token at once: right for a suspected leak,
    wrong for a planned rotation because it takes down the replacement minted moments
    earlier.

    ``revoked`` is the number of rows actually changed, so revoking an unknown or
    already-revoked token reports 0 rather than a reassuring 1.

    The value in ``AUTH_SERVICE_TOKEN`` is handled specially in both forms, writing a
    revoked tombstone instead of a plain UPDATE, so "kill this now" works from a cold start
    (see ``revoke_configured_service_token``).
    """
    s = _require_auth_store()
    configured = config.AUTH_SERVICE_TOKEN or ""
    if body and body.token:
        if configured and tokens_match(body.token, configured):
            return {"revoked": await s.revoke_configured_service_token(body.token, set(_service_token_scope()))}
        return {"revoked": await s.revoke_service_token(body.token)}
    revoked = await s.revoke_all_service_tokens()
    if configured:
        revoked += await s.revoke_configured_service_token(configured, set(_service_token_scope()))
    return {"revoked": revoked}


async def report_legacy_password_hashes() -> int | None:
    """Log how many accounts still hold a pre-migration password hash, and return that
    count (None when it could not be determined). Never raises and never blocks startup.

    There is deliberately no force option. Revoking a pre-migration hash is the only way to
    finish one without the plaintext, and this codebase has no password-reset path to
    revoke one with: a force switch would not migrate anything, it would permanently lock
    out every account that has not logged in since the upgrade.
    """
    s = store
    if s is None:
        return None
    try:
        remaining = await s.count_legacy_passwords()
    except Exception:
        logger.warning("could not count pre-migration password hashes", exc_info=True)
        return None
    if remaining:
        logger.info(
            "auth: %d account(s) still hold a pre-migration password hash. Each authenticates on "
            "the first %d bytes of its password and is rewritten to the full password on its next "
            "successful login. An account that never logs in again keeps the old hash, which is "
            "the exposure it always had: there is no way to rehash a password without its "
            "plaintext, and no password-reset path to revoke one with.",
            remaining,
            _LEGACY_BCRYPT_MAX_BYTES,
        )
    else:
        logger.info("auth: no pre-migration password hashes remain")
    return remaining


async def bootstrap_admin() -> None:
    """Seed the bootstrap admin from config (once, at startup). Never overwrites an existing
    account's password. Safe under concurrent worker startups: the duplicate / write-lock
    races are handled instead of failing startup.

    A value the validators reject is refused, loudly, and no account is created -- but the
    process still starts, so a typo in one env var cannot take the whole API (and /health)
    down and leave nobody able to reach the service to fix it.

    An admin account left behind by an earlier run with weak credentials is deliberately NOT
    deleted: this runs at startup, unauthenticated, and removing the only admin account
    would lock every operator out of their own deployment.
    """
    email = (config.AUTH_ADMIN_EMAIL or "").strip().lower()
    password = config.AUTH_ADMIN_PASSWORD or ""
    if not email or not password:
        return
    # The config values are the one remaining path into the user table that does not go
    # through the validators, so a typo like AUTH_ADMIN_PASSWORD=x used to provision a
    # full-admin account with a 1-character password the signup endpoint would itself have
    # rejected.
    #
    # The rejection is a permanent, config-level fault: retrying it inside the write-lock
    # loop below would re-log the identical error and change nothing, so it returns before
    # reaching that loop. The loop still retries genuine transient faults (SQLite write
    # locks).
    #
    # The password is validated first, and unconditionally, so the advice given for an EMAIL
    # fault can still say whether the account behind it is on a weak password: a deploy that
    # ran pre-validator main can have both faults at once.
    password_error = _password_rejection(password)

    email_error = _validator_rejection(validate_email, email)
    if email_error:
        await _reject_bootstrap(
            "AUTH_ADMIN_EMAIL",
            email_error,
            hint=f"set it to a valid address (max {config.AUTH_MAX_EMAIL_LEN} characters) and restart",
            password_rejected=password_error,
        )
        return
    if password_error:
        await _reject_bootstrap(
            "AUTH_ADMIN_PASSWORD",
            password_error,
            hint=_password_hint(password_error),
            password_rejected=password_error,
        )
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


async def _reject_bootstrap(
    variable: str, reason: str, *, hint: str, password_rejected: str | None
) -> None:
    """Log that the configured bootstrap admin credentials were refused, and report (never
    delete) an account a previous run already created from the same bad value.

    Must stay loud and specific: the operator has to learn that AUTH_ADMIN_PASSWORD -- not
    the login endpoint -- is the thing to fix.
    """
    # A weak admin may already exist from a run that predates the validators. Removing it
    # here would be an unauthenticated, startup-time way to delete the only admin account
    # and lock every operator out, so the row is left alone and surfaced loudly instead.
    #
    # "Rotate" is demanded only when the configured value is genuinely the account's current
    # password; a value that merely fails validation says nothing about a healthy admin's.
    if store is not None:
        probe = (config.AUTH_ADMIN_EMAIL or "").strip().lower()
        existing = None
        try:
            existing = await store.get_user_by_email(probe) if probe else None
        except Exception as exc:  # noqa: BLE001 - a probe must never break startup
            logger.warning(
                "could not check whether admin %s already exists (%s: %s); reporting the rejected "
                "bootstrap config on its own merits.",
                probe, type(exc).__name__, exc,
            )
            existing = None
        if existing is not None and existing.role == "admin":
            # Rotation needs BOTH facts about the ACCOUNT, neither being "which variable
            # failed": the stored password must be the configured one (proving this account
            # came from the bad config), AND the password must itself have failed
            # validation. Requiring both means an email fault on a healthy admin never nags.
            if password_rejected and verify_password(
                config.AUTH_ADMIN_PASSWORD or "", existing.password_hash
            ):
                # The remedy here is a PASSWORD rotation, so the hint must be the password
                # one; passing `hint` through would send the operator to the wrong variable.
                logger.error(
                    "bootstrap admin %s is REJECTED by validation: %s rejected the configured %s: %s. "
                    "No account was created, and the pre-existing admin account (id %s) -- whose "
                    "current password IS the rejected AUTH_ADMIN_PASSWORD, so it was provisioned "
                    "from this non-compliant value before these checks existed -- was left in place. "
                    "Rotate that account's password out of band, and also correct %s: %s. "
                    "To rotate the password: %s.",
                    probe, _validator_name_for(variable), variable, reason, existing.id, variable,
                    hint, _password_hint("password must contain a letter and a digit"),
                )
            elif _validator_name_for(variable) != "validate_password":
                logger.error(
                    "bootstrap admin NOT created: %s rejected the configured %s: %s. The "
                    "pre-existing admin account (id %s) was left untouched and keeps its current "
                    "password, so it needs no rotation -- the fault is the configured address, not "
                    "the account. %s.",
                    _validator_name_for(variable), variable, reason, existing.id, hint,
                )
            else:
                logger.error(
                    "bootstrap admin NOT created: %s rejected the configured %s: %s. An unrelated "
                    "admin account (id %s) already exists on a different, valid password, so it was "
                    "left untouched and needs no rotation. %s.",
                    _validator_name_for(variable), variable, reason, existing.id, hint,
                )
            return
    logger.error(
        "bootstrap admin NOT created: %s rejected the configured %s: %s. This is a configuration "
        "error, not a transient fault, so it is not retried. Startup continues with no admin "
        "account: %s. The service is up, so you can fix the config and restart.",
        _validator_name_for(variable), variable, reason, hint,
    )


def _password_hint(reason: str) -> str:
    """The remediation hint for a ``validate_password`` reason, keyed on the reason so the
    advice can never contradict it -- one fixed hint would tell an operator to lengthen a
    password that was rejected for having no digit."""
    if "at least" in reason:
        return (
            f"set it to a password of at least {config.AUTH_PASSWORD_MIN_LEN} characters "
            "containing both a letter and a digit, then restart"
        )
    if "letter and a digit" in reason:
        return "set it to a password containing both a letter and a digit, then restart"
    return "set AUTH_ADMIN_PASSWORD to a password that satisfies the password policy, then restart"


def _validator_name_for(variable: str) -> str:
    return "validate_email" if variable.endswith("EMAIL") else "validate_password"
