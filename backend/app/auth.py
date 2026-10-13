

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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    "admin": {"chat:use", "analytics:read", "users:read", "users:manage", "feed:use"},
    "user": {"chat:use", "feed:use"},
}


SERVICE_USER_ID = "service-token"

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

















_PASSWORD_SCHEME = "$bcrypt-sha256$"



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


    user: UserOut


class SignupOut(BaseModel):


    message: str








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


    last_seen: float | None = None


@dataclass
class StoredServiceToken:


    token_hash: str
    scope: frozenset[str]
    created_at: float
    expires_at: float


class ServiceTokenOut(BaseModel):


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

    if len(password) < config.AUTH_PASSWORD_MIN_LEN:
        return f"password must be at least {config.AUTH_PASSWORD_MIN_LEN} characters"
    if not _has_letter_and_digit(password):
        return "password must contain a letter and a digit"
    return None


def validate_password(password: str) -> str:

    reason = _password_rejection(password)
    if reason:
        raise HTTPException(status_code=422, detail=reason)
    return password


def _validator_rejection(validator, value: str) -> str | None:

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

    return hashlib.sha256(password.encode("utf-8")).digest()


def hash_password(password: str) -> str:

    digest = bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("utf-8")
    return _PASSWORD_SCHEME + digest


def needs_rehash(hashed: str) -> bool:

    return not hashed.startswith(_PASSWORD_SCHEME)


def verify_password(password: str, hashed: str) -> bool:

    if hashed.startswith(_PASSWORD_SCHEME):
        stored, preimage = hashed[len(_PASSWORD_SCHEME):], _prehash(password)
    else:
        stored = hashed
        preimage = password.encode("utf-8")[:_LEGACY_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(preimage, stored.encode("utf-8"))
    except ValueError:



        return False










_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(presented: str, expected: str) -> bool:

    return secrets.compare_digest(
        presented.encode("utf-8", "surrogateescape"), expected.encode("utf-8", "surrogateescape")
    )


def _now() -> float:
    return time.time()


class DuplicateEmailError(Exception):
    pass


class AuthStore:


    def __init__(self, path: str):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:




        if self._db is not None:
            return
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




        cols = await self._db.execute_fetchall("PRAGMA table_info(users)")
        col_names = {row["name"] for row in cols} if cols else set()
        if "last_seen" not in col_names:
            await self._db.execute("ALTER TABLE users ADD COLUMN last_seen REAL")
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @asynccontextmanager
    async def _unit_of_work(self) -> AsyncIterator[aiosqlite.Connection]:

        db = await aiosqlite.connect(self._path, isolation_level=None)
        try:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute("BEGIN IMMEDIATE")
            yield db
            await db.commit()
        except BaseException:




            await db.rollback()
            raise
        finally:
            await db.close()

    async def _fetchone(self, query: str, params: tuple = ()):
        rows = await self._db.execute_fetchall(query, params)
        return rows[0] if rows else None

    async def _fetchall(self, query: str, params: tuple = ()):
        return await self._db.execute_fetchall(query, params)

    @staticmethod
    def _to_user(row) -> StoredUser:
        last_seen = row["last_seen"] if "last_seen" in row.keys() else None  # noqa: SIM118 (sqlite3.Row `in` tests values, not keys)
        return StoredUser(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"],
            name=row["name"] or "",
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=float(row["created_at"]),
            last_seen=float(last_seen) if last_seen is not None else None,
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
            async with self._unit_of_work() as db:
                await db.execute(
                    "INSERT INTO users (id, email, password_hash, name, role, is_active, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user.id, user.email, user.password_hash, user.name, user.role, 1, user.created_at),
                )
        except sqlite3.IntegrityError:





            raise DuplicateEmailError(email) from None
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



            where += (
                " AND NOT (role = 'admin'"
                " AND (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1"
                " AND (? OR ?))"
            )
            params.append(1 if role == "user" else 0)
            params.append(1 if is_active is False else 0)
        async with self._unit_of_work() as db:
            cur = await db.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE {where}", tuple(params)
            )
            return cur.rowcount

    async def delete_user(self, user_id: str, guard_last_admin: bool = False) -> int:

        where = "id = ?"
        params: tuple = (user_id,)
        if guard_last_admin:
            where += (
                " AND NOT (role = 'admin'"
                " AND (SELECT COUNT(*) FROM users WHERE role = 'admin') <= 1)"
            )
        async with self._unit_of_work() as db:
            cur = await db.execute(f"DELETE FROM users WHERE {where}", params)
            return cur.rowcount

    async def set_password(self, user_id: str, password_hash: str) -> None:

        async with self._unit_of_work() as db:
            await db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
            )

    async def upgrade_password_hash(self, user_id: str, observed_hash: str, new_hash: str) -> int:

        async with self._unit_of_work() as db:
            cur = await db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ? AND password_hash = ?",
                (new_hash, user_id, observed_hash),
            )
            return cur.rowcount

    async def count_legacy_passwords(self) -> int:

        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM users WHERE substr(password_hash, 1, ?) <> ?",
            (len(_PASSWORD_SCHEME), _PASSWORD_SCHEME),
        )
        return int(row["n"]) if row else 0

    async def count_admins(self) -> int:
        row = await self._fetchone("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
        return int(row["n"]) if row else 0

    async def touch_last_seen(self, user_id: str, now: float | None = None) -> None:

        async with self._unit_of_work() as db:
            await db.execute(
                "UPDATE users SET last_seen = ? WHERE id = ?",
                (now if now is not None else _now(), user_id),
            )

    async def account_stats(self, now: float | None = None) -> dict:

        stamp = now if now is not None else _now()
        today_start = int(stamp // 86400) * 86400

        signup_total = await self._fetchone("SELECT COUNT(*) AS n FROM users")
        signup_today = await self._fetchone(
            "SELECT COUNT(*) AS n FROM users WHERE created_at >= ?", (today_start,)
        )
        role_rows = await self._fetchall("SELECT role, COUNT(*) AS n FROM users GROUP BY role")
        disabled = await self._fetchone(
            "SELECT COUNT(*) AS n FROM users WHERE is_active = 0"
        )

        last_seen_rows = await self._fetchall(
            "SELECT last_seen FROM users WHERE last_seen IS NOT NULL"
        )
        ls_vals = [float(r["last_seen"]) for r in last_seen_rows]
        active_today = sum(1 for ls in ls_vals if ls >= today_start)
        active_7d = sum(1 for ls in ls_vals if ls >= stamp - 7 * 86400)
        active_30d = sum(1 for ls in ls_vals if ls >= stamp - 30 * 86400)


        signup_days: dict[str, int] = {}
        rows = await self._fetchall("SELECT created_at FROM users")
        for r in rows:
            day = time.strftime("%Y-%m-%d", time.gmtime(float(r["created_at"])))
            signup_days[day] = signup_days.get(day, 0) + 1
        today_iso = time.strftime("%Y-%m-%d", time.gmtime(stamp))
        days = []
        start = datetime.fromisoformat(today_iso).date() - timedelta(days=13)
        for i in range(14):
            d = (start + timedelta(days=i)).isoformat()
            days.append([d, signup_days.get(d, 0)])

        return {
            "signups": {
                "today": int(signup_today["n"]) if signup_today else 0,
                "total": int(signup_total["n"]) if signup_total else 0,
            },
            "role_distribution": {r["role"]: int(r["n"]) for r in role_rows},
            "disabled_accounts": int(disabled["n"]) if disabled else 0,
            "active_today": active_today,
            "active_last_7d": active_7d,
            "active_last_30d": active_30d,
            "signups_14d": days,
        }


    async def issue_token(self, user_id: str, ttl_days: int) -> str:

        raw = secrets.token_urlsafe(32)
        created = _now()
        async with self._unit_of_work() as db:
            await db.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (hash_token(raw), user_id, created, created + ttl_days * 86400),
            )
            await self._revict_tokens_over_cap(db, user_id)
        return raw

    async def _revict_tokens_over_cap(self, db, user_id: str) -> int:

        cap = int(getattr(config, "AUTH_MAX_ACTIVE_TOKENS_PER_USER", 0))
        if cap <= 0:
            return 0
        rows = await db.execute_fetchall(
            "SELECT token_hash FROM auth_tokens"
            " WHERE user_id = ? AND expires_at >= ?"
            " ORDER BY rowid ASC",
            (user_id, _now()),
        )
        surplus = len(rows) - cap
        if surplus <= 0:
            return 0
        victims = [row[0] for row in rows[:surplus]]
        for token_hash in victims:
            await db.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (token_hash,))
        return len(victims)

    async def active_token_count(self, user_id: str) -> int:

        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM auth_tokens WHERE user_id = ? AND expires_at >= ?",
            (user_id, _now()),
        )
        return int(row["n"]) if row else 0



    async def issue_service_token(self, scope: set[str], ttl_seconds: float) -> tuple[str, StoredServiceToken]:

        raw = secrets.token_urlsafe(32)
        created = _now()
        record = StoredServiceToken(
            token_hash=hash_token(raw),
            scope=frozenset(scope),
            created_at=created,
            expires_at=created + ttl_seconds,
        )
        async with self._unit_of_work() as db:
            await db.execute(
                "INSERT INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
                " VALUES (?, ?, ?, ?, NULL)",
                (record.token_hash, ",".join(sorted(record.scope)), record.created_at, record.expires_at),
            )
        return raw, record

    async def ensure_bootstrap_service_token(self, raw: str, scope: set[str], ttl_seconds: float) -> None:




        created = _now()
        async with self._unit_of_work() as db:
            await db.execute(
                "INSERT OR IGNORE INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
                " VALUES (?, ?, ?, ?, NULL)",
                (hash_token(raw), ",".join(sorted(scope)), created, created + ttl_seconds),
            )

    async def service_token_for(self, raw: str) -> StoredServiceToken | None:

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

        async with self._unit_of_work() as db:
            cur = await db.execute(
                "UPDATE auth_service_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
                (_now(), hash_token(raw)),
            )
            return cur.rowcount

    async def revoke_configured_service_token(self, raw: str, scope: set[str]) -> int:

        now = _now()
        async with self._unit_of_work() as db:
            cur = await db.execute(
                "INSERT INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(token_hash) DO UPDATE SET revoked_at = excluded.revoked_at"
                " WHERE auth_service_tokens.revoked_at IS NULL",
                (hash_token(raw), ",".join(sorted(scope)), now, now, now),
            )
            return cur.rowcount

    async def revoke_all_service_tokens(self) -> int:
        async with self._unit_of_work() as db:
            cur = await db.execute(
                "UPDATE auth_service_tokens SET revoked_at = ? WHERE revoked_at IS NULL", (_now(),)
            )
            return cur.rowcount

    async def purge_dead_service_tokens(self, keep_hash: str = "") -> int:

        now = _now()
        async with self._unit_of_work() as db:
            cur = await db.execute(
                "DELETE FROM auth_service_tokens"
                " WHERE (revoked_at IS NOT NULL OR expires_at < ?)"
                " AND (? = '' OR token_hash != ?)",
                (now, keep_hash, keep_hash),
            )
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
        async with self._unit_of_work() as db:
            await db.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?", (hash_token(raw_token),)
            )

    async def revoke_all_tokens(self, user_id: str) -> None:
        async with self._unit_of_work() as db:
            await db.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))

    async def change_password(self, user_id: str, new_password_hash: str, ttl_days: int) -> str:

        if self._db is None:
            raise RuntimeError("auth store is not connected")
        db = await aiosqlite.connect(self._path, isolation_level=None)
        try:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute("BEGIN IMMEDIATE")
            await db.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
            updated = await db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id)
            )
            if updated.rowcount == 0:





                raise ValueError(f"user {user_id} no longer exists")
            raw = secrets.token_urlsafe(32)
            created = _now()
            await db.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (hash_token(raw), user_id, created, created + ttl_days * 86400),
            )
            await db.commit()
        except BaseException:




            await db.rollback()
            raise
        finally:
            await db.close()
        return raw

    async def purge_expired_tokens(self) -> int:

        now = _now()
        async with self._unit_of_work() as db:
            cur = await db.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (now,))
            return cur.rowcount


def _require_auth_store() -> AuthStore:
    if store is None:
        raise HTTPException(status_code=503, detail="auth store not initialized")
    return store




_rate_client = None


def _rate_redis() -> aioredis.Redis:
    global _rate_client
    if _rate_client is None:







        _rate_client = aioredis.from_url(
            config.REDIS_URL,
            db=config.AUTH_RATE_LIMIT_REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _rate_client


async def close_rate_redis() -> None:

    global _rate_client
    if _rate_client is not None:
        await _rate_client.aclose()
        _rate_client = None


async def token_purge_loop() -> None:

    interval = config.AUTH_TOKEN_PURGE_INTERVAL_SECONDS
    while interval > 0:
        try:
            if store is not None:
                n = await store.purge_expired_tokens()
                if n:
                    logger.info("auth: purged %d expired token(s)", n)


                keep = hash_token(config.AUTH_SERVICE_TOKEN) if config.AUTH_SERVICE_TOKEN else ""
                m = await store.purge_dead_service_tokens(keep_hash=keep)
                if m:
                    logger.info("auth: purged %d dead service token(s)", m)
        except Exception:
            logger.exception("auth token purge failed")
        await asyncio.sleep(interval)


def _peer_is_local_proxy(peer: str | None) -> bool:

    if not peer:
        return False
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


def _trust_forwarded_for(peer: str | None) -> bool:

    configured = config.AUTH_TRUST_X_FORWARDED_FOR
    if configured is not None:
        return configured
    return _peer_is_local_proxy(peer)


def _client_ip(request: Request) -> str:

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

    try:
        rc = _rate_redis()





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







_local_rate_lock = threading.Lock()

_local_rate_counters: dict[str, tuple[int, float]] = {}













_LOCAL_RATE_MAX_KEYS = 20_000


def _local_rate_hit(key: str, window: int) -> int:

    now = time.monotonic()
    with _local_rate_lock:
        count, expiry = _local_rate_counters.get(key, (0, now + window))
        if now >= expiry:
            count, expiry = 0, now + window
        count += 1


        _local_rate_counters.pop(key, None)
        _local_rate_counters[key] = (count, expiry)
        if len(_local_rate_counters) > _LOCAL_RATE_MAX_KEYS:
            _prune_local_rate_counters(now)
        return count


def _prune_local_rate_counters(now: float) -> None:

    for key in [k for k, (_, expiry) in _local_rate_counters.items() if expiry <= now]:
        del _local_rate_counters[key]
    over = len(_local_rate_counters) - _LOCAL_RATE_MAX_KEYS
    if over <= 0:
        return

    for stale in list(_local_rate_counters)[:over]:
        del _local_rate_counters[stale]


def reset_local_rate_limits() -> None:

    with _local_rate_lock:
        _local_rate_counters.clear()


def public_rate_limit(
    action: str,
    limit_attr: str,
    *,
    fail_closed: bool = True,
) -> Callable[[Request], Awaitable[None]]:


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

    return request.cookies.get(config.AUTH_COOKIE_NAME) or None


def _host_only(authority: str) -> str:

    host = authority.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end]
        return host
    if host.count(":") > 1:



        return host
    return host.partition(":")[0]


def _origin_host(origin: str) -> str | None:

    try:
        return urlsplit(origin.strip()).hostname
    except ValueError:
        return None


def _request_host(request: Request) -> str | None:

    host = request.headers.get("host")
    return host.strip() if host and host.strip() else None


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


async def enforce_same_origin(request: Request) -> None:

    if request.method.upper() not in _UNSAFE_METHODS:
        return
    if (request.headers.get("sec-fetch-site") or "").strip().lower() == "cross-site":
        raise HTTPException(status_code=403, detail="cross-site request rejected")
    origin = request.headers.get("origin")
    if origin is None:
        return
    origin_host = _origin_host(origin)
    if not origin_host:


        raise HTTPException(status_code=403, detail="cross-site request rejected")
    expected = _request_host(request)
    if expected is None or _host_only(origin_host) != _host_only(expected):
        raise HTTPException(status_code=403, detail="cross-site request rejected")


def _set_session_cookie(response: Response, token: str) -> None:

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

    configured = int(getattr(config, "AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", 0) or 0)
    return float(configured if configured > 0 else 86400)


def _service_token_scope() -> frozenset[str]:

    known = {p for role in ROLE_PERMISSIONS.values() for p in role}
    requested = tuple(getattr(config, "AUTH_SERVICE_TOKEN_SCOPE", ()) or ())
    scope = {p for p in requested if p in known}
    if len(scope) != len(set(requested)):
        logger.warning("AUTH_SERVICE_TOKEN_SCOPE names unknown permissions; ignoring them")
    return frozenset(scope)


async def _resolve_service_token(raw: str) -> StoredServiceToken | None:

    s = _require_auth_store()
    record = await s.service_token_for(raw)
    if record is not None:
        return record
    if config.AUTH_SERVICE_TOKEN and tokens_match(raw, config.AUTH_SERVICE_TOKEN):
        await s.ensure_bootstrap_service_token(raw, set(_service_token_scope()), _service_token_ttl_seconds())
        return await s.service_token_for(raw)
    return None









_last_seen_touched: dict[str, float] = {}


def _last_seen_due(user_id: str, now: float) -> bool:
    interval = config.LAST_SEEN_TOUCH_INTERVAL_SECONDS
    if interval <= 0:
        return True
    last = _last_seen_touched.get(user_id)
    if last is None or now - last >= interval:
        _last_seen_touched[user_id] = now
        return True
    return False


async def _touch_last_seen(user_id: str) -> None:

    if not _last_seen_due(user_id, _now()):
        return
    s = store
    if s is None:
        return
    try:
        await s.touch_last_seen(user_id)
    except Exception:



        logger.debug("last_seen touch failed for %s", user_id, exc_info=True)


def _login_day_key(stamp: float | None = None) -> str:

    return f"analytics:login:day:{time.strftime('%Y-%m-%d', time.gmtime(stamp if stamp is not None else _now()))}"


async def _count_login() -> None:

    try:
        await _rate_redis().incr(_login_day_key())
    except Exception:
        logger.debug("login-day counter INCR failed", exc_info=True)


def _chat_audit_store():

    from app.chat import _require_store
    return _require_store()


async def _record_admin_audit(actor_id: str, action: str) -> None:

    try:
        await _chat_audit_store().record_admin_audit(actor_id, action)
    except Exception:
        logger.warning("admin audit write failed for %s", action, exc_info=True)



async def require_auth(request: Request) -> None:




    await enforce_same_origin(request)
    service = request.headers.get("x-service-token")
    if service:
        record = await _resolve_service_token(service)
        if record is not None:
            request.state.user = _service_user()
            request.state.user_id = SERVICE_USER_ID



            request.state.scope = record.scope
            request.state.service_token = service
            return



        logger.debug("auth: a presented service token did not resolve")
    token = _token_from_request(request)
    if token is None:
        raise HTTPException(status_code=401, detail="authentication required")
    user = await _require_auth_store().user_for_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    request.state.user = user
    request.state.user_id = user.id


    await _touch_last_seen(user.id)


def require_permission(permission: str):


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

    await _check_rate_limit(request, "signup", config.AUTH_SIGNUP_RATE_PER_MIN)
    email = validate_email(body.email)
    password = validate_password(body.password)
    name = validate_name(body.name)
    s = _require_auth_store()





    try:
        await s.create_user(email, password, name, role=SIGNUP_ROLE)
    except DuplicateEmailError:



        logger.info("signup for an already-registered address: reported as accepted")


    return SignupOut(message=SIGNUP_ACCEPTED_MESSAGE)


@router.post("/login", response_model=AuthOut)
async def login(
    body: LoginIn,
    request: Request,
    response: Response,
    _: None = Depends(enforce_same_origin),
):

    await _check_rate_limit(request, "login", config.AUTH_LOGIN_RATE_PER_MIN)
    email = validate_email(body.email)
    s = _require_auth_store()
    user = await s.get_user_by_email(email)



    password_ok = await asyncio.to_thread(
        verify_password,
        body.password,
        user.password_hash if user is not None else _DUMMY_PASSWORD_HASH,
    )
    if password_ok and user is not None and user.is_active:

        if needs_rehash(user.password_hash):



            await _upgrade_password_hash(s, user, body.password)
        token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)


        _set_session_cookie(response, token)


        await _touch_last_seen(user.id)
        await _count_login()
        return AuthOut(user=UserOut.from_user(user))




    await _check_account_rate_limit(
        request, "login", config.AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN, email
    )
    raise HTTPException(status_code=401, detail="invalid email or password")


async def _upgrade_password_hash(s: AuthStore, user: StoredUser, password: str) -> None:

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

    user = request.state.user
    s = _require_auth_store()
    stored = await s.get_user(user.id)
    if stored is None or not await asyncio.to_thread(
        verify_password, body.current_password, stored.password_hash
    ):
        raise HTTPException(status_code=400, detail="current password is incorrect")
    new_password = validate_password(body.new_password)


    new_hash = await asyncio.to_thread(hash_password, new_password)





    token = await s.change_password(user.id, new_hash, config.AUTH_TOKEN_TTL_DAYS)



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

    s = _require_auth_store()
    target = await _get_user_or_404(s, user_id)
    if body.role is not None and body.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail="invalid role")
    is_demote = target.role == "admin" and (body.role == "user" or body.is_active is False)
    name = validate_name(body.name) if body.name is not None else None
    rowcount = await s.update_user(user_id, name, body.role, body.is_active, guard_last_admin=is_demote)
    if is_demote and rowcount == 0:
        raise HTTPException(status_code=400, detail="cannot demote or deactivate the last admin")
    await _record_admin_audit(request.state.user_id, "users.patch")
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
    await _record_admin_audit(request.state.user_id, "users.delete")
    return {"ok": True}


@router.post("/users/{user_id}/tokens/revoke")
async def revoke_user_tokens(
    user_id: str,
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):

    await _require_auth_store().revoke_all_tokens(user_id)
    await _record_admin_audit(request.state.user_id, "users.revoke_tokens")
    return {"ok": True}


@router.post("/service-tokens", response_model=ServiceTokenOut)
async def mint_service_token(
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):

    raw, record = await _require_auth_store().issue_service_token(
        set(_service_token_scope()), _service_token_ttl_seconds()
    )
    await _record_admin_audit(request.state.user_id, "users.mint_service_token")
    return ServiceTokenOut(token=raw, scope=sorted(record.scope), expires_at=record.expires_at)


@router.post("/service-tokens/revoke")
async def revoke_service_tokens(
    request: Request,
    body: ServiceTokenRevokeIn | None = None,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):

    s = _require_auth_store()
    configured = config.AUTH_SERVICE_TOKEN or ""
    if body and body.token:
        if configured and tokens_match(body.token, configured):
            return {"revoked": await s.revoke_configured_service_token(body.token, set(_service_token_scope()))}
        return {"revoked": await s.revoke_service_token(body.token)}
    revoked = await s.revoke_all_service_tokens()
    if configured:
        revoked += await s.revoke_configured_service_token(configured, set(_service_token_scope()))
    await _record_admin_audit(request.state.user_id, "users.revoke_service_tokens")
    return {"revoked": revoked}



async def report_legacy_password_hashes() -> int | None:

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

    email = (config.AUTH_ADMIN_EMAIL or "").strip().lower()
    password = config.AUTH_ADMIN_PASSWORD or ""
    if not email or not password:
        return

















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






            if password_rejected and verify_password(
                config.AUTH_ADMIN_PASSWORD or "", existing.password_hash
            ):




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
