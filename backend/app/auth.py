"""Session-cookie + RBAC authentication for the API.

Signup issues no cookie and always answers the same thing (see the endpoint).
Login issues an opaque token (hashed with SHA-256 in storage, expiring after
AUTH_TOKEN_TTL_DAYS, individually revocable, several per user but capped at
AUTH_MAX_ACTIVE_TOKENS_PER_USER active ones, oldest revoked past the cap) and
delivers it ONLY as an HttpOnly cookie named ``config.AUTH_COOKIE_NAME``. The
token is never in a response body and is never read from a header: a token a
browser has to hold in script-readable storage is one XSS bug away from a
durable account takeover, and a header path stays reachable from ``fetch()``,
so keeping one alongside the cookie would leave the exfiltration surface open.

Because the cookie is attached by the browser automatically, every
cookie-authenticated unsafe request is forgeable by a page the user visits, so
``enforce_same_origin`` guards them (see its docstring). It runs inside
``require_auth``, which means coverage cannot be forgotten on a new route.

A role-based access-control layer maps roles to permissions; endpoints assert
the permission they need via ``require_permission``. A bootstrap admin account
is seeded from AUTH_ADMIN_EMAIL / AUTH_ADMIN_PASSWORD at startup.

Roles:
- ``admin`` — everything (chat, analytics, user management)
- ``user``  — chat only (the public-signup default)

Machine clients (eval scripts) may authenticate with the AUTH_SERVICE_TOKEN
header, which is a SCOPED, EXPIRING credential rather than an unconditional
admin bypass: it carries an explicit permission set, stops working after
AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS, and can be revoked or rotated. All inputs are
validated server-side.
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

# Module-level store; set by main.lifespan (and by tests).
store: "AuthStore | None" = None

VALID_ROLES = ("admin", "user")

# The only role public self-service signup can ever grant (see the signup
# docstring for why it is not configurable).
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

# How a password reaches bcrypt.
#
# bcrypt hashes AT MOST the first 72 bytes of its input and silently discards
# the rest, which would make every long password a prefix of itself. No
# application-level maximum can fix that, because the cut happens inside bcrypt
# rather than in validation -- which is why the policy has no maximum.
#
# So bcrypt is handed a fixed-width SHA-256 pre-image instead. The digest is
# always 32 bytes, so nothing is ever truncated, the prefix collision is gone,
# and a 200-byte passphrase is worth 200 bytes.
#
# It changes what a stored hash MEANS, so the two schemes must be tellable
# apart. A current hash is this marker followed by a plain bcrypt string;
# anything without the marker is a pre-migration ``bcrypt(raw[:72])``, which
# ``verify_password`` still accepts until its owner next logs in and it is
# rewritten in place (see ``login`` and ``_upgrade_password_hash``).
_PASSWORD_SCHEME = "$bcrypt-sha256$"

# The width of the credential a pre-migration hash represents. Kept ONLY so
# those hashes can still be verified.
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
    """The body of a successful ``POST /api/auth/login`` / ``/change-password``.

    Carries the user record and deliberately NOT the session token: publishing
    it in the body would hand every XSS on the site a durable account takeover,
    which is the exact exposure the cookie moves the credential out of.
    """

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
# and no error -- the message is their recovery route ("just sign in with your
# existing password"), and it is deliberately the same string a brand new
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
    # Populated after the last_seen migration; None before the column exists or
    # for a user who has never authenticated. Not surfaced in UserOut.
    last_seen: float | None = None


@dataclass
class StoredServiceToken:
    """A machine credential (X-Service-Token) as stored.

    ``scope`` is the explicit set of permissions the token may exercise -- it is
    NOT the full admin permission set, and it is enforced per route by
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


def _has_letter_and_digit(password: str) -> bool:
    """The composition rule, in one place so ``validate_password`` and the
    bootstrap guard cannot drift into disagreeing about what counts."""
    return bool(re.search(r"[A-Za-z]", password)) and bool(re.search(r"\d", password))


def _password_rejection(password: str) -> str | None:
    """Why this password is refused, or None when it is accepted.

    THE SINGLE SOURCE OF TRUTH for the password policy. Every path that can set
    a password -- ``signup``, ``change_password`` and ``bootstrap_admin`` -- must
    reach this function, so they cannot admit different passwords.

    Both rules are judged on the WHOLE value, because the whole value is what
    authenticates: ``hash_password`` hands bcrypt a fixed-width SHA-256
    pre-image, so no byte of the password is ever dropped. A length floor judged
    on the 72-byte prefix bcrypt could see used to be the weaker bound (30
    three-byte characters cleared a 30-character floor while only 24 of them
    became the credential), and pre-hashing removes the shortening that made it
    weaker.

    **Letter+digit** is a composition rule, not an entropy rule, and stays judged
    on the configured secret: judging it on a truncated prefix used to refuse
    credentials ``login`` already accepted.

    There is deliberately NO maximum. The 72-byte cut was a property of bcrypt,
    not a policy the application could enforce, and refusing over-long values is
    what stopped ``bootstrap_admin`` -- the only path that can ever create an
    admin -- from seeding one, while leaving the very same working passphrase
    unsettable through ``change_password``. Every byte of the value now counts.
    """
    if len(password) < config.AUTH_PASSWORD_MIN_LEN:
        return f"password must be at least {config.AUTH_PASSWORD_MIN_LEN} characters"
    if not _has_letter_and_digit(password):
        return "password must contain a letter and a digit"
    return None


def validate_password(password: str) -> str:
    """Validate a password (length + letter/digit), raising 422 on violation.

    The raising form the signup and change-password endpoints call. It returns
    the password unchanged rather than a normalised variant, so a caller can
    never store something other than the exact value that was validated.
    """
    reason = _password_rejection(password)
    if reason:
        raise HTTPException(status_code=422, detail=reason)
    return password


def _validator_rejection(validator, value: str) -> str | None:
    """Return why ``validator`` rejects ``value``, or None when it accepts it.

    ``validate_email`` / ``validate_password`` are written for the signup
    endpoints and signal failure by raising ``HTTPException``. Callers that
    need the reason as text (rather than as a 422 response) use this.
    """
    try:
        validator(value)
    except HTTPException as exc:
        return str(exc.detail)
    return None


def validate_name(name: str) -> str:
    """Trim + validate an optional display name, raising 422 on violation."""
    name = (name or "").strip()
    if len(name) > config.AUTH_MAX_NAME_LEN:
        raise HTTPException(status_code=422, detail=f"name too long (max {config.AUTH_MAX_NAME_LEN} chars)")
    if any(ord(c) < 32 for c in name):
        raise HTTPException(status_code=422, detail="name contains invalid characters")
    return name


def _prehash(password: str) -> bytes:
    """The fixed-width pre-image bcrypt is actually handed.

    SHA-256 rather than a second bcrypt: a second bcrypt would only move the
    truncation window somewhere else, and a plain digest removes it outright.
    32 bytes whatever the password's length, so the 72-byte cut can never bite.
    """
    return hashlib.sha256(password.encode("utf-8")).digest()


def hash_password(password: str) -> str:
    """Hash a password in the current scheme: the scheme marker followed by a
    plain bcrypt string over ``_prehash(password)``."""
    digest = bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("utf-8")
    return _PASSWORD_SCHEME + digest


def needs_rehash(hashed: str) -> bool:
    """Whether ``hashed`` is a pre-migration hash a verified password should be
    rewritten into the current scheme.

    A pure test of the stored marker, so the two can never disagree about which
    scheme a row is in. Callers only ask after a successful verify, and a row
    whose hash is unreadable or empty can never reach that point, because a hash
    that verifies is a hash bcrypt produced.
    """
    return not hashed.startswith(_PASSWORD_SCHEME)


def verify_password(password: str, hashed: str) -> bool:
    """Check a password against a stored hash of EITHER scheme.

    The scheme is read from the STORED hash and only from the stored hash; the
    two schemes are never both tried and the caller cannot choose between them.
    That is what makes the migration safe in both directions:

    - A current-scheme row is only ever checked against the pre-image, so
      nobody can authenticate it with a raw 72-byte prefix.
    - A pre-migration row is only ever checked against ``raw[:72]`` -- the
      credential it was created to represent -- so an account that predates the
      change keeps logging in unchanged.

    Trying both, or picking the scheme by what the caller submitted, would leave
    the shorter of the two as a working alternative for whichever row an
    attacker targeted. Dispatching on the row also keeps the cost at exactly
    one bcrypt either way, so the two row kinds are indistinguishable by timing.
    """
    if hashed.startswith(_PASSWORD_SCHEME):
        stored, preimage = hashed[len(_PASSWORD_SCHEME):], _prehash(password)
    else:
        stored = hashed
        preimage = password.encode("utf-8")[:_LEGACY_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(preimage, stored.encode("utf-8"))
    except ValueError:
        # Not a bcrypt string at all: a corrupt row, an empty placeholder, a
        # truncated write. Same answer as a wrong password, and never an
        # exception out of an auth path.
        return False


# A bcrypt hash of a per-process random secret, computed once at import by the
# same ``hash_password`` -- and therefore the same ``gensalt()`` cost factor --
# that produced every stored hash. Logging in against an address that has no
# account verifies the supplied password against this constant, so that path
# costs the same wall-clock time as a wrong-password login. Without it the
# missing short-circuit is a remote account-existence oracle even though both
# paths return the identical 401 body. It must never be a cheaper hash: the
# cost factor is the whole point, so it is derived rather than hard-coded.
_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(presented: str, expected: str) -> bool:
    """Constant-time equality of two credential strings.

    ``secrets.compare_digest`` raises ``TypeError`` when a ``str`` argument
    holds a non-ASCII character, and an ``X-Service-Token`` header is
    attacker-controlled bytes: a raw request carrying one would turn the
    comparison into an unhandled ``TypeError`` and the route into a 500.
    Comparing the UTF-8 encodings keeps the content-comparison property
    that matters: both sides still go through one ``compare_digest`` call
    over bytes, so the comparison walks the content without an early exit on
    a differing byte. (``compare_digest`` itself does return early when the
    two lengths differ; the configured token's length is not a secret, and
    that is unchanged from the ``str`` comparison.) A non-ASCII value simply
    is not the expected token, and ``surrogateescape`` makes the encode
    total, so no byte sequence a server can decode into the header raises
    here either.
    """
    return secrets.compare_digest(
        presented.encode("utf-8", "surrogateescape"), expected.encode("utf-8", "surrogateescape")
    )


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
        # Idempotent: connect() may be called more than once (the migration test
        # re-runs it to prove re-entry is safe), and a second call must not
        # clobber the live connection, which would abandon its aiosqlite worker
        # thread (non-daemon) and hang interpreter shutdown.
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
        # Migration for existing databases created before the last_seen column:
        # guard on its OWN name, exactly like the chat-store pattern, so a
        # partial schema is repaired rather than relied on. Single statement, so
        # no BEGIN IMMEDIATE needed; the shared connection autocommits DDL.
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
        """Run a group of statements as ONE atomic, durable unit.

        A dedicated connection, not the shared one: a transaction is scoped to a
        connection, not to a coroutine, so a unit left open on ``self._db`` is
        committed or rolled back by whichever unrelated request in the worker
        happens to call ``commit()``/``rollback()`` next -- publishing a
        half-finished write early, or discarding a finished one.
        ``isolation_level=None`` keeps this connection's explicit BEGIN from
        nesting inside an implicit one, and ``BEGIN IMMEDIATE`` takes the WAL
        write lock up front so a mid-transaction lock upgrade cannot fail past
        the busy timeout.

        ``foreign_keys`` is per-connection, so it must be set here too.
        """
        db = await aiosqlite.connect(self._path, isolation_level=None)
        try:
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute("BEGIN IMMEDIATE")
            yield db
            await db.commit()
        except BaseException:
            # BaseException, not Exception: a request cancelled out from under
            # this coroutine (client gone, worker shutting down) must not
            # abandon an open write transaction, which is what leaves the file
            # locked against every other connection.
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
            # Duplicate email under concurrency (e.g. concurrent worker
            # bootstrap). The unit has already rolled itself back by the time
            # this runs, so nothing is left open on any connection -- and no
            # rollback is issued here, because that would reach the shared
            # connection and discard whatever another request had pending.
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
        async with self._unit_of_work() as db:
            cur = await db.execute(
                f"UPDATE users SET {', '.join(sets)} WHERE {where}", tuple(params)
            )
            return cur.rowcount

    async def delete_user(self, user_id: str, guard_last_admin: bool = False) -> int:
        """Remove one account and every credential that authenticates it.

        One statement does the whole job: the cascade on ``auth_tokens.user_id``
        takes the tokens with the user row, so there is no second write that
        could fail and strand this one. That matters because the two used to
        share a commit on the shared connection with no rollback, so a failure
        between them left the connection holding the WAL write lock -- the
        other three workers then failed every write with ``database is locked``
        until this one exited.
        """
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
        """Overwrite one user's stored hash on its own.

        A caller that also has to invalidate credentials must not use this: a
        password write that commits without the matching revocation leaves
        every previously issued token alive behind a password the owner has
        just changed. ``change_password`` is the atomic form.
        """
        async with self._unit_of_work() as db:
            await db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
            )

    async def upgrade_password_hash(self, user_id: str, observed_hash: str, new_hash: str) -> int:
        """Replace a pre-migration hash with a current one, but only while the
        row still holds the hash that was just verified. Returns rows changed.

        The compare-and-swap on ``observed_hash`` is what makes the
        opportunistic upgrade safe to run from a login. Losing the race to a
        concurrent writer is the correct outcome -- the winning writer's hash is
        the newer one -- so the rowcount is returned for the caller to log, not
        retried: a retry would reopen the same race.

        One statement, so it is atomic on its own; the unit still wraps it so a
        cancellation between the write and its commit rolls back rather than
        stranding the connection on the WAL write lock.
        """
        async with self._unit_of_work() as db:
            cur = await db.execute(
                "UPDATE users SET password_hash = ? WHERE id = ? AND password_hash = ?",
                (new_hash, user_id, observed_hash),
            )
            return cur.rowcount

    async def count_legacy_passwords(self) -> int:
        """How many accounts still hold a pre-migration hash.

        An upper bound on the at-risk set, not an exact one: a row records
        nothing about how long the original password was, so a pre-migration
        account whose password fitted inside bcrypt's window is counted here
        too. Nothing in the row can tell those apart.

        ``substr`` with a bound parameter rather than ``NOT LIKE``: ``%`` and
        ``_`` are LIKE wildcards, so a scheme marker containing one would
        silently match the wrong rows.
        """
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM users WHERE substr(password_hash, 1, ?) <> ?",
            (len(_PASSWORD_SCHEME), _PASSWORD_SCHEME),
        )
        return int(row["n"]) if row else 0

    async def count_admins(self) -> int:
        row = await self._fetchone("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
        return int(row["n"]) if row else 0

    async def touch_last_seen(self, user_id: str, now: float | None = None) -> None:
        """Write ``users.last_seen`` for one user. Should only be called after the
        caller has throttled (see ``_touch_last_seen``). Single statement, so it is
        atomic on its own; the unit still wraps the write so a cancelled request
        rolls back rather than stranding the connection on the WAL write lock."""
        async with self._unit_of_work() as db:
            await db.execute(
                "UPDATE users SET last_seen = ? WHERE id = ?",
                (now if now is not None else _now(), user_id),
            )

    async def account_stats(self, now: float | None = None) -> dict:
        """SQLite-derived user/account counts for ``/analytics/users``.

        None of these fields is per-user auth text -- they are aggregate counts
        and the report is admin-only. ``now`` is injectable for tests."""
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
        # active via last_seen; only rows that have a last_seen value count.
        last_seen_rows = await self._fetchall(
            "SELECT last_seen FROM users WHERE last_seen IS NOT NULL"
        )
        ls_vals = [float(r["last_seen"]) for r in last_seen_rows]
        active_today = sum(1 for ls in ls_vals if ls >= today_start)
        active_7d = sum(1 for ls in ls_vals if ls >= stamp - 7 * 86400)
        active_30d = sum(1 for ls in ls_vals if ls >= stamp - 30 * 86400)

        # last 14 days of signups, asc by date.
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
        """Mint a bearer token and return it in plaintext (only its SHA-256 is
        stored). Enforces the per-user active-token cap, so a caller cannot
        grow the table by logging in repeatedly -- see ``_revict_tokens_over_cap``.

        The mint and the eviction are ONE unit. Publishing the new row first
        meant two ways to be over the cap with nothing left to notice: a worker
        killed between the two commits, which no later login comes back to undo
        (enforcement runs from here and nowhere else), and an eviction that lost
        the write lock and raised, which turned a login into a 500 while its row
        stayed -- occupying a cap slot and costing a real session its place. So
        the caller gets the token only after both are durable, or gets neither.
        """
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
        """Delete the user's surplus unexpired tokens on ``db`` and return how
        many. No commit: the caller's unit is what makes it durable.

        Ordered by rowid ALONE. SQLite assigns rowid monotonically at INSERT,
        so eviction order cannot move when the wall clock does -- whereas
        ordering by ``created_at`` would let a backwards step (a suspended host,
        an NTP correction) put the row just minted ahead of the ones it belongs
        behind, and this pass would then delete the token the caller is about to
        be handed. That token would authenticate for exactly one request and
        nobody would see why. ``created_at`` stays for reporting; it just no
        longer decides who gets evicted.
        """
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
        async with self._unit_of_work() as db:
            await db.execute(
                "INSERT INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
                " VALUES (?, ?, ?, ?, NULL)",
                (record.token_hash, ",".join(sorted(record.scope)), record.created_at, record.expires_at),
            )
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
        async with self._unit_of_work() as db:
            await db.execute(
                "INSERT OR IGNORE INTO auth_service_tokens (token_hash, scope, created_at, expires_at, revoked_at)"
                " VALUES (?, ?, ?, ?, NULL)",
                (hash_token(raw), ",".join(sorted(scope)), created, created + ttl_seconds),
            )

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

    async def revoke_service_token(self, raw: str) -> int:
        """Soft-revoke one machine credential. Marks the row rather than
        deleting it: the row is the tombstone that stops a revoked configured
        value being re-seeded with a fresh lifetime (see
        ``ensure_bootstrap_service_token``). Returns how many rows it actually
        revoked -- 0 for an unknown or already-revoked token, which the caller
        must be able to see rather than assume.
        """
        async with self._unit_of_work() as db:
            cur = await db.execute(
                "UPDATE auth_service_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
                (_now(), hash_token(raw)),
            )
            return cur.rowcount

    async def revoke_configured_service_token(self, raw: str, scope: set[str]) -> int:
        """Kill the env-configured value, including from a cold start.

        Seeding is lazy, so a configured value that has never been presented has
        no row at all. The plain UPDATE in ``revoke_service_token`` then matches
        nothing, reports 0, and the very next request seeds the value live with
        a full fresh ``AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS`` -- so "revoke this
        now" silently does nothing precisely when an operator is trying to
        contain a leak of a credential that is not yet in use.

        So when the value has no row, INSERT a tombstone: a row that exists and
        is already revoked. Seeding is INSERT OR IGNORE, so a later request
        cannot overwrite it, and a value that already has a row is left exactly
        as it is, so this can only ever make a credential deader, never younger.

        Both outcomes are one statement, so there is no ordering left to get
        wrong. They used to be INSERT OR IGNORE followed by a delegated
        ``revoke_service_token``, which meant the statement that actually
        revoked anything was the second commit: against the normal case -- an
        in-use credential an operator is containing -- the first commit
        committed an empty transaction, and a worker killed before the second
        left a leaked credential fully live while the call reported 1. The
        ``WHERE`` on the DO UPDATE is what keeps that from becoming a rewrite
        instead: an already-dead row is left alone and not counted.

        Returns the number of rows this call changed (1 for a fresh tombstone,
        1 for revoking a live row, 0 if it was already dead).
        """
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
        """Delete service-token rows that can no longer authenticate anything --
        revoked, or past their expiry -- so a long rotation history cannot grow
        the table without bound the way an unrotated one would.

        ``keep_hash`` is excluded, and it must be: that row is the tombstone
        which stops a revoked or expired configured ``AUTH_SERVICE_TOKEN`` from
        being re-seeded with a fresh lifetime on the next request. Delete it and
        the reaper would hand the credential straight back, which is the
        permanent-grant failure this table exists to prevent.
        """
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
        """Resolve a raw session token (the auth cookie's value) to an active
        user, or None when the token is unknown, expired, or the account is
        disabled."""
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
        """Store a new password, revoke every existing token and mint a
        replacement as ONE durable unit. Returns the replacement token.

        These three writes only mean anything together, so they are applied
        inside a single ``BEGIN IMMEDIATE`` .. ``COMMIT``: SQLite keeps an
        uncommitted transaction invisible to every other connection and
        discards it outright if the process dies.

        The shared connection cannot hold one open safely for the length of this
        change: it serialises every request in the worker, so another
        coroutine's ``commit()`` landing between two of these statements would
        publish a half-finished password change early, and its ``rollback()``
        would throw this one away. Hence a short-lived dedicated connection to
        the same file: WAL lets it read and write alongside the shared one,
        SQLite's own write lock serialises it against other writers, and
        ``BEGIN IMMEDIATE`` takes that lock up front so this waits out the same
        5 s busy timeout rather than failing on a mid-transaction lock upgrade.
        It is opened with ``isolation_level=None`` so this ``BEGIN`` is the
        connection's only transaction, not a nested one sqlite3 would refuse.

        The statements are ALSO ordered revoke -> set hash -> mint. That is
        defence in depth, not the guarantee: no interruption point leaves a
        changed password standing next to a token minted before it.

        The account has to still exist when the hash write runs. The route reads
        the user, spends a bcrypt round on the old password, and only then calls
        in here, so an admin DELETE landing in that window used to reach the end
        of this transaction as a no-op UPDATE -- silently -- and mint a session
        for a user_id nothing references. Hence the rowcount check below, which
        fails the whole unit, and the ``foreign_keys`` pragma, which is
        per-connection and so was off here even though it is on the shared one.

        The per-user token cap is not re-applied here: every other token was
        deleted in this same transaction, so the user can hold at most the one
        row inserted below.
        """
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
                # Zero rows is not a successful password change, it is a change
                # to no user at all. Reporting it as success would hand the
                # caller a session cookie that 401s on the very next request and
                # leave behind a token row for an account that no longer exists.
                # The surrounding handler rolls the unit back for this raise.
                raise ValueError(f"user {user_id} no longer exists")
            raw = secrets.token_urlsafe(32)
            created = _now()
            await db.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (hash_token(raw), user_id, created, created + ttl_days * 86400),
            )
            await db.commit()
        except BaseException:
            # BaseException, not Exception: a request cancelled out from under
            # this coroutine (client gone, worker shutting down) must not
            # abandon an open write transaction, which is what leaves the file
            # locked against every other connection.
            await db.rollback()
            raise
        finally:
            await db.close()
        return raw

    async def purge_expired_tokens(self) -> int:
        """Delete rows whose expiry has passed. Parameterized to avoid SQL
        injection; returns the number of rows removed. Run periodically so the
        table can't grow without bound as tokens expire."""
        now = _now()
        async with self._unit_of_work() as db:
            cur = await db.execute("DELETE FROM auth_tokens WHERE expires_at < ?", (now,))
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
        # Pin the DB explicitly so the rate-limit counters never silently land
        # in DB 0, which is the query cache and is flushed during deploys --
        # a flush there would reset every bucket and briefly disable the
        # throttle. The ``db`` kwarg overrides any db segment in REDIS_URL, so
        # this is safe whether or not the URL carries a db index. This client is
        # module-local to the limiter (closed by close_rate_redis), so pinning
        # it cannot move anyone else's data.
        _rate_client = aioredis.from_url(
            config.REDIS_URL,
            db=config.AUTH_RATE_LIMIT_REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
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
    """Background task: purge dead token rows -- expired user tokens, and
    service tokens that are revoked or expired. Never raises. Disabled when
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
    subject: str | None = None,
) -> None:
    """Enforce a rate limit with Redis INCR+EXPIRE.

    ``fail_closed`` selects what happens when the limiter's Redis is
    unreachable. The auth endpoints keep the default (False) and fall back to
    a bounded in-process limiter -- see ``_consume_counter``; the public search
    surface passes ``fail_closed=True`` and is answered 503 instead, because an
    unrated request against ``/search`` or ``/analytics/click`` is precisely
    the full-corpus scraping and analytics-poisoning vector these limits exist
    to close -- there, no answer is not an acceptable fallback.

    ``window_seconds`` defaults to the auth window so the existing auth call
    sites keep their current 60s window; the public limits pass their own.

    ``subject`` overrides the bucket identity from the client IP to a caller
    supplied one, typically an authenticated user id. A per-IP bucket cannot
    bound one account sitting behind a shared NAT or proxy address, which is
    the shape a deliberate flood takes, so an endpoint that mints per-account
    state is limited on both axes. It changes ONLY which string is counted --
    the counter, window, Redis path and in-process fallback are identical, so
    the two axes cannot drift apart in enforcement.
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
    """Enforce a rate limit keyed on the submitted account rather than the
    source address, so rotating IPs cannot buy an attacker a fresh bucket.

    The key is the address folded to lower case and stripped, so one account
    cannot be handed a second bucket by changing case or padding.

    THIS HELPER PERFORMS NO LOOKUP: given the same submitted string it computes
    the same key, the same counter update and the same answer whether or not
    the address has an account, which is what keeps it from being an
    account-existence oracle. Its caller decides WHETHER to call it, and that
    is where existence enters -- login only counts a failed credential check,
    which is not an enumeration channel because the only thing that skips the
    increment is the correct password.
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
    process. When it is unreachable the two obvious behaviours are both wrong:
    admitting the request turns an inducible outage into an unlimited
    credential-stuffing window, and rejecting every one turns it into a total
    login outage that hands a denial-of-service to whoever can disturb Redis
    without helping anyone guess a password.

    So the fallback is a third option: a bounded in-process limiter with the
    SAME limit. The degraded posture is "single-process limiting" rather than
    "no limiting" or "no service". It stays bounded under an unbounded key
    flood (see ``_local_rate_hit``), and it is not a hidden fail-open: past the
    limit the caller still gets 429, exactly as it would from Redis. The
    per-worker weakening is inherent and accepted deliberately: a gunicorn
    deployment multiplies the effective limit by its worker count during an
    outage, which is still a finite bound rather than none.
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
# than the first. Windows that have already closed are reclaimed first.
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
    # Head first == least recently used first, and then only in recency order.
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
    polled by load balancers and orchestrators, so failing its limiter closed
    would pull healthy nodes out of rotation for a dependency the service does
    not require to be ready. It still counts and still answers 429 -- only a
    broken limiter store is tolerated there.
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

    Same counter, window, Redis path and in-process fallback as
    ``public_rate_limit``, so neither axis can drift into weaker enforcement
    than the other. Only the bucket identity and the key prefix differ: this
    one counts the authenticated user id, under ``user:rl`` so it can never
    collide with the per-IP bucket even when the action name matches.

    Compose this WITH ``public_rate_limit`` on an endpoint that mints
    per-account persistent state: a per-IP bucket alone cannot bound one
    account behind a shared address, and a per-account bucket alone cannot
    bound one account rotating addresses.

    Depends on ``require_auth`` rather than reading ``request.state`` blindly,
    so the user id is guaranteed resolved before the counter is keyed.
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


# --- dependencies ---


def _token_from_request(request: Request) -> str | None:
    """The session token on this request, read from the auth cookie alone.

    There is deliberately no ``Authorization: Bearer`` branch. A header
    credential is one the app must hand to script, and that path stays
    reachable from ``fetch()``; keeping it alongside the cookie would leave
    the very exfiltration surface this migration closes.
    """
    return request.cookies.get(config.AUTH_COOKIE_NAME) or None


def _host_only(authority: str) -> str:
    """Lowercase an authority and drop its port, IPv6 literals included.

    ``host``, ``host:port``, ``[::1]:8001`` and a bare ``::1`` all reduce to
    their host. A split on the FIRST colon would turn ``[::1]:8001`` into ``[``
    and let any bracketed address match any other, so brackets are peeled
    before the port. The bare form matters too: ``urlsplit(...).hostname``
    returns an IPv6 address with its brackets already removed, and that value
    is fed straight back through here for the comparison — without this the
    two sides of an IPv6 comparison reduce differently (``::1`` vs ``''``) and
    a legitimate same-origin request is refused.

    Port is not part of the comparison because a port is not a security
    boundary for a cookie or for CSRF: the dev stack legitimately serves the
    frontend on :3000 and the API on :8001 under one host.
    """
    host = authority.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end]
        return host
    if host.count(":") > 1:
        # More than one colon and no brackets: a bare IPv6 literal. It cannot
        # carry a port (RFC 3986 requires brackets for that), so there is
        # nothing to drop and splitting would leave the empty string.
        return host
    return host.partition(":")[0]


def _origin_host(origin: str) -> str | None:
    """The host an ``Origin`` header names, or None when it names none.

    ``urlsplit(...).hostname`` already lowercases and drops the port and is
    IPv6-safe. It returns None for the literal ``Origin: null`` that a
    sandboxed iframe or a privacy browser sends; that is reported as None so
    the caller rejects it rather than treating "no host" as "same host".
    """
    try:
        return urlsplit(origin.strip()).hostname
    except ValueError:
        return None


def _request_host(request: Request) -> str | None:
    """The host this request was addressed to, from the ``Host`` header alone.

    ``X-Forwarded-Host`` is deliberately NOT consulted. It is not one of the
    Fetch spec's forbidden header names, so a page can set it, and the shipped
    nginx config neither overwrites nor strips it. A host the client chooses
    cannot be the basis of the CSRF comparison, since the client chooses
    ``Origin`` too. ``Host`` is the right basis because ``TrustedHostMiddleware``
    already constrains it to this deployment's own allow-list, and a cross-site
    page cannot pick it — the browser sets it to the victim's own domain.
    """
    host = request.headers.get("host")
    return host.strip() if host and host.strip() else None


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


async def enforce_same_origin(request: Request) -> None:
    """Reject a cross-site unsafe request: 403 when the browser says it is
    cross-site, or when the request's own headers disagree about the host.

    The auth cookie is attached by the browser whether or not the page means
    to send it, so a hostile page can make an authenticated state-changing
    request that rides the user's session. Two signals are checked, and they
    are not redundant with one another — neither covers the other's gap:

    - ``Sec-Fetch-Site: cross-site`` is set by the browser and cannot be set
      by script, so on its own it is decisive for every current browser. It says
      nothing about a client that omits it (curl, an eval script, a non-browser
      agent), and that is the gap the second signal closes. It is checked first
      because it needs no host comparison and stays correct behind any proxy.
    - ``Origin`` is compared against the request's own ``Host``. This is a
      self-consistency check, not an allow-list, and it is the signal that still
      stands when ``Sec-Fetch-Site`` is absent or has been tampered with. In
      turn it catches nothing when the client omits ``Origin`` as well, and it is
      only as trustworthy as the ``Host`` it is compared to — which is why
      ``_request_host`` refuses to let a client-settable ``X-Forwarded-Host``
      pick that answer. Hosts are compared with ports stripped because the app
      sits behind TLS termination and cannot trust ``request.url.scheme``.

    ``CORS_ORIGINS`` is deliberately NOT used here: it is a localhost-only dev
    default that does not contain the production host, so an allow-list built
    from it would 403 every real request.

    The honest limitation: a request with NEITHER header is allowed through.
    Every browser sends both on an unsafe method, so their absence means a
    non-browser client (curl, an eval script) which has no ambient cookie to
    ride in the first place. This is therefore not fail-closed — but the
    browser is the only thing that attaches cookies unasked, and demanding
    these headers outright would break every non-browser caller for no added
    protection.

    Scope: applied to every cookie-authenticated request via ``require_auth``
    and to ``POST /api/auth/login``. The routes with neither ``require_auth``
    nor a direct ``enforce_same_origin`` are the public ``GET /search`` and
    ``GET /facets``, the health probes, and the two public unsafe routes
    ``POST /analytics/click`` and ``POST /api/auth/signup``. The unguarded ones
    are all either safe methods, which the guard ignores anyway, or
    unauthenticated endpoints that have no session to ride.
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
        # Covers the literal "null" from a sandboxed iframe or privacy
        # browser: it names no host, so it can never be shown to be ours.
        raise HTTPException(status_code=403, detail="cross-site request rejected")
    expected = _request_host(request)
    if expected is None or _host_only(origin_host) != _host_only(expected):
        raise HTTPException(status_code=403, detail="cross-site request rejected")


def _set_session_cookie(response: Response, token: str) -> None:
    """Attach the session cookie. No ``domain``: host-only keeps it working
    on a bare-IP deployment, where any Domain would have to name an address
    the operator may not control."""
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
    """Expire the session cookie.

    Every attribute must match ``_set_session_cookie`` exactly: a browser
    treats a deletion whose name or path differs from the original as a
    different cookie, and the live one would then be left in place.
    """
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
    if config.AUTH_SERVICE_TOKEN and tokens_match(raw, config.AUTH_SERVICE_TOKEN):
        await s.ensure_bootstrap_service_token(raw, set(_service_token_scope()), _service_token_ttl_seconds())
        return await s.service_token_for(raw)
    return None


# --- last_seen throttling + admin audit (analytics contract) ---

# In-process per-user last_seen touch throttle. KEYED BY USER ID, so it can only
# grow as large as the account table (a user id is only ever produced by a
# successful authentication), never by an anonymous flood. A user authenticating
# from several gunicorn workers touches each one's set, which only over-writes
# (it can never under-report). Cleared only when the throttled interval elapses.
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
    """Best-effort, throttled write of ``users.last_seen``. Never raises."""
    if not _last_seen_due(user_id, _now()):
        return
    s = store
    if s is None:
        return
    try:
        await s.touch_last_seen(user_id)
    except Exception:
        # A missed last_seen is a lost DAU tick, not an auth outage: the user is
        # already authenticated by the time this runs, so the write can never
        # fail the request.
        logger.debug("last_seen touch failed for %s", user_id, exc_info=True)


def _login_day_key(stamp: float | None = None) -> str:
    """Redis key for the per-day successful-login counter."""
    return f"analytics:login:day:{time.strftime('%Y-%m-%d', time.gmtime(stamp if stamp is not None else _now()))}"


async def _count_login() -> None:
    """INCR today's successful-login counter. Best-effort, never raises."""
    try:
        await _rate_redis().incr(_login_day_key())
    except Exception:
        logger.debug("login-day counter INCR failed", exc_info=True)


def _chat_audit_store():
    """Lazily resolve the chat store for the admin audit trail.

    Avoids a module-load cycle (chat.py imports app.auth) by importing at call
    time, once main.py has imported both modules."""
    from app.chat import _require_store
    return _require_store()


async def _record_admin_audit(actor_id: str, action: str) -> None:
    """Best-effort admin audit write. Never raises and never breaks the action."""
    try:
        await _chat_audit_store().record_admin_audit(actor_id, action)
    except Exception:
        logger.warning("admin audit write failed for %s", action, exc_info=True)



async def require_auth(request: Request) -> None:
    """Validate the request's credentials and stash the user on request.state.
    Accepts the auth cookie (user tokens) or ``X-Service-Token`` (a scoped,
    expiring machine credential).

    A service token that is revoked or past its expiry is NOT honoured: it
    falls through to the cookie path and ends as a 401, rather than being
    granted admin because the environment still mentions it.
    """
    # Guard FIRST, so a cross-site request is refused whichever credential it
    # presents. Inside this dependency rather than on each route: a new
    # cookie-authenticated endpoint then cannot forget the guard.
    await enforce_same_origin(request)
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
        # Debug, not warning: this sits on an unauthenticated, attacker-
        # controlled path, so at warning level any anonymous request with a
        # junk header writes a log line -- a log-flood amplifier.
        logger.debug("auth: a presented service token did not resolve")
    token = _token_from_request(request)
    if token is None:
        raise HTTPException(status_code=401, detail="authentication required")
    user = await _require_auth_store().user_for_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    request.state.user = user
    request.state.user_id = user.id
    # Real user credential: record activity (throttled, best-effort). The
    # service-token path above returns early, so it is excluded here.
    await _touch_last_seen(user.id)


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
    sets no cookie: a session present only for fresh addresses would be the
    oracle all over again, and minting one for an existing account would hand
    an anonymous caller someone else's session.

    The role is hardcoded to 'user': public signups always land with the least
    privilege. There is deliberately no configuration knob here — a role that
    can be flipped by an env var (or a request field) turns a config mistake
    into a full account compromise.
    """
    await _check_rate_limit(request, "signup", config.AUTH_SIGNUP_RATE_PER_MIN)
    email = validate_email(body.email)
    password = validate_password(body.password)
    name = validate_name(body.name)
    s = _require_auth_store()
    # No pre-flight "does this address exist" lookup: create_user hashes the
    # password and attempts the INSERT either way, so a duplicate costs the
    # same wall-clock time as a fresh registration. A pre-check would skip
    # that bcrypt work and leak existence through timing. The users.email
    # UNIQUE constraint is the single source of truth.
    try:
        await s.create_user(email, password, name, role=SIGNUP_ROLE)
    except DuplicateEmailError:
        # Already registered (including losing a concurrent-creation race).
        # Swallow it into the same success-shaped answer a fresh address gets
        # and leave the stored row untouched.
        logger.info("signup for an already-registered address: reported as accepted")
    # No cookie is set here on purpose (see the docstring): minting a session
    # on signup is what the anti-enumeration rule forbids.
    return SignupOut(message=SIGNUP_ACCEPTED_MESSAGE)


@router.post("/login", response_model=AuthOut)
async def login(
    body: LoginIn,
    request: Request,
    response: Response,
    _: None = Depends(enforce_same_origin),
):
    """Exchange email+password for a session cookie.

    The opaque token is issued, stored hashed, and handed to the browser only
    as an HttpOnly cookie; it is deliberately absent from the response body.

    Guarded by ``enforce_same_origin`` even though it is unauthenticated:
    without it a hostile page could force a login with the *attacker's*
    credentials, so the victim's subsequent authenticated actions would post
    into the attacker's account ("login CSRF").

    An unknown address and a known address with a wrong password are
    indistinguishable to the caller: the identical 401 status and body, and
    the identical full bcrypt verify cost, because the unknown-address path
    verifies the supplied password against a fixed dummy hash at the same
    cost factor instead of skipping the check. A deactivated account is
    verified the same way. The per-account rate limit is keyed on the
    submitted address alone, so a registered and an unregistered address reach
    the same counter, the same 429 and the same amount of work.

    That per-account limit counts FAILED attempts only, and is applied after
    the credential check rather than before it. Counting every attempt, and
    gating on the counter first, made the throttle an account-lockout weapon:
    twenty anonymous wrong-password requests against a known address, from
    twenty source addresses, would lock the real owner out of their own
    account indefinitely without ever guessing a password. A correct password
    must never be rate-limited, so it never touches the counter.
    WHAT THIS COUNTER IS AND IS NOT: it bounds the RATE of attempts directed
    at one account, which the per-IP limit cannot see. It does NOT bound an
    attacker's COST, because being refused is free -- the bcrypt verify has
    already been paid. The per-IP limiter bounds attacker cost, and anything
    that makes this counter do that job has to consult it BEFORE the verify,
    which would reinstate both the account-lockout primitive and the existence
    timing oracle.
    A successful login also REWRITES the stored hash if it predates the
    pre-image scheme. That is deliberate: the plaintext exists in this request
    and nowhere else, so this is the only moment a pre-migration credential can
    be re-expressed. It never revokes anything and can never fail the login.
    """
    await _check_rate_limit(request, "login", config.AUTH_LOGIN_RATE_PER_MIN)
    email = validate_email(body.email)
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
    if password_ok and user is not None and user.is_active:
        # A real user with a real password is never counted or rate-limited.
        if needs_rehash(user.password_hash):
            # Opportunistic migration. The plaintext exists only in this
            # request, so this is the one moment the pre-migration hash can be
            # re-expressed in the current scheme.
            await _upgrade_password_hash(s, user, body.password)
        token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)
        # The session is delivered ONLY as the HttpOnly cookie: nothing in the
        # body for script on the page to read and exfiltrate.
        _set_session_cookie(response, token)
        # Analytics: a successful login is activity. Both are best-effort and
        # must never fail the login.
        await _touch_last_seen(user.id)
        await _count_login()
        return AuthOut(user=UserOut.from_user(user))

    # Failed. Count it against the address, keyed on the submitted string alone
    # (no lookup feeds this), so a registered and an unregistered address are
    # indistinguishable here too.
    await _check_account_rate_limit(
        request, "login", config.AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN, email
    )
    raise HTTPException(status_code=401, detail="invalid email or password")


async def _upgrade_password_hash(s: AuthStore, user: StoredUser, password: str) -> None:
    """Re-store ``user``'s just-verified password in the current scheme.

    Only ever called after ``verify_password`` accepted ``password`` against
    ``user.password_hash``, so the credential cannot change meaning here:
    nothing is revoked, because this rewrites how one secret is stored, not
    which secret it is -- the opposite of ``change_password``.

    Never raises. A failure here is logged and dropped: the row is untouched
    and its pre-migration hash still authenticates, so a housekeeping write can
    never turn into an authentication outage. The user id is logged rather than
    the address so this line cannot be read back as a record of who has
    successfully authenticated.
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

    For a service token that revocation is real: the token is marked revoked,
    so it stops working immediately. For a user session the token is read from
    the cookie, which is the only place it can now be: re-parsing a header here
    would leave the stored token live and turn logout into a silent no-op that
    still answers ``{"ok": true}``.
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
    """Change the current user's password after verifying the old one. Revokes
    every other token the user holds and re-issues the session cookie with a
    fresh token, so this session stays signed in and no other one does.

    The new password is judged by the same policy ``signup`` and
    ``bootstrap_admin`` use (see ``_password_rejection``), and stored the same
    way, so an admin seeded with a passphrase of any length can re-apply that
    very passphrase here instead of being 422'd out of its own credential.

    "Invalidates" holds even if the worker is killed mid-request: the hash
    write, the revocation and the replacement token commit together or not at
    all -- see ``AuthStore.change_password``.
    """
    user = request.state.user
    s = _require_auth_store()
    stored = await s.get_user(user.id)
    if stored is None or not await asyncio.to_thread(
        verify_password, body.current_password, stored.password_hash
    ):
        raise HTTPException(status_code=400, detail="current password is incorrect")
    new_password = validate_password(body.new_password)
    # Hash first, then take the write lock: bcrypt is the slow part and has no
    # business being spent holding it.
    new_hash = await asyncio.to_thread(hash_password, new_password)
    # ONE transaction for the hash write, the revocation and the replacement
    # token. Do not split this back into set_password / revoke_all_tokens /
    # issue_token: that leaves a durable new password valid alongside
    # still-authenticating pre-existing tokens whenever the worker dies between
    # them.
    token = await s.change_password(user.id, new_hash, config.AUTH_TOKEN_TTL_DAYS)
    # Revocation above killed every token this user held, including the one in
    # the cookie, so the cookie has to be re-issued with the new token or the
    # session dies on the user's very next request.
    _set_session_cookie(response, token)
    return AuthOut(user=UserOut.from_user(stored))


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
    await _record_admin_audit(request.state.user_id, "users.patch")
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
    await _record_admin_audit(request.state.user_id, "users.delete")
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
    await _record_admin_audit(request.state.user_id, "users.revoke_tokens")
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
    await _record_admin_audit(request.state.user_id, "users.mint_service_token")
    return ServiceTokenOut(token=raw, scope=sorted(record.scope), expires_at=record.expires_at)


@router.post("/service-tokens/revoke")
async def revoke_service_tokens(
    request: Request,
    body: ServiceTokenRevokeIn | None = None,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("users:manage")),
):
    """Revoke a service token: the other half of rotation.

    Pass the token to retire and only that one dies, which is what makes
    rotation safe to perform in the order an operator naturally reaches for --
    mint the replacement, move consumers onto it, *then* kill the old one,
    without a window in which no credential works. Posting no body at all (or an
    empty one) revokes every live service token at once; that is the right move
    for a suspected leak, and the wrong one for a planned rotation because it
    would take down the replacement minted moments earlier.

    ``revoked`` is the number of rows actually changed, so revoking an unknown
    or already-revoked token reports 0 rather than a reassuring 1.

    The value in ``AUTH_SERVICE_TOKEN`` is handled specially in both forms. It
    is the one credential whose seeding is lazy, so before it has ever been
    presented there is no row to UPDATE and a plain revoke would report 0 while
    leaving it live. Revoking it writes a revoked tombstone instead, so "kill
    this now" works from a cold start.
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
    await _record_admin_audit(request.state.user_id, "users.revoke_service_tokens")
    return {"revoked": revoked}



async def report_legacy_password_hashes() -> int | None:
    """Log how many accounts still hold a pre-migration password hash, and
    return that count (None when it could not be determined).

    The migration is completed by each account owner logging in, and nothing
    here can finish it for them: a hash that means "the first 72 bytes" can
    only become a pre-image hash by someone supplying the plaintext. What the
    operator can do is SEE whether it is draining.

    There is deliberately no force option alongside this. Revoking a
    pre-migration hash is the only way to finish one without the plaintext, and
    this codebase has no password-reset path to revoke one with: a force switch
    would not migrate anything, it would permanently lock out every account
    that has not logged in since the upgrade. Reporting the remainder costs
    nothing and locks nobody out.

    Never raises and never blocks startup. A count that fails is not worth
    failing a boot over, and the next restart tries again.
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
    """Seed the bootstrap admin from config (once, at startup). Never overwrites
    an existing account's password. Safe under concurrent worker startups: the
    duplicate / write-lock races are handled instead of failing startup (which
    would restart-loop the worker).

    The configured credentials go through the same ``validate_email`` /
    ``validate_password`` guards as every other path into the user table. A
    config value the validators reject is refused, loudly, and no account is
    created -- but the process still starts, so a typo in one env var cannot take
    the whole API (and /health) down and leave nobody able to reach the service
    to fix it. An operator must correct the config and restart.

    An admin account left behind by an earlier run with weak credentials is
    deliberately NOT deleted: this runs at startup, unauthenticated, and
    removing the only admin account would lock every operator out of their own
    deployment. Such an account is reported instead, so it gets rotated.
    """
    email = (config.AUTH_ADMIN_EMAIL or "").strip().lower()
    password = config.AUTH_ADMIN_PASSWORD or ""
    if not email or not password:
        return
    # The config values are the one remaining path into the user table that does
    # not go through the validators, so a typo like AUTH_ADMIN_PASSWORD=x used to
    # provision a full-admin account with a 1-character password that the signup
    # endpoint would itself have rejected. Run both through those validators.
    #
    # The rejection is a permanent, config-level fault: retrying it five times
    # inside the write-lock loop below would re-log the identical error five
    # times and change nothing, so it returns before reaching that loop. The
    # loop still retries genuine transient faults (SQLite write locks) as before.
    # The password is validated first, and unconditionally, so the advice given
    # for an EMAIL fault can still say whether the account behind it is on a
    # weak password: reporting only the address would hide a live 1-character
    # admin password.
    #
    # Length is judged on the whole configured value, and so is everything else
    # about it: hash_password hands bcrypt a fixed-width pre-image, so no byte
    # of this passphrase is dropped.
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
    """Log that the configured bootstrap admin credentials were refused, and
    report (never delete) an account a previous run already created from the
    same bad value.

    Must stay loud and specific: the operator reading the log has to learn that
    AUTH_ADMIN_PASSWORD -- not the login endpoint -- is the thing to fix, so the
    message names the variable, the validator that rejected it, the validator's
    own reason, and the fix.
    """
    # A weak admin may already exist from a run that predates the validators.
    # Removing it here would be an unauthenticated, startup-time way to delete
    # the only admin account and lock every operator out, so the row is left
    # alone and surfaced loudly for out-of-band rotation instead.
    #
    # "Rotate" is demanded only when the configured value is genuinely the
    # account's current password. A config value that merely fails validation
    # says nothing about a healthy admin's password, so demanding rotation of an
    # unrelated account on every worker restart would be a false alarm.
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
            # Rotation is warranted by a conjunction of two facts about the
            # ACCOUNT, neither of which is "which variable failed": the stored
            # password must actually be the configured one (proving this account
            # was provisioned from the bad config), AND the password must itself
            # have failed validation. Requiring both means an email fault on a
            # healthy admin never nags.
            if password_rejected and verify_password(
                config.AUTH_ADMIN_PASSWORD or "", existing.password_hash
            ):
                # The remedy here is a PASSWORD rotation, so the hint must be the
                # password one. Passing through `hint` unchanged would pair
                # "Rotate that account's password" with an email remedy, sending
                # the operator to fix the wrong variable.
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
                # The account is on a different, valid password, so the fault is
                # purely the configured address: say that, and do not send the
                # operator to rotate a password that is not the problem.
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
    """The remediation hint for a ``validate_password`` reason.

    Keyed on the reason so the advice can never contradict it -- one fixed hint
    would tell an operator to lengthen a password that was rejected for having
    no digit. It enumerates every reason ``_password_rejection`` can return; the
    too-long case is gone entirely now that the policy has no maximum, so a
    "lengthen it" branch here would have nothing left to describe.
    """
    if "at least" in reason:
        return (
            f"set it to a password of at least {config.AUTH_PASSWORD_MIN_LEN} characters "
            "containing both a letter and a digit, then restart"
        )
    if "letter and a digit" in reason:
        return "set it to a password containing both a letter and a digit, then restart"
    # A future reason must not fall through to advice that could be wrong, so
    # the fallback stays reason-agnostic.
    return "set AUTH_ADMIN_PASSWORD to a password that satisfies the password policy, then restart"


def _validator_name_for(variable: str) -> str:
    return "validate_email" if variable.endswith("EMAIL") else "validate_password"
