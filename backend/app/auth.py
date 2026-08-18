"""Token + RBAC authentication for the API.

Open signup/login issue opaque bearer tokens (hashed with SHA-256 in storage,
expiring after AUTH_TOKEN_TTL_DAYS, individually revocable, multiple per user).
A role-based access-control layer maps roles to permissions; endpoints assert
the permission they need via ``require_permission``. A bootstrap admin account
is seeded from AUTH_ADMIN_EMAIL / AUTH_ADMIN_PASSWORD at startup.

Roles:
- ``admin`` — everything (chat, analytics, user management)
- ``user``  — chat only (the public-signup default)

Machine clients (eval scripts) may bypass via the AUTH_SERVICE_TOKEN header,
which acts as an admin user. All inputs are validated server-side.
"""

import hashlib
import logging
import os
import re
import secrets
import time
import uuid
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

# Role -> permissions. Single source of truth for access control; add a
# resource-scoped permission here and assert it on the route that needs it.
ROLE_PERMISSIONS: ClassVar[dict[str, set[str]]] = {
    "admin": {"chat:use", "analytics:read", "users:read", "users:manage"},
    "user": {"chat:use"},
}

# Id used as the user_id for service-token requests (eval scripts, ops).
SERVICE_USER_ID = "service-token"

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_MAX_PASSWORD_LEN = 128


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


@dataclass
class StoredUser:
    id: str
    email: str
    password_hash: str
    name: str
    role: str
    is_active: bool
    created_at: float


def validate_email(email: str) -> str:
    """Normalize + validate an email address, raising 422 on any violation."""
    email = (email or "").strip().lower()
    if not email or len(email) > config.AUTH_MAX_EMAIL_LEN or not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="invalid email address")
    return email


def validate_password(password: str) -> str:
    """Validate a password (length + letter/digit), raising 422 on violation."""
    if not password or len(password) < config.AUTH_PASSWORD_MIN_LEN or len(password) > _MAX_PASSWORD_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"password must be {config.AUTH_PASSWORD_MIN_LEN}-{_MAX_PASSWORD_LEN} characters",
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


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


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
        user = StoredUser(
            id=uuid.uuid4().hex,
            email=email,
            password_hash=hash_password(password),
            name=name,
            role=role,
            is_active=True,
            created_at=_now(),
        )
        await self._db.execute(
            "INSERT INTO users (id, email, password_hash, name, role, is_active, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user.id, user.email, user.password_hash, user.name, user.role, 1, user.created_at),
        )
        await self._db.commit()
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

    async def update_user(self, user_id: str, name: str | None, role: str | None, is_active: bool | None) -> None:
        sets, params = [], []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if role is not None:
            sets.append("role = ?")
            params.append(role)
        if is_active is not None:
            sets.append("is_active = ?")
            params.append(1 if is_active else 0)
        if not sets:
            return
        params.append(user_id)
        await self._db.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", tuple(params))
        await self._db.commit()

    async def delete_user(self, user_id: str) -> None:
        await self._db.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
        await self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self._db.commit()

    async def set_password(self, user_id: str, password_hash: str) -> None:
        await self._db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        await self._db.commit()

    async def count_admins(self) -> int:
        row = await self._fetchone("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
        return int(row["n"]) if row else 0

    async def issue_token(self, user_id: str, ttl_days: int) -> str:
        raw = secrets.token_urlsafe(32)
        created = _now()
        expires = created + ttl_days * 86400
        await self._db.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (hash_token(raw), user_id, created, expires),
        )
        await self._db.commit()
        return raw

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


def _client_ip(request: Request) -> str:
    """Client IP from the first X-Forwarded-For hop (nginx), else the socket."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _check_rate_limit(request: Request, action: str, limit_per_min: int) -> None:
    """Enforce a per-IP rate limit with Redis INCR+EXPIRE. Fails open (no 429)
    when Redis is unreachable so auth availability never depends on it."""
    if limit_per_min <= 0:
        return
    key = f"auth:rl:{action}:{_client_ip(request)}"
    try:
        rc = _rate_redis()
        n = await rc.incr(key)
        if n == 1:
            await rc.expire(key, config.AUTH_RATE_WINDOW_SECONDS)
        if n > limit_per_min:
            raise HTTPException(
                status_code=429,
                detail="Too many attempts. Please try again shortly.",
                headers={"Retry-After": str(config.AUTH_RATE_WINDOW_SECONDS)},
            )
    except HTTPException:
        raise
    except Exception:
        logger.warning("auth rate limiter unavailable for %s", action, exc_info=True)


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


async def require_auth(request: Request) -> None:
    """Validate the request's credentials and stash the user on request.state.
    Accepts ``Authorization: Bearer <token>`` (user tokens) or
    ``X-Service-Token`` (machine bypass, acts as admin)."""
    service = request.headers.get("x-service-token")
    if service and config.AUTH_SERVICE_TOKEN and secrets.compare_digest(service, config.AUTH_SERVICE_TOKEN):
        request.state.user = _service_user()
        request.state.user_id = SERVICE_USER_ID
        return
    token = _token_from_request(request)
    if token is None:
        raise HTTPException(status_code=401, detail="authentication required")
    user = await _require_auth_store().user_for_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    request.state.user = user
    request.state.user_id = user.id


def require_permission(permission: str):
    """Dependency factory: require ``permission`` (see ROLE_PERMISSIONS)."""

    async def checker(request: Request) -> None:
        user = getattr(request.state, "user", None)
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        if permission not in ROLE_PERMISSIONS.get(user.role, ()):
            raise HTTPException(status_code=403, detail="forbidden")

    return checker


# --- endpoints ---


@router.post("/signup", response_model=AuthOut)
async def signup(body: SignupIn, request: Request):
    """Create an account (public). Returns a bearer token. Validated server-side:
    email format + uniqueness, password strength, name limits."""
    await _check_rate_limit(request, "signup", config.AUTH_SIGNUP_RATE_PER_MIN)
    email = validate_email(body.email)
    password = validate_password(body.password)
    name = validate_name(body.name)
    s = _require_auth_store()
    if await s.get_user_by_email(email) is not None:
        raise HTTPException(status_code=409, detail="an account with this email already exists")
    user = await s.create_user(email, password, name, role=config.AUTH_DEFAULT_ROLE)
    token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)
    return AuthOut(token=token, user=UserOut.from_user(user))


@router.post("/login", response_model=AuthOut)
async def login(body: LoginIn, request: Request):
    """Exchange email+password for a bearer token. Invalid credentials always
    return the same generic 401 (no account enumeration)."""
    await _check_rate_limit(request, "login", config.AUTH_LOGIN_RATE_PER_MIN)
    email = validate_email(body.email)
    s = _require_auth_store()
    user = await s.get_user_by_email(email)
    if user is None or not verify_password(body.password, user.password_hash) or not user.is_active:
        raise HTTPException(status_code=401, detail="invalid email or password")
    token = await s.issue_token(user.id, config.AUTH_TOKEN_TTL_DAYS)
    return AuthOut(token=token, user=UserOut.from_user(user))


@router.get("/me", response_model=UserOut)
async def me(request: Request, _: None = Depends(require_auth)):
    return UserOut.from_user(request.state.user)


@router.post("/logout")
async def logout(request: Request, _: None = Depends(require_auth)):
    """Revoke the current token (server-side)."""
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
    if stored is None or not verify_password(body.current_password, stored.password_hash):
        raise HTTPException(status_code=400, detail="current password is incorrect")
    new_password = validate_password(body.new_password)
    await s.set_password(user.id, hash_password(new_password))
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
    if target.role == "admin" and (body.role == "user" or body.is_active is False) and await s.count_admins() <= 1:
        raise HTTPException(status_code=400, detail="cannot demote or deactivate the last admin")
    name = validate_name(body.name) if body.name is not None else None
    await s.update_user(user_id, name, body.role, body.is_active)
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
    if target.role == "admin" and await s.count_admins() <= 1:
        raise HTTPException(status_code=400, detail="cannot delete the last admin")
    await s.delete_user(user_id)
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


async def bootstrap_admin() -> None:
    """Seed the bootstrap admin from config (once, at startup). Never overwrites
    an existing account's password."""
    email = (config.AUTH_ADMIN_EMAIL or "").strip().lower()
    password = config.AUTH_ADMIN_PASSWORD or ""
    if not email or not password:
        return
    s = _require_auth_store()
    if await s.get_user_by_email(email) is not None:
        return
    await s.create_user(email, password, "Administrator", role="admin")
    logger.info("bootstrapped admin account %s", email)
