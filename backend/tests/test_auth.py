import asyncio
import hashlib
import inspect
import logging
import re
import sqlite3
import statistics
import time
from types import SimpleNamespace

import pytest
from conftest import auth_cookie, session_cookie_value
from fastapi import Depends, HTTPException
from rate_limit_fake import RateLimitRedisFake

from app import auth
from app import config as config_module
from app.auth import AuthStore, DuplicateEmailError, StoredUser, bootstrap_admin
from app.config import config


def _headers(*pairs) -> dict:
    return {k: v for k, v in pairs}


def _req(headers: dict | None = None, state=None, cookies: dict | None = None, method: str = "GET") -> SimpleNamespace:
    """A stand-in Request: require_auth reads the session cookie and runs the
    same-origin guard, which is scoped to unsafe methods."""
    return SimpleNamespace(
        headers=headers or {}, client=None, state=state or SimpleNamespace(), cookies=cookies or {}, method=method
    )


@pytest.fixture
def store(tmp_path) -> AuthStore:
    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    yield s
    asyncio.run(s.close())


def test_password_hashes_never_plaintext(store):
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    assert user.password_hash != "secret12"
    # scheme marker followed by a plain bcrypt string
    assert re.match(r"^\$bcrypt-sha256\$\$2[aby]\$", user.password_hash)
    assert auth.verify_password("secret12", user.password_hash)
    assert not auth.verify_password("wrong12", user.password_hash)
    # hash-only on disk: the raw password never appears in the db or its WAL
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
    # No "too long" case: the policy has no maximum, since an unbounded input
    # costs no extra hashing work once a fixed-width SHA-256 pre-image is
    # hashed in place of the password.
    for bad in ("", "short1", "nodigits", "12345678"):
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
    row = asyncio.run(store._fetchone(
        "SELECT token_hash FROM auth_tokens WHERE user_id = ?", (user.id,)))
    assert row is not None and row["token_hash"] == auth.hash_token(raw)

    resolved = asyncio.run(store.user_for_token(raw))
    assert resolved is not None and resolved.id == user.id

    asyncio.run(store._db.execute(
        "UPDATE auth_tokens SET expires_at = ? WHERE token_hash = ?",
        (auth._now() - 1, auth.hash_token(raw)),
    ))
    asyncio.run(store._db.commit())
    assert asyncio.run(store.user_for_token(raw)) is None

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


def test_bootstrap_admin_refuses_weak_password_and_names_the_reason(store, monkeypatch, caplog):
    """A bootstrap password the signup path would reject must not provision a
    full-admin account."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "x")
    monkeypatch.setattr(auth, "store", store)

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())  # must not raise: the worker still starts

    assert asyncio.run(store.list_users()) == []
    assert asyncio.run(store.get_user_by_email("admin@x.co")) is None

    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a refused bootstrap admin must be logged, not silently skipped"
    joined = "\n".join(errors)
    assert "AUTH_ADMIN_PASSWORD" in joined
    assert "validate_password" in joined
    # Pin the validator's OWN reason verbatim: the remediation hint also
    # contains "8", so a numeric check would pass on an unrelated string.
    assert "password must be at least" in joined
    assert "NOT created" in joined


def test_bootstrap_admin_refuses_malformed_email_and_names_the_reason(store, monkeypatch, caplog):
    """A malformed AUTH_ADMIN_EMAIL must create nothing and name validate_email
    in the log."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "not-an-email")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "adminpass1")
    monkeypatch.setattr(auth, "store", store)

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    assert asyncio.run(store.list_users()) == []
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    joined = "\n".join(errors)
    assert "AUTH_ADMIN_EMAIL" in joined
    assert "validate_email" in joined
    assert "invalid email address" in joined


def test_bootstrap_admin_rejection_is_not_retried_five_times(store, monkeypatch, caplog):
    """A rejected config value is a permanent fault: it must not enter the
    write-lock retry loop, while a transient write lock still retries."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "x")
    monkeypatch.setattr(auth, "store", store)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(auth.asyncio, "sleep", fake_sleep)

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    rejects = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.ERROR and "AUTH_ADMIN_PASSWORD" in r.getMessage()
    ]
    assert len(rejects) == 1, f"expected one rejection log, got {len(rejects)}"
    assert sleeps == [], "a permanent config fault must not enter the retry loop"

    # control: a genuine transient write-lock fault still retries 5x
    calls = {"create": 0}

    async def locked_create(email, password, name, role):
        calls["create"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "adminpass1")
    monkeypatch.setattr(store, "create_user", locked_create)
    sleeps2 = []

    async def fake_sleep2(seconds):
        sleeps2.append(seconds)

    monkeypatch.setattr(auth.asyncio, "sleep", fake_sleep2)
    asyncio.run(auth.bootstrap_admin())
    assert calls["create"] == 5 and sleeps2 == [1, 1, 1, 1]


def test_bootstrap_admin_keeps_existing_weak_admin_but_warns(store, monkeypatch, caplog):
    """A weak existing admin must survive startup -- deleting the only admin is
    an unauthenticated lockout -- but be reported for out-of-band rotation."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "x")
    monkeypatch.setattr(auth, "store", store)
    # a weak admin already exists
    asyncio.run(store.create_user("admin@x.co", "x", "Administrator", role="admin"))

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    existing = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert existing is not None and existing.role == "admin"
    assert auth.verify_password("x", existing.password_hash)
    assert len(asyncio.run(store.list_users())) == 1

    joined = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR
    )
    assert "REJECTED" in joined
    assert "Rotate" in joined
    assert existing.id in joined


def test_bootstrap_admin_accepts_long_passphrase_and_creates_loggable_admin(store, monkeypatch):
    """A passphrase longer than bcrypt's old 72-byte window is serviceable.
    bootstrap_admin is the ONLY path that can create an admin (signup hardcodes
    SIGNUP_ROLE, and PATCH /users needs an admin token that cannot exist yet),
    so refusing it leaves a fresh deploy unadministrable while /health is green."""
    long_pw = "Passphrase1234" + "a" * 89  # 103 chars
    assert len(long_pw.encode()) > auth._LEGACY_BCRYPT_MAX_BYTES
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", long_pw)
    monkeypatch.setattr(auth, "store", store)

    asyncio.run(bootstrap_admin())

    admin = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert admin is not None, "a long passphrase must still bootstrap an admin"
    assert admin.role == "admin" and admin.is_active
    assert auth.verify_password(long_pw, admin.password_hash)
    assert len(asyncio.run(store.list_users())) == 1


def test_bootstrap_admin_long_multibyte_passphrase_does_not_crash(store, monkeypatch):
    """A multi-byte passphrase must not crash bootstrap: the whole value is
    encoded and digested, so nothing is decoded from a cut buffer."""
    pw = "Passphrase1234" + "a" * 57 + "é" * 20
    assert len(pw.encode()) > auth._LEGACY_BCRYPT_MAX_BYTES
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", pw)
    monkeypatch.setattr(auth, "store", store)

    asyncio.run(bootstrap_admin())  # must not raise

    admin = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert admin is not None and admin.role == "admin"
    assert auth.verify_password(pw, admin.password_hash)
    # the whole value is the pre-image, not a byte-prefix of it
    assert auth._prehash(pw) == hashlib.sha256(pw.encode("utf-8")).digest()


@pytest.mark.parametrize(
    "password",
    [
        "a" * 80 + "1",  # the only digit sits past where bcrypt used to cut
        "Passphrase" + "a" * 62 + "123",  # digits start past that point
    ],
)
def test_bootstrap_admin_accepts_digit_past_the_bcrypt_cut(store, monkeypatch, password):
    """The letter+digit rule is judged on the whole configured value, not on any
    byte prefix of it, so every byte of it is part of the credential."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", password)
    monkeypatch.setattr(auth, "store", store)

    asyncio.run(bootstrap_admin())

    admin = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert admin is not None, "a digit late in the passphrase must not block the admin"
    assert admin.role == "admin"
    assert auth.verify_password(password, admin.password_hash)


def test_bootstrap_admin_refuses_genuinely_composition_free_password(store, monkeypatch, caplog):
    """Composition is still enforced on the whole value, so a long password with
    no letter and no digit is refused."""
    password = "\U0001F600" * 100
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", password)
    monkeypatch.setattr(auth, "store", store)

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    assert asyncio.run(store.list_users()) == []
    joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "letter and a digit" in joined


def test_bootstrap_does_not_demand_rotation_for_an_email_rejection(store, monkeypatch, caplog):
    """A dotless address can never validate, so an email-axis rejection fires on
    every worker start; demanding a PASSWORD rotation there is unresolvable."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@localhost")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "healthy-pass1")
    monkeypatch.setattr(auth, "store", store)
    asyncio.run(store.create_user("admin@localhost", "healthy-pass1", "Administrator", role="admin"))

    for _ in range(2):  # every worker start
        with caplog.at_level(logging.ERROR, logger="auth"):
            asyncio.run(bootstrap_admin())

    existing = asyncio.run(store.get_user_by_email("admin@localhost"))
    assert auth.verify_password("healthy-pass1", existing.password_hash)
    joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "Rotate" not in joined, "an email rejection must never demand a password rotation"
    assert "REJECTED" not in joined
    assert "AUTH_ADMIN_EMAIL" in joined
    assert "address" in joined
    assert len(asyncio.run(store.list_users())) == 1


def test_bootstrap_admin_still_blocks_a_sub_floor_password_however_long_it_is(store, monkeypatch):
    """Length is judged on the whole password, and refusing an under-floor one
    must not take startup down with it."""
    password = "a" * 200 + "1"
    monkeypatch.setattr(auth.config, "AUTH_PASSWORD_MIN_LEN", len(password) + 1)
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", password)
    monkeypatch.setattr(auth, "store", store)

    asyncio.run(bootstrap_admin())

    assert asyncio.run(store.list_users()) == []


@pytest.mark.parametrize(
    "password",
    [
        "x",  # min-length reason
        "nodigitsbutlong",  # letter-and-digit reason
    ],
)
def test_bootstrap_rejection_hint_matches_the_reason(store, monkeypatch, caplog, password):
    """The remediation must never contradict the reason it accompanies; the
    reason is read back off the policy itself, so drift fails the test."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", password)
    monkeypatch.setattr(auth, "store", store)

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    assert asyncio.run(store.list_users()) == []
    joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    with pytest.raises(HTTPException) as exc:
        auth.validate_password(password)
    assert exc.value.detail in joined
    # the advice must not push the operator toward the other failure mode
    if "at least" in exc.value.detail:
        assert "letter and a digit" in joined
    else:
        assert "at least" not in joined



def test_all_three_set_paths_agree_on_one_policy():
    """``signup``, ``change_password`` and ``bootstrap_admin`` must all delegate
    to one policy function: a second copy of the rules is the bug. The check is
    that each endpoint adds no bound of its own."""
    for name in ("signup", "change_password"):
        src = inspect.getsource(getattr(auth, name))
        assert "validate_password(" in src, f"{name} must go through validate_password"
        for stray in ("AUTH_PASSWORD_MIN_LEN", "too long", "at least"):
            assert stray not in src, f"{name} restates a bound ({stray}); it must delegate only"

    boot = inspect.getsource(auth.bootstrap_admin)
    assert "_password_rejection(" in boot and "validate_password(" not in boot
    # and the raising form delegates rather than restating the rules
    vp = inspect.getsource(auth.validate_password)
    assert "_password_rejection(" in vp
    assert "too long" not in vp, "the raw 72-byte cap must not come back"


def test_a_truncated_passwords_prefix_no_longer_authenticates_it():
    """Two passwords sharing a 72-byte prefix are the same credential under
    bcrypt truncation -- which is why the value is pre-hashed, not truncated."""
    long_pw = "Passphrase1234" + "a" * 89
    prefix = long_pw[: auth._LEGACY_BCRYPT_MAX_BYTES]
    assert len(long_pw.encode()) > auth._LEGACY_BCRYPT_MAX_BYTES
    stored = auth.hash_password(long_pw)
    # the prefix is still itself an acceptable password...
    assert auth._password_rejection(prefix) is None
    # ...it just no longer authenticates the longer one
    assert not auth.verify_password(prefix, stored)
    assert not auth.verify_password("Z" + prefix[1:], stored)


def test_minimum_is_judged_on_the_whole_password(monkeypatch):
    """The length floor is judged on the whole value: a 30-character password of
    3-byte characters is 30 characters of credential, not 24."""
    pw = "a" + "１" * 29  # 30 chars, 88 bytes
    assert len(pw) == 30 and len(pw.encode()) > auth._LEGACY_BCRYPT_MAX_BYTES
    monkeypatch.setattr(auth.config, "AUTH_PASSWORD_MIN_LEN", 30)
    # 30 characters is 30 characters of credential now, so the floor is met ...
    assert auth._password_rejection(pw) is None
    # ... and one character short of it is still refused, on the whole value
    short = pw[:-1]
    assert auth._password_rejection(short) == "password must be at least 30 characters"
    with pytest.raises(HTTPException) as exc:
        auth.validate_password(short)
    assert "at least 30" in str(exc.value.detail)


def test_a_sub_floor_password_is_refused_by_every_set_path(tmp_path, monkeypatch, store):
    """The minimum rule must hold at the ENDPOINTS, not just in the helper, and
    every leg asserts the rejection REASON: a bare status check could not tell
    "under the floor" from an unrelated refusal."""
    pw = "a" + "１" * 28  # 29 characters
    monkeypatch.setattr(auth.config, "AUTH_PASSWORD_MIN_LEN", 30)
    reason = "password must be at least 30 characters"
    assert auth._password_rejection(pw) == reason

    client, s = _auth_app(tmp_path)
    try:
        r = client.post("/api/auth/signup", json={"email": "a@b.co", "password": pw})
        assert r.status_code == 422, "signup admitted a sub-floor effective credential"
        assert r.json()["detail"] == reason

        # The seeded account's password CLEARS the raised floor, so the 422
        # below can only be about the sub-floor one.
        seeded = "Goodpassword9-abcdefghijklmnop"
        cookie = _session(client, "user@x.co", seeded)
        cr = client.post(
            "/api/auth/change-password",
            cookies=cookie,
            json={"current_password": seeded, "new_password": pw},
        )
        assert cr.status_code == 422, "change_password admitted a sub-floor credential"
        assert cr.json()["detail"] == reason
        # and the existing credential is untouched by the refused change
        assert client.post(
            "/api/auth/login", json={"email": "user@x.co", "password": seeded}
        ).status_code == 200
    finally:
        auth.store = None
        asyncio.run(s.close())

    # The `store` fixture and _auth_app share one tmp_path database, so assert
    # on the ADMIN specifically rather than on the table being empty.
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", pw)
    monkeypatch.setattr(auth, "store", store)
    asyncio.run(bootstrap_admin())
    assert asyncio.run(store.get_user_by_email("admin@x.co")) is None, (
        "bootstrap admitted a sub-floor credential"
    )


def test_change_password_accepts_the_long_passphrase_bootstrap_accepted(tmp_path):
    """An admin's own working passphrase must stay settable through
    change_password, and must authenticate afterwards."""
    long_pw = "Passphrase1234" + "a" * 89  # 103 bytes
    assert len(long_pw.encode()) > auth._LEGACY_BCRYPT_MAX_BYTES
    client, s = _auth_app(tmp_path)
    try:
        cookie = _session(client, "admin@x.co")
        r = client.post(
            "/api/auth/change-password",
            cookies=cookie,
            json={"current_password": "secret12", "new_password": long_pw},
        )
        assert r.status_code == 200, r.text
        # change_password revokes every token and re-issues the session, so the
        # new credential arrives as a fresh Set-Cookie and never in the body.
        new_cookie = auth_cookie(session_cookie_value(r))
        assert "token" not in r.json()
        assert client.post(
            "/api/auth/change-password", cookies=new_cookie,
            json={"current_password": long_pw, "new_password": "secret99"},
        ).status_code == 200
        assert client.post("/api/auth/login", json={"email": "admin@x.co", "password": "secret99"}).status_code == 200
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_accepts_long_password_without_revealing_it(tmp_path):
    """``signup`` shares bootstrap's unbounded policy, and nothing about the
    password may leak into the fixed message that makes the endpoint
    non-enumerable."""
    long_pw = "Passphrase1234" + "a" * 89
    client, s = _auth_app(tmp_path)
    try:
        body = _signup(client, "long@x.co", long_pw)
        assert body == {"message": auth.SIGNUP_ACCEPTED_MESSAGE}
        assert long_pw not in str(body)
        assert client.post("/api/auth/login", json={"email": "long@x.co", "password": long_pw}).status_code == 200
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_no_set_path_still_drops_a_passwords_tail(tmp_path, store, monkeypatch, caplog):
    """What each set path stores is the whole password, and none of the three
    ever writes the password to the auth log."""
    long_pw = "Passphrase1234" + "a" * 89
    caplog.set_level(logging.DEBUG, logger="auth")
    client, s = _auth_app(tmp_path)
    try:
        _signup(client, "tail@x.co", long_pw)
        signed_up = asyncio.run(s.get_user_by_email("tail@x.co")).password_hash
        assert auth.verify_password(long_pw, signed_up)
        assert not auth.verify_password(long_pw[: auth._LEGACY_BCRYPT_MAX_BYTES], signed_up)

        # The auth logger's only record on a set path is the already-registered
        # notice, so the second signup is what makes the log check non-vacuous.
        assert _signup(client, "tail@x.co", long_pw) == {
            "message": auth.SIGNUP_ACCEPTED_MESSAGE
        }

        cookie = _session(client, "tail2@x.co")
        r = client.post(
            "/api/auth/change-password",
            cookies=cookie,
            json={"current_password": "secret12", "new_password": long_pw},
        )
        assert r.status_code == 200
        changed = asyncio.run(s.get_user_by_email("tail2@x.co")).password_hash
        assert auth.verify_password(long_pw, changed)
        assert not auth.verify_password(long_pw[: auth._LEGACY_BCRYPT_MAX_BYTES], changed)
    finally:
        auth.store = None
        asyncio.run(s.close())

    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", long_pw)
    monkeypatch.setattr(auth, "store", store)
    asyncio.run(bootstrap_admin())
    seeded = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert auth.verify_password(long_pw, seeded.password_hash)
    assert not auth.verify_password(long_pw[: auth._LEGACY_BCRYPT_MAX_BYTES], seeded.password_hash)

    # Guard first: without a captured auth record the check below would pass on
    # an empty capture and prove nothing.
    assert any(r.name == "auth" for r in caplog.records), (
        "nothing was captured from the auth logger, so the check below is vacuous"
    )
    assert long_pw not in caplog.text, "a set path must never write the password to the log"


def test_bootstrap_no_rotate_alarm_for_unrelated_healthy_admin(store, monkeypatch, caplog):
    """'Rotate' must appear only when the configured value really is that
    account's current password; a config value that merely fails validation says
    nothing about a healthy admin's password."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "x")
    monkeypatch.setattr(auth, "store", store)
    # but the existing admin is on a strong, valid password
    asyncio.run(store.create_user("admin@x.co", "healthy-pass1", "Administrator", role="admin"))

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    existing = asyncio.run(store.get_user_by_email("admin@x.co"))
    assert auth.verify_password("healthy-pass1", existing.password_hash)
    joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "REJECTED" not in joined
    assert "Rotate" not in joined
    assert "needs no rotation" in joined


def test_bootstrap_probe_failure_is_logged_not_swallowed(store, monkeypatch, caplog):
    """A failed existence probe must be logged, not replaced by the generic
    'no admin account' line with the weak-admin warning silently missing."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "x")
    monkeypatch.setattr(auth, "store", store)

    async def boom(email):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "get_user_by_email", boom)

    with caplog.at_level(logging.WARNING, logger="auth"):
        asyncio.run(bootstrap_admin())  # must not raise

    joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "could not check whether admin" in joined
    assert "OperationalError" in joined
    assert "database is locked" in joined


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


def test_require_auth_accepts_session_cookie_and_rejects_missing(store, monkeypatch):
    monkeypatch.setattr(auth, "store", store)
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    raw = asyncio.run(store.issue_token(user.id, 7))

    async def with_cookie():
        req = _req({}, cookies=auth_cookie(raw))
        await auth.require_auth(req)
        return req.state.user_id

    assert asyncio.run(with_cookie()) == user.id

    async def expect_401(**kwargs):
        await auth.require_auth(_req(**kwargs))

    # The old Authorization header is no longer a transport: a genuinely valid
    # token sent in it is refused, not merely shadowed.
    for bad_kwargs in ({}, {"cookies": auth_cookie("garbage")}, {"headers": {"authorization": f"Bearer {raw}"}}):
        with pytest.raises(HTTPException) as e:
            asyncio.run(expect_401(**bad_kwargs))
        assert e.value.status_code == 401


def test_service_token_acts_as_admin(store, monkeypatch):
    # A service token is a stored, scoped, expiring credential, so it needs the
    # auth store to resolve.
    monkeypatch.setattr(auth, "store", store)
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


def test_service_token_non_ascii_header_is_401_not_500(store, monkeypatch):
    """``secrets.compare_digest`` raises ``TypeError`` on non-ASCII ``str`` input,
    so the token is compared as bytes and a mismatched header is a 401, not a
    500. Driven through a raw ASGI scope because an httpx/TestClient call
    cannot send a non-ASCII header ``str`` at all."""
    from fastapi import FastAPI

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-tok-123")

    app = FastAPI()

    @app.get("/private")
    async def private(_auth: None = Depends(auth.require_auth)):
        return {"ok": True}

    async def drive(raw_header: bytes) -> int:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "path": "/private",
            "raw_path": b"/private",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "client": ("127.0.0.1", 5000),
            "server": ("test", 80),
            "headers": [(b"host", b"test"), (b"x-service-token", raw_header)],
        }
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)  # a TypeError here propagates: that IS the bug
        return next(m for m in sent if m["type"] == "http.response.start")["status"]

    async def scenario():
        # The fix is not "reject more": a wrong ASCII token is still a plain 401.
        assert await drive(b"svc-tok-123") == 200
        assert await drive(b"nope") == 401
        # latin-1 (what a server decodes a raw 0xE9 byte into) and utf-8 both
        # reach the comparison as non-ASCII, and both are ordinary 401s.
        assert await drive(b"svc-tok-\xe9") == 401
        assert await drive("svc-tok-é".encode()) == 401
        assert await store.service_token_for("svc-tok-\xe9") is None

    asyncio.run(scenario())


def test_revoke_service_token_non_ascii_is_not_a_crash(store, monkeypatch):
    """The revoke body is equally attacker-controlled, so a non-ASCII token must
    be a plain "revoked 0" rather than raise out of the endpoint."""
    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(auth.config, "AUTH_SERVICE_TOKEN", "svc-tok-123")
    body = auth.ServiceTokenRevokeIn(token="svc-tok-é")

    async def scenario():
        result = await auth.revoke_service_tokens(SimpleNamespace(), body, None, None)
        assert result == {"revoked": 0}
        # Not the configured token, so the ordinary path ran and left it alone.
        assert await store.service_token_for(body.token) is None
        assert await store.service_token_for("svc-tok-123") is None

    asyncio.run(scenario())


def test_rate_limit_429_and_reset(monkeypatch):
    # The shared fake models SET NX EX / INCR for real, so this asserts the
    # production limiter's window bookkeeping rather than a hand-rolled counter.
    fake = RateLimitRedisFake()
    monkeypatch.setattr(auth, "_rate_client", fake)
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
    assert fake.violations == [], "the limiter must establish every counter with a window TTL"
    key = "auth:rl:login:1.2.3.4"
    assert fake.ttl(key) is not None, "a counter with no TTL would lock this IP out forever"

    fake.advance(auth.config.AUTH_RATE_WINDOW_SECONDS + 1)
    asyncio.run(attempt())
    assert fake.counters[key] == 1


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


def _signup(client, email, password="secret12", name=""):
    """POST /signup and return the parsed body, asserting the 200. Signup is
    tokenless for a fresh address and an already-registered one alike, so a test
    needing a session must log in afterwards."""
    r = client.post("/api/auth/signup", json={"email": email, "password": password, "name": name})
    assert r.status_code == 200, r.text
    return r.json()


def _session(client, email, password="secret12", name=""):
    """Register the address, then log in and return the auth cookie."""
    _signup(client, email, password, name)
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return auth_cookie(session_cookie_value(r))



def test_signup_login_me_flow(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        data = _signup(client, "  New@Example.com ", name="Alice")
        # Signup is tokenless by design: a token here would tell an anonymous
        # caller that the address was free.
        assert set(data) == {"message"}
        assert data["message"] == auth.SIGNUP_ACCEPTED_MESSAGE

        # login is how a session is obtained, and normalizes the same address.
        # The token rides in an HttpOnly cookie and never in the body.
        login = client.post("/api/auth/login", json={"email": "  New@Example.com ", "password": "secret12"})
        assert login.status_code == 200
        assert login.json()["user"]["email"] == "new@example.com"
        assert login.json()["user"]["role"] == "user"
        assert "token" not in login.json()
        cookie = auth_cookie(session_cookie_value(login))

        assert client.get("/api/auth/me", cookies=cookie).json()["email"] == "new@example.com"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_validation_errors(tmp_path, monkeypatch):
    # The case list exceeds AUTH_SIGNUP_RATE_PER_MIN, and the limit is enforced
    # even with the limiter's Redis down, so it is raised to leave the loop
    # measuring the validation rules.
    monkeypatch.setattr(auth.config, "AUTH_SIGNUP_RATE_PER_MIN", 100)
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


def test_signup_duplicate_email_indistinguishable(tmp_path):
    """A duplicate signup must be indistinguishable from a fresh one: the same
    status and the same body, including when the address case varies."""
    client, s = _auth_app(tmp_path)
    try:
        first = _signup(client, "dup@x.co")
        again = client.post("/api/auth/signup", json={"email": "DUP@x.co", "password": "secret12"})
        assert again.status_code == 200
        assert again.json() == first
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_duplicate_leaves_existing_account_untouched(tmp_path):
    """The generic duplicate answer must not touch, replace or take over the
    account: the stored row is byte-identical afterwards, no second row appears,
    and the password the duplicate signup offered never becomes valid."""
    client, s = _auth_app(tmp_path)
    try:
        _signup(client, "dup@x.co", password="secret12", name="Owner")
        before = asyncio.run(s.get_user_by_email("dup@x.co"))
        assert before is not None

        dup = client.post(
            "/api/auth/signup",
            json={"email": "DUP@x.co", "password": "attacker9", "name": "Attacker"},
        )
        assert dup.status_code == 200

        after = asyncio.run(s.get_user_by_email("dup@x.co"))
        assert after == before  # no re-hash, rename, re-activation or id change
        assert asyncio.run(s.list_users()) == [auth.UserOut.from_user(before)]

        assert client.post("/api/auth/login", json={"email": "dup@x.co", "password": "secret12"}).status_code == 200
        assert client.post("/api/auth/login", json={"email": "dup@x.co", "password": "attacker9"}).status_code == 401
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_role_ignores_env_default_role(tmp_path, monkeypatch):
    """A hostile AUTH_DEFAULT_ROLE=admin env must not escalate a public signup.
    Both observable surfaces are checked -- the persisted row and what a real
    session reports via /me -- because a response-only fix is still a full
    compromise."""
    monkeypatch.setattr(auth.config, "AUTH_DEFAULT_ROLE", "admin", raising=False)
    client, s = _auth_app(tmp_path)
    try:
        body = _signup(client, "env@x.co", name="E")
        assert "role" not in body["message"]

        stored = asyncio.run(s.get_user_by_email("env@x.co"))
        assert stored is not None
        assert stored.role == "user"

        login = client.post("/api/auth/login", json={"email": "env@x.co", "password": "secret12"})
        me = client.get("/api/auth/me", cookies=auth_cookie(session_cookie_value(login)))
        assert me.status_code == 200
        assert me.json()["role"] == "user"
        assert me.json()["id"] == stored.id
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_signup_ignores_role_in_request_payload(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        # The account must NOT pre-exist: a duplicate POST short-circuits into
        # the swallowed DuplicateEmailError branch and never reaches create_user.
        r = client.post(
            "/api/auth/signup",
            json={"email": "sneaky@x.co", "password": "secret12", "name": "S", "role": "admin"},
        )
        assert r.status_code == 200
        assert asyncio.run(s.get_user_by_email("sneaky@x.co")).role == "user"
        login = client.post("/api/auth/login", json={"email": "sneaky@x.co", "password": "secret12"})
        assert client.get("/api/auth/me", cookies=auth_cookie(session_cookie_value(login))).json()["role"] == "user"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_login_invalid_credentials_identical_401(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        _signup(client, "a@x.co")
        bad1 = client.post("/api/auth/login", json={"email": "a@x.co", "password": "wrong12"})
        bad2 = client.post("/api/auth/login", json={"email": "nobody@x.co", "password": "secret12"})
        assert bad1.status_code == bad2.status_code == 401
        assert bad1.json() == bad2.json()  # identical message: no account enumeration
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_login_unknown_email_verifies_against_dummy_hash(tmp_path, monkeypatch):
    """``user is not None and await verify_password(...)`` short-circuits, so the
    unknown-address path skipped the bcrypt entirely and answered faster than a
    wrong password: a remote account-existence oracle behind an identical 401.
    Exactly one verify must run, against the dummy hash, which carries the same
    cost factor as a stored one."""
    client, s = _auth_app(tmp_path)
    try:
        _signup(client, "known@x.co")

        verified = []
        real_verify = auth.verify_password

        def recording_verify(password, hashed):
            verified.append(hashed)
            return real_verify(password, hashed)

        monkeypatch.setattr(auth, "verify_password", recording_verify)

        unknown = client.post("/api/auth/login", json={"email": "nobody@x.co", "password": "secret12"})
        assert unknown.status_code == 401
        assert verified == [auth._DUMMY_PASSWORD_HASH]
        assert not real_verify("secret12", auth._DUMMY_PASSWORD_HASH)  # the dummy is not a back door

        # the dummy hash costs what a real one costs, and is in the same scheme
        # so it takes the same verify path
        def cost_of(hashed: str) -> str:
            body = hashed.removeprefix(auth._PASSWORD_SCHEME)
            return body.split("$")[2]  # "$2b$<rounds>$..."

        assert auth._DUMMY_PASSWORD_HASH.startswith(auth._PASSWORD_SCHEME)
        assert cost_of(auth._DUMMY_PASSWORD_HASH) == cost_of(auth.hash_password("secret12"))

        verified.clear()
        wrong = client.post("/api/auth/login", json={"email": "known@x.co", "password": "wrong12"})
        assert wrong.status_code == 401
        assert len(verified) == 1
        assert verified[0] != auth._DUMMY_PASSWORD_HASH
        assert asyncio.run(s.get_user_by_email("known@x.co")).password_hash == verified[0]
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_login_unknown_vs_wrong_password_timing_comparable(tmp_path, monkeypatch):
    """Loose wall-clock guard on the dummy verify: the bug made the unknown
    address faster, so one-sided medians fail on the old code (a few ms vs
    ~100ms) while tolerating a loaded CI box. The per-IP limiter is off because
    the 14 logins here exceed the default AUTH_LOGIN_RATE_PER_MIN."""
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)
    client, s = _auth_app(tmp_path)
    try:
        _signup(client, "known@x.co")

        def median_ms(email: str) -> float:
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                r = client.post("/api/auth/login", json={"email": email, "password": "wrong12"})
                samples.append((time.perf_counter() - start) * 1000)
                assert r.status_code == 401
            return statistics.median(samples)

        wrong_password = median_ms("known@x.co")
        unknown_email = median_ms("nobody@x.co")
        assert unknown_email >= 0.5 * wrong_password, (
            f"unknown-address login took {unknown_email:.1f}ms vs {wrong_password:.1f}ms "
            "for a wrong password: the bcrypt verify was skipped"
        )
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_login_deactivated_account_pays_a_real_verify(tmp_path, monkeypatch):
    """The is_active check runs after the verify, so a deactivated account cannot
    become a third, cheaper branch separating it from 'no account'."""
    client, s = _auth_app(tmp_path)
    try:
        _signup(client, "off@x.co")
        uid = asyncio.run(s.get_user_by_email("off@x.co")).id
        asyncio.run(s.update_user(uid, None, "user", False))

        verified = []
        real_verify = auth.verify_password

        def recording_verify(password, hashed):
            verified.append(hashed)
            return real_verify(password, hashed)

        monkeypatch.setattr(auth, "verify_password", recording_verify)

        r = client.post("/api/auth/login", json={"email": "off@x.co", "password": "secret12"})
        assert r.status_code == 401
        assert r.json() == {"detail": "invalid email or password"}
        # the correct password was genuinely checked before the rejection
        assert verified == [asyncio.run(s.get_user_by_email("off@x.co")).password_hash]
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_me_requires_auth_and_logout_revokes(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        assert client.get("/api/auth/me").status_code == 401
        cookie = _session(client, "a@x.co")
        token = cookie[config.AUTH_COOKIE_NAME]
        assert client.get("/api/auth/me", cookies=cookie).status_code == 200
        assert client.post("/api/auth/logout", cookies=cookie).json() == {"ok": True}
        # Replaying the raw token is 401: revocation is server-side, not merely
        # a cookie the client happened to drop.
        assert client.get("/api/auth/me", cookies=auth_cookie(token)).status_code == 401
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_change_password_invalidates_other_tokens(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        cookie = _session(client, "a@x.co")

        wrong = client.post("/api/auth/change-password", cookies=cookie, json={"current_password": "nope12", "new_password": "secret21"})
        assert wrong.status_code == 400

        ok = client.post("/api/auth/change-password", cookies=cookie, json={"current_password": "secret12", "new_password": "secret21"})
        assert ok.status_code == 200
        # the rotated session is re-issued as a cookie, never in the body
        assert "token" not in ok.json()
        new_cookie = auth_cookie(session_cookie_value(ok))
        assert client.get("/api/auth/me", cookies=cookie).status_code == 401
        assert client.get("/api/auth/me", cookies=new_cookie).status_code == 200
        assert client.post("/api/auth/login", json={"email": "a@x.co", "password": "secret21"}).status_code == 200
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_concurrent_create_duplicate_race_no_poison(tmp_path):
    """Two connections racing the same UNIQUE email must yield exactly one
    success and one DuplicateEmailError. Each attempt gets its own
    AuthStore/connection: one SQLite connection must never be shared across
    concurrent coroutines."""
    db_path = str(tmp_path / "auth.db")

    async def attempt(name: str):
        s = AuthStore(db_path)
        await s.connect()
        try:
            return await s.create_user("race@x.co", "secret12", name, "user")
        finally:
            await s.close()

    async def main():
        # prime the schema once on a throwaway connection
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


def test_signup_duplicate_race_is_indistinguishable(tmp_path):
    """Two concurrent signups for one address must both look like a plain
    success. Raced as two coroutines on one event loop: the store's SQLite
    connection is bound to the thread it was opened on, so a cross-thread
    TestClient is flaky."""
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
                # both requests are told the same thing ...
                assert [r.status_code for r in results] == [200, 200]
                assert results[0].json() == results[1].json() == {
                    "message": auth.SIGNUP_ACCEPTED_MESSAGE
                }
                # ... and the race left exactly one account, which works
                assert len(await s.list_users()) == 1
                login = await client.post(
                    "/api/auth/login", json={"email": "same@x.co", "password": "secret12"}
                )
                assert login.status_code == 200
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
        uh = _session(client, "user@x.co")
        ah = _session(client, "boss@x.co", name="Boss")
        asyncio.run(s.update_user(asyncio.run(s.get_user_by_email("boss@x.co")).id, None, "admin", None))

        
        

        assert client.get("/api/auth/users", cookies=uh).status_code == 403
        assert client.patch("/api/auth/users/some-id", cookies=uh, json={"role": "user"}).status_code == 403
        listing = client.get("/api/auth/users", cookies=ah)
        assert listing.status_code == 200 and len(listing.json()) == 2
        assert client.get("/api/auth/users/nope", cookies=ah).status_code == 404
        uid = listing.json()[0]["id"]
        assert client.patch(f"/api/auth/users/{uid}", cookies=ah, json={"role": "superuser"}).status_code == 422
        user_id = asyncio.run(s.get_user_by_email("user@x.co")).id
        assert client.patch(f"/api/auth/users/{user_id}", cookies=ah, json={"role": "admin"}).status_code == 200
        assert client.post(f"/api/auth/users/{user_id}/tokens/revoke", cookies=ah).json() == {"ok": True}
        assert client.get("/api/auth/me", cookies=uh).status_code == 401
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_verify_password_malformed_hash_returns_false():
    """A malformed stored password hash raises ValueError inside bcrypt, which
    verify_password must swallow as a plain False."""
    assert auth.verify_password("secret12", "not-a-bcrypt-hash") is False
    assert auth.verify_password("secret12", "") is False


def test_create_user_generic_error_rolls_back_and_raises(store, monkeypatch):
    """A non-integrity INSERT failure must roll back so the connection never holds
    an open write transaction, then re-raise."""
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
    user = asyncio.run(store.create_user("a@b.co", "secret12", "A", "user"))
    token = asyncio.run(store.issue_token(user.id, 7))
    assert asyncio.run(store.user_for_token(token)) is not None
    asyncio.run(store.delete_user(user.id))
    assert asyncio.run(store.get_user(user.id)) is None
    assert asyncio.run(store.user_for_token(token)) is None


def test_issue_token_error_rolls_back_and_raises(store, monkeypatch):
    """A token INSERT failure must roll back before re-raising."""
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
    # Auto (the shipped default) with a client that is not behind a local proxy:
    # the socket peer stays authoritative, so a direct caller cannot forge an IP
    # to escape its own rate-limit bucket.
    req = _req({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
    req.client = SimpleNamespace(host="1.2.3.4")
    assert auth._client_ip(req) == "1.2.3.4"
    # Auto with a loopback peer -- the reference deploy, where nginx on this host
    # forwards to 127.0.0.1. The rightmost (nginx-appended) XFF hop wins, so a
    # client cannot spoof its IP by prepending a forged address.
    req = _req({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})
    req.client = SimpleNamespace(host="127.0.0.1")
    assert auth._client_ip(req) == "10.0.0.1"
    req = _req({"x-forwarded-for": " 203.0.113.9 "})
    req.client = SimpleNamespace(host="::1")
    assert auth._client_ip(req) == "203.0.113.9"
    # No forwarded header at all: the socket peer is authoritative, so the
    # caller keys on its own address rather than on anything it claims.
    req = _req({})
    req.client = SimpleNamespace(host="1.2.3.4")
    assert auth._client_ip(req) == "1.2.3.4"
    assert auth._client_ip(_req({})) == "unknown"


def test_client_ip_trust_setting_overrides_the_auto_peer_check(monkeypatch):
    req = _req({"x-forwarded-for": "10.0.0.1"})
    req.client = SimpleNamespace(host="127.0.0.1")
    monkeypatch.setattr(auth.config, "AUTH_TRUST_X_FORWARDED_FOR", False)
    assert auth._client_ip(req) == "127.0.0.1"

    req = _req({"x-forwarded-for": "10.0.0.1"})
    req.client = SimpleNamespace(host="203.0.113.9")
    monkeypatch.setattr(auth.config, "AUTH_TRUST_X_FORWARDED_FOR", True)
    assert auth._client_ip(req) == "10.0.0.1"


def test_xff_trust_env_parsing_is_three_state(monkeypatch):
    """'auto' (and an unset var, as in the .env shipped by setup.sh) resolves to
    the auto behaviour; an unrecognised value falls back to auto rather than
    silently picking a side."""
    monkeypatch.delenv("PROBE_FLAG", raising=False)
    assert config_module._env_tristate("PROBE_FLAG") is None
    for value in ("auto", "AUTO", "", "  "):
        monkeypatch.setenv("PROBE_FLAG", value)
        assert config_module._env_tristate("PROBE_FLAG") is None, value
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("PROBE_FLAG", value)
        assert config_module._env_tristate("PROBE_FLAG") is True, value
    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("PROBE_FLAG", value)
        assert config_module._env_tristate("PROBE_FLAG") is False, value
    # A typo must not resolve to the unsafe (trust-everything) side.
    monkeypatch.setenv("PROBE_FLAG", "yse")
    assert config_module._env_tristate("PROBE_FLAG") is None


def _promote_to_admin(client, s, email):
    uid = asyncio.run(s.get_user_by_email(email)).id
    asyncio.run(s.update_user(uid, None, "admin", None))
    return uid


def test_get_user_endpoint(tmp_path):
    client, s = _auth_app(tmp_path)
    try:
        ah = _session(client, "boss@x.co")
        uid = _promote_to_admin(client, s, "boss@x.co")
        
        r = client.get(f"/api/auth/users/{uid}", cookies=ah)
        assert r.status_code == 200
        assert r.json()["email"] == "boss@x.co"
        assert r.json()["role"] == "admin"
        assert client.get("/api/auth/users/nope", cookies=ah).status_code == 404
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_patch_user_last_admin_guard_endpoint(tmp_path):
    """PATCH cannot demote or deactivate the last active admin -- self-lockout
    protection."""
    client, s = _auth_app(tmp_path)
    try:
        ah = _session(client, "boss@x.co")
        uid = _promote_to_admin(client, s, "boss@x.co")
        
        assert client.patch(f"/api/auth/users/{uid}", cookies=ah, json={"role": "user"}).status_code == 400
        assert client.patch(f"/api/auth/users/{uid}", cookies=ah, json={"is_active": False}).status_code == 400
        # once a second admin exists the guard releases
        asyncio.run(s.create_user("other@x.co", "secret12", "O", "admin"))
        assert client.patch(f"/api/auth/users/{uid}", cookies=ah, json={"role": "user"}).status_code == 200
        assert asyncio.run(s.get_user(uid)).role == "user"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_delete_user_last_admin_guard_endpoint(tmp_path):
    """DELETE cannot remove the last active admin -- self-lockout protection."""
    client, s = _auth_app(tmp_path)
    try:
        ah = _session(client, "boss@x.co")
        uid = _promote_to_admin(client, s, "boss@x.co")
        
        assert client.delete(f"/api/auth/users/{uid}", cookies=ah).status_code == 400
        # a second admin releases the guard
        asyncio.run(s.create_user("other@x.co", "secret12", "O", "admin"))
        assert client.delete(f"/api/auth/users/{uid}", cookies=ah).status_code == 200
        assert asyncio.run(s.get_user(uid)) is None
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_bootstrap_admin_gives_up_after_write_lock_retries(store, monkeypatch):
    """bootstrap_admin retries 5 times on a write lock, sleeping between
    attempts, then gives up gracefully instead of failing startup."""
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


def test_bootstrap_still_flags_a_weak_admin_behind_an_email_fault(store, monkeypatch, caplog):
    """The email rejection must not swallow the weak-admin warning: the account's
    stored password really IS the rejected config value, so rotation is the
    right advice. Deciding by which variable failed would hide a live
    1-character admin password."""
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@localhost")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", "x")
    monkeypatch.setattr(auth, "store", store)
    # this admin was provisioned with the 1-character password
    asyncio.run(store.create_user("admin@localhost", "x", "Administrator", role="admin"))

    with caplog.at_level(logging.ERROR, logger="auth"):
        asyncio.run(bootstrap_admin())

    joined = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "Rotate" in joined, "a live weak admin password must still be flagged"
    assert "AUTH_ADMIN_EMAIL" in joined


@pytest.mark.parametrize(
    "password",
    [
        "a1" + "b" * (auth.config.AUTH_PASSWORD_MIN_LEN - 3),  # one under the minimum
        "a1" + "b" * (auth.config.AUTH_PASSWORD_MIN_LEN - 2),  # exactly the minimum
    ],
)
def test_bootstrap_enforces_the_minimum_length_boundary(store, monkeypatch, password):
    """An admin password shorter than AUTH_PASSWORD_MIN_LEN is refused rather
    than provisioned. Every case has a letter AND a digit, so the length rule is
    the only thing under test."""
    assert len(password) in (
        auth.config.AUTH_PASSWORD_MIN_LEN - 1,
        auth.config.AUTH_PASSWORD_MIN_LEN,
    )
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", password)
    monkeypatch.setattr(auth, "store", store)

    asyncio.run(bootstrap_admin())

    admin = asyncio.run(store.get_user_by_email("admin@x.co"))
    if len(password) < auth.config.AUTH_PASSWORD_MIN_LEN:
        assert admin is None, "a password one under the minimum must be refused"
        assert asyncio.run(store.list_users()) == []
    else:
        # exactly at the minimum is valid and must still bootstrap, so the
        # boundary is a real edge rather than an accident of the fixture
        assert admin is not None and admin.role == "admin"
        assert auth.verify_password(password, admin.password_hash)
