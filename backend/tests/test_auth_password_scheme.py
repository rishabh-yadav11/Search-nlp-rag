"""bcrypt is handed a SHA-256 pre-image, so nothing truncates at 72 bytes.

Two things have to hold at once:

1. The credential is the WHOLE password: a value differing only after byte 72
   is a different password, and the prefix does not authenticate it.
2. Every pre-migration credential still authenticates, because a legacy hash is
   ``bcrypt(raw[:72])`` while the current scheme is ``bcrypt(sha256(raw))``.
   Changing what a stored hash MEANS would otherwise lock out every existing
   account, including the only admin on a fresh deploy.

``_legacy_hash`` below builds the pre-migration form byte-for-byte. That is the
whole migration surface: no schema change, no bulk rewrite, no reset mail.
"""
import asyncio
import inspect
import logging
import re

import bcrypt
import pytest
from conftest import auth_cookie, session_cookie_value

from app import auth
from app.auth import AuthStore

# 103 bytes, every one past the 72-byte window bcrypt truncated at.
LONG_PW = "Passphrase1234" + "a" * 89
LONG_PREFIX = LONG_PW[: auth._LEGACY_BCRYPT_MAX_BYTES]
# Same first 72 bytes as LONG_PW, different after -- the collision truncation permitted.
LONG_TWINS = LONG_PREFIX + "bbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

SHORT_PW = "secret12"
ROTATED_PW = "rotated99"


def _legacy_hash(password: str) -> str:
    """A pre-migration stored hash: ``bcrypt(password[:72])``."""
    return bcrypt.hashpw(
        password.encode("utf-8")[: auth._LEGACY_BCRYPT_MAX_BYTES], bcrypt.gensalt()
    ).decode("utf-8")


@pytest.fixture
def store(tmp_path) -> AuthStore:
    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    yield s
    asyncio.run(s.close())


def _auth_app(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    s = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(s.connect())
    auth.store = s
    app = FastAPI()
    app.include_router(auth.router)
    return TestClient(app), s


def _downgrade_to_legacy(db_path, email, password):
    """Rewrite an existing account's hash into the pre-migration form."""
    s = AuthStore(str(db_path))
    asyncio.run(s.connect())

    async def go():
        user = await s.get_user_by_email(email)
        assert user is not None, f"{email} was never seeded"
        await s.set_password(user.id, _legacy_hash(password))

    try:
        asyncio.run(go())
    finally:
        asyncio.run(s.close())


# --- the collision is gone -------------------------------------------------


def test_a_72_byte_prefix_does_not_authenticate_the_longer_password():
    """``_legacy_hash(LONG_PW)`` is what the pre-migration code stored and the
    prefix genuinely does open it; the current scheme opens neither value."""
    legacy = _legacy_hash(LONG_PW)
    assert auth.verify_password(LONG_PREFIX, legacy), (
        "the fixture must reproduce the pre-migration defect, or this test proves nothing"
    )

    current = auth.hash_password(LONG_PW)
    assert not auth.verify_password(LONG_PREFIX, current), (
        "the 72-byte prefix authenticated a longer password: the truncation is back"
    )
    assert not auth.verify_password(LONG_PW, auth.hash_password(LONG_PREFIX))


def test_two_passwords_differing_only_past_byte_72_are_distinct_credentials():
    """Under truncation these two hashed to the same thing and either one
    logged the other in; now everything past byte 72 is part of the credential."""
    a, b = auth.hash_password(LONG_PW), auth.hash_password(LONG_TWINS)
    assert not auth.verify_password(LONG_TWINS, a)
    assert not auth.verify_password(LONG_PW, b)
    assert LONG_PW[: auth._LEGACY_BCRYPT_MAX_BYTES] == LONG_TWINS[: auth._LEGACY_BCRYPT_MAX_BYTES]
    assert auth.verify_password(LONG_PW, a) and auth.verify_password(LONG_TWINS, b)


def test_a_long_password_registered_today_still_authenticates_tomorrow(tmp_path):
    """The stored form is self-describing, so a fresh process reading the
    database from scratch verifies against the pre-image and the account still
    logs in -- nothing here is in-memory state a restart would lose."""
    db = str(tmp_path / "auth.db")
    client, s = _auth_app(tmp_path)
    try:
        r = client.post("/api/auth/signup", json={"email": "long@x.co", "password": LONG_PW})
        assert r.status_code == 200
        stored = asyncio.run(s.get_user_by_email("long@x.co")).password_hash
        assert stored.startswith(auth._PASSWORD_SCHEME)
        assert not auth.verify_password(LONG_PREFIX, stored)
    finally:
        auth.store = None
        asyncio.run(s.close())

    # A brand new store over the same file, as a restarted worker would open it.
    reopened = AuthStore(db)
    asyncio.run(reopened.connect())
    try:
        row = asyncio.run(reopened.get_user_by_email("long@x.co"))
        assert row is not None
        assert auth.verify_password(LONG_PW, row.password_hash)
        assert not auth.verify_password(LONG_PREFIX, row.password_hash)
        assert not auth.verify_password(LONG_TWINS, row.password_hash)
    finally:
        asyncio.run(reopened.close())


# --- the migration ---------------------------------------------------------


def test_a_credential_stored_before_the_change_still_authenticates(tmp_path):
    """A row written in the pre-migration form logs in unchanged."""
    client, s = _auth_app(tmp_path)
    try:
        asyncio.run(s.create_user("old@x.co", SHORT_PW, "Old", "user"))
        legacy = _legacy_hash(SHORT_PW)
        user = asyncio.run(s.get_user_by_email("old@x.co"))
        asyncio.run(s.set_password(user.id, legacy))

        r = client.post("/api/auth/login", json={"email": "old@x.co", "password": SHORT_PW})
        assert r.status_code == 200, r.text
        # the session comes back as the HttpOnly cookie and never in the body
        assert auth_cookie(session_cookie_value(r))
        assert "token" not in r.json()
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_a_legacy_credential_is_rehashed_on_its_next_successful_login(tmp_path):
    """Legacy hashes are migrated by the login itself: the plaintext exists in
    exactly one request, so that request is the only place the old hash can be
    re-expressed."""
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("old@x.co", SHORT_PW, "Old", "user"))
        legacy = _legacy_hash(SHORT_PW)
        asyncio.run(s.set_password(user.id, legacy))
        assert asyncio.run(s.count_legacy_passwords()) == 1

        assert client.post(
            "/api/auth/login", json={"email": "old@x.co", "password": SHORT_PW}
        ).status_code == 200

        after = asyncio.run(s.get_user_by_email("old@x.co")).password_hash
        assert after.startswith(auth._PASSWORD_SCHEME)
        assert after != legacy
        assert not auth.needs_rehash(after)
        assert asyncio.run(s.count_legacy_passwords()) == 0
        assert auth.verify_password(SHORT_PW, after)
        # and a second login is a no-op rather than another rewrite
        assert client.post(
            "/api/auth/login", json={"email": "old@x.co", "password": SHORT_PW}
        ).status_code == 200
        assert asyncio.run(s.get_user_by_email("old@x.co")).password_hash == after
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_migrating_a_long_legacy_credential_closes_its_prefix_collision(tmp_path):
    """A long passphrase stored in the legacy form is a credential its own
    72-byte prefix opens; the login that migrates it is the moment that stops
    being true -- same row, same password, prefix refused from then on."""
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("long@x.co", LONG_PW, "Long", "user"))
        asyncio.run(s.set_password(user.id, _legacy_hash(LONG_PW)))
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PREFIX}
        ).status_code == 200
        assert asyncio.run(s.count_legacy_passwords()) == 0, "the prefix login already migrated it"

        # Reset to the pre-migration state, then migrate with the real password.
        asyncio.run(s.set_password(user.id, _legacy_hash(LONG_PW)))
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PW}
        ).status_code == 200
        migrated = asyncio.run(s.get_user_by_email("long@x.co")).password_hash
        assert migrated.startswith(auth._PASSWORD_SCHEME)
        assert auth.verify_password(LONG_PW, migrated)
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PREFIX}
        ).status_code == 401
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PW}
        ).status_code == 200, "migration locked the owner out of their own account"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_an_upgraded_account_no_longer_accepts_its_72_byte_prefix(tmp_path):
    """A "verify the pre-image, else try the legacy form too" fallback would keep
    the collision alive for exactly the accounts that were fixed."""
    client, s = _auth_app(tmp_path)
    try:
        assert client.post(
            "/api/auth/signup", json={"email": "long@x.co", "password": LONG_PW}
        ).status_code == 200
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PW}
        ).status_code == 200
        assert not auth.needs_rehash(
            asyncio.run(s.get_user_by_email("long@x.co")).password_hash
        )
        for attack in (LONG_PREFIX, LONG_TWINS, LONG_PREFIX + "1"):
            r = client.post("/api/auth/login", json={"email": "long@x.co", "password": attack})
            assert r.status_code == 401, f"{attack[:16]!r}... still opened a migrated account"
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_a_failed_login_never_upgrades_the_stored_hash(tmp_path):
    """The upgrade is a credential rewrite, so it must not be reachable by
    anyone who cannot already authenticate -- otherwise a failed guess is a
    free way to churn rows."""
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("old@x.co", SHORT_PW, "Old", "user"))
        legacy = _legacy_hash(SHORT_PW)
        asyncio.run(s.set_password(user.id, legacy))

        for guess in ("wrong12", SHORT_PW + "x", SHORT_PW[:-1], ""):
            r = client.post("/api/auth/login", json={"email": "old@x.co", "password": guess})
            assert r.status_code in (401, 422), guess
        assert asyncio.run(s.get_user_by_email("old@x.co")).password_hash == legacy
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_the_upgrade_is_a_compare_and_swap_on_the_hash_that_was_verified(store):
    """The write must land only while the row still holds the hash that was just
    checked; anything else overwrites a credential the owner has replaced."""
    user = asyncio.run(store.create_user("a@x.co", SHORT_PW, "A", "user"))
    legacy = _legacy_hash(SHORT_PW)
    asyncio.run(store.set_password(user.id, legacy))
    newer = auth.hash_password(ROTATED_PW)
    asyncio.run(store.set_password(user.id, newer))

    stale = asyncio.run(store.upgrade_password_hash(user.id, legacy, auth.hash_password(SHORT_PW)))
    assert stale == 0, "a stale observation overwrote a newer credential"
    assert asyncio.run(store.get_user(user.id)).password_hash == newer
    assert auth.verify_password(ROTATED_PW, newer)
    assert not auth.verify_password(SHORT_PW, newer)

    fresh = auth.hash_password(ROTATED_PW + "x")
    assert asyncio.run(store.upgrade_password_hash(user.id, newer, fresh)) == 1
    assert asyncio.run(store.get_user(user.id)).password_hash == fresh


def test_a_concurrent_password_change_is_not_clobbered_by_a_login_upgrade(tmp_path, monkeypatch):
    """A concurrent writer's credential is derived from a secret the owner has
    moved on from, so writing it would resurrect a password they believe is
    gone: the compare-and-swap has to lose."""
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("old@x.co", SHORT_PW, "Old", "user"))
        asyncio.run(s.set_password(user.id, _legacy_hash(SHORT_PW)))

        real_get = s.get_user_by_email

        async def racing_get(email):
            row = await real_get(email)
            if row is not None and email == "old@x.co":
                # Another worker finishes a password change after we read the row.
                await s.set_password(row.id, auth.hash_password(ROTATED_PW))
            return row

        monkeypatch.setattr(s, "get_user_by_email", racing_get)

        r = client.post("/api/auth/login", json={"email": "old@x.co", "password": SHORT_PW})
        assert r.status_code == 200, r.text
        final = asyncio.run(s.get_user_by_email("old@x.co")).password_hash
        assert auth.verify_password(ROTATED_PW, final), "the concurrent change was clobbered"
        assert not auth.verify_password(SHORT_PW, final), (
            "the login wrote back a password the owner had already replaced"
        )
    finally:
        auth.store = None
        asyncio.run(s.close())


def test_a_failed_upgrade_write_does_not_fail_the_login(tmp_path, monkeypatch, caplog):
    """The migration is housekeeping and must never become an outage: if the
    rewrite cannot be written, the legacy hash still authenticates and the
    upgrade is retried on the next login."""
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("old@x.co", SHORT_PW, "Old", "user"))
        legacy = _legacy_hash(SHORT_PW)
        asyncio.run(s.set_password(user.id, legacy))

        async def broken_upgrade(*_a, **_kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(s, "upgrade_password_hash", broken_upgrade)
        with caplog.at_level(logging.WARNING, logger="auth"):
            r = client.post("/api/auth/login", json={"email": "old@x.co", "password": SHORT_PW})
        assert r.status_code == 200, "a failed migration write cost the user their login"
        assert auth_cookie(session_cookie_value(r)), "a failed migration write cost the user their session"
        assert asyncio.run(s.get_user_by_email("old@x.co")).password_hash == legacy
        assert auth.verify_password(SHORT_PW, legacy), "the row must still authenticate"
        assert "could not upgrade" in "\n".join(x.getMessage() for x in caplog.records)
    finally:
        auth.store = None
        asyncio.run(s.close())


# --- the fallback is not a choice the caller can make ----------------------


@pytest.mark.parametrize("scheme", ["current", "legacy"])
def test_verification_runs_exactly_one_bcrypt_on_either_scheme(monkeypatch, scheme):
    """Exactly one ``checkpw`` per verify, whatever the row holds: two would
    re-offer the legacy pre-image for migrated rows and make not-yet-migrated
    rows distinguishable by timing."""
    hashed = auth.hash_password(SHORT_PW) if scheme == "current" else _legacy_hash(SHORT_PW)
    calls = []
    real = auth.bcrypt.checkpw

    def recording(password, hashed_password):
        calls.append(password)
        return real(password, hashed_password)

    monkeypatch.setattr(auth.bcrypt, "checkpw", recording)
    assert auth.verify_password(SHORT_PW, hashed)
    assert len(calls) == 1, f"{scheme}: verify tried {len(calls)} candidates"
    if scheme == "current":
        assert calls[0] == auth._prehash(SHORT_PW)
    else:
        assert calls[0] == SHORT_PW.encode()[: auth._LEGACY_BCRYPT_MAX_BYTES]


def test_the_scheme_is_decided_by_the_stored_hash_not_by_the_caller():
    """The dispatch reads the stored marker and nothing else, so no request
    field, header or value shape can steer a row onto the other scheme."""
    src = inspect.getsource(auth.verify_password)
    assert "hashed.startswith(_PASSWORD_SCHEME)" in src
    # A second checkpw would be a second candidate scheme, which is the property above.
    assert src.count("bcrypt.checkpw(") == 1


def test_needs_rehash_reads_the_stored_marker_only():
    assert auth.needs_rehash(auth.hash_password(SHORT_PW)) is False
    assert auth.needs_rehash(_legacy_hash(SHORT_PW)) is True
    # A row that is not a hash at all can never verify, so the upgrade is never reached.
    assert auth.needs_rehash("") is True
    assert auth.verify_password(SHORT_PW, "") is False


def test_a_corrupt_stored_hash_is_refused_rather_than_raised():
    """The two-scheme dispatch must not turn a bad row into a 500."""
    for bad in ("", "not-a-hash", auth._PASSWORD_SCHEME, auth._PASSWORD_SCHEME + "nonsense"):
        assert auth.verify_password(SHORT_PW, bad) is False


# --- the operator's view ---------------------------------------------------


def test_startup_reports_the_accounts_still_on_a_pre_migration_hash(store, monkeypatch, caplog):
    """Dormant accounts can never be migrated, because a hash that means "the
    first 72 bytes" only becomes a pre-image hash if someone supplies the
    plaintext, and this service has no password-reset path. The operator has to
    be able to see the remainder."""
    monkeypatch.setattr(auth, "store", store)
    # Deliberately uneven, so counting the wrong side of the marker cannot give the same number.
    for email in ("old1@x.co", "old2@x.co", "old3@x.co"):
        u = asyncio.run(store.create_user(email, SHORT_PW, "Old", "user"))
        asyncio.run(store.set_password(u.id, _legacy_hash(SHORT_PW)))
    asyncio.run(store.create_user("new@x.co", SHORT_PW, "New", "user"))

    with caplog.at_level(logging.INFO, logger="auth"):
        remaining = asyncio.run(auth.report_legacy_password_hashes())
    assert remaining == 3
    line = "\n".join(x.getMessage() for x in caplog.records)
    assert "3 account(s) still hold a pre-migration password hash" in line
    assert "next successful login" in line
    assert "never logs in again" in line
    # It counts rows; it never reveals which account, let alone any credential.
    assert "old1@x.co" not in line and SHORT_PW not in line


def test_the_report_says_so_when_there_is_nothing_left_to_migrate(store, monkeypatch, caplog):
    """A deploy with nothing pending says so, rather than leaving the line
    unreadable as "the migration must not have started"."""
    monkeypatch.setattr(auth, "store", store)
    asyncio.run(store.create_user("new@x.co", SHORT_PW, "New", "user"))

    with caplog.at_level(logging.INFO, logger="auth"):
        assert asyncio.run(auth.report_legacy_password_hashes()) == 0
    assert "no pre-migration password hashes remain" in "\n".join(
        x.getMessage() for x in caplog.records
    )


def test_the_report_never_fails_startup(store, monkeypatch, caplog):
    """A boot is not worth failing over a count, and there may be no store yet."""
    monkeypatch.setattr(auth, "store", None)
    assert asyncio.run(auth.report_legacy_password_hashes()) is None

    async def boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(auth, "store", store)
    monkeypatch.setattr(store, "count_legacy_passwords", boom)
    with caplog.at_level(logging.WARNING, logger="auth"):
        assert asyncio.run(auth.report_legacy_password_hashes()) is None
    assert "could not count pre-migration password hashes" in "\n".join(
        x.getMessage() for x in caplog.records
    )


# --- the bootstrap admin ---------------------------------------------------


def test_a_fresh_deploy_is_administrable_through_the_migration(tmp_path, monkeypatch):
    """Both deploys are administrable: a fresh one seeds its admin in the current
    scheme, and one carrying a pre-migration admin credential -- the only account
    there is -- migrates on the operator's ordinary login."""
    db = str(tmp_path / "auth.db")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", LONG_PW)

    s = AuthStore(db)
    asyncio.run(s.connect())
    monkeypatch.setattr(auth, "store", s)
    asyncio.run(auth.bootstrap_admin())
    fresh = asyncio.run(s.get_user_by_email("admin@x.co"))
    assert fresh is not None and fresh.role == "admin"
    assert fresh.password_hash.startswith(auth._PASSWORD_SCHEME)
    assert not auth.needs_rehash(fresh.password_hash)
    assert asyncio.run(s.count_legacy_passwords()) == 0
    asyncio.run(s.close())

    # (b) the same deploy as it looked before the change: a pre-migration hash.
    _downgrade_to_legacy(db, "admin@x.co", LONG_PW)

    client, s2 = _auth_app(tmp_path)
    try:
        r = client.post("/api/auth/login", json={"email": "admin@x.co", "password": LONG_PW})
        assert r.status_code == 200, r.text
        migrated = asyncio.run(s2.get_user_by_email("admin@x.co")).password_hash
        assert migrated.startswith(auth._PASSWORD_SCHEME)
        assert auth.verify_password(LONG_PW, migrated)
        assert not auth.verify_password(LONG_PREFIX, migrated)
    finally:
        auth.store = None
        asyncio.run(s2.close())



def test_a_long_password_is_stored_as_the_full_credential(store):
    """Asserted rather than assumed: if the pre-image ever went back to being the
    password itself, the 72-byte cut would return silently and every other
    behavioural test here would still pass on short passwords."""
    stored = auth.hash_password(LONG_PW)
    assert re.match(r"^\$bcrypt-sha256\$\$2[aby]\$", stored)
    assert len(auth._prehash(LONG_PW)) == 32
    assert len(auth._prehash(LONG_PW)) < auth._LEGACY_BCRYPT_MAX_BYTES
    user = asyncio.run(store.create_user("a@x.co", LONG_PW, "A", "user"))
    assert user.password_hash.startswith(auth._PASSWORD_SCHEME)
    assert LONG_PW not in user.password_hash
