"""#387: bcrypt is handed a SHA-256 pre-image, so nothing truncates at 72 bytes.

Two things have to hold at once, and this file is the proof of both:

1. The credential is the WHOLE password. A value differing only after byte 72 is
   a different password, and the prefix does not authenticate it.
2. Every credential stored before the change still authenticates, because a
   pre-migration hash is ``bcrypt(raw[:72])`` and the new scheme is
   ``bcrypt(sha256(raw))``. Changing what a stored hash MEANS would otherwise
   lock out every existing account, including the only admin on a fresh deploy.

The pre-migration hash is built by ``_legacy_hash`` below, which is byte-for-byte
what the pre-#387 code produced. That is the whole migration surface: no schema
change, no bulk rewrite, no reset mail.
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

# 103 bytes, every one of them past the window bcrypt used to truncate at.
LONG_PW = "Passphrase1234" + "a" * 89
LONG_PREFIX = LONG_PW[: auth._LEGACY_BCRYPT_MAX_BYTES]
# Same first 72 bytes as LONG_PW, different after. This pair IS the collision
# bcrypt truncation permitted: pre-#387 they hashed identically.
LONG_TWINS = LONG_PREFIX + "bbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

SHORT_PW = "secret12"
ROTATED_PW = "rotated99"


def _legacy_hash(password: str) -> str:
    """A pre-migration stored hash, exactly as the code before #387 built it."""
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
    """Rewrite an existing account's hash into the pre-migration form.

    This is what the same deploy looked like the day before the change: the row
    is there, under the same id, holding ``bcrypt(raw[:72])``.
    """
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
    """The bug this issue exists to remove.

    ``_legacy_hash(LONG_PW)`` is what the pre-#387 code stored, and the prefix
    genuinely does open it -- that is the pinned defect. The current scheme
    stores neither as the other's credential.
    """
    legacy = _legacy_hash(LONG_PW)
    assert auth.verify_password(LONG_PREFIX, legacy), (
        "the fixture must reproduce the pre-migration defect, or this test proves nothing"
    )

    current = auth.hash_password(LONG_PW)
    assert not auth.verify_password(LONG_PREFIX, current), (
        "the 72-byte prefix authenticated a longer password: the truncation is back"
    )
    # and the other way round: the longer value is not a super-credential for
    # an account whose real password is the short one
    assert not auth.verify_password(LONG_PW, auth.hash_password(LONG_PREFIX))


def test_two_passwords_differing_only_past_byte_72_are_distinct_credentials():
    """Acceptance: "a password differing only after byte 72 is a distinct
    credential", and "length beyond 72 bytes increases effective entropy".

    Under truncation these two hashed to the same thing and either one logged
    the other in. Now neither does, which is the whole point: everything after
    byte 72 is part of the credential again.
    """
    a, b = auth.hash_password(LONG_PW), auth.hash_password(LONG_TWINS)
    assert not auth.verify_password(LONG_TWINS, a)
    assert not auth.verify_password(LONG_PW, b)
    # the difference is entirely past the window, and each authenticates itself
    assert LONG_PW[: auth._LEGACY_BCRYPT_MAX_BYTES] == LONG_TWINS[: auth._LEGACY_BCRYPT_MAX_BYTES]
    assert auth.verify_password(LONG_PW, a) and auth.verify_password(LONG_TWINS, b)


def test_a_long_password_registered_today_still_authenticates_tomorrow(tmp_path):
    """Acceptance: the stored form is self-describing, so it survives a restart.

    A >72-byte password registered now is written as the scheme marker plus a
    bcrypt string. Reopening the database from scratch -- a fresh process, a
    fresh store, tomorrow -- reads that marker back and verifies against the
    pre-image, so the account still logs in. Nothing about the upgrade is
    in-memory state a restart would lose.
    """
    db = str(tmp_path / "auth.db")
    client, s = _auth_app(tmp_path)
    try:
        r = client.post("/api/auth/signup", json={"email": "long@x.co", "password": LONG_PW})
        assert r.status_code == 200
        stored = asyncio.run(s.get_user_by_email("long@x.co")).password_hash
        assert stored.startswith(auth._PASSWORD_SCHEME)
        # the whole password is the credential now, not its first 72 bytes
        assert not auth.verify_password(LONG_PREFIX, stored)
    finally:
        auth.store = None
        asyncio.run(s.close())

    # "Tomorrow": a brand new store over the same file, as a restarted worker
    # would open it.
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
    """Acceptance: "every credential stored before the change still
    authenticates". A row written by the pre-#387 code logs in unchanged."""
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
    """Acceptance: legacy hashes are migrated by the login itself.

    No bulk rewrite, no operator action, no reset mail: the plaintext exists in
    exactly one request, so that request is the only place the old hash can be
    re-expressed. Afterwards the row is in the current scheme and the same
    password still opens it.
    """
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
        # the same password, unchanged, still authenticates after the rewrite
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
    """The point of the whole change, end to end on one account.

    A long passphrase stored before #387 is a credential its own 72-byte
    prefix opens. The login that migrates it is the moment that stops being
    true: the same row, the same id, the same password the owner types -- and
    the prefix is refused from then on. The account was never locked out; it
    stopped being open to a shorter credential.
    """
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("long@x.co", LONG_PW, "Long", "user"))
        asyncio.run(s.set_password(user.id, _legacy_hash(LONG_PW)))
        # before: the prefix really does open it
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PREFIX}
        ).status_code == 200
        assert asyncio.run(s.count_legacy_passwords()) == 0, "the prefix login already migrated it"

        # reset to the pre-migration state, then migrate with the real password
        asyncio.run(s.set_password(user.id, _legacy_hash(LONG_PW)))
        assert client.post(
            "/api/auth/login", json={"email": "long@x.co", "password": LONG_PW}
        ).status_code == 200
        migrated = asyncio.run(s.get_user_by_email("long@x.co")).password_hash
        assert migrated.startswith(auth._PASSWORD_SCHEME)
        assert auth.verify_password(LONG_PW, migrated)
        # same account, same row, same password -- and the prefix is now refused
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
    """The fallback must not survive the upgrade.

    The tempting shortcut is "verify the pre-image, and if that fails try the
    legacy form too". That would keep the collision alive for exactly the
    accounts that were fixed, so this pins the direction: after the upgrade the
    value that used to open the account is refused.
    """
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
    """Only a verified password may rewrite the row.

    The upgrade is a credential rewrite, so it must not be reachable by anyone
    who cannot already authenticate -- otherwise the write is a free way to
    churn rows at no cost, since a guess that fails the verify must not be
    allowed to reach the write.
    """
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
    """The upgrade's transactional half, at the statement that does the write.

    It must land only while the row still holds the hash that was just checked.
    Anything else is a lost update: a credential the owner has since replaced
    would be overwritten with one derived from a secret they no longer use.
    """
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

    # the write still lands when the observation really is the live value
    fresh = auth.hash_password(ROTATED_PW + "x")
    assert asyncio.run(store.upgrade_password_hash(user.id, newer, fresh)) == 1
    assert asyncio.run(store.get_user(user.id)).password_hash == fresh


def test_a_concurrent_password_change_is_not_clobbered_by_a_login_upgrade(tmp_path, monkeypatch):
    """The same race, end to end through ``login``.

    Another writer commits a new credential in the window between this request
    reading the row and its upgrade landing. The login has by then verified the
    OLD password against the OLD hash, so its ``new_hash`` is derived from a
    secret the owner has already moved on from -- writing it would resurrect a
    password the owner believes is gone. The compare-and-swap has to lose.
    """
    client, s = _auth_app(tmp_path)
    try:
        user = asyncio.run(s.create_user("old@x.co", SHORT_PW, "Old", "user"))
        asyncio.run(s.set_password(user.id, _legacy_hash(SHORT_PW)))

        real_get = s.get_user_by_email

        async def racing_get(email):
            row = await real_get(email)
            if row is not None and email == "old@x.co":
                # another worker finishes a password change after we read the row
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
    """The migration is housekeeping and must never become an outage.

    If the rewrite cannot be written, the row keeps a pre-migration hash that
    still authenticates and the user still gets their session. The upgrade is
    retried on the next login.
    """
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
    """Neither scheme is ever tried twice, and the caller cannot ask for both.

    Exactly one ``checkpw`` per verify, whatever the row holds. Two would mean
    the legacy pre-image is still being offered as an alternative to a migrated
    row -- the collision, for exactly the accounts that were fixed -- and two
    would also double the cost of a login against a not-yet-migrated account,
    making those rows distinguishable by timing from migrated ones.
    """
    hashed = auth.hash_password(SHORT_PW) if scheme == "current" else _legacy_hash(SHORT_PW)
    calls = []
    real = auth.bcrypt.checkpw

    def recording(password, hashed_password):
        calls.append(password)
        return real(password, hashed_password)

    monkeypatch.setattr(auth.bcrypt, "checkpw", recording)
    assert auth.verify_password(SHORT_PW, hashed)
    assert len(calls) == 1, f"{scheme}: verify tried {len(calls)} candidates"
    # whatever the row declares is the only credential that is ever built
    if scheme == "current":
        assert calls[0] == auth._prehash(SHORT_PW)
    else:
        assert calls[0] == SHORT_PW.encode()[: auth._LEGACY_BCRYPT_MAX_BYTES]


def test_the_scheme_is_decided_by_the_stored_hash_not_by_the_caller():
    """No input can steer a row onto the other scheme.

    The dispatch reads the stored marker and nothing else, so there is no
    request field, header or value shape that makes a migrated row check itself
    against a raw 72-byte prefix, or a pre-migration row check itself against a
    pre-image it was never created from.
    """
    src = inspect.getsource(auth.verify_password)
    assert "hashed.startswith(_PASSWORD_SCHEME)" in src
    # one comparison against the supplied password: a second checkpw would be a
    # second candidate scheme, which is the property above
    assert src.count("bcrypt.checkpw(") == 1


def test_needs_rehash_reads_the_stored_marker_only():
    assert auth.needs_rehash(auth.hash_password(SHORT_PW)) is False
    assert auth.needs_rehash(_legacy_hash(SHORT_PW)) is True
    # a row that is not a hash at all is reported as needing one; it can never
    # verify, so the upgrade is never reached for it
    assert auth.needs_rehash("") is True
    assert auth.verify_password(SHORT_PW, "") is False


def test_a_corrupt_stored_hash_is_refused_rather_than_raised():
    """The two-scheme dispatch must not turn a bad row into a 500."""
    for bad in ("", "not-a-hash", auth._PASSWORD_SCHEME, auth._PASSWORD_SCHEME + "nonsense"):
        assert auth.verify_password(SHORT_PW, bad) is False


# --- the operator's view ---------------------------------------------------


def test_startup_reports_the_accounts_still_on_a_pre_migration_hash(store, monkeypatch, caplog):
    """Accounts that never log in again cannot be migrated for the operator, so
    the operator has to be able to see that they still exist.

    A hash that means "the first 72 bytes" can only become a pre-image hash by
    someone supplying the plaintext. This service has no password-reset path to
    revoke one with instead, so forcing the issue would not migrate anything --
    it would lock out every dormant account permanently. Reporting the
    remainder is what the operator actually has to act on.
    """
    monkeypatch.setattr(auth, "store", store)
    # deliberately uneven: more rows on the old scheme than on the current one,
    # so counting the wrong side of the marker cannot produce the same number
    for email in ("old1@x.co", "old2@x.co", "old3@x.co"):
        u = asyncio.run(store.create_user(email, SHORT_PW, "Old", "user"))
        asyncio.run(store.set_password(u.id, _legacy_hash(SHORT_PW)))
    asyncio.run(store.create_user("new@x.co", SHORT_PW, "New", "user"))

    with caplog.at_level(logging.INFO, logger="auth"):
        remaining = asyncio.run(auth.report_legacy_password_hashes())
    assert remaining == 3
    line = "\n".join(x.getMessage() for x in caplog.records)
    assert "3 account(s) still hold a pre-migration password hash" in line
    # the report says how the remainder gets finished, and that it is not forced
    assert "next successful login" in line
    assert "never logs in again" in line
    # it counts rows; it never reveals which account, let alone any credential
    assert "old1@x.co" not in line and SHORT_PW not in line


def test_the_report_says_so_when_there_is_nothing_left_to_migrate(store, monkeypatch, caplog):
    """The other half of the operator's view: a deploy with nothing pending says
    so, rather than staying silent and leaving the line unreadable as "the
    migration must not have started"."""
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
    """Acceptance: a fresh deploy is administrable throughout the migration.

    Two deploys, both administrable, on either side of the change. A new one
    seeds its admin in the current scheme. One that has been running since
    before it keeps its pre-migration admin credential -- the only account
    there is -- and logs in with what the operator always used, and that login
    is what migrates it.
    """
    db = str(tmp_path / "auth.db")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_EMAIL", "admin@x.co")
    monkeypatch.setattr(auth.config, "AUTH_ADMIN_PASSWORD", LONG_PW)

    # (a) fresh deploy
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

    # (b) the same deploy as it looked before the change: the admin row carries
    # a pre-migration hash, and the operator logs in with what they always did.
    _downgrade_to_legacy(db, "admin@x.co", LONG_PW)

    client, s2 = _auth_app(tmp_path)
    try:
        r = client.post("/api/auth/login", json={"email": "admin@x.co", "password": LONG_PW})
        assert r.status_code == 200, r.text
        migrated = asyncio.run(s2.get_user_by_email("admin@x.co")).password_hash
        assert migrated.startswith(auth._PASSWORD_SCHEME)
        assert auth.verify_password(LONG_PW, migrated)
        # and the prefix that used to open it does not any more
        assert not auth.verify_password(LONG_PREFIX, migrated)
    finally:
        auth.store = None
        asyncio.run(s2.close())



def test_a_long_password_is_stored_as_the_full_credential(store):
    """The stored hash is a marker plus a bcrypt string over a 32-byte pre-image.

    Asserted rather than assumed: if the pre-image ever went back to being the
    password itself, the 72-byte cut would return silently and every other
    behavioural test in this file would still pass on short passwords.
    """
    stored = auth.hash_password(LONG_PW)
    assert re.match(r"^\$bcrypt-sha256\$\$2[aby]\$", stored)
    assert len(auth._prehash(LONG_PW)) == 32
    assert len(auth._prehash(LONG_PW)) < auth._LEGACY_BCRYPT_MAX_BYTES
    user = asyncio.run(store.create_user("a@x.co", LONG_PW, "A", "user"))
    assert user.password_hash.startswith(auth._PASSWORD_SCHEME)
    # the password itself is nowhere in what is stored
    assert LONG_PW not in user.password_hash
