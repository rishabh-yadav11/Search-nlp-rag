

from __future__ import annotations

import itertools
import os

_PASSWORD = "Password1"
_EMAIL_COUNTER = itertools.count()


def _unique_email() -> str:
    return f"smoke-{next(_EMAIL_COUNTER)}-{os.getpid()}@example.test"


def _do_signup_login(app_client, email: str, password: str = _PASSWORD):
    r = app_client.post("/api/auth/signup", json={"email": email, "password": password, "name": "Smoke"})
    assert r.status_code == 200, r.text
    r = app_client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


def test_signup_login_me_round_trip(app_client) -> None:
    app_client.cookies.clear()
    email = _unique_email()
    auth = _do_signup_login(app_client, email)
    assert auth["user"]["email"] == email
    assert auth["user"]["role"] == "user"
    me = app_client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_wrong_password_rejected(app_client) -> None:
    app_client.cookies.clear()
    email = _unique_email()
    app_client.post("/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "W"})
    r = app_client.post("/api/auth/login", json={"email": email, "password": "WrongPass9"})
    assert r.status_code == 401


def test_unauthenticated_me_401(app_client) -> None:
    app_client.cookies.clear()
    r = app_client.get("/api/auth/me")
    assert r.status_code == 401


def test_logout_revokes_session(app_client) -> None:
    app_client.cookies.clear()
    _do_signup_login(app_client, _unique_email())
    assert app_client.get("/api/auth/me").status_code == 200
    assert app_client.post("/api/auth/logout").status_code == 200
    assert app_client.get("/api/auth/me").status_code == 401


def test_change_password_reissues_session(app_client) -> None:
    app_client.cookies.clear()
    email = _unique_email()
    _do_signup_login(app_client, email)
    r = app_client.post(
        "/api/auth/change-password",
        json={"current_password": _PASSWORD, "new_password": "BrandNew9"},
    )
    assert r.status_code == 200, r.text
    app_client.cookies.clear()
    r = app_client.post("/api/auth/login", json={"email": email, "password": "BrandNew9"})
    assert r.status_code == 200
    assert app_client.get("/api/auth/me").status_code == 200


def test_user_cannot_list_users(app_client) -> None:
    app_client.cookies.clear()
    _do_signup_login(app_client, _unique_email())
    r = app_client.get("/api/auth/users")
    assert r.status_code == 403


def test_admin_user_crud_and_gating(app_client, admin_headers) -> None:
    app_client.cookies.clear()
    email = _unique_email()
    app_client.post("/api/auth/signup", json={"email": email, "password": _PASSWORD, "name": "CRUD"})
    app_client.post("/api/auth/login", json={"email": email, "password": _PASSWORD})
    uid = app_client.get("/api/auth/me").json()["id"]


    assert app_client.get("/api/auth/users", headers=admin_headers).status_code == 200
    assert app_client.get(f"/api/auth/users/{uid}", headers=admin_headers).status_code == 200
    patched = app_client.patch(f"/api/auth/users/{uid}", json={"name": "Renamed"}, headers=admin_headers)
    assert patched.status_code == 200
    assert patched.json()["name"] == "Renamed"


    assert app_client.post(f"/api/auth/users/{uid}/tokens/revoke", headers=admin_headers).status_code == 200
    assert app_client.get("/api/auth/me").status_code == 401


    assert app_client.delete(f"/api/auth/users/{uid}", headers=admin_headers).status_code == 200
    assert app_client.get(f"/api/auth/users/{uid}", headers=admin_headers).status_code == 404


    assert app_client.get("/api/auth/users").status_code == 401


def test_service_token_mint_and_revoke(app_client, admin_headers) -> None:
    app_client.cookies.clear()
    mint = app_client.post("/api/auth/service-tokens", headers=admin_headers)
    assert mint.status_code == 200
    raw = mint.json()["token"]
    assert raw
    scope = mint.json().get("scope") or []
    assert "users:read" in scope

    fresh = {"X-Service-Token": raw}
    assert app_client.get("/api/auth/users", headers=fresh).status_code == 200

    rev = app_client.post("/api/auth/service-tokens/revoke", json={"token": raw}, headers=fresh)
    assert rev.status_code == 200
    assert rev.json()["revoked"] == 1

    assert app_client.get("/api/auth/users", headers=fresh).status_code == 401
