"""Session-cookie contract and CSRF guard.

The auth credential moved from ``Authorization: Bearer <token>`` to an HttpOnly
cookie so that no script on the page can read it. A cookie is attached by the
browser whether or not the page means to send it, so every state-changing
request became forgeable; ``enforce_same_origin`` is the compensating control.
Both halves are pinned here through the real app.
"""

import asyncio

import pytest
from conftest import auth_cookie, session_cookie_value
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.auth import AuthStore
from app.config import config

PASSWORD = "secret12"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The real auth router over a throwaway store, plus a minimal authenticated
    unsafe route so the CSRF cases post to a real cookie-guarded endpoint."""
    # The Redis-backed rate limiter is not what these tests are about, and with
    # Redis unreachable every login pays a connection timeout (and the limiter
    # fails open, so it would not gate anything anyway).
    monkeypatch.setattr(auth.config, "AUTH_LOGIN_RATE_PER_MIN", 0)
    monkeypatch.setattr(auth.config, "AUTH_SIGNUP_RATE_PER_MIN", 0)
    store = AuthStore(str(tmp_path / "auth.db"))
    asyncio.run(store.connect())
    auth.store = store
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(_echo_router())
    try:
        yield TestClient(app)
    finally:
        auth.store = None
        asyncio.run(store.close())


def _echo_router():
    """A minimal authenticated unsafe route, standing in for /api/chat/sessions:
    it exists so the guard is exercised on a real cookie-authenticated POST."""
    from fastapi import APIRouter, Depends, Request

    router = APIRouter(prefix="/api/chat", tags=["chat"])

    @router.post("/sessions")
    async def create_session(request: Request, _: None = Depends(auth.require_auth)):
        return {"user_id": request.state.user_id}

    return router


def _register(client, email):
    r = client.post("/api/auth/signup", json={"email": email, "password": PASSWORD, "name": "A"})
    assert r.status_code == 200, r.text
    return client.post("/api/auth/login", json={"email": email, "password": PASSWORD})


def _logged_in(client, email="a@example.com"):
    """Register, log in, and return (cookie dict, raw token, login response)."""
    r = _register(client, email)
    assert r.status_code == 200, r.text
    return auth_cookie(session_cookie_value(r)), session_cookie_value(r), r


def _set_cookie_attrs(response) -> dict:
    """The auth cookie's Set-Cookie line, split into lowercased attributes."""
    raw = next(h for h in response.headers.get_list("set-cookie") if h.startswith(f"{config.AUTH_COOKIE_NAME}="))
    parts = [p.strip() for p in raw.split(";")]
    attrs = {}
    for part in parts[1:]:
        key, _, value = part.partition("=")
        attrs[key.strip().lower()] = value.strip()
    return attrs


# --- the cookie itself ---


def test_login_sets_httponly_secure_lax_root_cookie(client):
    """The credential arrives as HttpOnly + Secure + SameSite=Lax on Path=/ with
    a Max-Age, and with no Domain (host-only, so bare-IP deploys work)."""
    _, _, r = _logged_in(client)
    raw = next(h for h in r.headers.get_list("set-cookie") if h.startswith(f"{config.AUTH_COOKIE_NAME}="))
    attrs = _set_cookie_attrs(r)
    assert "httponly" in attrs
    assert "secure" in attrs
    assert attrs["samesite"].lower() == "lax"
    assert attrs["path"] == "/"
    assert int(attrs["max-age"]) == config.AUTH_COOKIE_MAX_AGE_SECONDS
    assert "domain" not in raw.lower()


def test_login_body_carries_no_token(client):
    """The response body has no ``token`` key at all — publishing it would put
    the credential back in front of every script on the page."""
    _, _, r = _logged_in(client)
    body = r.json()
    assert "token" not in body
    assert set(body) == {"user"}
    assert body["user"]["email"] == "a@example.com"


def test_credential_is_not_reachable_from_javascript(client):
    """Nothing the response hands to script authenticates: the only cookie set
    is HttpOnly, and a request carrying no cookie is 401."""
    _, _, r = _logged_in(client)
    cookies = r.headers.get_list("set-cookie")
    assert cookies
    for header in cookies:
        if header.startswith(f"{config.AUTH_COOKIE_NAME}="):
            assert "httponly" in header.lower()
        else:
            pytest.fail(f"a script-readable cookie was set: {header}")
    assert r.json().get("token") is None
    # A browser hands script only the body, the status, and non-HttpOnly
    # cookies; none of those authenticate.
    assert client.get("/api/auth/me").status_code == 401


def test_authenticated_request_works_with_the_cookie_alone(client):
    """A valid session cookie with no Authorization header authenticates."""
    cookie, _, _ = _logged_in(client)
    me = client.get("/api/auth/me", cookies=cookie)
    assert me.status_code == 200, me.text
    assert me.json()["email"] == "a@example.com"


def test_authorization_bearer_header_is_no_longer_accepted(client):
    """A genuinely valid token in the old header is now 401: the header path is
    gone, not merely shadowed by the cookie."""
    _, token, _ = _logged_in(client)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
    # ...and the same token still works as a cookie, so the 401 above is the
    # transport being refused rather than the token being bad.
    assert client.get("/api/auth/me", cookies=auth_cookie(token)).status_code == 200


# --- CSRF guard ---


def test_cross_site_origin_is_rejected(client):
    """A cookie-authenticated POST naming a foreign Origin is 403."""
    cookie, _, _ = _logged_in(client)
    r = client.post("/api/chat/sessions", cookies=cookie, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_cross_site_fetch_metadata_is_rejected_without_origin(client):
    """Sec-Fetch-Site: cross-site is refused on its own, with no Origin present:
    the two signals are independent."""
    cookie, _, _ = _logged_in(client)
    r = client.post("/api/chat/sessions", cookies=cookie, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_sibling_subdomain_origin_is_rejected(client):
    """A registrable-domain sibling is still a different host: evil.example.com
    on Host example.com must not pass a suffix or substring check."""
    cookie, _, _ = _logged_in(client)
    r = client.post(
        "/api/chat/sessions",
        cookies=cookie,
        headers={"Origin": "https://evil.example.com", "Host": "example.com"},
    )
    assert r.status_code == 403


def test_null_origin_is_rejected(client):
    """Origin: null (sandboxed iframe, privacy browser) names no host, so it can
    never be shown to be ours and is refused."""
    cookie, _, _ = _logged_in(client)
    r = client.post("/api/chat/sessions", cookies=cookie, headers={"Origin": "null"})
    assert r.status_code == 403


def test_same_origin_request_is_allowed(client):
    """An Origin whose host equals the request host is not refused."""
    cookie, _, _ = _logged_in(client)
    r = client.post(
        "/api/chat/sessions",
        cookies=cookie,
        headers={"Origin": "https://example.com", "Host": "example.com"},
    )
    assert r.status_code == 200, r.text


def test_port_mismatch_is_allowed(client):
    """Host localhost:8001 with Origin http://localhost:3000 is the real dev
    topology (frontend :3000 -> backend :8001) and must pass: a port is not a
    security boundary for a cookie, so it is not compared."""
    cookie, _, _ = _logged_in(client)
    r = client.post(
        "/api/chat/sessions",
        cookies=cookie,
        headers={"Origin": "http://localhost:3000", "Host": "localhost:8001"},
    )
    assert r.status_code == 200, r.text


def test_absent_origin_and_fetch_metadata_is_allowed(client):
    """Neither header present is allowed, documenting the deliberate limit: every
    browser sends both on an unsafe method, so their absence means a non-browser
    client, which has no ambient cookie to ride. This is not fail-closed."""
    cookie, _, _ = _logged_in(client)
    r = client.post("/api/chat/sessions", cookies=cookie)
    assert r.status_code == 200, r.text


def test_safe_methods_are_not_guarded(client):
    """A cross-site GET with a valid cookie still works: the guard is scoped to
    unsafe methods, so a cross-site link cannot lock a user out of the app."""
    cookie, _, _ = _logged_in(client)
    r = client.get("/api/auth/me", cookies=cookie, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 200, r.text
    assert r.json()["email"] == "a@example.com"


def test_login_is_guarded_against_login_csrf(client):
    """A cross-site login POST is 403, so a hostile page cannot force a victim to
    authenticate into the attacker's account."""
    _register(client, "victim@example.com")
    r = client.post(
        "/api/auth/login",
        json={"email": "victim@example.com", "password": PASSWORD},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403


# --- session lifecycle ---


def test_logout_clears_the_cookie_and_revokes_the_token(client):
    """Logout expires the cookie in the browser AND revokes the token server-side:
    replaying the old value is 401, so it is not just a client-side wipe."""
    cookie, token, _ = _logged_in(client)
    r = client.post("/api/auth/logout", cookies=cookie)
    assert r.status_code == 200, r.text
    raw = next(h for h in r.headers.get_list("set-cookie") if h.startswith(f"{config.AUTH_COOKIE_NAME}="))
    value = raw.split(";", 1)[0].partition("=")[2]
    expired = "max-age=0" in raw.lower() or value == ""
    assert expired, f"logout did not expire the cookie: {raw}"
    # The name and path must match the setter or the browser keeps the original.
    assert config.AUTH_COOKIE_NAME in raw
    assert "path=/" in raw.lower()
    # Server-side revocation, independent of whatever the browser does.
    assert client.get("/api/auth/me", cookies=auth_cookie(token)).status_code == 401


def test_change_password_rotates_the_cookie(client):
    """Change-password re-issues the cookie: the new token works and the old one
    is refused, so the session neither dies nor survives the rotation."""
    cookie, old_token, _ = _logged_in(client)
    r = client.post(
        "/api/auth/change-password",
        cookies=cookie,
        json={"current_password": PASSWORD, "new_password": "secret34"},
    )
    assert r.status_code == 200, r.text
    new_token = session_cookie_value(r)
    assert new_token != old_token
    assert "token" not in r.json()
    new_cookie = auth_cookie(new_token)
    assert client.get("/api/auth/me", cookies=new_cookie).status_code == 200
    assert client.get("/api/auth/me", cookies=auth_cookie(old_token)).status_code == 401
