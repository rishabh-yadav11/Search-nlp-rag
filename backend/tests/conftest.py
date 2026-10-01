import os
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

for _path in (BACKEND_DIR, SCRIPTS_DIR, TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture
def fake_cache():
    """Factory returning the ``FakeCache`` class; each call builds a fresh store.

    A test passes ``get_result=``/``get_error=`` to make ``get()`` return or
    raise something specific.
    """
    from _support import FakeCache

    return FakeCache


@pytest.fixture(autouse=True)
def _reset_auth_rate_limits():
    """Clear the auth rate limiter's in-process counters around every test.

    No Redis runs in the test environment, so the limiter's bounded
    in-process fallback is the live limiter here, and its state lives on the
    module -- without this reset one test's logins 429 another's.
    """
    from app import auth

    auth.reset_local_rate_limits()
    yield
    auth.reset_local_rate_limits()


@pytest.fixture
def parse_config(monkeypatch):
    """Return a loader for a private copy of ``app/config.py`` parsed under a
    controlled environment.

    ``app.config`` reads its knobs with ``os.getenv`` at import time, so the
    only way to ask "what does the app parse when the environment says X?" is
    to execute config.py again. ``load_dotenv()`` runs inside it and would
    re-populate ``os.environ`` from a developer's ``backend/.env``, and the
    ambient environment is replaced outright by the caller's mapping.

    The load is private: no other module's ``config`` object is touched.
    """

    def _parse(**env):
        import importlib.util
        from pathlib import Path

        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        monkeypatch.setattr(os, "environ", dict(env))
        spec = importlib.util.spec_from_file_location(
            "config_probe", Path(BACKEND_DIR) / "app" / "config.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config

    return _parse

def auth_cookie(token: str) -> dict:
    """The cookie a browser attaches for this session token.

    The credential is HttpOnly, so tests authenticate by cookie, never by an
    ``Authorization`` header.
    """
    from app.config import config

    return {config.AUTH_COOKIE_NAME: token}


def login_cookie(client, email: str, password: str = "secret12") -> dict:
    """Log in through the real endpoint and return the resulting cookie dict."""
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return auth_cookie(session_cookie_value(r))


def session_cookie_value(response) -> str:
    """The token carried by a response's ``Set-Cookie`` for the auth cookie."""
    from app.config import config

    for header in response.headers.get_list("set-cookie"):
        name, _, rest = header.partition("=")
        if name.strip() == config.AUTH_COOKIE_NAME:
            return rest.split(";")[0]
    raise AssertionError(
        f"login set no {config.AUTH_COOKIE_NAME} cookie: {response.headers.get_list('set-cookie')}"
    )
