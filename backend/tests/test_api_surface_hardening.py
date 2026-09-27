"""Tests for the API attack surface being closed (#291).

Two things are asserted here:

* the interactive docs and the generated schema are not served at all, because
  they publish the complete route list and request/response models (including
  the mass-assignable ``UserPatchIn``) to any unauthenticated caller;
* the ``Host`` header is validated, and — just as important — the allow-list the
  app actually ships with still accepts every host this project legitimately uses.
  A ``TrustedHostMiddleware`` with a wrong allow-list answers 400 to *every*
  request, which is a worse outage than the one it prevents, so the defaults are
  pinned here explicitly.
"""

import os
import socket
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app import config as config_module
from app import main as main_module
from app.config import _default_route_addresses, _machine_hosts, _parse_allowed_hosts, config

# raise_server_exceptions=False so a 400/404 from middleware surfaces as a
# response instead of propagating.
_client = TestClient(main_module.app, raise_server_exceptions=False)

# Every host a real client of this project may legitimately present: the
# Starlette TestClient default, the loopback dev stack, and whatever the
# deployment derives from CORS_ORIGINS.
_LOCAL_HOSTS = ("testserver", "localhost", "127.0.0.1", "localhost:3000", "localhost:8001")


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_and_schema_are_not_served(path):
    """No /docs, /redoc or /openapi.json — they are reconnaissance, not docs."""
    r = _client.get(path)
    assert r.status_code == 404, f"{path} is still served ({r.status_code})"
    assert "openapi" not in r.text.lower()


def test_schema_still_generable_in_process():
    """The schema is only hidden from HTTP; `app.openapi()` still works locally.

    This is what docs/API.md tells developers to run to regenerate the schema,
    so the doc stays true now that the route is gone.
    """
    schema = main_module.app.openapi()
    assert len(schema["paths"]) > 10
    assert "UserPatchIn" in schema["components"]["schemas"]


@pytest.mark.parametrize(
    "host",
    ["evil.example.com", "attacker.test", "localhost.evil.com", "127.0.0.1.evil.com"],
)
def test_unknown_host_is_rejected(host):
    r = _client.get("/live", headers={"Host": host})
    assert r.status_code == 400
    assert r.text == "Invalid host header"


@pytest.mark.parametrize("host", _LOCAL_HOSTS)
def test_dev_and_test_hosts_still_work(host):
    r = _client.get("/live", headers={"Host": host})
    assert r.status_code == 200, f"Host {host} was rejected by the allow-list"


@pytest.mark.parametrize("host", config.ALLOWED_HOSTS)
def test_every_shipped_allowed_host_is_accepted(host):
    """Each entry of the default allow-list must actually work.

    Parametrised over the parsed config rather than a hardcoded copy, so the
    test follows the default instead of drifting from it.
    """
    r = _client.get("/live", headers={"Host": f"{host}:8001"})
    assert r.status_code == 200, f"allow-list entry {host!r} does not match its own Host header"


def test_default_allow_list_covers_dev_and_test_hosts():
    bare_local = {h.split(":")[0] for h in _LOCAL_HOSTS}
    assert bare_local <= {h.split(":")[0] for h in config.ALLOWED_HOSTS}
    assert config.ALLOWED_HOSTS  # never empty: an empty list 400s every request


def test_default_allow_list_derives_hosts_from_cors_origins():
    """A deployment whose public origin is in CORS_ORIGINS is covered for free."""
    assert _parse_allowed_hosts(None, ("https://api.example.com", "http://localhost:3000")) == (
        "localhost",
        "127.0.0.1",
        "testserver",
        "api.example.com",
    )


def test_default_allow_list_covers_this_boxes_own_identity():
    """The box's own name and IPs must be in the default allow-list.

    Production is same-origin through nginx behind a `server_name _` catch-all
    vhost, which forwards whatever Host the client used, and the deployed
    CORS_ORIGINS is left at its localhost default. So for a site reached by IP
    or by the box's own name, the CORS-derived list covers nothing and every
    public request would 400.
    """
    assert set(_machine_hosts()) <= set(config.ALLOWED_HOSTS)


@pytest.mark.parametrize("host", _machine_hosts())
def test_box_identity_hosts_are_accepted(host):
    r = _client.get("/live", headers={"Host": host})
    assert r.status_code == 200, f"Host {host!r} is the box's own identity but was rejected"


def test_default_allow_list_covers_the_default_route_address(monkeypatch):
    """A NAT'd box is reached at its public address, not the bound private one.

    nginx forwards the client's `Host` through (`server_name _;` plus
    `proxy_set_header Host $host`), so on a cloud host the `Host` a real browser
    sends is the public address — which `getaddrinfo(gethostname())` does not
    report. Without it in the default allow-list the whole site answers 400.
    """
    monkeypatch.setattr(config_module, "_default_route_addresses", lambda: ("203.0.113.7",))
    assert "203.0.113.7" in _parse_allowed_hosts(None, _machine_hosts())


def test_default_route_address_is_read_from_the_routing_table(monkeypatch):
    """The probe is a real socket, not a guess: it reports what the kernel picks."""
    seen: list[tuple] = []

    class _Probe:
        def __init__(self, family, kind):
            seen.append((family, kind))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def connect(self, peer):
            seen.append(peer)

        def getsockname(self):
            return ("203.0.113.7", 53)

    monkeypatch.setattr(config_module.socket, "socket", _Probe)
    assert _default_route_addresses() == ("203.0.113.7", "203.0.113.7")
    assert (socket.AF_INET, socket.SOCK_DGRAM) in seen
    assert ("8.8.8.8", 53) in seen


def test_default_route_probe_failure_degrades_instead_of_raising(monkeypatch):
    """No default route (or a sandboxed import) must not stop the app importing."""

    def boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(config_module.socket, "socket", boom)
    assert _default_route_addresses() == ()
    assert _parse_allowed_hosts(None, _machine_hosts())[:3] == (
        "localhost",
        "127.0.0.1",
        "testserver",
    )


def test_default_route_probe_never_raises_at_import(monkeypatch):
    """No probe failure may escape: this runs while the module is imported.

    The route probe is a best-effort nicety, so *any* failure has to cost one
    allowed host and nothing more. Two paths are pinned here because neither is
    an OSError, so a narrow ``except OSError`` misses both and the API refuses
    to boot over a cosmetic detail:

    * a build without IPv6 has no ``socket.AF_INET6`` at all, and reading the
      attribute outside the guard raises AttributeError;
    * an unusable address family raises TypeError from ``socket.socket()``.
    """

    class NoIPv6:
        """A socket module without AF_INET6, as on an IPv6-less build.

        Delegates every other attribute to the real socket module, so only the
        missing constant is simulated and the name probes still work.
        """

        AF_INET = socket.AF_INET
        SOCK_DGRAM = socket.SOCK_DGRAM

        def __getattr__(self, name):
            if name == "AF_INET6":
                # The one simulated absence. Must raise rather than fall
                # through, or the fake would hand back the real constant and
                # the test would pass without exercising anything.
                raise AttributeError(name)
            return getattr(socket, name)

        @staticmethod
        def socket(family, socktype):
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    monkeypatch.setattr(config_module, "socket", NoIPv6())
    assert _default_route_addresses()  # the IPv4 probe still worked
    assert _parse_allowed_hosts(None, _machine_hosts())[:3] == (
        "localhost",
        "127.0.0.1",
        "testserver",
    )


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: TypeError("AF_INET unavailable in this build"),
        lambda: AttributeError("socket module has no attribute 'AF_INET6'"),
        lambda: RuntimeError("sandboxed socket module"),
    ],
    ids=["typeerror", "attributeerror", "runtimeerror"],
)
def test_route_probe_survives_non_oserror_failures(monkeypatch, make_exc):
    """Only the "never raise" contract matters, so the guard is broad."""

    def boom(*args, **kwargs):
        raise make_exc()

    monkeypatch.setattr(config_module.socket, "socket", boom)
    assert _default_route_addresses() == ()
    assert _parse_allowed_hosts(None, _machine_hosts())[:3] == (
        "localhost",
        "127.0.0.1",
        "testserver",
    )


def test_machine_hosts_degrade_instead_of_raising(monkeypatch):
    """This runs at import: a name-resolution failure must not take the app down."""

    def boom(*args, **kwargs):
        raise OSError("name resolution unavailable")

    monkeypatch.setattr(config_module.socket, "getaddrinfo", boom)
    hosts = _machine_hosts()
    assert socket.gethostname() in hosts
    assert "" not in hosts


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: OSError("probe failed"),
        lambda: UnicodeDecodeError("utf-8", b"\xff", 0, 1, "undecodable hostname"),
        lambda: ValueError("probe failed"),
    ],
    ids=["oserror", "unicode", "valueerror"],
)
def test_machine_hosts_survive_any_probe_failure(monkeypatch, make_exc):
    """A hostname probe that fails in any way must not stop the app importing.

    A non-decodable hostname raises UnicodeDecodeError, not OSError; missing it
    would mean the API refuses to boot over a cosmetic detail.
    """

    def boom(*args, **kwargs):
        raise make_exc()

    monkeypatch.setattr(config_module.socket, "getaddrinfo", boom)
    monkeypatch.setattr(config_module.socket, "gethostname", boom)
    monkeypatch.setattr(config_module.socket, "getfqdn", boom)
    # Still produces a closed, usable allow-list rather than raising.
    assert _parse_allowed_hosts(None, _machine_hosts())[:3] == (
        "localhost",
        "127.0.0.1",
        "testserver",
    )


def test_allow_list_entries_are_normalised():
    """Ports, case and duplicates are collapsed: the Host header is compared bare."""
    assert _parse_allowed_hosts("Example.COM:443, example.com , api.example.com") == (
        "example.com",
        "api.example.com",
    )


@pytest.mark.parametrize("raw", ["*", " * ", "example.com,*"])
def test_wildcard_cannot_disable_the_check(raw):
    """`*` would silently turn the middleware off — the knob must refuse it."""
    with pytest.raises(ValueError, match="disable the Host header check"):
        _parse_allowed_hosts(raw)


@pytest.mark.parametrize("raw", [",", ",,", "::1"])
def test_unusable_allow_list_is_rejected_loudly(raw):
    """Better a startup error than a list that matches nothing and 400s the site."""
    with pytest.raises(ValueError):
        _parse_allowed_hosts(raw)


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_blank_allow_list_falls_back_to_the_defaults_not_to_open(raw):
    """Unset/blank must fall back to the closed default, never to an empty list."""
    hosts = _parse_allowed_hosts(raw, ("http://localhost:3000",))
    assert "testserver" in hosts
    assert "*" not in hosts


def test_wildcard_env_fails_at_config_load():
    """End to end: ALLOWED_HOSTS=* must not import into a disabled check.

    Run in a subprocess: the value is read at import time, and reloading
    app.config in-process would rebind the `config` object that app.health,
    app.auth and others hold a reference to, corrupting unrelated tests.
    """
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", "import app.config"],
        check=False,
        cwd=backend_dir,
        env={**os.environ, "ALLOWED_HOSTS": "*"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "ALLOWED_HOSTS" in proc.stderr


def test_allow_list_is_logged_without_logging_config():
    """The effective allow-list must actually reach stderr in a worker.

    This is the only clue an operator gets when ALLOWED_HOSTS is wrong and every
    request answers 400, so it has to survive the process manager's logging
    setup: gunicorn and uvicorn configure only their own loggers, leaving the
    root logger without a handler, where anything below WARNING is dropped.
    A bare subprocess import is exactly that environment.
    """
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        check=False,
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TrustedHost allowed hosts: " in proc.stderr
    for host in config.ALLOWED_HOSTS:
        assert host in proc.stderr
