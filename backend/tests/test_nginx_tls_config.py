"""Structural tests for the nginx config setup.sh generates.

`render_nginx_config on|off` emits; the env-resolving `nginx_site_config`
decides the mode and delegates. Asserting against the real functions keeps
this file from drifting from the config nginx is handed. Nothing here runs
`run_nginx`/`run_tls` or the script with a stage argument -- those need sudo,
certbot and the deploy host, so the suite stays runnable offline.
"""

import os
import re
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import pytest

SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"

PUBLIC_PORT = 8080
API_PORT = 18001
NEXT_PORT = 13000

OPENSSL = shutil.which("openssl")
requires_openssl = pytest.mark.skipif(
    OPENSSL is None,
    reason="openssl is what mints, and what validates, the certificate pairs below",
)

_CA_CONF = """\
[ ca ]
default_ca = CA_default
[ CA_default ]
dir = ./ca
database = $dir/index.txt
new_certs_dir = $dir/newcerts
serial = $dir/serial
default_md = sha256
policy = pol
email_in_dn = no
unique_subject = no
[ pol ]
commonName = supplied
"""


def _openssl(cwd, *args):
    proc = subprocess.run(["openssl", *args], cwd=cwd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"openssl {' '.join(args)} failed:\n{proc.stderr}"
    return proc.stdout


@cache
def _valid_pair(domain):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        _openssl(
            scratch, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(scratch / "key.pem"), "-out", str(scratch / "cert.pem"),
            "-days", "30", "-subj", f"/CN={domain}",
        )
        return (scratch / "cert.pem").read_text(), (scratch / "key.pem").read_text()


@cache
def _expired_pair(domain):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        (scratch / "ca" / "newcerts").mkdir(parents=True)
        (scratch / "ca" / "index.txt").write_text("")
        (scratch / "ca" / "serial").write_text("1000\n")
        (scratch / "ca" / "openssl.cnf").write_text(_CA_CONF)
        _openssl(scratch, "genrsa", "-out", str(scratch / "ca-key.pem"), "2048")
        _openssl(
            scratch, "req", "-x509", "-key", str(scratch / "ca-key.pem"),
            "-out", str(scratch / "ca.pem"), "-days", "3650", "-subj", "/CN=test-ca",
        )
        _openssl(
            scratch, "req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(scratch / "key.pem"),
            "-out", str(scratch / "req.csr"), "-subj", f"/CN={domain}",
        )
        _openssl(
            scratch, "ca", "-batch", "-config", str(scratch / "ca" / "openssl.cnf"),
            "-cert", str(scratch / "ca.pem"), "-keyfile", str(scratch / "ca-key.pem"),
            "-in", str(scratch / "req.csr"), "-out", str(scratch / "cert.pem"),
            "-startdate", "20200101000000Z", "-enddate", "20200201000000Z", "-notext",
        )
        return (scratch / "cert.pem").read_text(), (scratch / "key.pem").read_text()


def _shipped_default(name):
    match = re.search(rf'^{name}="\$\{{{name}:-(\d+)\}}"$', SETUP_SH.read_text(), re.MULTILINE)
    assert match, f"setup.sh no longer declares {name} with a default port"
    return int(match.group(1))


def test_port_pins_differ_from_the_shipped_defaults():
    for name, pinned in (("PUBLIC_PORT", PUBLIC_PORT), ("API_PORT", API_PORT), ("NEXT_PORT", NEXT_PORT)):
        default = _shipped_default(name)
        assert pinned != default, (
            f"{name} is pinned to {pinned}, which is also the shipped default, so a template "
            f"baking in {default} would pass every test here unnoticed"
        )

DOMAIN = "search.example.com"
ACME_PATH = "/.well-known/acme-challenge/"
REDIRECT_TARGET = "https://$host$request_uri"

API_LOCATIONS = (
    "/search",
    "/health",
    "/live",
    "/ready",
    "/readyz",
    "/facets",
    "/api",
    "/api/chat/",
    "/recommend/",
    "/analytics/click",
    "/analytics/summary",
    "/analytics/chat",
)
FRONTEND_LOCATIONS = ("/analytics", "/")


def _webroot(tmp_path):
    return tmp_path / "certbot-webroot"


def _missing_cert_root(tmp_path):
    return tmp_path / "no-such-letsencrypt"


def _certified_root(tmp_path, domain, *, pair=None):
    root = tmp_path / "letsencrypt"
    live = root / "live" / domain
    live.mkdir(parents=True)
    if pair is None and OPENSSL:
        pair = _valid_pair(domain)
    cert, key = pair or ("-----BEGIN CERTIFICATE-----\n", "-----BEGIN PRIVATE KEY-----\n")
    (live / "fullchain.pem").write_text(cert)
    (live / "privkey.pem").write_text(key)
    return root


def _broken_cert_root(tmp_path, domain, how):
    if how == "expired":
        return _certified_root(tmp_path, domain, pair=_expired_pair(domain))
    root = _certified_root(tmp_path, domain)
    live = root / "live" / domain
    if how == "no_cert":
        (live / "fullchain.pem").unlink()
    elif how == "no_key":
        (live / "privkey.pem").unlink()
    elif how == "empty_key":
        (live / "privkey.pem").write_text("")
    elif how == "empty_cert":
        (live / "fullchain.pem").write_text("")
    elif how == "unparseable_cert":
        (live / "fullchain.pem").write_text("-----BEGIN CERTIFICATE-----\nnot base64\n")
    elif how == "unreadable_cert":
        (live / "fullchain.pem").chmod(0o000)
    else:
        raise AssertionError(f"unknown breakage {how!r}")
    return root


def _env(tmp_path, **over):
    env = dict(os.environ)
    env.update(
        {
            "NGINX_TLS": "off",
            "LE_DOMAIN": "",
            "LE_ROOT": str(_missing_cert_root(tmp_path)),
            "CERTBOT_WEBROOT": str(_webroot(tmp_path)),
            "PUBLIC_PORT": str(PUBLIC_PORT),
            "API_PORT": str(API_PORT),
            "NEXT_PORT": str(NEXT_PORT),
            "NGINX_CONF": str(tmp_path / "nginx" / "not-installed"),
            "NGINX_LINK": str(tmp_path / "nginx" / "not-linked"),
        }
    )
    env.update(over)
    return env


def _call(env, function):
    return subprocess.run(
        ["bash", "-c", f'source "{SETUP_SH}"\n{function}'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _rendered(tmp_path, mode, **over):
    proc = _call(_env(tmp_path, **over), f"render_nginx_config {mode}")
    assert proc.returncode == 0, f"render_nginx_config {mode} exited {proc.returncode}: {proc.stderr}"
    return proc.stdout


def _site_config(tmp_path, **over):
    proc = _call(_env(tmp_path, **over), "nginx_site_config")
    assert proc.returncode == 0, f"nginx_site_config exited {proc.returncode}: {proc.stderr}"
    return proc.stdout


def _server_blocks(config):
    blocks, current, depth = [], None, 0
    for line in config.splitlines():
        if current is None:
            if line.strip().startswith("server "):
                current, depth = [line], line.count("{") - line.count("}")
        else:
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blocks.append("\n".join(current))
                current = None
    assert current is None, "unbalanced braces: a server block never closed"
    return blocks


def _listen_ports(block):
    return {int(p) for p in re.findall(r"^\s*listen\s+(\d+)", block, re.MULTILINE)}


def _server_on(config, port):
    matches = [b for b in _server_blocks(config) if port in _listen_ports(b)]
    assert len(matches) == 1, f"expected exactly one server on port {port}:\n{config}"
    return matches[0]


def _directive(block, name):
    match = re.search(rf"^\s*{re.escape(name)}\s+(\S+);", block, re.MULTILINE)
    return match.group(1) if match else None


def _server_name(block):
    return _directive(block, "server_name")


def _location_paths(block):
    paths = set()
    for match in re.finditer(r"^\s*location\s+([^\n{]*)\{", block, re.MULTILINE):
        tokens = match.group(1).split()
        if tokens and tokens[0] in ("^~", "=", "~", "~*"):
            tokens = tokens[1:]
        if tokens:
            paths.add(tokens[-1])
    return paths


def _location_body(block, path):
    lines = block.splitlines()
    opener = re.compile(rf"^\s*location\s+(\S+\s+)?{re.escape(path)}\s*\{{")
    for at, line in enumerate(lines):
        if opener.match(line):
            depth, body = line.count("{") - line.count("}"), [line]
            if depth == 0:
                return line
            for following in lines[at + 1 :]:
                body.append(following)
                depth += following.count("{") - following.count("}")
                if depth == 0:
                    return "\n".join(body)
    raise AssertionError(f"no location {path!r} in:\n{block}")


def test_setup_script_is_syntactically_valid():
    proc = subprocess.run(["bash", "-n", str(SETUP_SH)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"bash -n {SETUP_SH} failed:\n{proc.stderr}"


def test_sourcing_setup_sh_runs_no_stage(tmp_path):
    proc = subprocess.run(
        ["bash", "-c", f'source "{SETUP_SH}"\necho SOURCED-OK'],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"sourcing setup.sh must not fail or exit: {proc.stderr}"
    assert proc.stdout == "SOURCED-OK\n", f"sourcing must print nothing of its own, got: {proc.stdout!r}"


def test_render_nginx_config_takes_the_mode_positionally_not_from_the_env(tmp_path):
    on = _rendered(tmp_path, "on", NGINX_TLS="off", LE_DOMAIN=DOMAIN)
    assert "listen 443" in on, f"render_nginx_config on must emit TLS whatever NGINX_TLS says:\n{on}"

    off = _rendered(tmp_path, "off", NGINX_TLS="on", LE_DOMAIN=DOMAIN)
    assert "listen 443" not in off, f"render_nginx_config off must emit no TLS server:\n{off}"
    assert "ssl_" not in off, f"render_nginx_config off must emit no ssl_ directive:\n{off}"


def test_render_nginx_config_does_not_probe_the_certificate_store(tmp_path):
    config = _rendered(tmp_path, "on", LE_DOMAIN=DOMAIN, LE_ROOT=str(_missing_cert_root(tmp_path)))
    cert = _directive(_server_on(config, 443), "ssl_certificate")
    assert cert, f"render_nginx_config on must emit ssl_certificate unconditionally:\n{config}"
    assert cert.startswith(str(_missing_cert_root(tmp_path))), (
        f"the certificate path must be interpolated from LE_ROOT, not hardcoded: {cert}"
    )


def test_auto_mode_with_a_certificate_matches_render_nginx_config_on(tmp_path):
    knobs = {"LE_DOMAIN": DOMAIN, "LE_ROOT": str(_certified_root(tmp_path, DOMAIN))}
    auto = _site_config(tmp_path, NGINX_TLS="auto", **knobs)
    assert auto == _rendered(tmp_path, "on", **knobs), (
        "with a domain and a readable certificate, NGINX_TLS=auto must render exactly "
        "render_nginx_config on"
    )


def test_plain_mode_emits_no_tls_at_all(tmp_path):
    config = _rendered(tmp_path, "off")
    assert "listen 443" not in config, f"TLS leaked into plain mode:\n{config}"
    assert "ssl_" not in config, f"plain mode must emit no ssl_ directive at all:\n{config}"
    assert "return 30" not in config, f"plain mode must not redirect:\n{config}"


def test_plain_mode_serves_the_whole_site_on_the_public_port(tmp_path):
    config = _rendered(tmp_path, "off")
    blocks = _server_blocks(config)
    assert len(blocks) == 1, f"plain mode is a single server, got {len(blocks)}:\n{config}"
    assert _listen_ports(blocks[0]) == {PUBLIC_PORT}, f"plain mode serves $PUBLIC_PORT:\n{config}"
    assert _server_name(blocks[0]) == "_", "with no domain there is only the catch-all name to match"
    paths = _location_paths(blocks[0])
    missing = (set(API_LOCATIONS) | set(FRONTEND_LOCATIONS)) - paths
    assert not missing, f"plain mode dropped proxied routes: {sorted(missing)}"
    assert ACME_PATH in paths, f"the ACME challenge is served over plain HTTP too:\n{config}"


def test_proxied_ports_come_from_the_env_knobs(tmp_path):
    config = _rendered(tmp_path, "off")
    assert f"127.0.0.1:{API_PORT}" in config, f"API routes must proxy to $API_PORT:\n{config}"
    assert f"127.0.0.1:{NEXT_PORT}" in config, f"frontend routes must proxy to $NEXT_PORT:\n{config}"


def _api_location_bodies(block):
    found = []
    for path in _location_paths(block):
        body = _location_body(block, path)
        if f"127.0.0.1:{API_PORT}" in body:
            found.append((path, body))
    return sorted(found)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_every_api_location_forwards_the_public_host(tmp_path, mode):
    config = _rendered(tmp_path, mode, LE_DOMAIN=DOMAIN)
    for block in _server_blocks(config):
        for path, body in _api_location_bodies(block):
            assert "proxy_set_header Host $host;" in body, (
                f"location {path} proxies to the API without forwarding the public "
                f"Host, so the CSRF guard will 403 it in production:\n{body}"
            )


def test_the_host_header_is_asserted_on_every_api_location_not_just_api(tmp_path):
    block = _server_on(_rendered(tmp_path, "off"), PUBLIC_PORT)
    api_paths = {path for path, _ in _api_location_bodies(block)}
    assert api_paths == set(API_LOCATIONS), (
        f"the API-location set changed; update API_LOCATIONS so it stays the full list. "
        f"got {sorted(api_paths)}, expected {sorted(API_LOCATIONS)}"
    )


def test_tls_mode_adds_a_tls_server(tmp_path):
    config = _rendered(tmp_path, "on", LE_DOMAIN=DOMAIN, LE_ROOT=str(_certified_root(tmp_path, DOMAIN)))
    blocks = _server_blocks(config)
    assert len(blocks) == 2, f"TLS mode needs exactly two servers, got {len(blocks)}:\n{config}"

    by_port = {}
    for block in blocks:
        ports = _listen_ports(block)
        assert len(ports) == 1, f"a server must listen on exactly one port:\n{block}"
        by_port[ports.pop()] = block
    assert set(by_port) == {PUBLIC_PORT, 443}, (
        f"expected a $PUBLIC_PORT server and a 443 server, got {sorted(by_port)}:\n{config}"
    )

    http, https = by_port[PUBLIC_PORT], by_port[443]
    assert _server_name(http) == _server_name(https) == DOMAIN, (
        f"both servers must answer for {DOMAIN} or the redirect lands on nothing: "
        f"{_server_name(http)!r} vs {_server_name(https)!r}"
    )
    assert _server_name(http) != "_", "TLS mode knows the domain, so it must not serve the catch-all name"

    cert = _directive(https, "ssl_certificate")
    key = _directive(https, "ssl_certificate_key")
    assert cert and key, f"the 443 server must point at a certificate and a key:\n{https}"
    assert str(tmp_path) in cert, f"ssl_certificate must come from LE_ROOT, not a real store: {cert}"
    assert Path(cert).is_file(), f"ssl_certificate points at {cert}, which does not exist"
    assert Path(key).is_file(), f"ssl_certificate_key points at {key}, which does not exist"
    assert "return 30" not in https, f"the TLS server must serve the site, not redirect:\n{https}"


def test_auto_mode_falls_back_to_plain_when_there_is_no_certificate(tmp_path):
    missing = str(_missing_cert_root(tmp_path))
    auto = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=missing)
    plain = _rendered(tmp_path, "off", LE_DOMAIN=DOMAIN, LE_ROOT=missing)
    assert auto == plain, "auto with no certificate at all must render the plain site, byte for byte"
    assert "listen 443" not in auto, f"a 443 server without a certificate:\n{auto}"
    assert "ssl_" not in auto, f"an ssl_ directive without a certificate:\n{auto}"


def test_acme_challenge_is_still_served_over_http_under_the_redirect(tmp_path):
    http = _server_on(_rendered(tmp_path, "on", LE_DOMAIN=DOMAIN), PUBLIC_PORT)
    assert "return 30" in http, f"this test is about the redirect being present:\n{http}"

    opener = re.search(rf"^\s*location\s+(\S+\s+)?{re.escape(ACME_PATH)}\s*\{{", http, re.MULTILINE)
    assert opener, f"no ACME challenge location in the http server:\n{http}"
    assert opener.group(1), f"the ACME location needs the ^~ modifier, got: {opener.group(0)!r}"
    assert opener.group(1).strip() == "^~", f"the ACME location must use ^~, got: {opener.group(0)!r}"

    body = _location_body(http, ACME_PATH)
    assert f"root {_webroot(tmp_path)}" in body, f"the challenge must be served from CERTBOT_WEBROOT:\n{body}"
    assert "return" not in body, f"the challenge must be served, not redirected:\n{body}"


def test_redirect_preserves_host_and_path(tmp_path):
    http = _server_on(_rendered(tmp_path, "on", LE_DOMAIN=DOMAIN), PUBLIC_PORT)
    targets = re.findall(r"return\s+30[1278]\s+(\S+);", http)
    assert targets == [REDIRECT_TARGET], (
        f"the http server must issue exactly one redirect to {REDIRECT_TARGET}, got {targets}"
    )


def test_the_two_servers_serve_the_same_routes(tmp_path):
    plain = _location_paths(_server_on(_rendered(tmp_path, "off", LE_DOMAIN=DOMAIN), PUBLIC_PORT))
    secure = _location_paths(_server_on(_rendered(tmp_path, "on", LE_DOMAIN=DOMAIN), 443))
    expected = plain - {ACME_PATH}
    assert expected, "the plain server served nothing to compare against"
    assert secure == expected, (
        "the tls server drifted from the plain one: "
        f"only-plain={sorted(expected - secure)} only-tls={sorted(secure - expected)}"
    )


def test_tls_mode_refuses_to_render_a_config_for_a_missing_certificate(tmp_path):
    root = _missing_cert_root(tmp_path)
    proc = _call(_env(tmp_path, NGINX_TLS="on", LE_DOMAIN=DOMAIN, LE_ROOT=str(root)), "nginx_site_config")
    expected = str(root / "live" / DOMAIN / "fullchain.pem")
    assert proc.returncode != 0, f"NGINX_TLS=on without a certificate must fail, stdout was:\n{proc.stdout}"
    assert proc.stdout == "", f"nothing may reach stdout when the certificate is missing: {proc.stdout!r}"
    assert expected in proc.stderr, f"the error must name the missing certificate {expected}:\n{proc.stderr}"


def test_tls_mode_refuses_to_render_a_config_for_a_missing_private_key(tmp_path):
    root = _broken_cert_root(tmp_path, DOMAIN, "no_key")
    proc = _call(_env(tmp_path, NGINX_TLS="on", LE_DOMAIN=DOMAIN, LE_ROOT=str(root)), "nginx_site_config")

    expected = str(root / "live" / DOMAIN / "privkey.pem")
    assert proc.returncode != 0, f"NGINX_TLS=on without a private key must fail, stdout was:\n{proc.stdout}"
    assert proc.stdout == "", f"nothing may reach stdout when the key is missing: {proc.stdout!r}"
    assert expected in proc.stderr, f"the error must name the missing key {expected}:\n{proc.stderr}"


@pytest.mark.parametrize("how", ["no_key", "empty_key", "empty_cert"])
def test_auto_mode_refuses_a_pair_that_is_not_there(how, tmp_path):
    root = _broken_cert_root(tmp_path, DOMAIN, how)
    config = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))

    assert 443 not in {port for block in _server_blocks(config) for port in _listen_ports(block)}, (
        f"a certificate pair that is {how} must not be rendered as a TLS server:\n{config}"
    )
    assert "return 301 https://" not in config, f"nothing to redirect to:\n{config}"


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root can read any file, so the permission difference cannot be modelled this way",
)
def test_a_key_only_its_owner_can_read_still_renders(tmp_path):
    root = _certified_root(tmp_path, DOMAIN)
    key = root / "live" / DOMAIN / "privkey.pem"
    key.chmod(0o000)
    try:
        assert key.is_file() and key.stat().st_size > 0, "the key must be present and non-empty"
        assert not os.access(key, os.R_OK), (
            f"the premise needs a key this user cannot read, but {key} is readable"
        )

        config = _site_config(tmp_path, NGINX_TLS="on", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))
    finally:
        key.chmod(0o600)

    ports = {port for block in _server_blocks(config) for port in _listen_ports(block)}
    assert 443 in ports, f"a root-only key must not stop the HTTPS config being written:\n{config}"


@requires_openssl
@pytest.mark.parametrize("how", ["unparseable_cert", "expired"])
def test_auto_mode_keeps_tls_for_a_pair_that_is_only_unusable(how, tmp_path):
    """A downgrade here silently strips a live :443 server, and nothing on ./setup.sh nginx runs certbot to put TLS back."""
    root = _broken_cert_root(tmp_path, DOMAIN, how)
    config = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))

    ports = {port for block in _server_blocks(config) for port in _listen_ports(block)}
    assert 443 in ports, (
        f"a complete pair that merely cannot be parsed must not cost the site its "
        f"HTTPS server: {how}\n{config}"
    )
    assert "return 301 https://" in config, f"the redirect must stay too:\n{config}"


@requires_openssl
def test_auto_mode_keeps_tls_for_a_pair_openssl_still_accepts(tmp_path):
    root = _certified_root(tmp_path, DOMAIN)
    config = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))

    ports = {port for block in _server_blocks(config) for port in _listen_ports(block)}
    assert 443 in ports, f"a usable certificate must keep serving over TLS:\n{config}"


def test_cert_state_is_unknown_when_openssl_is_missing(tmp_path):
    root = _certified_root(tmp_path, DOMAIN)
    env = _env(tmp_path, LE_DOMAIN=DOMAIN, LE_ROOT=str(root))
    lean_path = tmp_path / "lean-path"
    lean_path.mkdir()
    for tool in ("bash", "dirname", "env"):
        os.symlink(shutil.which(tool), lean_path / tool)
    env["PATH"] = str(lean_path)

    premise = _call(env, "command -v openssl")
    assert premise.returncode != 0, f"openssl is still reachable on the lean PATH: {premise.stdout!r}"

    proc = _call(env, "nginx_tls_cert_state")

    assert proc.returncode == 0, f"nginx_tls_cert_state failed without openssl: {proc.stderr}"
    assert proc.stdout.strip() == "unknown", (
        f"with no openssl the state must be unknown, not a verdict; got {proc.stdout.strip()!r}"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file, so 'unreadable' is unreachable")
@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("no_cert", "missing"),
        ("empty_cert", "empty"),
        ("unreadable_cert", "unreadable"),
        ("unparseable_cert", "corrupt"),
    ],
)
def test_cert_state_tells_a_lapsed_certificate_from_an_unusable_one(how, expected, tmp_path):
    root = _broken_cert_root(tmp_path, DOMAIN, how)
    cert = root / "live" / DOMAIN / "fullchain.pem"
    try:
        proc = _call(_env(tmp_path, LE_DOMAIN=DOMAIN, LE_ROOT=str(root)), "nginx_tls_cert_state")
    finally:
        if cert.exists():
            cert.chmod(0o600)

    assert proc.returncode == 0, f"nginx_tls_cert_state failed: {proc.stderr}"
    assert proc.stdout.strip() == expected, (
        f"a certificate that is {how} should read as {expected!r}, got {proc.stdout.strip()!r}"
    )


@pytest.mark.parametrize("mode", ["off", "on"])
def test_rendered_config_is_structurally_sound(tmp_path, mode):
    """Structural only: `nginx -t` needs the deploy host, and the unquoted heredoc hides a bad escape from source-level checks."""
    config = _rendered(tmp_path, mode, LE_DOMAIN=DOMAIN)

    assert config.count("{") == config.count("}"), f"unbalanced braces:\n{config}"
    for line in config.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert stripped.endswith((";", "{", "}")), f"unterminated directive: {line!r}"

    blocks = _server_blocks(config)
    seen = set()
    for block in blocks:
        ports = _listen_ports(block)
        assert len(ports) == 1, f"a server must listen on exactly one port:\n{block}"
        port = ports.pop()
        assert port not in seen, f"two server blocks both listen on {port}"
        seen.add(port)
        names = re.findall(r"^\s*server_name\s+(\S+);", block, re.MULTILINE)
        assert len(names) == 1, f"a server needs exactly one server_name, got {names}:\n{block}"


def test_an_unrelated_letsencrypt_entry_is_never_adopted_as_the_domain(tmp_path):
    """/etc/letsencrypt is shared, so "the only entry under live/" is not evidence of anything."""
    foreign = tmp_path / "letsencrypt" / "live" / "someone-elses-blog.example.org"
    foreign.mkdir(parents=True)
    (foreign / "fullchain.pem").write_text("-----BEGIN CERTIFICATE-----\nnot ours\n")

    env = _env(tmp_path, LE_ROOT=str(tmp_path / "letsencrypt"))

    proc = _call(env, "echo \"[$LE_DOMAIN][$LE_DOMAIN_RECOVERED]\"")

    assert proc.stdout.strip() == "[][0]", (
        f"a domain was adopted from an unrelated entry: {proc.stdout!r}"
    )


def test_the_domain_is_recovered_from_the_installed_config(tmp_path):
    root = _certified_root(tmp_path, DOMAIN)
    conf = tmp_path / "nginx" / "sites-available" / "site"
    conf.parent.mkdir(parents=True)
    conf.write_text(
        "server {\n"
        f"    ssl_certificate {root}/live/{DOMAIN}/fullchain.pem;\n"
        f"    ssl_certificate_key {root}/live/{DOMAIN}/privkey.pem;\n"
        "}\n"
    )
    env = _env(tmp_path, LE_ROOT=str(root), NGINX_CONF=str(conf))

    proc = _call(env, "echo \"[$LE_DOMAIN][$LE_DOMAIN_RECOVERED][$LE_CERT]\"")

    assert proc.stdout.strip() == f"[{DOMAIN}][1][{root}/live/{DOMAIN}/fullchain.pem]", (
        f"the domain in use was not recovered from the installed config: {proc.stdout!r}"
    )
