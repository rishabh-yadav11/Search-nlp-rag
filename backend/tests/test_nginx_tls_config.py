"""Structural tests for the nginx config setup.sh generates (issue #244).

setup.sh used to emit a single `server { listen 80; ... }` block, so passwords,
bearer tokens, X-Service-Token and chat bodies all crossed the network in
cleartext, and the Strict-Transport-Security header the frontend sets was
ignored -- browsers only honour HSTS delivered over https, so over plain HTTP it
was decoration. The fix renders the same proxy body from one place and adds a
443 server once a certificate exists; `./setup.sh tls` obtains that certificate
through certbot's webroot plugin.

The generator is split in two, and these tests use both:

* `render_nginx_config on|off` is the emitting half. The mode arrives as a
  positional parameter, only the path/port knobs come from the environment, and
  it never reads NGINX_TLS or probes the certificate store -- so either mode
  renders deterministically, with nothing created under /etc.
* `nginx_site_config` is the env-resolving wrapper. It decides the mode from
  NGINX_TLS / LE_DOMAIN / cert readability, refuses to emit anything when the
  mode is on and the certificate cannot be read, and delegates.

Asserting against the real function keeps this file from drifting away from the
config nginx is actually handed: deleting the `^~` on the ACME location, dropping
`$request_uri` from the redirect, or letting the two servers drift apart all turn
it red.

Nothing here runs `run_nginx`, `run_tls` or the script as a program with a stage
argument. Those need sudo, certbot and the deploy host, and this suite has to
stay runnable offline, so the validity checks below are structural only --
balanced braces, terminated directives, one listen and one server_name per
server, no two servers on the same port. They catch a malformed render; they
cannot replace nginx's own parser.
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

# Ports are pinned to values that differ from the shipped defaults, so a config
# that hardcodes them fails here instead of passing by coincidence. That only
# holds while the pins really are different, which is what
# test_port_pins_differ_from_the_shipped_defaults guards.
PUBLIC_PORT = 8080
API_PORT = 18001
NEXT_PORT = 13000

OPENSSL = shutil.which("openssl")
requires_openssl = pytest.mark.skipif(
    OPENSSL is None,
    reason="openssl is what mints, and what validates, the certificate pairs below",
)

# `openssl ca` is the only way to get a leaf with a notAfter in the past:
# `req -x509` refuses a non-positive -days. The dates are fixed, so the
# certificate it produces is expired permanently, not just today.
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
    """(cert, key) PEM text for a real, unexpired self-signed pair.

    Real rather than placeholder text because setup.sh now asks openssl whether
    the leaf is still in date: a file that merely exists says nothing about
    whether it can be served. Minted once per session, then copied into each
    test's own tmp_path.
    """
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
    """(cert, key) PEM text for a well-formed pair whose leaf has expired."""
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
    """The port setup.sh falls back to when the environment says nothing."""
    match = re.search(rf'^{name}="\$\{{{name}:-(\d+)\}}"$', SETUP_SH.read_text(), re.MULTILINE)
    assert match, f"setup.sh no longer declares {name} with a default port"
    return int(match.group(1))


def test_port_pins_differ_from_the_shipped_defaults():
    """The claim every assertion in this module rests on.

    A pin equal to the shipped default makes a template that hardcodes that
    default indistinguishable from one that interpolates the knob, so the pins
    have to stay off the defaults for this file to mean anything.
    """
    for name, pinned in (("PUBLIC_PORT", PUBLIC_PORT), ("API_PORT", API_PORT), ("NEXT_PORT", NEXT_PORT)):
        default = _shipped_default(name)
        assert pinned != default, (
            f"{name} is pinned to {pinned}, which is also the shipped default, so a template "
            f"baking in {default} would pass every test here unnoticed"
        )

# A domain with a cert present, and the ACME path that must keep answering on
# port 80 for renewals to keep working once the redirect exists.
DOMAIN = "search.example.com"
ACME_PATH = "/.well-known/acme-challenge/"
REDIRECT_TARGET = "https://$host$request_uri"

# Routes proxied to gunicorn, and routes proxied to the frontend. Dropping one
# silently 404s it in production, so both sets are asserted in full.
API_LOCATIONS = (
    "/search",
    "/health",
    "/live",
    "/ready",
    "/readyz",
    "/facets",
    "/api",
    "/recommend/",
    "/analytics/click",
    "/analytics/summary",
    "/analytics/chat",
)
FRONTEND_LOCATIONS = ("/analytics", "/")


# --------------------------------------------------------------------------
# fixtures for the knobs the generator reads
# --------------------------------------------------------------------------


def _webroot(tmp_path):
    return tmp_path / "certbot-webroot"


def _missing_cert_root(tmp_path):
    """A letsencrypt root that is never created, so 'no certificate' is certain
    regardless of what the machine running the suite happens to have."""
    return tmp_path / "no-such-letsencrypt"


def _certified_root(tmp_path, domain, *, pair=None):
    """A letsencrypt root holding a cert/key pair for `domain`, under tmp_path
    so nothing here can touch a real certificate store.

    Without openssl there is no way to mint one, and setup.sh's own rule
    degrades to the file test in that case, so the placeholder it is happy
    with is written instead. Both paths mean the same thing to the script.
    """
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
    """A letsencrypt root holding a pair nginx or certbot could not actually
    serve, which is the case "auto" has to refuse."""
    if how == "expired":
        return _certified_root(tmp_path, domain, pair=_expired_pair(domain))
    root = _certified_root(tmp_path, domain)
    live = root / "live" / domain
    if how == "no_key":
        (live / "privkey.pem").unlink()
    elif how == "empty_key":
        (live / "privkey.pem").write_text("")
    elif how == "empty_cert":
        (live / "fullchain.pem").write_text("")
    elif how == "unparseable_cert":
        (live / "fullchain.pem").write_text("-----BEGIN CERTIFICATE-----\nnot base64\n")
    else:
        raise AssertionError(f"unknown breakage {how!r}")
    return root


def _env(tmp_path, **over):
    """A complete environment for the generator.

    Every knob is set explicitly, defaults included: the mode must be decided
    by what the test asks for, never by a value leaked in from the environment
    pytest happens to be started in.
    """
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
        }
    )
    env.update(over)
    return env


def _call(env, function):
    """Source setup.sh, then call one pure generator function.

    Only the renderers are ever invoked. run_nginx, run_tls and the script
    driven with a stage argument all need sudo, certbot and the deploy host.
    """
    return subprocess.run(
        ["bash", "-c", f'source "{SETUP_SH}"\n{function}'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _rendered(tmp_path, mode, **over):
    """Render one specific mode. No NGINX_TLS, no certificate lookup."""
    proc = _call(_env(tmp_path, **over), f"render_nginx_config {mode}")
    assert proc.returncode == 0, f"render_nginx_config {mode} exited {proc.returncode}: {proc.stderr}"
    return proc.stdout


def _site_config(tmp_path, **over):
    """Render through the wrapper, which resolves the mode from the env."""
    proc = _call(_env(tmp_path, **over), "nginx_site_config")
    assert proc.returncode == 0, f"nginx_site_config exited {proc.returncode}: {proc.stderr}"
    return proc.stdout


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------


def _server_blocks(config):
    """Top-level `server { ... }` blocks, located by brace depth so a nested
    location block is never mistaken for a server."""
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
    """The single server block listening on `port`."""
    matches = [b for b in _server_blocks(config) if port in _listen_ports(b)]
    assert len(matches) == 1, f"expected exactly one server on port {port}:\n{config}"
    return matches[0]


def _directive(block, name):
    """The single value of `name` in a block, or None if it is absent."""
    match = re.search(rf"^\s*{re.escape(name)}\s+(\S+);", block, re.MULTILINE)
    return match.group(1) if match else None


def _server_name(block):
    return _directive(block, "server_name")


def _location_paths(block):
    """Every location path in a block, with match modifiers (^~, =, ~) dropped
    so the two servers compare on what they serve, not on how they match it."""
    paths = set()
    for match in re.finditer(r"^\s*location\s+([^\n{]*)\{", block, re.MULTILINE):
        tokens = match.group(1).split()
        if tokens and tokens[0] in ("^~", "=", "~", "~*"):
            tokens = tokens[1:]
        if tokens:
            paths.add(tokens[-1])
    return paths


def _location_body(block, path):
    """The full text of one location, sliced out by relative brace depth."""
    lines = block.splitlines()
    opener = re.compile(rf"^\s*location\s+(\S+\s+)?{re.escape(path)}\s*\{{")
    for at, line in enumerate(lines):
        if opener.match(line):
            depth, body = line.count("{") - line.count("}"), [line]
            for following in lines[at + 1 :]:
                body.append(following)
                depth += following.count("{") - following.count("}")
                if depth == 0:
                    return "\n".join(body)
    raise AssertionError(f"no location {path!r} in:\n{block}")


# --------------------------------------------------------------------------
# the script itself
# --------------------------------------------------------------------------


def test_setup_script_is_syntactically_valid():
    """`bash -n` is the only parse available without the deploy host."""
    proc = subprocess.run(["bash", "-n", str(SETUP_SH)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"bash -n {SETUP_SH} failed:\n{proc.stderr}"


def test_sourcing_setup_sh_runs_no_stage(tmp_path):
    """Sourcing must only define functions.

    Every renderer here is called from a subshell that sources this script, so
    if the top-level argument parsing still ran unguarded it would print usage
    and exit before any function was reachable.
    """
    proc = subprocess.run(
        ["bash", "-c", f'source "{SETUP_SH}"\necho SOURCED-OK'],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"sourcing setup.sh must not fail or exit: {proc.stderr}"
    assert proc.stdout == "SOURCED-OK\n", f"sourcing must print nothing of its own, got: {proc.stdout!r}"


# --------------------------------------------------------------------------
# the seam itself
# --------------------------------------------------------------------------


def test_render_nginx_config_takes_the_mode_positionally_not_from_the_env(tmp_path):
    """The mode is an argument, not a second source of truth.

    If the emitter went back to reading NGINX_TLS, a caller that resolved the
    mode and passed it in could be silently overridden by the environment.
    """
    on = _rendered(tmp_path, "on", NGINX_TLS="off", LE_DOMAIN=DOMAIN)
    assert "listen 443" in on, f"render_nginx_config on must emit TLS whatever NGINX_TLS says:\n{on}"

    off = _rendered(tmp_path, "off", NGINX_TLS="on", LE_DOMAIN=DOMAIN)
    assert "listen 443" not in off, f"render_nginx_config off must emit no TLS server:\n{off}"
    assert "ssl_" not in off, f"render_nginx_config off must emit no ssl_ directive:\n{off}"


def test_render_nginx_config_does_not_probe_the_certificate_store(tmp_path):
    """The emitter has no cert-existence gate; the wrapper owns it.

    Two different gates in two halves is how they drift -- so the emitter is
    required to emit the same TLS config whether or not anything is on disk, and
    the refusal is asserted separately against nginx_site_config.
    """
    config = _rendered(tmp_path, "on", LE_DOMAIN=DOMAIN, LE_ROOT=str(_missing_cert_root(tmp_path)))
    cert = _directive(_server_on(config, 443), "ssl_certificate")
    assert cert, f"render_nginx_config on must emit ssl_certificate unconditionally:\n{config}"
    assert cert.startswith(str(_missing_cert_root(tmp_path))), (
        f"the certificate path must be interpolated from LE_ROOT, not hardcoded: {cert}"
    )


def test_auto_mode_with_a_certificate_matches_render_nginx_config_on(tmp_path):
    """The wrapper must resolve the mode and delegate, not re-render."""
    knobs = {"LE_DOMAIN": DOMAIN, "LE_ROOT": str(_certified_root(tmp_path, DOMAIN))}
    auto = _site_config(tmp_path, NGINX_TLS="auto", **knobs)
    assert auto == _rendered(tmp_path, "on", **knobs), (
        "with a domain and a readable certificate, NGINX_TLS=auto must render exactly "
        "render_nginx_config on"
    )


# --------------------------------------------------------------------------
# plain HTTP (the rollback posture, and the default with no certificate)
# --------------------------------------------------------------------------


def test_plain_mode_emits_no_tls_at_all(tmp_path):
    """NGINX_TLS=off is the one-flag rollback, so it must be a clean plain site.

    Half a rollback -- a 443 server, an ssl_ directive or a redirect left behind
    -- takes the site down on the reload that follows it.
    """
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
    """The rendered config must interpolate the ports it is given.

    Because API_PORT and NEXT_PORT are pinned away from the shipped defaults,
    a literal 8001 or 3000 baked into the template fails here instead of
    quietly proxying to whatever happens to answer on the host. They used to be
    pinned to the defaults themselves, which made this assertion pass for a
    template that had hardcoded them.
    """
    config = _rendered(tmp_path, "off")
    assert f"127.0.0.1:{API_PORT}" in config, f"API routes must proxy to $API_PORT:\n{config}"
    assert f"127.0.0.1:{NEXT_PORT}" in config, f"frontend routes must proxy to $NEXT_PORT:\n{config}"


# --------------------------------------------------------------------------
# TLS mode
# --------------------------------------------------------------------------


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
    """A domain with nothing to serve it on must not produce a broken half-TLS
    site; the operator gets plain HTTP and the tls stage to fix it."""
    missing = str(_missing_cert_root(tmp_path))
    auto = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=missing)
    plain = _rendered(tmp_path, "off", LE_DOMAIN=DOMAIN, LE_ROOT=missing)
    assert auto == plain, "auto with no readable certificate must render the plain site, byte for byte"
    assert "listen 443" not in auto, f"a 443 server without a certificate:\n{auto}"
    assert "ssl_" not in auto, f"an ssl_ directive without a certificate:\n{auto}"


def test_acme_challenge_is_still_served_over_http_under_the_redirect(tmp_path):
    """The point of the whole issue: renewals keep working.

    Once :80 redirects, the challenge would be redirected to https -- where
    certbot's renewal request does not follow -- and the certificate would
    quietly expire. The challenge must therefore stay on :80, and it needs the
    `^~` modifier so it wins over the catch-all `location /` that redirects.
    """
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
    """A redirect that drops the host or the path breaks every deep link."""
    http = _server_on(_rendered(tmp_path, "on", LE_DOMAIN=DOMAIN), PUBLIC_PORT)
    targets = re.findall(r"return\s+30[1278]\s+(\S+);", http)
    assert targets == [REDIRECT_TARGET], (
        f"the http server must issue exactly one redirect to {REDIRECT_TARGET}, got {targets}"
    )


def test_the_two_servers_serve_the_same_routes(tmp_path):
    """The consistency claim, checked rather than asserted in a comment.

    :80 serves the ACME challenge plus a redirect; :443 serves the site. Any
    route present in one and missing from the other is drift between the two
    copies of the proxy body.
    """
    plain = _location_paths(_server_on(_rendered(tmp_path, "off", LE_DOMAIN=DOMAIN), PUBLIC_PORT))
    secure = _location_paths(_server_on(_rendered(tmp_path, "on", LE_DOMAIN=DOMAIN), 443))
    expected = plain - {ACME_PATH}
    assert expected, "the plain server served nothing to compare against"
    assert secure == expected, (
        "the tls server drifted from the plain one: "
        f"only-plain={sorted(expected - secure)} only-tls={sorted(secure - expected)}"
    )


# --------------------------------------------------------------------------
# guards against emitting an unusable config
# --------------------------------------------------------------------------


def test_tls_mode_refuses_to_render_a_config_for_a_missing_certificate(tmp_path):
    """The whole point of the cert-existence gate.

    A config naming a certificate that is not there is not a warning, it is a
    failed `nginx -t` and a site that will not reload -- so the generator must
    write nothing at all and fail instead.
    """
    root = _missing_cert_root(tmp_path)
    proc = _call(_env(tmp_path, NGINX_TLS="on", LE_DOMAIN=DOMAIN, LE_ROOT=str(root)), "nginx_site_config")
    expected = str(root / "live" / DOMAIN / "fullchain.pem")
    assert proc.returncode != 0, f"NGINX_TLS=on without a certificate must fail, stdout was:\n{proc.stdout}"
    assert proc.stdout == "", f"nothing may reach stdout when the certificate is missing: {proc.stdout!r}"
    assert expected in proc.stderr, f"the error must name the missing certificate {expected}:\n{proc.stderr}"


def test_tls_mode_refuses_to_render_a_config_for_a_missing_private_key(tmp_path):
    """The other half of the pair. `nginx -t` reads privkey.pem from the same
    config it reads the certificate from, so a config pointing at a missing key
    fails exactly as hard, and the generator must not write it."""
    root = _broken_cert_root(tmp_path, DOMAIN, "no_key")
    proc = _call(_env(tmp_path, NGINX_TLS="on", LE_DOMAIN=DOMAIN, LE_ROOT=str(root)), "nginx_site_config")

    expected = str(root / "live" / DOMAIN / "privkey.pem")
    assert proc.returncode != 0, f"NGINX_TLS=on without a private key must fail, stdout was:\n{proc.stdout}"
    assert proc.stdout == "", f"nothing may reach stdout when the key is missing: {proc.stdout!r}"
    assert expected in proc.stderr, f"the error must name the missing key {expected}:\n{proc.stderr}"


@requires_openssl
@pytest.mark.parametrize(
    "how",
    ["no_key", "empty_key", "empty_cert", "unparseable_cert", "expired"],
)
def test_auto_mode_refuses_a_certificate_pair_it_cannot_serve(how, tmp_path):
    """"auto" has to mean usable, not present.

    Every one of these leaves a certificate in place that nginx or certbot
    cannot actually serve: a half-written pair from an interrupted run, text
    that is not a certificate, or a leaf that has already expired. Keeping TLS
    on any of them leaves a site that is broken in a way nobody notices, so
    the answer has to be plain HTTP -- and because this verdict is only ever a
    pre-flight, certbot runs next and puts TLS straight back.
    """
    root = _broken_cert_root(tmp_path, DOMAIN, how)
    config = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))

    assert 443 not in {port for block in _server_blocks(config) for port in _listen_ports(block)}, (
        f"a certificate pair that is {how} must not be rendered as a TLS server:\n{config}"
    )
    assert "return 301 https://" not in config, f"nothing to redirect to:\n{config}"


@requires_openssl
def test_auto_mode_keeps_tls_for_a_pair_openssl_still_accepts(tmp_path):
    """The other side of the same rule, so the check cannot pass by always
    answering "off"."""
    root = _certified_root(tmp_path, DOMAIN)
    config = _site_config(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))

    ports = {port for block in _server_blocks(config) for port in _listen_ports(block)}
    assert 443 in ports, f"a usable certificate must keep serving over TLS:\n{config}"


def test_auto_mode_does_not_downgrade_when_openssl_is_missing(tmp_path):
    """Expiry is the one check that needs a tool, and it is not worth a
    downgrade: without openssl setup.sh cannot read an expiry verdict, so the
    file test stands alone rather than assuming the site is broken."""
    root = _certified_root(tmp_path, DOMAIN)
    env = _env(tmp_path, NGINX_TLS="auto", LE_DOMAIN=DOMAIN, LE_ROOT=str(root))
    # A PATH holding only what is needed to reach and source the script:
    # `dirname` for SCRIPT_DIR, `bash` for this call, `env` for the subprocess
    # lookup. `command -v openssl` then cannot succeed.
    lean_path = tmp_path / "lean-path"
    lean_path.mkdir()
    for tool in ("bash", "dirname", "env"):
        os.symlink(shutil.which(tool), lean_path / tool)
    env["PATH"] = str(lean_path)

    proc = _call(env, "nginx_tls_mode")

    assert proc.returncode == 0, f"nginx_tls_mode failed without openssl: {proc.stderr}"
    assert proc.stdout.strip() == "on", (
        f"a present, non-empty pair must not be downgraded just because openssl is absent, "
        f"got {proc.stdout.strip()!r}"
    )


@pytest.mark.parametrize("mode", ["off", "on"])
def test_rendered_config_is_structurally_sound(tmp_path, mode):
    """Shape checks, in both modes. Not a substitute for `nginx -t`."""
    config = _rendered(tmp_path, mode, LE_DOMAIN=DOMAIN)

    assert config.count("{") == config.count("}"), f"unbalanced braces:\n{config}"
    for line in config.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert stripped.endswith((";", "{", "}")), f"unterminated directive: {line!r}"

    blocks = _server_blocks(config)
    assert blocks, f"no server block rendered:\n{config}"
    seen = set()
    for block in blocks:
        ports = _listen_ports(block)
        assert len(ports) == 1, f"a server must listen on exactly one port:\n{block}"
        port = ports.pop()
        assert port not in seen, f"two server blocks both listen on {port}"
        seen.add(port)
        names = re.findall(r"^\s*server_name\s+(\S+);", block, re.MULTILINE)
        assert len(names) == 1, f"a server needs exactly one server_name, got {names}:\n{block}"
