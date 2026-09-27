"""Tests for the `tls` stage of setup.sh: what it does, and in what order.

`run_tls` obtains a certificate with certbot and then installs the HTTPS
config, and `run_nginx` is the function that puts a config in front of a live
site. What those functions actually *do* is invisible in the rendered config,
so they are exercised here by running the real functions with `sudo`,
`certbot`, `systemctl` and `nginx` replaced by stubs placed first on PATH.

1. The ACME challenge location must be installed BEFORE certbot runs. On a
   host whose nginx config predates it, the challenge token falls through
   `location /` to Next.js, 404s, and Let's Encrypt validation fails on the
   first run of `./setup.sh tls`. So the stage installs the config (which
   serves the challenge in both modes) first, then runs certbot, then switches
   the site to TLS.

2. Issuance uses certbot's webroot plugin, never `--standalone`, which needs
   port 80 to itself and would therefore fight the running site.

3. A re-run that fails to renew must NOT cost the site the HTTPS it already
   has. The pre-flight only has to drop to plain HTTP when there is no usable
   certificate, because port 80 serves the challenge either way.

4. `nginx -t` gates the live site: an accepted config is installed and nginx
   is reloaded, a rejected one is rolled back with no reload and a non-zero
   exit. It fails for reasons that have nothing to do with this site, so the
   rollback has to work when this config is perfectly fine.

Nothing here can reach the network, write to /etc, or touch a real certificate
store. The `sudo` stub carries out file operations only against paths inside
the test's own sandbox, skips anything naming the deploy host, and never runs
`systemctl`; the `certbot` stub mints a throwaway certificate under the test's
own `LE_ROOT`. The assertions are about the bytes that reach disk and the
sequence of privileged operations the stage requests.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"

# `nginx -t` is the gate that decides whether a config reaches the live site,
# so the stub is not a blind recorder: it reads the config that was actually
# installed and checks it the way nginx would before letting it through, and
# records the verdict. Those checks are exercised directly by
# test_the_nginx_stub_rejects_what_nginx_would_reject; STUB_NGINX_T_EXIT
# separately forces a rejection, for the common real reason this gate fires --
# `nginx -t` validates *every* config on the host, so a broken file in an
# unrelated site is enough to fail it.
_NGINX = """#!/bin/sh
printf 'nginx %s\n' "$*" >> "$STUB_LOG"
if [ "$1" != "-t" ]; then
    exit 0
fi
if [ -n "$STUB_NGINX_T_EXIT" ] && [ "$STUB_NGINX_T_EXIT" != "0" ]; then
    printf 'nginx -t rejected\n' >> "$STUB_LOG"
    exit "$STUB_NGINX_T_EXIT"
fi
if [ ! -f "$NGINX_CONF" ] || [ ! -s "$NGINX_CONF" ]; then
    printf 'nginx -t rejected (no config)\n' >> "$STUB_LOG"
    exit 1
fi
opens=$(grep -c '{' "$NGINX_CONF" || true)
closes=$(grep -c '}' "$NGINX_CONF" || true)
if [ "$opens" != "$closes" ]; then
    printf 'nginx -t rejected (unbalanced braces)\n' >> "$STUB_LOG"
    exit 1
fi
# Existence and non-emptiness, not readability: `nginx -t` runs as root, and
# certbot's privkey.pem is 0600 root:root, so a readability test here would
# reject a config real nginx loads happily -- the same unprivileged-reader
# mistake setup.sh's own gate had.
#
# Both halves. Real nginx refuses a config whose ssl_certificate points at
# nothing, and a stub that only checked the key would let a cert-side gate
# regression through with the key-side assertion noticing instead. The required
# space after the directive name is load-bearing: without it, the pattern for
# ssl_certificate also matches the first 17 characters of ssl_certificate_key.
for directive in ssl_certificate ssl_certificate_key; do
    for f in $(sed -n "s/^ *$directive \\+//p" "$NGINX_CONF" | tr -d ';'); do
        if [ ! -f "$f" ] || [ ! -s "$f" ]; then
            printf 'nginx -t rejected (no %s at %s)\n' "$directive" "$f" >> "$STUB_LOG"
            exit 1
        fi
    done
done
# Real nginx also *parses* the certificate, so a file that is present,
# non-empty and not a certificate is refused too -- which is the difference
# between a corrupt cert triggering the rollback and being installed on a live
# site. Skipped when there is no openssl, where the presence check stands alone.
if command -v openssl >/dev/null 2>&1; then
    for f in $(sed -n "s/^ *ssl_certificate \\+//p" "$NGINX_CONF" | tr -d ';'); do
        if [ -z "$(openssl x509 -noout -enddate -in "$f" 2>/dev/null)" ]; then
            printf 'nginx -t rejected (unparseable certificate at %s)\n' "$f" >> "$STUB_LOG"
            exit 1
        fi
    done
fi
printf 'nginx -t accepted\n' >> "$STUB_LOG"
exit 0
"""

# `systemctl is-enabled certbot.timer` must fail here: most hosts do not run
# the packaged timer, and that is exactly the branch where the stage has to
# install a renewal job by hand. Reporting success would make the stage claim
# renewal is handled when nothing would ever run.
_SYSTEMCTL = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
if [ "$1" = "is-enabled" ]; then
    exit 1
fi
exit 0
"""

# `sudo` records every command, and carries out the file operations -- but only
# when the operation's TARGET, resolved, lands inside the test's own sandbox
# (STUB_ROOT). That is what lets the rollback tests have a real previous config
# to restore and a real installed config to inspect, while
# `rm -f /etc/nginx/sites-enabled/default`, which names a path on the deploy
# host, is recorded and skipped.
#
# Containment is decided on the resolved path, not on a substring: a substring
# test is satisfied by `$STUB_ROOT/../../escape` and the kernel then resolves
# the `..` after the check has already passed. `realpath -m` resolves the path
# as it will actually be used, and works on paths that do not exist yet, which
# is the normal case for the config being installed.
#
# It is the TARGET that is checked because it is the only argument a file
# utility can write. Sources are routinely mktemp files in /tmp, outside the
# sandbox, and reading one is harmless.
#
# `systemctl` is recorded and never run: reloading a real nginx is not this
# suite's to do. For certbot it defers to the stub so the exit status
# propagates the way sudo's would.
_SUDO = """#!/bin/sh
if [ "$1" = "certbot" ]; then
    shift
    exec certbot "$@"
fi
if [ "$1" = "crontab" ] && [ -n "$2" ] && [ -f "$2" ]; then
    # Record what would land in root's crontab, not just the path to it.
    printf 'crontab %s\n' "$(cat "$2")" >> "$STUB_LOG"
    exit 0
fi
printf 'sudo %s\n' "$*" >> "$STUB_LOG"
[ "$1" = "systemctl" ] && exit 0
case "$1" in
    cp|install|mv|ln|rm|mkdir|chmod|touch)
        target=""
        for a in "$@"; do
            case "$a" in
                -*) continue ;;
            esac
            target="$a"
        done
        if [ -z "$target" ]; then
            exit 0
        fi
        root=$(realpath -m -- "$STUB_ROOT") || exit 0
        resolved=$(realpath -m -- "$target") || exit 0
        case "$resolved" in
            "$root"|"$root"/*) exec "$@" ;;
            *) exit 0 ;;
        esac
        ;;
    # Anything else here is a stub on PATH that only inspects and reports.
    *) exec "$@" ;;
esac
"""

# Stands in for certbot: snapshots the config that is live at the moment it is
# invoked, records its arguments and, on success, leaves the certificate where
# certbot would leave it. The snapshot is what proves the challenge was
# servable *before* issuance was attempted, rather than merely that an install
# was requested somewhere earlier in the log.
_CERTBOT = """#!/bin/sh
printf 'certbot %s\n' "$*" >> "$STUB_LOG"
if [ -n "$STUB_CERTBOT_SNAPSHOT" ] && [ -f "$NGINX_CONF" ]; then
    cp "$NGINX_CONF" "$STUB_CERTBOT_SNAPSHOT"
    printf 'certbot saw a config on disk\n' >> "$STUB_LOG"
else
    printf 'certbot saw NO config on disk\n' >> "$STUB_LOG"
fi
if [ "$STUB_CERTBOT_EXIT" != "0" ]; then
    exit "$STUB_CERTBOT_EXIT"
fi
mkdir -p "$LE_ROOT/live/$LE_DOMAIN"
# A real, non-empty pair: setup.sh now judges a certificate by what is in it,
# not by whether the path exists, so a zero-byte placeholder would be a state
# real certbot never leaves behind. EC because the key is never served, only
# read, and this keeps the suite quick. Without openssl, the placeholder is
# all that is available -- and is then also all setup.sh requires.
if command -v openssl >/dev/null 2>&1; then
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
        -keyout "$LE_ROOT/live/$LE_DOMAIN/privkey.pem" \
        -out "$LE_ROOT/live/$LE_DOMAIN/fullchain.pem" \
        -days 30 -subj "/CN=$LE_DOMAIN" >/dev/null 2>&1
fi
if [ ! -s "$LE_ROOT/live/$LE_DOMAIN/privkey.pem" ]; then
    printf '-----BEGIN CERTIFICATE-----\n' > "$LE_ROOT/live/$LE_DOMAIN/fullchain.pem"
    printf '-----BEGIN PRIVATE KEY-----\n' > "$LE_ROOT/live/$LE_DOMAIN/privkey.pem"
fi
exit 0
"""


def _write_stub(directory, name, body):
    path = directory / name
    path.write_text(body)
    path.chmod(0o755)
    return path


def _stubs(tmp_path, *, with_certbot=True, nginx_t_exit=0):
    """Build a stub directory plus the log path the stubs record into.

    The log is cleared here, so it always describes exactly the run that
    follows. A test that runs a stage twice would otherwise see both runs'
    commands concatenated, and an assertion like "this run did not reload
    nginx" would be answered by the previous run's reload.
    """
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    log.unlink(missing_ok=True)
    _write_stub(bindir, "sudo", _SUDO)
    _write_stub(bindir, "nginx", _NGINX)
    _write_stub(bindir, "systemctl", _SYSTEMCTL)
    if with_certbot:
        _write_stub(bindir, "certbot", _CERTBOT)
    return bindir, log


def _base_env(tmp_path, bindir, log, **over):
    """Every knob the nginx and tls stages can write to, redirected into
    tmp_path; the shipped defaults point at /etc/nginx and /var/www."""
    for parent in ("sites-available", "sites-enabled"):
        (tmp_path / "nginx" / parent).mkdir(parents=True, exist_ok=True)
    env = {
        "LE_DOMAIN": "search.example.com",
        "LE_EMAIL": "ops@example.com",
        "LE_ROOT": str(tmp_path / "letsencrypt"),
        "CERTBOT_WEBROOT": str(tmp_path / "webroot"),
        "NGINX_CONF": str(tmp_path / "nginx" / "sites-available" / "site"),
        "NGINX_LINK": str(tmp_path / "nginx" / "sites-enabled" / "site"),
        # The sandbox the sudo stub is willing to act inside.
        "STUB_ROOT": str(tmp_path),
        "STUB_LOG": str(log),
        "STUB_CERTBOT_EXIT": "0",
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
    }
    env.update(over)
    return env


def _calls(log):
    return log.read_text().splitlines() if log.exists() else []


def _run(tmp_path, function, env):
    proc = subprocess.run(
        ["/usr/bin/bash", "-c", f'source "{SETUP_SH}"\n{function}\n'],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    return proc


def _run_tls(tmp_path, *, certbot_exit=0, extra_env=None):
    """Run the real run_tls against stubs. Returns (code, calls, stdout, stderr)."""
    bindir, log = _stubs(tmp_path)
    env = _base_env(
        tmp_path,
        bindir,
        log,
        STUB_CERTBOT_EXIT=str(certbot_exit),
        STUB_CERTBOT_SNAPSHOT=str(tmp_path / "at-certbot-time.conf"),
        **(extra_env or {}),
    )

    proc = _run(tmp_path, "run_tls", env)
    return proc.returncode, _calls(log), proc.stdout, proc.stderr


def _run_nginx(tmp_path, *, nginx_t_exit=0, previous_config=None, extra_env=None):
    """Run the real run_nginx against stubs, with a real previous config in
    place so the rollback path has something to restore.

    Returns (code, calls, installed_config, stdout, stderr).
    """
    bindir, log = _stubs(tmp_path, with_certbot=False, nginx_t_exit=nginx_t_exit)
    env = _base_env(tmp_path, bindir, log, STUB_NGINX_T_EXIT=str(nginx_t_exit), **(extra_env or {}))
    conf = Path(env["NGINX_CONF"])
    if previous_config is not None:
        conf.write_text(previous_config)

    proc = _run(tmp_path, "run_nginx", env)
    return proc.returncode, _calls(log), (conf.read_text() if conf.exists() else None), proc.stdout, proc.stderr


def _index_of(calls, needle):
    for i, call in enumerate(calls):
        if needle in call:
            return i
    return -1


def test_challenge_is_served_before_certbot_runs(tmp_path):
    """certbot must not be reached before nginx serves the challenge.

    Otherwise the token 404s on the first ever run of `./setup.sh tls`, because
    the host's existing config has no `/.well-known/acme-challenge/` location.
    """
    code, calls, _, stderr = _run_tls(tmp_path)

    assert code == 0, f"run_tls failed: {stderr}"
    conf_at = _index_of(calls, "install")
    certbot_at = _index_of(calls, "certbot certonly")
    assert conf_at != -1, f"the nginx config was never installed: {calls}"
    assert certbot_at != -1, f"certbot was never invoked: {calls}"
    assert conf_at < certbot_at, (
        f"config installed at {conf_at} but certbot ran at {certbot_at}; the ACME "
        f"challenge would 404 and issuance would fail on a host that does not yet "
        f"serve it: {calls}"
    )


def test_the_config_certbot_answered_to_served_the_challenge(tmp_path):
    """The bytes, not the log: what was actually on disk at the moment certbot
    was asked to validate must already answer the ACME path. Ordering in a log
    only proves an install was requested; this proves it had happened."""
    code, calls, _, stderr = _run_tls(tmp_path)

    assert code == 0, f"run_tls failed: {stderr}"
    live = (tmp_path / "at-certbot-time.conf").read_text()
    assert "certbot saw a config on disk" in calls, f"certbot ran with no config installed: {calls}"
    assert "/.well-known/acme-challenge/" in live, (
        f"the config live when certbot ran could not serve the challenge:\n{live}"
    )


def test_tls_is_installed_only_after_the_certificate_exists(tmp_path):
    """The :443 config goes in after certbot succeeds, never before."""
    code, calls, _, stderr = _run_tls(tmp_path)

    assert code == 0, f"run_tls failed: {stderr}"
    installs = [i for i, c in enumerate(calls) if c.startswith("sudo install")]
    certbot_at = _index_of(calls, "certbot certonly")
    assert len(installs) == 2, f"expected a pre-flight and a final install: {calls}"
    assert installs[0] < certbot_at < installs[1], (
        f"the HTTPS config must be installed after certbot succeeds: {calls}"
    )


def test_certbot_is_never_called_standalone(tmp_path):
    """`--standalone` binds port 80 itself, which fights the running site."""
    _, calls, _, _ = _run_tls(tmp_path)

    issuance = [c for c in calls if "certbot certonly" in c]
    assert issuance, f"certbot was never invoked: {calls}"
    for call in issuance:
        assert "--standalone" not in call, f"standalone issuance would take port 80: {call}"
    assert "--webroot" in issuance[0], f"no webroot issuance: {calls}"


def test_issuance_is_non_interactive_repeatable_and_reloaded(tmp_path):
    """A re-run must not burn the rate limit, and a renewal must reach nginx."""
    _, calls, _, _ = _run_tls(tmp_path)
    issuance = next(c for c in calls if "certonly" in c)

    assert "--non-interactive" in issuance
    assert "--agree-tos" in issuance
    # Without this, every re-run counts against Let's Encrypt's rate limits.
    assert "--keep-until-expiring" in issuance
    assert "--email ops@example.com" in issuance
    # Without a deploy hook a renewed certificate is never picked up by nginx.
    assert "--deploy-hook systemctl reload nginx" in issuance


def test_renewal_is_wired_not_left_to_chance(tmp_path):
    """The packaged timer is stubbed as absent, so the fallback must install a
    renewal job; otherwise the certificate quietly expires."""
    code, calls, _, stderr = _run_tls(tmp_path)

    assert code == 0, f"run_tls failed: {stderr}"
    renewal = [c for c in calls if c.startswith("crontab ")]
    assert renewal, f"nothing was scheduled, so the certificate would expire: {calls}"
    assert any("certbot renew" in line for line in renewal), f"no renewal job: {renewal}"


def test_failed_issuance_keeps_serving_and_explains_itself(tmp_path):
    """A certbot hiccup must not take the site down, and must not be silent."""
    code, calls, _, stderr = _run_tls(tmp_path, certbot_exit=1)

    assert code == 1
    # The pre-flight config is in place, so the site keeps serving plain HTTP.
    assert _index_of(calls, "sudo install") != -1, f"site left unconfigured: {calls}"
    # No certificate was produced, so nothing may reload nginx as if it had.
    after = calls[_index_of(calls, "certbot certonly") :]
    # Match the reload the stage issues, exactly. A substring would also match
    # the text of certbot's own "--deploy-hook systemctl reload nginx"
    # argument, and the earlier whole-line comparison against a bare
    # "systemctl reload nginx" never matched anything at all.
    assert not any(c.strip() == "sudo systemctl reload nginx" for c in after), (
        f"nginx was reloaded as though issuance had succeeded: {calls}"
    )
    assert "certbot failed" in stderr
    # "certbot failed" alone leaves the operator no idea what to check.
    assert ".well-known/acme-challenge/" in stderr
    assert "port 80" in stderr



def _installed_https_servers(config):
    return config.count("listen 443 ssl")


def _existing_pair(tmp_path):
    """A letsencrypt root already holding a usable certificate, as it would be
    on any host that has run `./setup.sh tls` successfully once."""
    live = tmp_path / "letsencrypt" / "live" / "search.example.com"
    live.mkdir(parents=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "ec",
            "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
            "-keyout", str(live / "privkey.pem"),
            "-out", str(live / "fullchain.pem"),
            "-days", "30", "-subj", "/CN=search.example.com",
        ],
        check=True,
        capture_output=True,
    )



def _expired_pair(tmp_path):
    """A letsencrypt root holding a complete pair whose leaf has already lapsed.

    `openssl req -x509` refuses a non-positive -days, so the dates come from
    `openssl ca` and are fixed in the past: this stays expired for good rather
    than only today.
    """
    live = tmp_path / "letsencrypt" / "live" / "search.example.com"
    live.mkdir(parents=True, exist_ok=True)
    ca = tmp_path / "mini-ca"
    (ca / "newcerts").mkdir(parents=True)
    (ca / "index.txt").write_text("")
    (ca / "serial").write_text("1000\n")
    (ca / "openssl.cnf").write_text(
        "[ ca ]\ndefault_ca = CA_default\n[ CA_default ]\n"
        "dir = .\ndatabase = $dir/index.txt\nnew_certs_dir = $dir/newcerts\n"
        "serial = $dir/serial\ndefault_md = sha256\npolicy = pol\n"
        "email_in_dn = no\nunique_subject = no\n[ pol ]\ncommonName = supplied\n"
    )

    def openssl(*args):
        proc = subprocess.run(["openssl", *args], cwd=ca, check=True, capture_output=True)
        return proc.stdout

    openssl("genrsa", "-out", str(ca / "ca-key.pem"), "2048")
    openssl("req", "-x509", "-key", str(ca / "ca-key.pem"), "-out", str(ca / "ca.pem"),
            "-days", "3650", "-subj", "/CN=expired-cert-test-ca")
    openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", str(live / "privkey.pem"),
            "-out", str(ca / "req.csr"), "-subj", "/CN=search.example.com")
    openssl("ca", "-batch", "-config", str(ca / "openssl.cnf"), "-cert", str(ca / "ca.pem"),
            "-keyfile", str(ca / "ca-key.pem"), "-in", str(ca / "req.csr"),
            "-out", str(live / "fullchain.pem"),
            "-startdate", "20200101000000Z", "-enddate", "20200201000000Z", "-notext")



@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a real certificate")
def test_rerun_keeps_https_when_certbot_fails(tmp_path):
    """A certbot hiccup must never cost the site its HTTPS.

    The pre-flight only has to drop to plain HTTP on a first run, because the
    :80 server serves the ACME challenge in both modes. Forcing it to plain HTTP
    unconditionally meant a re-run rewrote a working HTTPS site to cleartext and
    reloaded nginx, so a transient certbot failure during a routine re-run left
    credentials and chat content crossing the wire in the clear -- the exact
    exposure this stage exists to close.
    """
    _existing_pair(tmp_path)
    code, calls, _, stderr = _run_tls(tmp_path, certbot_exit=1)

    conf = (tmp_path / "nginx" / "sites-available" / "site").read_text()
    assert code == 1, f"a certbot failure must still be reported as one: {stderr}"
    assert _installed_https_servers(conf) == 1, (
        f"the live TLS server was removed by a re-run that only had to renew: {calls}"
    )
    assert "return 301 https://" in conf, f"the redirect to https was dropped too: {calls}"
    # And the operator must not be left believing the site went plaintext.
    assert "plain-HTTP config is still installed" not in stderr, (
        f"the site is still on HTTPS; telling the operator otherwise is the bug: {stderr}"
    )
    assert "TLS config is still installed" in stderr, f"the failure must say what is actually serving: {stderr}"


def test_rerun_downgrades_when_there_is_no_usable_certificate(tmp_path):
    """The other side of the same rule, so the pre-flight cannot be "fixed" by
    always keeping TLS: with nothing to serve TLS with, the challenge still has
    to be servable, so plain HTTP it is -- and that is what gets reported."""
    code, calls, _, stderr = _run_tls(tmp_path, certbot_exit=1)

    conf = (tmp_path / "nginx" / "sites-available" / "site").read_text()
    assert code == 1
    assert _index_of(calls, "sudo install") != -1, (
        f"a downgrade that writes nothing leaves the site unconfigured: {calls}"
    )
    assert _installed_https_servers(conf) == 0, f"no certificate exists, so there is nothing to serve: {conf}"
    assert "/.well-known/acme-challenge/" in conf, f"the challenge must still be servable: {conf}"
    assert "plain-HTTP config is still installed" in stderr, f"the report must match reality: {stderr}"


def test_accepted_config_is_installed_and_nginx_is_reloaded(tmp_path):
    """The ordinary path: a config nginx validates goes live and takes effect."""
    previous = "# the config that is live right now\nserver { listen 80; }\n"
    code, calls, installed, _, stderr = _run_nginx(tmp_path, previous_config=previous)

    assert code == 0, f"run_nginx failed: {stderr}"
    assert "nginx -t accepted" in calls, f"the config was never validated: {calls}"
    assert installed is not None and installed != previous, "the new config was never installed"
    assert "server {" in installed and installed.count("{") == installed.count("}")
    assert any(c.strip() == "sudo systemctl reload nginx" for c in calls), (
        f"an accepted config must be reloaded into nginx: {calls}"
    )


def test_rejected_config_is_rolled_back_and_nginx_is_not_reloaded(tmp_path):
    """`nginx -t` is the only thing standing between a typo and a dead site.

    It fails often for reasons that have nothing to do with this site -- a
    broken file in an unrelated vhost is enough, because nginx validates every
    config it has. So when it says no, the config that was already serving has
    to be put back, no reload may be issued (a reload would load the rejected
    config anyway), and the stage has to fail loudly.
    """
    previous = "# the config that is live right now\nserver { listen 80; }\n"
    code, calls, installed, _, stderr = _run_nginx(tmp_path, nginx_t_exit=1, previous_config=previous)

    assert code != 0, f"a rejected config must fail the stage, not proceed: {calls}"
    assert installed == previous, (
        f"the config live before this run was not restored; it now reads:\n{installed}"
    )
    assert not any(c.strip() == "sudo systemctl reload nginx" for c in calls), (
        f"nginx was reloaded onto a config it had just rejected: {calls}"
    )
    assert "rolling back" in stderr, f"the operator must be told the config was rolled back: {stderr}"


def test_rollback_restores_the_config_byte_for_byte(tmp_path):
    """The restored file is the one that was serving, exactly: nginx is
    reloaded from it on the next run, so a truncated or re-rendered backup would
    be a different config from the one that was working."""
    previous = "# live config\nserver {\n    listen 80;\n    server_name _;\n}\n"
    _, _, installed, _, _ = _run_nginx(tmp_path, nginx_t_exit=1, previous_config=previous)

    assert installed == previous



@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_standalone_nginx_keeps_https_when_the_certificate_has_expired(tmp_path):
    """`./setup.sh nginx` on its own must not quietly undo TLS.

    This is the path `./setup.sh nginx` and `./setup.sh all` take, and neither
    runs certbot -- so a rule that answers "off" for a lapsed certificate had no
    way to put TLS back. It removed the live `listen 443 ssl` server and the
    redirect, reloaded nginx, and exited 0, handing a working encrypted site
    back in cleartext because a clock ran out. The certificate is reported as
    expired instead, and `./setup.sh tls` renews it.
    """
    _expired_pair(tmp_path)
    code, calls, installed, _, stderr = _run_nginx(tmp_path)
    assert code == 0, f"run_nginx failed: {stderr}"
    assert _installed_https_servers(installed) == 1, (
        f"an expired certificate cost the site its HTTPS server: {calls}"
    )
    assert "return 301 https://" in installed, f"the redirect to https was dropped too: {calls}"
    assert any(c.strip() == "sudo systemctl reload nginx" for c in calls), (
        f"a valid config must still be reloaded: {calls}"
    )
    # Reported, not silently tolerated.
    assert "has expired" in stderr, f"the operator must be told the certificate has lapsed: {stderr}"
    assert "./setup.sh tls" in stderr, f"and told how to renew it: {stderr}"
    assert "plain HTTP" not in stderr, (
        f"the site is on HTTPS; the warning must not claim otherwise: {stderr}"
    )




def test_the_sudo_stub_refuses_a_dot_dot_escape(tmp_path):
    """The sandbox has to be a resolved path, not a prefix of a string.

    A substring test is satisfied by `$STUB_ROOT/../../escape`, and the kernel
    then resolves the `..` long after the check passed -- so a stub that is
    escapable is a trap for whoever adds the next test, even though no command
    setup.sh actually issues contains one.
    """
    bindir, log = _stubs(tmp_path, with_certbot=False)
    outside = tmp_path.parent / "escape-rel"
    escape = f"{tmp_path}/nginx/../../escape-rel"
    assert not outside.exists(), f"the escape target must not exist beforehand: {outside}"

    proc = subprocess.run(
        ["sudo", "install", "-m", "644", str(__file__), escape],
        env=_base_env(tmp_path, bindir, log),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, "the stub records and declines; it does not fail the caller"
    assert not outside.exists(), (
        f"the stub wrote outside its sandbox: {outside}\n{_calls(log)}"
    )
    # Recorded, so the refusal is visible rather than a silent no-op.
    assert any("escape-rel" in line for line in _calls(log)), (
        f"the refused command must still be recorded: {_calls(log)}"
    )


@pytest.mark.parametrize(
    ("config", "rejected"),
    [
        ("server {\n    listen 443 ssl;\n}\n", False),
        ("server {\n    listen 443 ssl;\n", True),
        ("", True),
    ],
)
def test_the_nginx_stub_rejects_what_nginx_would_reject(config, rejected, tmp_path):
    """The stub's own checks, with nothing forcing the outcome.

    Every other rejection test drives the verdict through STUB_NGINX_T_EXIT, so
    without this the validation inside the stub is decorative code that never
    decides anything. A balanced, non-empty config is accepted; unbalanced
    braces and an empty file are not, which is what real `nginx -t` does.
    """
    bindir, log = _stubs(tmp_path, with_certbot=False)
    env = _base_env(tmp_path, bindir, log)
    conf = Path(env["NGINX_CONF"])
    if config:
        conf.write_text(config)

    proc = subprocess.run(
        [str(bindir / "nginx"), "-t"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if rejected:
        assert proc.returncode != 0, f"the stub accepted a config nginx would refuse: {config!r}"
        assert any("nginx -t rejected" in line for line in _calls(log)), (
            f"the reason must be recorded: {_calls(log)}"
        )
    else:
        assert proc.returncode == 0, f"the stub rejected a sound config: {_calls(log)}"
        assert "nginx -t accepted" in "\n".join(_calls(log)), f"the verdict must be recorded: {_calls(log)}"


def test_the_nginx_stub_rejects_a_config_naming_a_key_that_is_not_there(tmp_path):
    """A config whose ssl_certificate_key points at nothing is not loadable."""
    bindir, log = _stubs(tmp_path, with_certbot=False)
    env = _base_env(tmp_path, bindir, log)
    conf = Path(env["NGINX_CONF"])
    conf.write_text(f"server {{\n    ssl_certificate_key {tmp_path}/nope/privkey.pem;\n}}\n")

    proc = subprocess.run(
        [str(bindir / "nginx"), "-t"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0, "a config naming a missing key must not pass `nginx -t`"


def test_standalone_nginx_downgrades_when_the_pair_is_really_absent(tmp_path):
    """The case that genuinely is a downgrade, kept deliberate.

    With no certificate and no key there is nothing to serve TLS with, so
    plain HTTP is the honest outcome -- but the warning has to name the pair
    that is missing, not claim "no readable certificate", which was true of a
    certificate that was present, non-empty and merely expired.
    """
    code, calls, installed, _, stderr = _run_nginx(tmp_path)

    assert code == 0
    assert _index_of(calls, "sudo install") != -1, (
        f"a downgrade that writes nothing leaves the site unconfigured: {calls}"
    )
    assert _installed_https_servers(installed) == 0, f"there is no pair to serve: {installed}"
    assert "plain HTTP" in stderr, f"the downgrade must be stated: {stderr}"
    assert "fullchain.pem" in stderr and "privkey.pem" in stderr, (
        f"the warning must name both halves it could not find: {stderr}"
    )
    assert "no readable certificate" not in stderr, (
        f"readability is not what this mode is decided on: {stderr}"
    )


@pytest.mark.parametrize(
    ("env", "expected"),
    [({"LE_DOMAIN": ""}, "LE_DOMAIN"), ({"LE_EMAIL": ""}, "LE_EMAIL")],
)
def test_refuses_to_start_without_its_prerequisites(tmp_path, env, expected):
    """No domain, no contact address -> no privileged action at all."""
    code, calls, _, stderr = _run_tls(tmp_path, extra_env=env)

    assert code == 1
    assert expected in stderr
    assert calls == [], f"the stage acted without its prerequisites: {calls}"


def test_refuses_when_certbot_is_missing(tmp_path):
    """Without certbot there is no TLS; say so instead of half-configuring."""
    bindir, log = _stubs(tmp_path, with_certbot=False)
    # Only `dirname` is reachable: setup.sh uses it on line 4 while being
    # sourced, and run_tls returns before invoking anything else. Nothing
    # outside this directory can be found, so a real certbot is unreachable.
    os.symlink(shutil.which("dirname"), bindir / "dirname")

    proc = subprocess.run(
        ["/usr/bin/bash", "-c", f'source "{SETUP_SH}"\nrun_tls\n'],
        env={
            **os.environ,
            "LE_DOMAIN": "search.example.com",
            "LE_EMAIL": "ops@example.com",
            "PATH": str(bindir),
            "STUB_LOG": str(log),
            "STUB_CERTBOT_EXIT": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "certbot is not installed" in proc.stderr
    assert not log.exists() or log.read_text().strip() == ""


def test_routine_rerun_without_le_domain_keeps_serving_https(tmp_path):
    """A re-run from an ordinary shell must not undo TLS.

    LE_DOMAIN is read from the environment and written nowhere, so
    `./setup.sh nginx` -- and the nginx stage inside `./setup.sh all` -- run
    from a shell that does not export it used to see no domain, resolve "auto"
    to plain HTTP, rewrite a live HTTPS site to cleartext and exit 0 with
    nothing on stderr. The certificate was sitting right there.

    Two things make that safe now: the domain is recovered from the installed
    config, and "auto" prefers what is already serving over what a probe says.
    """
    _existing_pair(tmp_path)
    # First run: bring the site up on TLS, the way ./setup.sh tls would leave it.
    code, _, installed, _, stderr = _run_nginx(tmp_path, extra_env={"NGINX_TLS": "on"})
    assert code == 0, f"the TLS install failed: {stderr}"
    assert _installed_https_servers(installed) == 1, f"expected a TLS config: {installed}"

    # Second run: same shell, LE_DOMAIN not exported, NGINX_TLS back to auto.
    code, calls, installed, stdout, stderr = _run_nginx(tmp_path, extra_env={"LE_DOMAIN": ""})

    assert code == 0, f"the re-run failed: {stderr}"
    assert _installed_https_servers(installed) == 1, (
        f"a routine re-run stripped the live HTTPS server: {calls}"
    )
    assert "return 301 https://" in installed, f"the redirect was dropped too: {calls}"
    assert "serving: https" in stdout, f"the operator must be told it is still encrypted:\n{stdout}"
    assert "plain HTTP" not in stderr, f"and not told the opposite: {stderr}"
    # The domain was found, not configured, and saying so is the point.
    assert "recovered" in stdout, (
        f"a recovered domain must be reported as recovered, not implied to be configured:\n{stdout}"
    )


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_a_lost_certificate_does_not_strip_the_live_https_server(tmp_path):
    """The case the status-quo default actually exists for.

    The certificate is gone -- cleaned up, restored from a backup, moved -- but
    nginx is still serving the one it loaded, and the installed config still
    names `:443`. Nothing here is a reason to take the site off TLS: the
    operator gets a loud failure and an untouched config, because the stage
    cannot emit a correct one without the pair. Rewriting to plain HTTP and
    reloading would be wrong twice over: it discards a working posture, and it
    does it silently.
    """
    _existing_pair(tmp_path)
    code, _, installed, _, stderr = _run_nginx(tmp_path, extra_env={"NGINX_TLS": "on"})
    assert code == 0, f"the TLS install failed: {stderr}"
    live_tls = installed

    # The pair disappears underneath the running site.
    live = tmp_path / "letsencrypt" / "live" / "search.example.com"
    (live / "fullchain.pem").unlink()
    (live / "privkey.pem").unlink()

    code, calls, installed, _, stderr = _run_nginx(tmp_path)

    assert code != 0, f"a config naming a certificate that is gone must fail, not be replaced: {calls}"
    assert installed == live_tls, (
        f"the config that was serving must be left exactly as it was:\n{installed}"
    )
    assert not any(c.strip() == "sudo systemctl reload nginx" for c in calls), (
        f"nginx must not be reloaded onto a config that was never written: {calls}"
    )
    assert "no certificate at" in stderr, f"the missing pair must be named: {stderr}"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_a_corrupt_certificate_is_not_reported_as_expired(tmp_path):
    """An unparseable certificate needs its own advice.

    `openssl x509 -checkend` exits non-zero for a lapsed notAfter, for text that
    is not a certificate, and for a file it cannot open. Reporting all three as
    "expired" sent the operator to re-run a command that could not help: the
    tls stage issues with --keep-until-expiring, which leaves anything certbot
    cannot parse exactly as it is, so the loop repeats and reports success
    forever.
    """
    _existing_pair(tmp_path)
    (tmp_path / "letsencrypt" / "live" / "search.example.com" / "fullchain.pem").write_text(
        "-----BEGIN CERTIFICATE-----\nnot a certificate\n"
    )

    code, _, installed, _, stderr = _run_nginx(tmp_path, extra_env={"NGINX_TLS": "on"})

    # The mode is unchanged -- a complete pair still gets a TLS server, and
    # `nginx -t` is what refuses it -- but the operator is told the real cause.
    assert "cannot be read as a" in stderr, f"the corrupt file must be named: {stderr}"
    assert "has expired" not in stderr, f"a corrupt certificate is not an expired one: {stderr}"
    assert "sudo rm -f" in stderr, f"the remedy has to say what to do about the file: {stderr}"
    assert "--keep-until-expiring" in stderr, (
        f"the operator must be told why re-running will not fix it: {stderr}"
    )
    # nginx rejects it, so the previous config is restored and the stage fails.
    assert code != 0, f"a config nginx refuses must fail the stage: {installed}"
    assert "rolling back" in stderr, f"the rollback must be announced: {stderr}"


def _tls_site_installed(tmp_path):
    """A host already serving :443, as `./setup.sh tls` would have left it."""
    _existing_pair(tmp_path)
    code, _, installed, _, stderr = _run_nginx(tmp_path, extra_env={"NGINX_TLS": "on"})
    assert code == 0, f"the TLS install failed: {stderr}"
    assert _installed_https_servers(installed) == 1, f"expected a TLS config: {installed}"
    return installed


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_tls_stage_can_reissue_when_the_pair_is_gone(tmp_path):
    """`./setup.sh tls` is the repair tool, so it must work on exactly the hosts
    that need repairing.

    A live HTTPS site whose certificate was deleted, truncated or replaced used
    to be unrepairable: the status-quo default resolved the pre-flight to TLS,
    the gate then refused to render a config naming a certificate that is not
    there, and the stage returned 1 BEFORE certbot ran -- while telling the
    operator to run the command they had just run.
    """
    _tls_site_installed(tmp_path)
    live = tmp_path / "letsencrypt" / "live" / "search.example.com"
    (live / "fullchain.pem").unlink()
    (live / "privkey.pem").unlink()

    code, calls, _, stderr = _run_tls(tmp_path)

    assert _index_of(calls, "certbot certonly") != -1, (
        f"the stage must still ask certbot for a certificate: {calls}\n{stderr}"
    )
    assert code == 0, f"the stage failed instead of repairing the site: {stderr}"
    conf = (tmp_path / "nginx" / "sites-available" / "site").read_text()
    assert _installed_https_servers(conf) == 1, f"TLS was not restored: {conf}"
    assert "return 301 https://" in conf, f"the redirect was not restored: {conf}"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_tls_stage_can_reissue_when_the_certificate_is_corrupt(tmp_path):
    """The same, for a certificate that is present, non-empty and not one."""
    _tls_site_installed(tmp_path)
    (tmp_path / "letsencrypt" / "live" / "search.example.com" / "fullchain.pem").write_text(
        "-----BEGIN CERTIFICATE-----\nnot a certificate\n"
    )

    code, calls, _, stderr = _run_tls(tmp_path)

    assert _index_of(calls, "certbot certonly") != -1, (
        f"the stage must still ask certbot for a certificate: {calls}\n{stderr}"
    )
    assert "cannot be read as a" in stderr, f"the corrupt file must be named: {stderr}"
    assert code == 0, f"the stage failed instead of repairing the site: {stderr}"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_tls_stage_can_reissue_after_the_documented_corrupt_remedy(tmp_path):
    """A remedy the script prints must be one that actually works.

    The corrupt warning tells the operator to `rm` the file and re-run the tls
    stage. Under the previous behaviour that second line could not succeed:
    after the `rm` the pre-flight still resolved to TLS (the installed config
    still said `listen 443 ssl`), the gate refused, and the stage returned 1
    before certbot. So the script was handing out a two-line recipe whose
    second line was guaranteed to fail. This executes exactly what is printed.
    """
    _tls_site_installed(tmp_path)
    cert = tmp_path / "letsencrypt" / "live" / "search.example.com" / "fullchain.pem"
    cert.write_text("-----BEGIN CERTIFICATE-----\nnot a certificate\n")

    _, _, _, stderr = _run_tls(tmp_path)
    remedy = next((line.split()[-1] for line in stderr.splitlines() if "sudo rm -f" in line), None)
    assert remedy == str(cert), (
        f"the corrupt warning must print the path to remove, got {remedy!r}\n{stderr}"
    )

    # Now do exactly what the script said to do.
    Path(remedy).unlink()
    code, calls, _, stderr = _run_tls(tmp_path)

    assert _index_of(calls, "certbot certonly") != -1, f"certbot was never reached: {calls}\n{stderr}"
    assert code == 0, f"the documented remedy does not work: {stderr}"
    conf = (tmp_path / "nginx" / "sites-available" / "site").read_text()
    assert _installed_https_servers(conf) == 1, f"TLS was not restored: {conf}"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint a certificate")
def test_the_corrupt_diagnosis_survives_the_tls_preflight(tmp_path):
    """A file certbot will skip must still be named during the pre-flight.

    TLS_BOOTSTRAP=1 silences "you are being served over plain HTTP", because
    certbot is genuinely about to make that irrelevant. It must NOT silence
    this: the tls stage is issued with --keep-until-expiring, so certbot skips
    a file it cannot parse and leaves it exactly as it is. The pre-flight is
    often the only run that notices, so suppressing the diagnosis there is
    suppressing it at the one moment it is needed -- and the operator is left
    re-running a command that reports success and changes nothing.
    """
    _tls_site_installed(tmp_path)
    (tmp_path / "letsencrypt" / "live" / "search.example.com" / "fullchain.pem").write_text(
        "-----BEGIN CERTIFICATE-----\nnot a certificate\n"
    )

    # The pre-flight alone: TLS_BOOTSTRAP=1, and nothing after it.
    bindir, log = _stubs(tmp_path)
    proc = subprocess.run(
        ["/usr/bin/bash", "-c", f'source "{SETUP_SH}"\nTLS_BOOTSTRAP=1 run_nginx\n'],
        env={**os.environ, **_base_env(tmp_path, bindir, log)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert "cannot be read as a" in proc.stderr, (
        f"the corrupt file must be named even in the pre-flight: {proc.stderr}"
    )
    assert "served over plain HTTP" not in proc.stderr, (
        f"the plain-HTTP advisory is the one thing the pre-flight may silence: {proc.stderr}"
    )


def test_the_nginx_stub_rejects_a_config_naming_a_missing_certificate(tmp_path):
    """The stub must check the certificate as well as the key.

    Checking only the key meant a regression in setup.sh's cert-side gate was
    still caught -- but by the key-side assertion, on which error message came
    out, not by the gate being absent. A stub that accepts a config nginx
    refuses is not a stand-in for nginx.
    """
    bindir, log = _stubs(tmp_path, with_certbot=False)
    env = _base_env(tmp_path, bindir, log)
    conf = Path(env["NGINX_CONF"])
    good = tmp_path / "letsencrypt" / "live" / "search.example.com"
    good.mkdir(parents=True)
    (good / "privkey.pem").write_text("key\n")
    conf.write_text(
        "server {\n"
        f"    ssl_certificate {tmp_path}/nowhere/fullchain.pem;\n"
        f"    ssl_certificate_key {good / 'privkey.pem'};\n"
        "}\n"
    )

    proc = subprocess.run(
        [str(bindir / "nginx"), "-t"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0, "a config naming a missing certificate must not pass `nginx -t`"
    assert any("no ssl_certificate at" in line for line in _calls(log)), (
        f"the missing half must be named: {_calls(log)}"
    )


# --------------------------------------------------------------------------
# the class invariant, enumerated
# --------------------------------------------------------------------------
#
# Three separate fixes each closed one downgrade path, which is the signature of
# a class rather than three coincidences. The rule is short: an ORDINARY
# invocation must never change what is already serving. This enumerates the
# states that decide the posture, so a fourth path cannot open unnoticed.
#
# The invariant, for a host whose installed config already serves :443: either
# the :443 server and the 301 are still there afterwards, or the stage failed
# having written nothing and reloaded nothing. Never: it exited 0 having
# replaced them with plain HTTP.

_TLS_DOMAIN = "search.example.com"  # the domain _base_env already configures


def _install_serving_tls(tmp_path):
    """Put a :443 config in place and return the bytes that were installed."""
    _existing_pair(tmp_path)
    code, _, installed, _, stderr = _run_nginx(tmp_path, extra_env={"NGINX_TLS": "on"})
    assert code == 0, f"could not stage a TLS host: {stderr}"
    assert _installed_https_servers(installed) == 1, f"expected a TLS config: {installed}"
    return installed


def _break_pair(tmp_path, how):
    """Damage the installed pair in one of the enumerated ways."""
    live = tmp_path / "letsencrypt" / "live" / _TLS_DOMAIN
    if how == "no_pair":
        (live / "fullchain.pem").unlink()
        (live / "privkey.pem").unlink()
    elif how == "key_only":
        (live / "fullchain.pem").unlink()
    elif how == "cert_only":
        (live / "privkey.pem").unlink()
    elif how == "corrupt":
        (live / "fullchain.pem").write_text("-----BEGIN CERTIFICATE-----\nnope\n")
    elif how == "empty":
        (live / "fullchain.pem").write_text("")
    elif how == "expired":
        _expired_pair(tmp_path)  # overwrites the pair in place, lapsed
    else:
        raise AssertionError(f"unknown breakage {how!r}")


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl to mint certificates")
@pytest.mark.parametrize(
    ("pair", "le_domain", "mode", "le_root"),
    [
        ("valid", _TLS_DOMAIN, "auto", "normal"),
        ("valid", "", "auto", "normal"),
        ("valid", _TLS_DOMAIN, "on", "normal"),
        ("valid", _TLS_DOMAIN, "auto", "absent"),
        ("valid", "", "auto", "absent"),
        ("valid", _TLS_DOMAIN, "auto", "empty"),
        ("valid", _TLS_DOMAIN, "off", "normal"),
        ("expired", _TLS_DOMAIN, "auto", "normal"),
        ("expired", "", "auto", "normal"),
        ("corrupt", _TLS_DOMAIN, "auto", "normal"),
        ("corrupt", "", "auto", "normal"),
        ("empty", _TLS_DOMAIN, "auto", "normal"),
        ("no_pair", _TLS_DOMAIN, "auto", "normal"),
        ("no_pair", "", "auto", "normal"),
        ("key_only", _TLS_DOMAIN, "auto", "normal"),
        ("cert_only", _TLS_DOMAIN, "auto", "normal"),
    ],
)
def test_an_ordinary_rerun_never_changes_what_is_serving(pair, le_domain, mode, le_root, tmp_path):
    """A re-run either keeps the site encrypted or fails without touching it.

    `NGINX_TLS=off` is the one row that is expected to move the site, and it is
    listed so the sweep states the exception rather than leaving it implicit.
    """
    before = _install_serving_tls(tmp_path)
    assert "listen 443 ssl" in before
    if pair != "valid":
        _break_pair(tmp_path, pair)

    env = {"NGINX_TLS": mode, "LE_DOMAIN": le_domain}
    if le_root == "absent":
        env["LE_ROOT"] = str(tmp_path / "nowhere")
    elif le_root == "empty":
        env["LE_ROOT"] = str(tmp_path)
    code, calls, installed, _, stderr = _run_nginx(tmp_path, extra_env=env)
    where = f"{pair}/{'domain' if le_domain else 'no-domain'}/{mode}/{le_root}"

    if mode == "off":
        # The deliberate opt-in, and the only thing that may remove TLS.
        assert _installed_https_servers(installed) == 0, f"{where}: an explicit off must take effect"
        return

    if code == 0:
        assert "listen 443 ssl" in installed, (
            f"{where}: the re-run exited 0 having taken a live HTTPS site off TLS:\n{installed}\n{stderr}"
        )
        assert "return 301 https://" in installed, (
            f"{where}: the redirect was dropped by a re-run that exited 0:\n{installed}"
        )
    else:
        assert installed == before, (
            f"{where}: the stage failed but still changed the config that was serving:\n{installed}"
        )
        assert not any(c.strip() == "sudo systemctl reload nginx" for c in calls), (
            f"{where}: the stage failed and reloaded anyway: {calls}"
        )
        assert "ERROR" in stderr, f"{where}: a failure must say so: {stderr}"
