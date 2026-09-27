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
# records that it did. STUB_NGINX_T_EXIT forces a rejection for the common real
# reason this gate fires -- `nginx -t` validates *every* config on the host, so
# a broken file in an unrelated site is enough to fail it.
_NGINX = """#!/bin/sh
printf 'nginx %s\n' "$*" >> "$STUB_LOG"
if [ "$1" != "-t" ]; then
    exit 0
fi
if [ -n "$STUB_NGINX_T_EXIT" ] && [ "$STUB_NGINX_T_EXIT" != "0" ]; then
    printf 'nginx -t rejected\\n' >> "$STUB_LOG"
    exit "$STUB_NGINX_T_EXIT"
fi
if [ ! -s "$NGINX_CONF" ]; then
    printf 'nginx -t rejected (no config)\\n' >> "$STUB_LOG"
    exit 1
fi
opens=$(grep -c '{' "$NGINX_CONF" || true)
closes=$(grep -c '}' "$NGINX_CONF" || true)
if [ "$opens" != "$closes" ]; then
    printf 'nginx -t rejected (unbalanced braces)\\n' >> "$STUB_LOG"
    exit 1
fi
for f in $(sed -n 's/^ *ssl_certificate_key *//p' "$NGINX_CONF" | tr -d ';'); do
    if [ ! -r "$f" ]; then
        printf 'nginx -t rejected (unreadable key %s)\\n' "$f" >> "$STUB_LOG"
        exit 1
    fi
done
printf 'nginx -t accepted\\n' >> "$STUB_LOG"
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
# against paths inside the test's own sandbox (STUB_ROOT). That is what lets the
# rollback tests have a real previous config to restore and a real installed
# config to inspect, while `rm -f /etc/nginx/sites-enabled/default`, which names
# a path on the deploy host, is recorded and skipped. `systemctl` is recorded
# and never run: reloading a real nginx is not this suite's to do. For certbot
# it defers to the stub so the exit status propagates the way sudo's would.
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
    # File utilities act only inside the sandbox. `rm -f
    # /etc/nginx/sites-enabled/default` names the deploy host, so it is recorded
    # and skipped rather than obeyed.
    cp|install|mv|ln|rm|mkdir|chmod|touch)
        # A sandbox path is followed either by more of the path or by the end
        # of the argument, never by anything else.
        case " $* " in
            *" $STUB_ROOT/"*|*" $STUB_ROOT "*) exec "$@" ;;
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
    """Build a stub directory plus the log path the stubs record into."""
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
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
