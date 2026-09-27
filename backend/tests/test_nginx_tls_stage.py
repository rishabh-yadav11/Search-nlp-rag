"""Tests for the `tls` stage of setup.sh: what it does, and in what order.

`run_tls` obtains a certificate with certbot and then installs the HTTPS
config. Two things about that are load-bearing and neither is visible in the
rendered config, so they are exercised here by running the real function with
`sudo`, `certbot`, `systemctl` and `nginx` replaced by recording stubs placed
first on PATH:

1. The ACME challenge location must be installed BEFORE certbot runs. On a
   host whose nginx config predates it, the challenge token falls through
   `location /` to Next.js, 404s, and Let's Encrypt validation fails on the
   first run of `./setup.sh tls`. So the stage installs the plain-HTTP config
   (which serves the challenge and otherwise is what the host already serves)
   first, then runs certbot, then switches the site to TLS.

2. Issuance uses certbot's webroot plugin, never `--standalone`, which needs
   port 80 to itself and would therefore fight the running site.

Nothing here can reach the network, write to /etc, or touch a real certificate
store: `sudo` records the command it was asked to run and returns, and the
`certbot` stub creates a throwaway certificate under the test's own `LE_ROOT`.
The assertions are about the sequence of privileged operations the stage
requests, which is the property that was wrong.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"

# Records "<args>" and does nothing else. Used for the commands the stage runs
# purely for their side effect on a real host (install, ln, reload, nginx -t).
_RECORDER = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG"
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

# `sudo` must not execute what it is asked to run: run_nginx would otherwise
# really write /etc/nginx. It records the command and, for certbot, defers to
# the stubbed certbot so the exit status propagates the way sudo's would.
_SUDO = """#!/bin/sh
if [ "$1" = "certbot" ]; then
    shift
    exec certbot "$@"
fi
if [ "$1" = "crontab" ] && [ -n "$2" ] && [ -f "$2" ]; then
    # Record what would land in root's crontab, not just the path to it.
    printf 'crontab %s\\n' "$(cat "$2")" >> "$STUB_LOG"
    exit 0
fi
printf 'sudo %s\\n' "$*" >> "$STUB_LOG"
exit 0
"""

# Stands in for certbot: records its arguments and, on success, leaves the
# certificate where certbot would leave it. That is what lets the stage's
# post-issuance TLS install proceed.
_CERTBOT = """#!/bin/sh
printf 'certbot %s\\n' "$*" >> "$STUB_LOG"
if [ "$STUB_CERTBOT_EXIT" != "0" ]; then
    exit "$STUB_CERTBOT_EXIT"
fi
mkdir -p "$LE_ROOT/live/$LE_DOMAIN"
: > "$LE_ROOT/live/$LE_DOMAIN/fullchain.pem"
: > "$LE_ROOT/live/$LE_DOMAIN/privkey.pem"
exit 0
"""


def _write_stub(directory, name, body):
    path = directory / name
    path.write_text(body)
    path.chmod(0o755)
    return path


def _stubs(tmp_path, *, with_certbot=True):
    """Build a stub directory plus the log path the stubs record into."""
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    _write_stub(bindir, "sudo", _SUDO)
    _write_stub(bindir, "nginx", _RECORDER)
    _write_stub(bindir, "systemctl", _SYSTEMCTL)
    if with_certbot:
        _write_stub(bindir, "certbot", _CERTBOT)
    return bindir, log


def _run_tls(tmp_path, *, certbot_exit=0, extra_env=None):
    """Run the real run_tls against stubs. Returns (code, calls, stdout, stderr)."""
    bindir, log = _stubs(tmp_path)
    env = {
        "LE_DOMAIN": "search.example.com",
        "LE_EMAIL": "ops@example.com",
        "LE_ROOT": str(tmp_path / "letsencrypt"),
        "CERTBOT_WEBROOT": str(tmp_path / "webroot"),
        # Every path the stage can write to is redirected into tmp_path; the
        # shipped defaults point at /etc/nginx and /var/www.
        "NGINX_CONF": str(tmp_path / "nginx" / "sites-available" / "site"),
        "NGINX_LINK": str(tmp_path / "nginx" / "sites-enabled" / "site"),
        "STUB_LOG": str(log),
        "STUB_CERTBOT_EXIT": str(certbot_exit),
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
    }
    env.update(extra_env or {})

    proc = subprocess.run(
        ["/usr/bin/bash", "-c", f'source "{SETUP_SH}"\nrun_tls\n'],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc.returncode, calls, proc.stdout, proc.stderr


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
    assert not any(c.strip() == "systemctl reload nginx" for c in after), (
        f"nginx was reloaded as though issuance had succeeded: {calls}"
    )
    assert "certbot failed" in stderr
    # "certbot failed" alone leaves the operator no idea what to check.
    assert ".well-known/acme-challenge/" in stderr
    assert "port 80" in stderr


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
