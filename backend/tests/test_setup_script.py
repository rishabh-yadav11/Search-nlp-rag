"""Tests for the AUTH_TRUST_X_FORWARDED_FOR migration in setup.sh.

setup.sh repairs an upgraded host's backend/.env in two steps: it appends the
shipped default when the key is missing, and warns when the operator has forced
the header to be trusted. The warning is a safety net -- a forced-true value
means X-Forwarded-For is trusted from any peer, so a client reaching :8001
directly can forge it to dodge a rate limit -- so it has to fire for every
spelling the application itself reads as forced-true, not just the literal
"true". Missing one means the operator gets the forgeable posture with no
warning at all.

The pattern is extracted from setup.sh and run through real grep, so these
tests cannot drift away from the script they are policing, and the expected
set is derived from config._env_tristate rather than restated here, so changing
the parser's truthy set turns this file red until setup.sh is updated too.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from app.config import _env_tristate

SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"

# The key the migration repairs, and the value it appends when absent.
FLAG = "AUTH_TRUST_X_FORWARDED_FOR"
SHIPPED_DEFAULT = "auto"

# Spellings grouped by what _env_tristate does with them. The truthy group is
# what the warning must cover; the others must stay silent.
TRUTHY_SPELLINGS = ("1", "true", "yes", "on")
FALSY_SPELLINGS = ("0", "false", "no", "off")
SILENT_SPELLINGS = ("auto", "", "maybe", "1 2", "yes-no")


def _guard_flags():
    """The -i/-E flags and the regex the warning branch of setup.sh greps with.

    Read out of the script rather than duplicated, so fixing the script is what
    makes these tests pass. Fails loudly if the branch moves or is renamed.
    """
    script = SETUP_SH.read_text()
    match = re.search(
        r"^\s*elif grep (?P<flags>-q[a-zA-Z]*) '(?P<pattern>[^']+)' \"\$ENV_FILE\"; then$",
        script,
        re.MULTILINE,
    )
    assert match, f"could not find the {FLAG} warning guard in {SETUP_SH}"
    assert FLAG in match.group("pattern"), "the guard must key on the flag it repairs"
    return match.group("flags"), match.group("pattern")


def _warns(env_line, flags, pattern):
    """Run the real grep against a one-line .env and report whether it matches."""
    if env_line is not None:
        contents = f"{env_line}\n"
    else:
        contents = "SOME_OTHER_SETTING=1\n"  # key absent
    proc = subprocess.run(
        ["grep", flags, "-e", pattern],
        input=contents,
        capture_output=True,
        text=True,
        check=False,
    )
    # grep exits 0 on match, 1 on no match, >1 on a usage error.
    assert proc.returncode in (0, 1), f"grep failed: {proc.returncode} {proc.stderr}"
    return proc.returncode == 0


@pytest.mark.parametrize("value", TRUTHY_SPELLINGS)
def test_warning_guard_matches_every_forced_true_spelling(value):
    """1/true/yes/on all force trust, so all must warn.

    This is the defect: matching only the literal "true" left an operator who
    wrote "1" or "yes" in the forgeable posture with nothing printed.
    """
    flags, pattern = _guard_flags()
    assert _warns(f"{FLAG}={value}", flags, pattern), f"forced-true {value!r} did not warn"


@pytest.mark.parametrize("value", ["TRUE", "True", "YES", "On", f"  {TRUTHY_SPELLINGS[1]}  "])
def test_warning_guard_is_case_and_whitespace_insensitive(value):
    """_env_tristate strips and lowercases before matching, so the guard must too."""
    flags, pattern = _guard_flags()
    assert _warns(f"{FLAG}={value}", flags, pattern), f"{value!r} did not warn"


@pytest.mark.parametrize("value", FALSY_SPELLINGS + SILENT_SPELLINGS)
def test_warning_guard_stays_silent_for_everything_else(value):
    """A falsy value is a deliberate local/direct choice and 'auto' is the new
    default; neither is the forgeable posture, so neither should nag."""
    flags, pattern = _guard_flags()
    assert not _warns(f"{FLAG}={value}", flags, pattern), f"{value!r} wrongly warned"


def test_warning_guard_stays_silent_when_the_key_is_absent():
    """An absent key is the case setup.sh appends to, not one it warns about."""
    flags, pattern = _guard_flags()
    assert not _warns(None, flags, pattern)


def test_guard_covers_exactly_the_spellings_the_parser_calls_true():
    """The guard and the parser must agree, or the warning covers the wrong set.

    Driven off _env_tristate itself: widen or narrow the parser's truthy set and
    this fails until setup.sh follows, which is the point -- the script and the
    application cannot each maintain their own idea of "forced true".
    """
    flags, pattern = _guard_flags()
    for candidate in TRUTHY_SPELLINGS + FALSY_SPELLINGS + SILENT_SPELLINGS:
        # Drive the real parser by putting the value in the environment.
        old = os.environ.get(FLAG)
        os.environ[FLAG] = candidate
        try:
            truthy = _env_tristate(FLAG) is True
        finally:
            if old is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = old
        assert _warns(f"{FLAG}={candidate}", flags, pattern) is truthy, (
            f"{candidate!r}: parser says truthy={truthy}, but the guard disagrees"
        )


def test_missing_key_is_appended_and_an_existing_value_is_never_rewritten():
    """The migration only ever appends, and only when the key is absent.

    Rewriting an operator's value would break the supported "proxy on another
    host" deployment, which is exactly what a forced-true setting is for.
    """
    lines = [line.strip() for line in SETUP_SH.read_text().splitlines()]
    append_line = f'echo "{FLAG}={SHIPPED_DEFAULT}" >> "$ENV_FILE"'
    assert append_line in lines, "setup.sh should append the shipped default when the key is absent"

    # The append must sit inside a presence check on the key, so it cannot run
    # for a value the operator already set.
    at = lines.index(append_line)
    opener = lines[at - 1]
    assert opener.startswith("if ! grep -q ") and f"^{FLAG}=" in opener, (
        f"the append must be guarded by a presence check on the key, got: {opener!r}"
    )

    # And nothing anywhere may rewrite an operator's value in place: silently
    # downgrading a forced-true would break the supported "proxy on another
    # host" deployment, which is exactly what a forced-true setting is for.
    # Only the guarded append may write to $ENV_FILE, and nothing may sed it in
    # place. (The warning lines mention the flag and >&2, so they are excluded
    # by requiring a write to $ENV_FILE or a sed.)
    writes = [
        line
        for line in lines
        if FLAG in line and ("sed -i" in line or ('"$ENV_FILE"' in line and ">>" in line))
    ]
    for line in writes:
        assert line == append_line, f"setup.sh must not rewrite the operator's value: {line!r}"




def test_the_nginx_vhost_refuses_the_uncached_probe():
    """`location /ready` is a PREFIX match, so without an explicit block a public
    GET /ready/deep is proxied to the API and is held dark only by the app's
    host-local check. /ready/deep is uncached and unrated by design, so a
    public dependency-probe amplifier deserves a second layer that does not live
    in the same file as the code it protects."""
    source = SETUP_SH.read_text()

    assert "location /ready/deep { return 404; }" in source


def test_the_cron_entry_probes_the_port_the_api_is_actually_bound_to():
    """API_PORT is a documented override (usage() lists it), and run_services
    binds pm2 to 127.0.0.1:$API_PORT. The watchdog's own default is 8001, so a
    cron entry that does not pass BASE leaves it probing a closed port on any
    other port -- and a watchdog that cannot reach the backend restarts it every
    five minutes, which is the outage it exists to prevent."""
    source = SETUP_SH.read_text()
    cron_lines = [line for line in source.splitlines() if "deploy/healthcheck.sh" in line]

    assert cron_lines, "the cron entry must still be installed by setup.sh"
    entry = cron_lines[0]
    assert "API_PORT" in entry, f"the cron entry must derive BASE from API_PORT: {entry!r}"
    assert "BASE=" in entry


def test_the_nginx_vhost_heredoc_contains_no_backticks():
    """The vhost is written through an UNQUOTED heredoc, so a backtick anywhere
    in it is a command substitution: setup.sh prints "command not found" to the
    operator and writes the mangled result into the deployed vhost. The fix is
    plain text in the comment -- quoting the heredoc delimiter is not available,
    since $API_PORT and the \\$remote_addr escapes depend on the unquoted form."""
    body = SETUP_SH.read_text().split("<<NGINX\n", 1)[1].split("\nNGINX\n", 1)[0]

    assert "`" not in body, "a backtick in the unquoted NGINX heredoc is executed"


# --- the deploy gate must wait on readiness, and must not destroy the deploy
# --- to find out (#279) ---------------------------------------------------
#
# `run_services` is EXECUTED here rather than grepped. Asserting the source
# contains a URL is the vacuous-test class this project keeps policing: it
# passes just as well with the gate moved in front of the teardown, which is
# precisely the regression a reviewer found -- a gate that can now legitimately
# fail, placed after `pm2 delete` of both services under `set -e`, turned a
# misconfigured key into a destroyed deployment with the frontend never coming
# back. The functions are extracted verbatim from setup.sh, so this cannot drift
# from the script it is policing.

CURL_STUB = """\
#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >>"$CURL_LOG"
case "$url" in
*/ready/deep)
    code="$READY_DEEP_CODE"
    if [ "$1" = "-fsS" ]; then
        # wait_http uses -f: no body on a failure, and a non-zero exit. 000 is
        # a refused connection, which is also a curl failure.
        printf '%s' "$code"
        [ "$code" -ge 400 ] 2>/dev/null && exit 22
        [ "$code" = "000" ] && exit 7
        exit 0
    fi
    # The diagnostic curl has no -f, so the body survives -- that is the whole
    # reason it is written that way.
    [ "$code" -ge 400 ] && printf '%s' "$READY_DEEP_BODY"
    exit 0
    ;;
*) code=200 ;;
esac
printf '%s' "$code"
[ "$code" -ge 400 ] 2>/dev/null && exit 22
exit 0
"""

PM2_STUB = """\
#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$PM2_LOG"
exit 0
"""

REAL_KEY = "AI" + "za" + "SyD-Example_Key" + "0123456789" + "abcdefghij"
PLACEHOLDER_KEY = "your" + "_key_here"


def _extract_function(name):
    """Pull one shell function out of setup.sh, verbatim."""
    lines = SETUP_SH.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    end = next(i for i, line in enumerate(lines[start:], start) if line == "}")
    return "\n".join(lines[start : end + 1])


READY_REPORT_BODY = (
    '{"ready": false, "checks": {"qdrant": {"ok": true}, "models": {"ok": true},'
    ' "redis": {"ok": true, "cache": "memory"}, "llm": {"ok": false, "reason": "placeholder"}}}'
)


def run_services(tmp_path, *, env_key=REAL_KEY, env_base="", ready_deep_code=200, report=READY_REPORT_BODY):
    """Run the real run_services against stub pm2/curl. Returns (rc, pm2_log,
    stdout+stderr)."""
    home = tmp_path / "host"
    (home / "backend").mkdir(parents=True)
    (home / "frontend").mkdir(parents=True)
    (home / "logs").mkdir()
    (home / "pid").mkdir()
    (home / "bin").mkdir()
    env_lines = [f"GEMINI_API_KEY={env_key}"]
    if env_base:
        env_lines.append(f"GEMINI_BASE_URL={env_base}")
    (home / "backend" / ".env").write_text("\n".join(env_lines) + "\n")
    for name, stub_body in (("curl", CURL_STUB), ("pm2", PM2_STUB)):
        stub = home / "bin" / name
        stub.write_text(stub_body)
        stub.chmod(0o755)

    pm2_log = tmp_path / "pm2.log"
    curl_log = tmp_path / "curl.log"
    harness = "\n".join(
        [
            "set -euo pipefail",
            f"cd {home}",
            'API_PORT=8001; NEXT_PORT=3000; GUNICORN_WORKERS=4',
            'API_MAX_MEMORY=5G; API_MAX_RESTARTS=10; FRONTEND_MAX_MEMORY=1G; RESTART_BACKOFF_MS=100',
            f'LOGS={home}/logs; PID_DIR={home}/pid; ENV_FILE={home}/backend/.env',
            f'VENV_PY={home}/bin/python; SCRIPT_DIR={home}',
            "sleep() { :; }",  # wait_http's 1s backoff would cost 30s per run
            "stage() { :; }",
            "ensure_pm2() { :; }",
            _extract_function("report_readiness_reason"),
            _extract_function("wait_http"),
            _extract_function("run_services"),
            'run_services && echo "RUN_SERVICES_RC=0" || echo "RUN_SERVICES_RC=$?"',
        ]
    )
    proc = subprocess.run(
        ["bash", "-c", harness],
        env={
            **os.environ,
            "PATH": f"{home}/bin:{os.environ['PATH']}",
            "PM2_LOG": str(pm2_log),
            "CURL_LOG": str(curl_log),
            "READY_DEEP_CODE": str(ready_deep_code),
            "READY_DEEP_BODY": report,
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc, (pm2_log.read_text() if pm2_log.exists() else "")


def _started(pm2_log, service):
    return any("start" in line and f"--name {service}" in line for line in pm2_log.splitlines())


def test_a_failing_readiness_gate_does_not_leave_the_frontend_stopped(tmp_path):
    """The regression a review found: with the gate in front of the frontend
    start, a not-ready deploy took the frontend down with it and `set -e`
    aborted before bringing it back. Both services must be started, and the
    dump saved, before anything is allowed to fail the deploy."""
    proc, pm2_log = run_services(tmp_path, ready_deep_code=503)

    assert "RUN_SERVICES_RC=1" in proc.stdout
    assert _started(pm2_log, "vccircle-backend")
    assert _started(pm2_log, "vccircle-frontend"), "a failed readiness gate must not stop the frontend from being deployed"
    assert "save" in pm2_log, "the pm2 dump must be saved before the gate can abort"
    assert "not ready after" in proc.stderr + proc.stdout


def test_the_gate_is_not_satisfied_by_the_liveness_stub(tmp_path):
    """A backend that answers 200 on /health and 503 on readiness is exactly
    the deployment this issue is about: it must not be declared done."""
    proc, pm2_log = run_services(tmp_path, ready_deep_code=503)

    assert "RUN_SERVICES_RC=1" in proc.stdout
    assert _started(pm2_log, "vccircle-frontend")


def test_a_ready_deployment_still_succeeds(tmp_path):
    """The control: the reordering must not have broken the happy path."""
    proc, pm2_log = run_services(tmp_path, ready_deep_code=200)

    assert "RUN_SERVICES_RC=0" in proc.stdout, proc.stdout + proc.stderr
    assert _started(pm2_log, "vccircle-backend")
    assert _started(pm2_log, "vccircle-frontend")


def test_a_gateway_deployment_with_a_non_google_key_is_not_blocked(tmp_path):
    """GEMINI_BASE_URL is configurable, so the pre-flight's shape rule applies
    only to Google's endpoint -- exactly as app.config.classify_gemini_api_key
    does. A false rejection here would block a working deploy."""
    # Assembled, not written out: a credential-shaped literal on a line naming a
    # key is what a secrets scanner reports (same reason as REAL_KEY above).
    home_key = "gateway-" + "token-0123" + "456789"
    proc, _pm2_log = run_services(
        tmp_path,
        env_key=home_key,
        env_base="https://llm-gateway.internal/v1",
        ready_deep_code=200,
    )

    assert "RUN_SERVICES_RC=0" in proc.stdout, proc.stdout + proc.stderr


# The drift guard: setup.sh must never hold a copy of the placeholder list.
#
# An earlier revision of this change added an `api_key_preflight` that re-implemented
# app.config.classify_gemini_api_key in bash. It was a second classifier, and a
# review proved it wrong: it carried 20 of the classifier's 28 sentinels, and the
# eight it missed (plus the repeated-filler rule) are all reachable when
# GEMINI_BASE_URL is a non-Google gateway -- a configuration this repo supports.
# A host in that shape passed the pre-flight, lost both pm2 services to the
# teardown, and only then learned from the gate that it was never ready.
#
# The pre-flight is gone. What is tested instead is the property that actually
# matters, against every sentinel the app rejects INCLUDING any added later:
# a not-ready deploy must never leave the frontend stopped.


def _all_app_placeholders():
    """Every value classify_gemini_api_key calls a placeholder, plus the
    repeated-filler shapes it rejects outside the sentinel list. Read from the
    app so a new sentinel cannot escape this test."""
    from app.config import _PLACEHOLDER_API_KEYS

    return sorted(_PLACEHOLDER_API_KEYS | {"x" * 9, "a-x-a-x-a-x", "0" * 9})


@pytest.mark.parametrize("placeholder", _all_app_placeholders())
def test_every_placeholder_the_app_rejects_still_deploys_both_services(tmp_path, placeholder):
    """Whatever the app makes of the key, the deploy is not left torn down: both
    services are started and the dump is saved before the gate can fail. The
    base URL is a non-Google gateway on purpose -- that is the path on which the
    shell-side shape rule was skipped, and therefore the path where the old
    pre-flight's gaps were reachable."""
    proc, pm2_log = run_services(
        tmp_path,
        env_key=placeholder,
        env_base="https://llm-gateway.internal/v1",
        ready_deep_code=503,
    )

    assert "RUN_SERVICES_RC=1" in proc.stdout
    assert _started(pm2_log, "vccircle-backend")
    assert _started(pm2_log, "vccircle-frontend"), f"{placeholder!r} tore the frontend down"
    assert "save" in pm2_log


def test_a_failed_gate_names_the_key_fault_from_the_app_not_from_shell(tmp_path):
    """The operator has to be told WHICH fault, and the only trustworthy source
    is the app: report_readiness_reason prints the response body, which carries
    checks.llm.reason. Its curl deliberately has no -f -- with -f the body is
    suppressed on exactly the >=400 response this function only ever sees, and
    the function becomes dead code."""
    proc, _pm2_log = run_services(tmp_path, env_key=PLACEHOLDER_KEY, ready_deep_code=503)
    output = proc.stdout + proc.stderr

    assert "not ready after" in output
    assert "placeholder" in output, f"the readiness reason never reached the operator: {output}"
    assert "readiness report" in output


def test_a_failed_gate_says_so_even_when_the_probe_returns_no_body(tmp_path):
    """A probe that cannot be reached at all must not print an empty report
    silently; the operator is told the difference between 'not ready' and 'no
    answer'."""
    # A refused connection: curl fails and writes no body (the string matters,
    # 000 is curl's "no answer" code -- the int 0 would str() to "0").
    proc, _pm2_log = run_services(tmp_path, ready_deep_code="000", report="")

    assert "no report body" in proc.stderr
    assert "not ready after" in proc.stderr
