"""Tests for the AUTH_TRUST_X_FORWARDED_FOR migration in setup.sh.

setup.sh repairs an upgraded host's backend/.env in two steps: it appends the
shipped default when the key is missing, and warns when the operator has forced
the header to be trusted. The warning is a safety net -- a forced-true value means
X-Forwarded-For is trusted from any peer, so a client reaching :8001 directly can
forge it to dodge a rate limit -- so it has to fire for every line an operator can
write that the application itself reads as forced-true, not just `KEY=true`.

The guard is EXECUTED here, not read out of the source and handed to grep:
candidates built as f"{FLAG}={value}" cannot express a quoted value or a blank
around the `=`, so a guard that matched nothing for those shapes still passed
every test. The expected answer is derived from python-dotenv -- the parser
app/config.py reads .env with -- and config._env_tristate, so the script and the
application cannot each hold their own idea of "forced true".
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from itertools import product
from pathlib import Path

import pytest
from dotenv import dotenv_values

from app.config import _FALSE_SPELLINGS, _TRUE_SPELLINGS, _env_tristate

SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"
ECOSYSTEM_JS = Path(__file__).resolve().parents[2] / "ecosystem.config.js"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node resolves ecosystem.config.js")


# The environment `run_services` hands `pm2 start ecosystem.config.js` at the
# defaults this module's harness runs it with.
_ECOSYSTEM_ENV = {
    "VCCIRCLE_ROOT": str(ECOSYSTEM_JS.parent),
    "API_PORT": "8001",
    "NEXT_PORT": "3000",
    "GUNICORN_WORKERS": "4",
    "MIN_UPTIME_MS": "30000",
    "API_MAX_MEMORY": "5G",
    "FRONTEND_MAX_MEMORY": "1G",
    "API_MAX_RESTARTS": "10",
    "RESTART_BACKOFF_MS": "100",
}


def _ecosystem_apps() -> list[dict]:
    """The pm2 apps `ecosystem.config.js` resolves to, as node computes them.

    `run_services` starts pm2 from this file rather than from inline argv, so what
    the running process gets is what the file resolves to once setup.sh's exports
    are applied. Resolving it with node rather than reading the source keeps these
    assertions about the EXECUTED definition."""
    proc = subprocess.run(
        [NODE, "-e", "console.log(JSON.stringify(require('./ecosystem.config.js').apps))"],
        cwd=str(ECOSYSTEM_JS.parent),
        env={**os.environ, **_ECOSYSTEM_ENV},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"ecosystem.config.js did not load: {proc.stderr}"
    return json.loads(proc.stdout)


def _ecosystem_app(service: str) -> dict:
    for app in _ecosystem_apps():
        if app["name"] == service:
            return app
    raise AssertionError(f"{service} is not defined in ecosystem.config.js")

# The key the migration repairs, and the value it appends when absent.
FLAG = "AUTH_TRUST_X_FORWARDED_FOR"
# The shell function that carries the whole repair, named so a rename in setup.sh
# fails in one place.
XFF_FUNCTION = "migrate_xff_trust"

SHIPPED_DEFAULT = "auto"
APPEND_LINE = f"{FLAG}={SHIPPED_DEFAULT}"

# Every way an operator writes the key that python-dotenv resolves to the same
# assignment. Each reaches config._env_tristate as a forced True -- which every
# test below proves against the parser rather than taking it on trust -- so each
# has to warn.
FORCED_TRUE_LINES = [
    f"{FLAG}=true",
    f'{FLAG}="true"',
    f"{FLAG}='true'",
    f"{FLAG} = true",
    f"{FLAG}= true",
    f"{FLAG} =true",
    f'{FLAG} = "true"',
    f"export {FLAG}=true",
    f'export {FLAG} = "yes"',
    f"\t{FLAG}=on",
    f"  {FLAG} = 1  ",
    f"{FLAG}=TRUE",
    f"{FLAG}=On",
    f'{FLAG}=" true "',
    f"{FLAG}=true # forced",
    f'{FLAG}="true" # forced',
    f"{FLAG}=true\r",
]

# Shapes that carry the key but not a forced true. Each has to stay silent: a
# warning the operator cannot act on is a warning they learn to ignore. The last
# four are the precision cases -- dotenv either refuses the line outright or keeps
# the punctuation as part of the value.
KEY_PRESENT_SILENT_LINES = [
    f"{FLAG}=auto",
    f"{FLAG}=false",
    f"{FLAG}=0",
    f"{FLAG}=",
    f"{FLAG}=maybe",
    f"{FLAG}=1 2",
    f"{FLAG}=truee",
    f"{FLAG}=true#note",
    f"{FLAG} = auto",
    f'export {FLAG}="false"',
    f'{FLAG}="true',  # unterminated: python-dotenv cannot parse the line
    f"{FLAG}=true\"",  # the quote is part of the value, so the value is 'true"'
    f"{FLAG}=\"true'",  # mismatched quotes: a parse error, not a forced true
]

# Lines with no assignment to our key at all -- the one case setup.sh is allowed to
# write to, since a .env that predates the per-IP rate limits has no trust setting
# and every proxied request then keys on the nginx peer. A commented-out line is the
# operator's own off switch and must read as absent.
KEY_ABSENT_LINES = [
    f"# {FLAG}=true",
    f"  # {FLAG} = \"true\"",
    f"{FLAG}_MAX=1",
    f"{FLAG}_URL=https://example.invalid",
    "SOME_OTHER_SETTING=1",
]

# Templates for the spelling sweep below. The value half of the guard is one
# alternation and the assignment half is a separate pattern built in front of it,
# so the sweep has to cross the two: three value forms against a plain, a spaced,
# an exported and a comment-tailed assignment.
LINE_TEMPLATES = [
    "{key}={value}",
    '{key}="{value}"',
    "{key}='{value}'",
    "{key} = {value}",
    'export {key} = "{value}"',
    "{key}={value} # note",
]


def _xff_function() -> str:
    """setup.sh's `migrate_xff_trust`, verbatim, or a loud failure.

    These tests run the function rather than a copy of its regex, so a rename, a
    move or a deletion is answered here rather than leaving a test that greps a
    pattern nothing calls."""
    source = SETUP_SH.read_text()
    assert f"{XFF_FUNCTION}() {{" in source, (
        f"setup.sh no longer defines {XFF_FUNCTION}(), so the forced-true "
        "warning has nowhere to live; the tests in this section EXECUTE it"
    )
    return _extract_function(XFF_FUNCTION)


XFF_HARNESS = """\
@FUNCTION@

# One candidate per line on stdin, in order. Each gets a .env holding exactly
# that line and the real function, and reports what the operator would see:
# warned or not, and whether their file came back byte-identical.
dir=$(mktemp -d)
count=0
while IFS= read -r line || [ -n "$line" ]; do
    count=$((count + 1))
    printf '%s\\n' "$line" > "$dir/$count.line"
done

probe() {
    local i=$1 line after verdict touched
    IFS= read -r line < "$dir/$i.line"
    printf '%s\\n' "$line" > "$dir/$i.env"
    migrate_xff_trust "$dir/$i.env" 2> "$dir/$i.err" > /dev/null
    if [ -s "$dir/$i.err" ]; then verdict=warn; else verdict=silent; fi
    # `$(<file)`, not `read`: read stops at the first newline, so an APPENDED
    # second line would compare equal and every probe would report the
    # operator's file as untouched -- which is precisely the write this has to
    # catch, since the presence check missing a shape is what let setup.sh
    # append `=auto` under a forced true. `$(<file)` is a bash builtin, so
    # reading the whole file costs no fork.
    after="$(<"$dir/$i.env")"
    if [ "$after" = "$line" ]; then touched=untouched; else touched=rewritten; fi
    printf '%s %s\\n' "$verdict" "$touched" > "$dir/$i.res"
}

# The sweep below runs tens of thousands of probes, and each one is two grep
# processes. Spread them over a few workers so the check stays cheap enough to
# keep exhaustive; the results are written per candidate and printed in order,
# so the parallelism cannot reorder the answer.
worker=1
while [ "$worker" -le "@WORKERS@" ]; do
    (
        i=$worker
        while [ "$i" -le "$count" ]; do
            probe "$i"
            i=$((i + @WORKERS@))
        done
    ) &
    worker=$((worker + 1))
done
wait

i=1
while [ "$i" -le "$count" ]; do
    IFS= read -r result < "$dir/$i.res"
    printf '%s\\n' "$result"
    i=$((i + 1))
done
rm -rf "$dir"
"""


def run_xff_guard(candidates, workers=8):
    """Run the real guard against one throwaway .env per candidate line.

    Returns one ``(warned, untouched, line)`` per candidate, in order. A single
    bash process handles the whole batch so a sweep of the spelling space costs
    one interpreter rather than thousands."""
    harness = (
        XFF_HARNESS.replace("@FUNCTION@", _xff_function()).replace("@WORKERS@", str(workers))
    )
    proc = subprocess.run(
        ["bash", "-c", harness],
        input="".join(f"{line}\n" for line in candidates),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"the {XFF_FUNCTION} harness failed:\n{proc.stderr}"
    results = [line.split() for line in proc.stdout.splitlines()]
    assert len(results) == len(candidates), (
        f"the harness answered {len(results)} of {len(candidates)} candidates"
    )
    return [
        (verdict == "warn", touched == "untouched", line)
        for (verdict, touched), line in zip(results, candidates)
    ]


def _config_reads_as_forced_true(line: str) -> bool:
    """What app.config makes of this .env LINE, through the parser it uses.

    The line is written to a real file and read back with python-dotenv -- the
    parser `load_dotenv()` uses, and the thing that strips the quotes, the `export`
    prefix and the blanks around the `=` before any value reaches `_env_tristate`.
    Asking about a LINE rather than about a value is the whole point: a value cannot
    be spelled two ways, and a test that only ever builds `f"{FLAG}={value}"` cannot
    see a shape the guard is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / ".env"
        env_file.write_text(line + "\n", encoding="utf-8")
        value = dotenv_values(env_file).get(FLAG)
    previous = os.environ.pop(FLAG, None)
    try:
        if value is not None:
            os.environ[FLAG] = value
        return _env_tristate(FLAG) is True
    finally:
        os.environ.pop(FLAG, None)
        if previous is not None:
            os.environ[FLAG] = previous


@pytest.mark.parametrize("line", FORCED_TRUE_LINES)
def test_every_forced_true_line_shape_warns(line):
    """A forced True in any shape is the forgeable posture, so all of them warn.

    The value half of this was covered first; the shape half was not, and
    `KEY="true"` reached the application as a forced True while the guard matched
    nothing at all."""
    (warned, untouched, _), = run_xff_guard([line])
    assert _config_reads_as_forced_true(line), (
        f"{line!r} is not a forced true to the app, so it is not this test's case"
    )
    assert warned, f"the operator wrote {line!r} and setup.sh warned about nothing"
    assert untouched, f"setup.sh must not touch an operator's value: {line!r}"


def test_the_quoted_and_spaced_lines_of_issue_388_warn():
    """The quoted and spaced spellings end to end.

    `AUTH_TRUST_X_FORWARDED_FOR="true"` and `AUTH_TRUST_X_FORWARDED_FOR = true` are
    read as a forced True by the application and matched by nothing in the script,
    which is a warning that fires only for the spelling nobody bothers to write.
    Asserted as a unit because these two are the reported defect; the parametrized
    test above carries the rest of the shapes."""
    lines = [f"{FLAG}=true", f'{FLAG}="true"', f"{FLAG} = true"]
    results = run_xff_guard(lines)
    assert [warned for warned, _, _ in results] == [True, True, True], results
    assert all(_config_reads_as_forced_true(line) for line in lines), (
        "the app no longer reads any of these as a forced true; this test's "
        "premise has changed and the shapes need re-measuring"
    )


@pytest.mark.parametrize("line", KEY_PRESENT_SILENT_LINES)
def test_a_value_that_is_not_a_forced_true_stays_silent(line):
    """`auto`, a falsy spelling and a typo are not the forgeable posture, and a
    warning the operator cannot act on trains them to skip the one that matters.

    The last four here are the precision cases: python-dotenv either refuses the
    line or keeps the punctuation as part of the value."""
    (warned, untouched, _), = run_xff_guard([line])
    assert not _config_reads_as_forced_true(line), (
        f"{line!r} IS a forced true to the app; move it to FORCED_TRUE_LINES"
    )
    assert not warned, f"setup.sh warned about {line!r}, which is not a forced true"
    assert untouched, f"setup.sh rewrote the operator's value: {line!r}"


@pytest.mark.parametrize("line", KEY_ABSENT_LINES)
def test_a_missing_key_is_appended_once_and_never_warned_about(line, tmp_path):
    """The absent case is the one case setup.sh may write to.

    A .env that predates the per-IP rate limits has no trust setting at all, so
    every proxied request keys on the nginx peer and the whole site shares one
    rate-limit bucket; appending the shipped default is what closes that. Run twice,
    because a presence check that keeps missing appends a second `=auto` on every
    deploy and python-dotenv resolves a repeated key to the last one. A commented-out
    line and a longer key that starts the same way both read as absent, and
    commenting the knob out is how an operator turns it off."""
    env_file = tmp_path / ".env"
    env_file.write_text(line + "\n", encoding="utf-8", newline="\n")
    script = "\n".join(
        [
            _xff_function(),
            f'{XFF_FUNCTION} "$ENV_FILE"',
            f'{XFF_FUNCTION} "$ENV_FILE"',
        ]
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "ENV_FILE": str(env_file)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", (
        f"setup.sh warned about a .env with no {FLAG} assignment: {line!r}\n{proc.stderr}"
    )
    assert env_file.read_text(encoding="utf-8").splitlines() == [line, APPEND_LINE], (
        f"setup.sh must append {APPEND_LINE} exactly once and leave the "
        f"operator's line alone, got {env_file.read_text(encoding='utf-8').splitlines()}"
    )


def _candidate_spellings():
    """A generated token space, not a hand-written list of the known spellings.

    The lists above pin the values anyone thought of and cannot catch one nobody
    did, which is how ``y`` reached the parser's truthy set while every test here
    stayed green: the guard would then stop warning about a value that forces
    X-Forwarded-For to be trusted from any peer."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 -_."
    # Every one- and two-character token: that already contains every spelling
    # a person would write, plus all the near-misses worth a warning.
    space = {"".join(p) for n in (1, 2) for p in product(alphabet, repeat=n)}
    narrow = "tfynos01 -"
    space |= {"".join(p) for p in product(narrow, repeat=3)}
    # Shapes a fixed-width sweep cannot produce.
    space |= {
        "", " ", "\t", "true ", " true", "  true  ", "TRUE", "True", "On", "ON",
        "1 2", "yes-no", "auto", "maybe", "enabled", "disable", "y", "t",
    }
    return space


def test_guard_and_parser_agree_on_every_spelling_in_both_directions():
    """setup.sh's guard must match precisely the values config reads as True,
    bidirectionally, because either drift is a defect in opposite directions:

    * the parser is widened (a spelling becomes forced-True) and the guard is not,
      so the operator is left trusting X-Forwarded-For from any peer with no warning;
    * the guard is widened, so the operator is nagged about a posture they are not in.

    Derived from ``_env_tristate`` reading a line through python-dotenv -- the same
    two steps app/config.py performs -- rather than from the literals at the top of
    this file, so neither side can hold its own idea of "forced true"."""
    candidates = [
        template.format(key=FLAG, value=value)
        for template in LINE_TEMPLATES
        for value in _candidate_spellings()
    ]
    results = run_xff_guard(candidates)
    wrong_way = [
        line for warned, _, line in results if warned and not _config_reads_as_forced_true(line)
    ]
    missing = [
        line for warned, _, line in results if not warned and _config_reads_as_forced_true(line)
    ]
    assert not wrong_way and not missing, (
        f"setup.sh's forced-true warning and config's truthy set disagree over "
        f"{len(candidates)} lines.\n"
        f"  warns but config does not read a forced True: {sorted(wrong_way)[:20]}\n"
        f"  a forced True with no warning, see #245: {sorted(missing)[:20]}\n"
        f"  truthy spellings are {sorted(_TRUE_SPELLINGS)}"
    )


def test_the_parser_and_the_guard_share_one_spelling_set():
    """The application must not be the only side with a list of its own.

    setup.sh's guard is a third copy of ``_TRUE_SPELLINGS`` in a language that
    cannot import the first two -- it runs before the venv is guaranteed to exist.
    That is a real coupling, so it is pinned here rather than left to a comment in
    the shell."""
    assert not _TRUE_SPELLINGS & _FALSE_SPELLINGS
    candidates = [
        template.format(key=FLAG, value=spelling)
        for template in LINE_TEMPLATES
        for spelling in sorted(_TRUE_SPELLINGS)
    ]
    for warned, _, line in run_xff_guard(candidates):
        assert warned, f"setup.sh does not warn about {line!r}"


def test_run_backend_still_calls_the_migration():
    """A guard nothing calls prints nothing.

    The rest of this section executes `migrate_xff_trust` directly, so the call
    site is checked here. The value is passed on, not read from a global, because a
    function that reaches into `$ENV_FILE` cannot be run against a throwaway file."""
    source = SETUP_SH.read_text()
    body = re.search(
        r"^run_backend\(\) \{\n(?P<body>.*?)\n\}", source, re.MULTILINE | re.DOTALL
    )
    assert body, "could not find run_backend() in setup.sh"
    assert re.search(rf'^\s*{XFF_FUNCTION} "\$ENV_FILE"$', body.group("body"), re.MULTILINE), (
        f"run_backend() never calls {XFF_FUNCTION} \"$ENV_FILE\", so the .env it "
        "just created is never repaired and never warned about"
    )



def test_the_nginx_vhost_refuses_the_uncached_probe():
    """`location /ready` is a PREFIX match, so without an explicit block a public
    GET /ready/deep is proxied to the API and is held dark only by the app's
    host-local check. /ready/deep is uncached and unrated by design, so a public
    dependency-probe amplifier deserves a second layer."""
    source = SETUP_SH.read_text()

    assert "location /ready/deep { return 404; }" in source


def test_the_cron_entry_probes_the_port_the_api_is_actually_bound_to():
    """API_PORT is a documented override and run_services binds pm2 to
    127.0.0.1:$API_PORT, while the watchdog's own default is 8001. A cron entry that
    omits BASE leaves it probing a closed port on any other port, and a watchdog
    that cannot reach the backend restarts it every five minutes."""
    source = SETUP_SH.read_text()
    cron_lines = [line for line in source.splitlines() if "deploy/healthcheck.sh" in line]

    assert cron_lines, "the cron entry must still be installed by setup.sh"
    entry = cron_lines[0]
    assert "API_PORT" in entry, f"the cron entry must derive BASE from API_PORT: {entry!r}"
    assert "BASE=" in entry


def test_the_nginx_vhost_heredoc_contains_no_backticks():
    """The vhost is written through an UNQUOTED heredoc, so a backtick anywhere in
    it is a command substitution and setup.sh writes the mangled result into the
    deployed vhost. Quoting the heredoc delimiter is not available, since $API_PORT
    and the \\$remote_addr escapes depend on the unquoted form."""
    body = SETUP_SH.read_text().split("<<NGINX\n", 1)[1].split("\nNGINX\n", 1)[0]

    assert "`" not in body, "a backtick in the unquoted NGINX heredoc is executed"


# --- the deploy gate must wait on readiness, and must not destroy the deploy ---
#
# `run_services` is EXECUTED here rather than grepped: asserting the source contains
# a URL is the vacuous-test class this project keeps policing -- it passes just as
# well with the gate moved in front of the teardown. The functions are extracted
# verbatim from setup.sh, so this cannot drift from the script it is policing.

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
    """Run the real run_services against stub pm2/curl. Returns (rc, pm2_log, stdout+stderr)."""
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
            'API_PORT=8001; NEXT_PORT=3000; GUNICORN_WORKERS=4; MIN_UPTIME_MS=30000',
            'API_MAX_MEMORY=5G; API_MAX_RESTARTS=10; FRONTEND_MAX_MEMORY=1G; RESTART_BACKOFF_MS=100',
            f'LOGS={home}/logs; PID_DIR={home}/pid; ENV_FILE={home}/backend/.env',
            f'VENV_PY={home}/bin/python; SCRIPT_DIR={home}',
            "sleep() { :; }",  # wait_http's 1s backoff would cost 30s per run
            "stage() { :; }",
            "ensure_pm2() { :; }",
            # run_services calls these, so the function under test can be the real
            # one. MIN_UPTIME_MS is the same reason: it is exported to `pm2 start`,
            # and `set -u` turns a missing one into a subshell that never reaches pm2.
            _extract_function("env_value"),
            _extract_function("harden_permissions"),
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
    """Did this run register `service` with pm2?

    `run_services` starts pm2 ONCE, from `ecosystem.config.js`, so the evidence is
    both halves: the invocation pm2 actually received (recorded by the stub) and the
    app that file defines under this name. Either half alone would pass for the
    wrong reason."""
    started = any(
        "start" in line and "ecosystem.config.js" in line
        for line in pm2_log.splitlines()
    )
    return started and any(app["name"] == service for app in _ecosystem_apps())


def test_a_failing_readiness_gate_does_not_leave_the_frontend_stopped(tmp_path):
    """With the gate in front of the frontend start, a not-ready deploy took the
    frontend down with it and `set -e` aborted before bringing it back. Both services
    must be started, and the dump saved, before anything is allowed to fail."""
    proc, pm2_log = run_services(tmp_path, ready_deep_code=503)

    assert "RUN_SERVICES_RC=1" in proc.stdout
    assert _started(pm2_log, "vccircle-backend")
    assert _started(pm2_log, "vccircle-frontend"), "a failed readiness gate must not stop the frontend from being deployed"
    assert "save" in pm2_log, "the pm2 dump must be saved before the gate can abort"
    assert "not ready after" in proc.stderr + proc.stdout


def test_the_gate_is_not_satisfied_by_the_liveness_stub(tmp_path):
    proc, pm2_log = run_services(tmp_path, ready_deep_code=503)

    assert "RUN_SERVICES_RC=1" in proc.stdout
    assert _started(pm2_log, "vccircle-frontend")


def test_a_ready_deployment_still_succeeds(tmp_path):
    """The control: the reordering must not have broken the happy path."""
    proc, pm2_log = run_services(tmp_path, ready_deep_code=200)

    assert "RUN_SERVICES_RC=0" in proc.stdout, proc.stdout + proc.stderr
    assert _started(pm2_log, "vccircle-backend")
    assert _started(pm2_log, "vccircle-frontend")


@needs_node
def test_the_frontend_is_registered_with_a_loopback_bind(tmp_path):
    """`run_services` must give the frontend a loopback bind, not just mention one.

    `run_services` starts pm2 from `ecosystem.config.js`, so the argv the frontend
    actually runs is what that file RESOLVES to once setup.sh's exports are applied
    -- and this resolves it with node rather than reading the text. `next start`
    binds every interface when no hostname is passed, so the flag is the whole
    control."""
    proc, pm2_log = run_services(tmp_path, ready_deep_code=200)

    assert "RUN_SERVICES_RC=0" in proc.stdout, proc.stdout + proc.stderr
    assert any(
        "start" in line and "ecosystem.config.js" in line
        for line in pm2_log.splitlines()
    ), f"pm2 was never asked to start the frontend:\n{pm2_log}"

    argv = _ecosystem_app("vccircle-frontend")["args"]

    assert "-H 127.0.0.1" in argv, (
        f"pm2 registers the frontend without a loopback bind, so `next start` "
        f"would listen on the wildcard: {argv!r}"
    )
    # The port must be the one the nginx template proxies to, or the site goes
    # down; run_services is invoked here with the shipped NEXT_PORT default.
    assert "-p 3000" in argv, f"the frontend is registered on an unexpected port: {argv!r}"


def test_a_gateway_deployment_with_a_non_google_key_is_not_blocked(tmp_path):
    """setup.sh must not judge the key at all, so it cannot refuse a deploy the app
    considers ready.

    GEMINI_BASE_URL is configurable, so a non-Google-shaped key is a supported
    configuration, and while this script had its own pre-flight that was one of the
    ways it could block a working deploy. The verdict now comes from the app, and
    this pins that setup.sh stays out of it."""
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
# A bash re-implementation of app.config.classify_gemini_api_key carried only 20
# of the classifier's 28 sentinels, and the eight it missed are all reachable
# when GEMINI_BASE_URL is a non-Google gateway. What is tested instead is the
# property that matters, against every sentinel the app rejects INCLUDING any
# added later: a not-ready deploy must never leave the frontend stopped.


def _all_app_placeholders():
    """Every value classify_gemini_api_key calls a placeholder, plus the
    repeated-filler shapes it rejects outside the sentinel list. Read from the
    app so a new sentinel cannot escape this test."""
    from app.config import _PLACEHOLDER_API_KEYS

    return sorted(_PLACEHOLDER_API_KEYS | {"x" * 9, "a-x-a-x-a-x", "0" * 9})


@pytest.mark.parametrize("placeholder", _all_app_placeholders())
def test_every_placeholder_the_app_rejects_still_deploys_both_services(tmp_path, placeholder):
    """Whatever the app makes of the key, the deploy is not left torn down: both
    services are started and the dump is saved before the gate can fail. The base
    URL is a non-Google gateway on purpose -- that is the path on which the
    shell-side shape rule was skipped."""
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
    """The operator has to be told WHICH fault, and the only trustworthy source is
    the app: report_readiness_reason prints the response body, which carries
    checks.llm.reason. Its curl deliberately has no -f -- with -f the body is
    suppressed on exactly the >=400 response this function only ever sees."""
    proc, _pm2_log = run_services(tmp_path, env_key=PLACEHOLDER_KEY, ready_deep_code=503)
    output = proc.stdout + proc.stderr

    assert "not ready after" in output
    assert "placeholder" in output, f"the readiness reason never reached the operator: {output}"
    assert "readiness report" in output


def test_a_failed_gate_says_so_even_when_the_probe_returns_no_body(tmp_path):
    """A probe that cannot be reached at all must not print an empty report silently;
    the operator is told the difference between 'not ready' and 'no answer'."""
    # A refused connection: curl fails and writes no body (the string matters,
    # 000 is curl's "no answer" code -- the int 0 would str() to "0").
    proc, _pm2_log = run_services(tmp_path, ready_deep_code="000", report="")

    assert "no report body" in proc.stderr
    assert "not ready after" in proc.stderr


# --- run_cron must converge on one healthcheck entry ------------------------
#
# run_cron reconciles the crontab by removing the lines it manages and re-adding
# its own. When the managed healthcheck line's TEXT changes, a `grep -vFx`
# (whole-line) filter stops matching the entry a previous revision wrote, so the
# stale copy survives every run: on a host with a non-default API_PORT it carries no
# BASE=, falls back to the watchdog's :8001 default, is read as "not alive" and
# restarts a perfectly healthy backend every five minutes.

CRONTAB_STUB = """\
#!/usr/bin/env bash
# `crontab -l` prints; `crontab <file>` installs. Never the real binary.
if [ "$1" = "-l" ]; then
    [ -f "$CRONTAB_FILE" ] || exit 1
    cat "$CRONTAB_FILE"
    exit 0
fi
cp "$1" "$CRONTAB_FILE"
exit 0
"""

INDEXER_PATH_MARKER = "update_index.py"


def _hc_entries(crontab_text):
    return [line for line in crontab_text.splitlines() if "deploy/healthcheck.sh" in line]


def _indexer_entries(crontab_text):
    return [line for line in crontab_text.splitlines() if INDEXER_PATH_MARKER in line]


def run_cron(tmp_path, seeded_lines, api_port="9001", runs=1):
    """Run the real run_cron `runs` times against a seeded crontab."""
    home = tmp_path / "host"
    (home / "logs").mkdir(parents=True)
    (home / "bin").mkdir()
    crontab_file = tmp_path / "crontab.txt"
    crontab_file.write_text("".join(line + "\n" for line in seeded_lines))
    stub = home / "bin" / "crontab"
    stub.write_text(CRONTAB_STUB)
    stub.chmod(0o755)

    script_dir = home / "app"
    (script_dir / "deploy").mkdir(parents=True)
    harness = "\n".join(
        [
            "set -euo pipefail",
            f"cd {home}",
            f"LOGS={home}/logs; SCRIPT_DIR={script_dir}; API_PORT={api_port}",
            f'VENV_PY={home}/bin/python',
            "stage() { :; }",
            _extract_function("run_cron"),
            "\n".join(["run_cron"] * runs),
        ]
    )
    proc = subprocess.run(
        ["bash", "-c", harness],
        env={**os.environ, "PATH": f"{home}/bin:{os.environ['PATH']}", "CRONTAB_FILE": str(crontab_file)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc, crontab_file.read_text()


def _previous_revision_entry(script_dir, log, api_port="9001"):
    """The entry run_cron writes with the same script but no BASE= -- the literal a
    whole-line filter can no longer match."""
    return f'*/5 * * * * HEALTHCHECK_WEBHOOK_URL="" LOG={log} {script_dir}/deploy/healthcheck.sh'


def test_run_cron_removes_the_entry_a_previous_revision_wrote(tmp_path):
    """A stale entry that carries no BASE= would probe :8001 and restart a healthy
    backend every five minutes. It must be gone, not merely joined by a second
    copy."""
    log = tmp_path / "healthcheck.log"
    script_dir = tmp_path / "host" / "app"
    proc, crontab = run_cron(
        tmp_path,
        seeded_lines=[_previous_revision_entry(script_dir, log)],
        api_port="9001",
    )

    assert proc.returncode == 0, proc.stderr
    entries = _hc_entries(crontab)
    assert len(entries) == 1, f"expected convergence to one entry, got {entries}"
    assert 'BASE="http://localhost:9001"' in entries[0], entries[0]


def test_run_cron_is_idempotent(tmp_path):
    proc, crontab = run_cron(tmp_path, seeded_lines=[], api_port="9001", runs=3)

    assert proc.returncode == 0, proc.stderr
    assert len(_hc_entries(crontab)) == 1
    assert len(_indexer_entries(crontab)) == 1


def test_run_cron_removes_a_duplicate_healthcheck_entry(tmp_path):
    log = tmp_path / "healthcheck.log"
    script_dir = tmp_path / "host" / "app"
    current = f'*/5 * * * * BASE="http://localhost:9001" HEALTHCHECK_WEBHOOK_URL="" LOG={log} {script_dir}/deploy/healthcheck.sh'
    _proc, crontab = run_cron(
        tmp_path,
        seeded_lines=[_previous_revision_entry(script_dir, log), current],
        api_port="9001",
    )

    assert len(_hc_entries(crontab)) == 1


def test_run_cron_preserves_a_users_own_crontab_lines(tmp_path):
    """The filter removes entries that run OUR script, not the user's crontab."""
    user_line = "0 3 * * * /usr/local/bin/backup.sh"
    proc, crontab = run_cron(tmp_path, seeded_lines=[user_line], api_port="9001")

    assert proc.returncode == 0, proc.stderr
    assert user_line in crontab.splitlines()
    assert len(_hc_entries(crontab)) == 1


def test_run_cron_preserves_a_hand_edited_indexer_entry(tmp_path):
    """The indexer line is still matched exactly, so a user who tweaked its
    schedule keeps their version -- the asymmetry is deliberate."""
    hand_edited = "*/20 * * * * nice -n 5 /venv/bin/python /app/backend/scripts/update_index.py >> /var/log/idx.log 2>&1"
    proc, crontab = run_cron(tmp_path, seeded_lines=[hand_edited], api_port="9001")

    assert proc.returncode == 0, proc.stderr
    assert hand_edited in crontab.splitlines()
    assert len(_indexer_entries(crontab)) == 2, "the hand-edited copy is kept and the managed one is added"


def test_run_cron_reclaims_a_users_own_healthcheck_line_by_design(tmp_path):
    """The accepted cost of filtering on the script path, pinned so it stays a decision.

    A user who added their own healthcheck.sh entry on a custom schedule has it
    replaced by the managed one. That is deliberate: `./setup.sh cron` is an
    explicit operator action and reclaims lines running a script it manages -- a
    stale entry from a previous revision is exactly what the convergence test above
    covers."""
    log = tmp_path / "healthcheck.log"
    script_dir = tmp_path / "host" / "app"
    custom = f"0 * * * * LOG={log} {script_dir}/deploy/healthcheck.sh"
    proc, crontab = run_cron(tmp_path, seeded_lines=[custom], api_port="9001")

    assert proc.returncode == 0, proc.stderr
    assert custom not in crontab.splitlines(), "the custom entry is expected to be reclaimed"
    entries = _hc_entries(crontab)
    assert len(entries) == 1
    assert entries[0].startswith("*/5 * * * *"), entries[0]


def test_run_cron_keeps_a_commented_out_healthcheck_entry(tmp_path):
    """Commenting the entry out is how an operator switches the watchdog off, so that
    marker must survive. A plain substring `grep -vF` on the script path deleted it,
    which would silently re-arm a watchdog the operator believes is disabled."""
    script_dir = tmp_path / "host" / "app"
    disabled = f"# disabled for now: {script_dir}/deploy/healthcheck.sh"
    proc, crontab = run_cron(tmp_path, seeded_lines=[disabled], api_port="9001")

    assert proc.returncode == 0, proc.stderr
    assert disabled in crontab.splitlines(), "a commented-out entry is the user's disable marker, not a managed line"


def test_run_cron_still_removes_an_active_healthcheck_entry_among_comments(tmp_path):
    """The narrowing must not become a loophole: an ACTIVE line that runs our
    script is still reclaimed even when comments sit around it."""
    log = tmp_path / "healthcheck.log"
    script_dir = tmp_path / "host" / "app"
    active = f"*/7 * * * * LOG={log} {script_dir}/deploy/healthcheck.sh"
    comment = f"# an old copy of the watchdog: {script_dir}/deploy/healthcheck.sh"
    proc, crontab = run_cron(tmp_path, seeded_lines=[active, comment], api_port="9001")

    assert proc.returncode == 0, proc.stderr
    assert active not in crontab.splitlines()
    assert comment in crontab.splitlines()
    active_lines = [line for line in crontab.splitlines() if not line.lstrip().startswith("#")]
    assert len(_hc_entries("\n".join(active_lines))) == 1
