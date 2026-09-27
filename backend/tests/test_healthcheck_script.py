"""Executes deploy/healthcheck.sh against stub `curl` and `pm2` binaries.

The watchdog is the consumer this issue is about: it used to probe /health, an
endpoint that cannot fail, so a dead Qdrant client, unloaded models or the
placeholder GEMINI_API_KEY shipped in .env.example were invisible to it and it
exited 0 forever. Asserting the script's text would not catch that (the text
already said "health"), so these tests run it and assert on what it decided to
do: restart, or not.

Each test gets a HOME of its own, which is where the script looks for `curl` and
`pm2` on PATH and where it writes its log, so nothing here touches the real
ones.
"""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/healthcheck.sh"

# Both doubles live in $HOME/.local/bin, which the script puts first on PATH
# precisely because cron has a minimal one.
CURL_STUB = """\
#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >>"$CURL_LOG"
for a in "$@"; do
    if [ "$a" = "-X" ]; then method=1; fi
done
if [ "${method:-0}" = "1" ]; then
    # Record the method, URL and arguments so a webhook delivery is observable.
    printf 'WEBHOOK %s %s\n' "$url" "$*" >>"$CURL_LOG"
    if [ -n "$WEBHOOK_FAILS" ]; then exit 7; fi
    printf 'ok'
    exit 0
fi
case "$url" in
*/health)
    if [ -f "$RESTARTED_MARKER" ]; then code="$HEALTH_AFTER_RESTART"; else code="$HEALTH_CODE"; fi
    ;;
*/ready/deep) code="$READY_CODE" ;;
*) code="000" ;;
esac
printf '%s' "$code"
# -f semantics: a non-2xx answer is an error for the caller, even though the
# http_code is still written to stdout.
[ "$code" -ge 400 ] 2>/dev/null && exit 22
exit 0
"""

PM2_STUB = """\
#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$PM2_LOG"
: >"$RESTARTED_MARKER"
exit 0
"""


class WatchdogRun:
    def __init__(self, proc, home, curl_log, pm2_log, log):
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        self.curl_log = curl_log
        self.pm2_log = pm2_log
        self.log = log

    @property
    def webhooks(self):
        return [line for line in self.curl_log.splitlines() if line.startswith("WEBHOOK ")]

    @property
    def probes(self):
        return [line for line in self.curl_log.splitlines() if line]

    @property
    def restarts(self):
        return [line for line in self.pm2_log.splitlines() if line]


_RUN_COUNTER = [0]


def run_watchdog(
    tmp_path,
    *,
    health="200",
    health_after_restart="200",
    ready="200",
    webhook="",
    state_file=None,
    cooldown="3600",
    webhook_fails="",
):
    _RUN_COUNTER[0] += 1
    home = tmp_path / f"run{_RUN_COUNTER[0]}" / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    for name, body in (("curl", CURL_STUB), ("pm2", PM2_STUB)):
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)

    curl_log = tmp_path / "curl.log"
    pm2_log = tmp_path / "pm2.log"
    log = tmp_path / "healthcheck.log"
    marker = tmp_path / "restarted"
    env = {
        **os.environ,
        "HOME": str(home),
        "BASE": "http://localhost:8001",
        "LOG": str(log),
        "CURL_LOG": str(curl_log),
        "PM2_LOG": str(pm2_log),
        "RESTARTED_MARKER": str(marker),
        "HEALTH_CODE": str(health),
        "HEALTH_AFTER_RESTART": str(health_after_restart),
        "READY_CODE": str(ready),
        # The script's real default is 8s; a test must not spend it.
        "RESTART_WAIT_SECONDS": "0",
        "HEALTHCHECK_WEBHOOK_URL": webhook,
        "STATE_FILE": str(state_file) if state_file else str(tmp_path / "state"),
        "ALERT_COOLDOWN_SECONDS": str(cooldown),
        "WEBHOOK_FAILS": webhook_fails,
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # the watchdog's exit code IS the assertion
    )
    return WatchdogRun(
        proc,
        home,
        curl_log.read_text() if curl_log.exists() else "",
        pm2_log.read_text() if pm2_log.exists() else "",
        log.read_text() if log.exists() else "",
    )


def test_a_healthy_backend_is_silent_and_untouched(tmp_path):
    run = run_watchdog(tmp_path, health="200", ready="200")

    assert run.returncode == 0
    assert run.stdout == ""
    assert run.restarts == [], "a healthy backend must never be restarted"


def test_a_dependency_outage_alerts_instead_of_restarting_a_live_backend(tmp_path):
    """The failure this issue is about: the process is up and the service is
    not. Restarting cannot bring Qdrant back or fix a placeholder key, and
    restart-storms are their own outage -- so alert, and do not restart."""
    run = run_watchdog(tmp_path, health="200", ready="503")

    assert run.returncode == 1
    assert run.restarts == [], "a live process must not be restarted for a dependency fault"
    assert "/ready/deep" in run.stdout, "the alert must name the probe that failed"
    assert "GEMINI_API_KEY" in run.stdout, "the alert must point at what a readiness 503 usually means"
    assert "ALERT" in run.log


def test_the_watchdog_asks_for_readiness_not_just_liveness(tmp_path):
    """Both probes are issued: /health is the restart decision, /ready/deep is
    the health decision, and neither one alone is the answer."""
    run = run_watchdog(tmp_path, health="200", ready="200")

    assert any(p.endswith("/health") for p in run.probes)
    assert any(p.endswith("/ready/deep") for p in run.probes)

def test_a_dead_process_is_restarted_and_reported_down_when_it_stays_down(tmp_path):
    # The restart does not bring it back, so the alert must name liveness --
    # the fault a restart could not fix.
    run = run_watchdog(tmp_path, health="000", health_after_restart="000", ready="000")

    assert run.returncode == 1
    assert len(run.restarts) == 1
    assert run.restarts[0] == "restart vccircle-backend --update-env"
    assert "/health" in run.stdout


def test_a_dead_process_that_comes_back_is_reported_recovered(tmp_path):
    run = run_watchdog(tmp_path, health="000", health_after_restart="200", ready="200")

    assert run.returncode == 0
    assert run.stdout == ""
    assert len(run.restarts) == 1
    assert "recovered after restart" in run.log


def test_a_connection_refused_watchdog_still_reaches_a_verdict(tmp_path):
    """curl writes 000 on a refused connection, which the script must read as
    "not alive" rather than as an empty answer."""
    run = run_watchdog(tmp_path, health="", ready="")

    assert run.returncode == 1
    assert len(run.restarts) == 1


def test_the_script_is_valid_bash():
    """Cheap, and it is the failure mode a stray edit to a shell script
    produces: the watchdog silently does nothing, every time."""
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30, check=False)

    assert proc.returncode == 0, textwrap.indent(proc.stderr, "  ")


def test_a_refused_readiness_probe_is_not_reported_as_a_dependency_failure(tmp_path):
    """403/404 is what the API answers when the probe cannot be reached the way
    it requires -- a reverse proxy in front, or a BASE pointing at the wrong
    port. That is a fact about BASE, not about Qdrant or the API key, and an
    alert naming the dependencies sends the operator after the wrong thing."""
    run = run_watchdog(tmp_path, health="200", ready="403")

    assert run.returncode == 1
    assert run.restarts == []
    assert "REFUSED" in run.stdout
    assert "GEMINI_API_KEY" not in run.stdout, "a refused probe is not a dependency verdict"
    assert "Qdrant" not in run.stdout


def test_a_real_readiness_failure_names_the_dependencies(tmp_path):
    run = run_watchdog(tmp_path, health="200", ready="503")

    assert "GEMINI_API_KEY" in run.stdout
    assert "Qdrant" in run.stdout


def test_a_broken_probe_says_so_instead_of_guessing(tmp_path):
    run = run_watchdog(tmp_path, health="200", ready="500")

    assert run.returncode == 1
    assert "failed internally" in run.stdout
    assert "GEMINI_API_KEY" not in run.stdout


# --- alerting once per fault, not once per run (the */5 cron cadence) ---


def test_a_persistent_fault_alerts_once_and_then_stays_quiet(tmp_path):
    """At the shipped */5 cadence a fault that is never fixed is ~288 identical
    webhook POSTs and ~288 cron mails a day. That is how the one alert that
    matters gets ignored, so the repeat is logged once and otherwise silent --
    while the exit code still reports the fault on every single run."""
    state = tmp_path / "state"
    first = run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    second = run_watchdog(tmp_path, health="200", ready="503", state_file=state)

    assert "ALERT" in first.stdout
    assert "still failing" in second.log
    assert "ALERT" not in second.stdout, "a repeat fault must not re-alert inside the cooldown"
    assert second.returncode == 1, "the run must still report the fault on its exit code"


def test_the_cooldown_expires_and_the_fault_is_reported_again(tmp_path):
    state = tmp_path / "state"
    run_watchdog(tmp_path, health="200", ready="503", state_file=state, cooldown="3600")
    again = run_watchdog(tmp_path, health="200", ready="503", state_file=state, cooldown="0")

    assert "ALERT" in again.stdout


def test_a_different_fault_always_re_alerts(tmp_path):
    """Suppression is per fault, not a blanket mute: a Qdrant outage that
    becomes a dead process is new information and must get through."""
    state = tmp_path / "state"
    run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    # The process now does not come back at all: a different fault, and one the
    # restart path has to report.
    other = run_watchdog(tmp_path, health="000", health_after_restart="000", ready="000", state_file=state)

    assert "ALERT" in other.stdout
    assert "down" in other.stdout or "ALERT: VCCircle backend down" in other.stdout


def test_a_healthy_run_clears_the_fault_state(tmp_path):
    """Otherwise the cooldown would mute the NEXT outage as well, which is the
    failure mode a naive "don't spam the webhook" fix always introduces."""
    state = tmp_path / "state"
    run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    assert state.exists()

    run_watchdog(tmp_path, health="200", ready="200", state_file=state)

    assert not state.exists()
    after_recovery = run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    assert "ALERT" in after_recovery.stdout, "a fault after a healthy run must alert immediately"


# --- the webhook path ---


def test_a_fault_is_posted_to_the_webhook_with_the_alert_text(tmp_path):
    run = run_watchdog(tmp_path, health="200", ready="503", webhook="https://hooks.example/abc")

    assert len(run.webhooks) == 1
    assert "https://hooks.example/abc" in run.webhooks[0]
    assert "POST" in run.webhooks[0]
    assert "GEMINI_API_KEY" in run.webhooks[0], "the alert body must be what the operator reads"


def test_a_healthy_backend_posts_nothing(tmp_path):
    run = run_watchdog(tmp_path, health="200", ready="200", webhook="https://hooks.example/abc")

    assert run.webhooks == []


def test_a_failing_webhook_never_swallows_the_alert(tmp_path):
    """The webhook is best effort and cannot decide anything: it is only ever
    posted on a path that has already failed, so the failure has to be visible
    in the log and the alert still has to reach the operator's terminal. An
    alerting service that is down must not turn a real outage into a silent one.
    """
    faulted = run_watchdog(tmp_path, health="200", ready="503", webhook="https://hooks.example/x", webhook_fails="1")

    assert faulted.returncode == 1
    assert "ALERT" in faulted.stdout, "a failed webhook must not swallow the alert"
    assert "webhook delivery failed" in faulted.log
    assert len(faulted.webhooks) == 1, "the delivery must have been attempted, not skipped"


def test_a_healthy_backend_never_uses_the_webhook(tmp_path):
    """So the webhook cannot fail a run that is passing, whatever the
    alerting service is doing."""
    healthy = run_watchdog(tmp_path, health="200", ready="200", webhook="https://hooks.example/x", webhook_fails="1")

    assert healthy.returncode == 0
    assert healthy.webhooks == []


# --- the state file is a cache, never a gate -------------------------------
#
# Suppressing a repeat alert is only safe if the state can never be the reason
# an alert is lost. Both of these were patch-introduced and both are worse than
# a noisy watchdog: a clock that moved backwards silenced the alert for longer
# than the configured cooldown, and a corrupt stamp aborted the run from inside
# alert() under `set -u` -- before log, before the webhook, before the alert.


def _state(text, tmp_path):
    state = tmp_path / "state"
    state.write_text(text)
    return state


def test_a_future_timestamp_does_not_suppress_the_alert(tmp_path):
    """An ntp/DST slip or a copied state file leaves a stamp ahead of us. That is
    not a cooldown, so it must not silence a real outage -- the previous
    arithmetic even reported suppressing for LONGER than the cooldown."""
    state = _state("live-not-ready:503 4102444800\n", tmp_path)  # 2100-01-01

    run = run_watchdog(tmp_path, health="200", ready="503", state_file=state, cooldown="3600")

    assert "ALERT" in run.stdout
    assert "suppressed" not in run.log
    assert run.returncode == 1


@pytest.mark.parametrize(
    "corrupt",
    ["live-not-ready:503 notanumber\n", "live-not-ready:503\n", "live-not-ready:503 -5\n", "live-not-ready:503 12.5\n"],
    ids=["word", "empty-stamp", "negative", "decimal"],
)
def test_a_corrupt_state_file_cannot_silence_the_alert(tmp_path, corrupt):
    """A non-numeric stamp expanded unquoted aborts the run under `set -u` from
    inside alert(), which killed the watchdog with no log line, no webhook and
    no alert. Anything that is not a plain non-negative integer means "never
    alerted"."""
    state = _state(corrupt, tmp_path)

    run = run_watchdog(tmp_path, health="200", ready="503", state_file=state, webhook="https://hooks.example/x")

    assert run.returncode == 1
    assert "ALERT" in run.stdout, "a corrupt state file must not swallow the alert"
    assert "unbound variable" not in run.stdout + run.stderr
    assert len(run.webhooks) == 1
    written = state.read_text().split()
    assert len(written) == 2 and written[1].isdigit(), f"the state was not rewritten with a numeric stamp: {written!r}"


def test_an_empty_state_file_is_treated_as_never_alerted(tmp_path):
    """A torn write leaves an empty file (now also prevented by the atomic
    write, but an operator can truncate one by hand)."""
    state = _state("", tmp_path)

    run = run_watchdog(tmp_path, health="200", ready="503", state_file=state)

    assert "ALERT" in run.stdout
    assert run.returncode == 1


def test_a_state_write_failure_still_alerts(tmp_path):
    """The state is a cache. Losing it must cost a duplicate alert at worst, not
    a silent one: the state file is made undirectory-uncopyable by pointing it
    at a path that cannot be created."""
    run = run_watchdog(
        tmp_path,
        health="200",
        ready="503",
        state_file=tmp_path / "state" / "nested" / "state",
        webhook="https://hooks.example/x",
    )

    assert "ALERT" in run.stdout, "a state write failure must not swallow the alert"
    assert run.returncode == 1
    assert len(run.webhooks) == 1
    # The operator is told deduplication is now broken, rather than the next
    # run silently re-alerting (or, worse, silently not).
    assert "will not be deduplicated" in run.log
