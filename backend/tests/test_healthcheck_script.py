"""Runs deploy/healthcheck.sh against stubbed curl/pm2 binaries and asserts what it decided to do."""

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/healthcheck.sh"

# The script APPENDS ~/.local/bin to PATH, so the stubs are prepended to PATH here instead.
CURL_STUB = """\
#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >>"$CURL_LOG"
for a in "$@"; do
    if [ "$a" = "-X" ]; then method=1; fi
done
if [ "${method:-0}" = "1" ]; then
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
    # The frontend has no health endpoint: probed at / leniently, any wire status counts.
    *) code="$FRONTEND_CODE" ;;
esac
printf '%s' "$code"
# -f semantics: a non-2xx answer exits 22 even though the code is still printed to stdout.
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
    frontend="200",
    frontend_after_restart=None,
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
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        "BASE": "http://localhost:8001",
        "LOG": str(log),
        "CURL_LOG": str(curl_log),
        "FRONTEND_BASE": "http://localhost:3000",
        "FRONTEND_CODE": str(frontend),
        "FRONTEND_CODE_AFTER_RESTART": str(
            frontend if frontend_after_restart is None else frontend_after_restart
        ),
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
        check=False,
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
    run = run_watchdog(tmp_path, health="200", ready="503")

    assert run.returncode == 1
    assert run.restarts == [], "a live process must not be restarted for a dependency fault"
    assert "/ready/deep" in run.stdout, "the alert must name the probe that failed"
    assert "GEMINI_API_KEY" in run.stdout, "the alert must point at what a readiness 503 usually means"
    assert "ALERT" in run.log


def test_the_watchdog_asks_for_readiness_not_just_liveness(tmp_path):
    run = run_watchdog(tmp_path, health="200", ready="200")

    assert any(p.endswith("/health") for p in run.probes)
    assert any(p.endswith("/ready/deep") for p in run.probes)

def test_a_dead_process_is_restarted_and_reported_down_when_it_stays_down(tmp_path):
    # A restart that did not bring it back leaves liveness as the fault.
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
    """curl writes 000 on a refused connection: "not alive", not an empty answer."""
    run = run_watchdog(tmp_path, health="", ready="")

    assert run.returncode == 1
    assert len(run.restarts) == 1


def test_the_script_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30, check=False)

    assert proc.returncode == 0, textwrap.indent(proc.stderr, "  ")


def test_a_refused_readiness_probe_is_not_reported_as_a_dependency_failure(tmp_path):
    """403/404 means BASE cannot reach the probe the way it requires: a fact about the target, not Qdrant or the API key."""
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


def test_a_persistent_fault_alerts_once_and_then_stays_quiet(tmp_path):
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
    """Suppression is keyed on the whole fault set, so a new fault always re-alerts."""
    state = tmp_path / "state"
    run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    other = run_watchdog(tmp_path, health="000", health_after_restart="000", ready="000", state_file=state)

    assert "ALERT" in other.stdout
    assert "down" in other.stdout or "ALERT: VCCircle backend down" in other.stdout


def test_a_healthy_run_clears_the_fault_state(tmp_path):
    state = tmp_path / "state"
    run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    assert state.exists()

    run_watchdog(tmp_path, health="200", ready="200", state_file=state)

    assert not state.exists()
    after_recovery = run_watchdog(tmp_path, health="200", ready="503", state_file=state)
    assert "ALERT" in after_recovery.stdout, "a fault after a healthy run must alert immediately"


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
    faulted = run_watchdog(tmp_path, health="200", ready="503", webhook="https://hooks.example/x", webhook_fails="1")

    assert faulted.returncode == 1
    assert "ALERT" in faulted.stdout, "a failed webhook must not swallow the alert"
    assert "webhook delivery failed" in faulted.log
    assert len(faulted.webhooks) == 1, "the delivery must have been attempted, not skipped"


def test_a_healthy_backend_never_uses_the_webhook(tmp_path):
    healthy = run_watchdog(tmp_path, health="200", ready="200", webhook="https://hooks.example/x", webhook_fails="1")

    assert healthy.returncode == 0
    assert healthy.webhooks == []


def _state(text, tmp_path):
    state = tmp_path / "state"
    state.write_text(text)
    return state


def test_a_future_timestamp_does_not_suppress_the_alert(tmp_path):
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
    state = _state(corrupt, tmp_path)

    run = run_watchdog(tmp_path, health="200", ready="503", state_file=state, webhook="https://hooks.example/x")

    assert run.returncode == 1
    assert "ALERT" in run.stdout, "a corrupt state file must not swallow the alert"
    assert "unbound variable" not in run.stdout + run.stderr
    assert len(run.webhooks) == 1
    written = state.read_text().split()
    assert len(written) == 2 and written[1].isdigit(), f"the state was not rewritten with a numeric stamp: {written!r}"


def test_an_empty_state_file_is_treated_as_never_alerted(tmp_path):
    state = _state("", tmp_path)

    run = run_watchdog(tmp_path, health="200", ready="503", state_file=state)

    assert "ALERT" in run.stdout
    assert run.returncode == 1


def test_a_state_write_failure_still_alerts(tmp_path):
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
    assert "will not be deduplicated" in run.log
