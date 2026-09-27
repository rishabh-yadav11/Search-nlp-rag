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

SCRIPT = Path(__file__).resolve().parents[2] / "deploy/healthcheck.sh"

# Both doubles live in $HOME/.local/bin, which the script puts first on PATH
# precisely because cron has a minimal one.
CURL_STUB = """\
#!/usr/bin/env bash
url="${@: -1}"
printf '%s\\n' "$url" >>"$CURL_LOG"
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
        self.curl_log = curl_log
        self.pm2_log = pm2_log
        self.log = log

    @property
    def probes(self):
        return [line for line in self.curl_log.splitlines() if line]

    @property
    def restarts(self):
        return [line for line in self.pm2_log.splitlines() if line]


def run_watchdog(tmp_path, *, health="200", health_after_restart="200", ready="200"):
    home = tmp_path / "home"
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
