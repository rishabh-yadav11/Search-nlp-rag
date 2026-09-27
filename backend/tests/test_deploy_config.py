"""`setup.sh services` and `ecosystem.config.js` must define the same pm2 processes.

Both files can start the API, and they describe the same two pm2 apps. When
`./setup.sh services` re-registers the backend, the process is rebuilt from the
inline `pm2 start` line in `setup.sh` -- NOT from `ecosystem.config.js`. So any
process option that exists only in `ecosystem.config.js` is silently dropped the
first time an operator follows the documented remedy, and the pm2 save then
persists the stripped definition. `max_memory_restart` is the dangerous one: the
backend loses its OOM auto-restart guard with no error anywhere.

These tests assert the two definitions agree on every process option, by
comparing the option VALUES rather than grepping for key names. A rename or
reorder on either side is fine; only a genuinely different value fails, which is
the case worth failing for.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SH = REPO_ROOT / "setup.sh"
ECOSYSTEM_JS = REPO_ROOT / "ecosystem.config.js"

# pm2 ecosystem key -> the pm2 CLI flag that expresses the same setting.
OPTION_KEYS = {
    "max_memory_restart": "--max-memory-restart",
    "max_restarts": "--max-restarts",
    "exp_backoff_restart_delay": "--exp-backoff-restart-delay",
}

# ecosystem.config.js app name -> the pm2 --name used by setup.sh
APP_NAMES = {
    "vccircle-backend": "vccircle-backend",
    "vccircle-frontend": "vccircle-frontend",
}


def _setup_sh_defaults() -> dict[str, str]:
    """`VAR="${VAR:-default}"` assignments in setup.sh, as {name: default}."""
    text = SETUP_SH.read_text()
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r'^(\w+)="\$\{\1:-([^}]*)\}"', text, re.MULTILINE)
    }


def _setup_sh_start_flags(app_name: str) -> dict[str, str]:
    """The pm2 options setup.sh passes for `app_name`, as {flag: resolved value}.

    Locates the real `pm2 start ... --name <app_name>` invocation and reads the
    flags before the bare `--` that separates pm2 options from the app argv, so
    app arguments (which use similar dashes) cannot be mistaken for pm2 options.
    Values that are shell variables are resolved through setup.sh's own defaults
    so that `--max-memory-restart "$API_MAX_MEMORY"` compares equal to the
    literal `"5G"` in ecosystem.config.js.
    """
    text = SETUP_SH.read_text()
    defaults = _setup_sh_defaults()

    start = re.search(
        r"pm2 start\b.*?--name\s+" + re.escape(app_name) + r"\b(?P<opts>.*?)--\s",
        text,
        re.DOTALL,
    )
    assert start is not None, f"no `pm2 start --name {app_name}` invocation in setup.sh"

    # The flags are written as `--flag value`, not `--flag=value`.
    tokens = shlex.split(start.group("opts"))
    resolved: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--") and i + 1 < len(tokens):
            resolved[tokens[i]] = tokens[i + 1].strip("\"'")
            i += 2
        else:
            i += 1

    # Resolve `$VAR` / `${VAR}` references against setup.sh's own defaults.
    out: dict[str, str] = {}
    for flag, value in resolved.items():
        m = re.fullmatch(r"\$\{(\w+)\}|\$(\w+)", value)
        if m:
            var = m.group(1) or m.group(2)
            assert var in defaults, (
                f"setup.sh passes {flag}={value} but declares no default for "
                f"${var}, so the drift guard cannot resolve it"
            )
            out[flag] = defaults[var]
        else:
            out[flag] = value
    return out


def _ecosystem_options(app_name: str) -> dict[str, str]:
    """The pm2 options ecosystem.config.js declares for `app_name`."""
    text = ECOSYSTEM_JS.read_text()
    block = re.search(
        r"\{\s*name:\s*\""
        + re.escape(app_name)
        + r"\".*?\n    \}",
        text,
        re.DOTALL,
    )
    assert block is not None, f"{app_name} is not defined in ecosystem.config.js"
    return {
        OPTION_KEYS[key]: m.group(1)
        for key, flag in OPTION_KEYS.items()
        if (m := re.search(rf"\b{key}:\s*\"?([^\",\n]+)\"?", block.group(0)))
    }


@pytest.mark.parametrize("app_name", sorted(APP_NAMES))
def test_setup_sh_passes_every_ecosystem_process_option(app_name: str) -> None:
    """No process option in ecosystem.config.js may be absent from setup.sh.

    This is the guard against the silent regression: adding an option to
    ecosystem.config.js and forgetting to pass it to the inline `pm2 start` in
    setup.sh would strip it on the next `./setup.sh services`.
    """
    declared = _ecosystem_options(app_name)
    assert declared, f"no recognised process options parsed for {app_name}"

    passed = _setup_sh_start_flags(app_name)
    missing = sorted(set(declared) - set(passed))
    assert not missing, (
        f"ecosystem.config.js defines {missing} for {app_name} but setup.sh's "
        f"`pm2 start --name {app_name}` does not pass them, so "
        f"`./setup.sh services` would drop them from the running process"
    )


@pytest.mark.parametrize("app_name", sorted(APP_NAMES))
def test_setup_sh_process_options_match_ecosystem_values(app_name: str) -> None:
    """The option values must be equal, not merely both present.

    Comparing values is what makes this a real guard: a renamed flag or a
    reordered argument line does not fail, but a drifted limit does.
    """
    declared = _ecosystem_options(app_name)
    passed = _setup_sh_start_flags(app_name)

    mismatched = {
        flag: (declared[flag], passed[flag])
        for flag in declared
        if flag in passed and declared[flag] != passed[flag]
    }
    assert not mismatched, (
        f"{app_name} process options disagree between ecosystem.config.js and "
        f"setup.sh (ecosystem, setup.sh): {mismatched}"
    )


def test_both_startup_paths_bind_the_api_to_loopback() -> None:
    """The API bind must be 127.0.0.1 in both files, and they must agree.

    `setup.sh services` and `pm2 start ecosystem.config.js` are two ways to start
    the same API; a wildcard bind surviving in either one re-exposes :8001.
    """
    ecosystem_bind = re.search(
        r"--bind\s+([0-9.]+):(\d+)", ECOSYSTEM_JS.read_text()
    )
    assert ecosystem_bind is not None, "ecosystem.config.js has no --bind"
    assert ecosystem_bind.group(1) == "127.0.0.1", (
        f"ecosystem.config.js binds the API to {ecosystem_bind.group(1)}; "
        "it must be 127.0.0.1 so the API is not reachable off-host"
    )

    # The host may be a literal or a shell variable; the port is always one.
    setup_bind = re.search(
        r'--bind\s+"?([0-9.]+|\$\w+):\$\{?(\w+)\}?"?', SETUP_SH.read_text()
    )
    assert setup_bind is not None, "setup.sh has no --bind for the API"

    defaults = _setup_sh_defaults()
    address = setup_bind.group(1)
    if address.startswith("$"):
        var = address[1:]
        assert var in defaults, f"setup.sh --bind host ${var} has no default"
        address = defaults[var]
    assert address == "127.0.0.1", f"setup.sh binds the API to {address}, not 127.0.0.1"

    assert ecosystem_bind.group(2) == defaults[setup_bind.group(2)], (
        "the two startup paths must agree on the API port"
    )
