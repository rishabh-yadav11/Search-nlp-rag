"""`setup.sh services` and `ecosystem.config.js` must define the same pm2 processes.

Both files can start the API, and they describe the same two pm2 apps. When
`./setup.sh services` re-registers the backend, the process is rebuilt from the
inline `pm2 start` line in `setup.sh` -- NOT from `ecosystem.config.js`. So any
process option that exists only in `ecosystem.config.js` is silently dropped the
first time an operator follows the documented remedy, and `pm2 save` then
persists the stripped definition. `max_memory_restart` is the dangerous one: the
backend loses its OOM auto-restart guard with no error anywhere.

These tests compare the two definitions option by option:

* every non-structural option `ecosystem.config.js` declares for an app must
  also be passed on that app's `pm2 start` line in `setup.sh`;
* and the VALUE must be equal, so a stale limit fails as loudly as a missing one.

The option set is read from `ecosystem.config.js` itself rather than from a
hardcoded list, so an option added there later is covered automatically. An
earlier version of this file enumerated only the three options that existed when
it was written, which meant adding a fourth option to the ecosystem file left the
suite green -- the exact drift it exists to catch.

For the value comparison, options that take no value (a bare flag such as
`--watch`) are checked for presence only, and a value that is a shell variable is
resolved through setup.sh's own `${VAR:-default}` declaration so that
`--max-memory-restart "$API_MAX_MEMORY"` compares equal to the literal `"5G"`.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SH = REPO_ROOT / "setup.sh"
ECOSYSTEM_JS = REPO_ROOT / "ecosystem.config.js"

# Keys in an ecosystem app block that are not pm2 process options: they describe
# the app itself and have no `pm2 start` CLI equivalent.
STRUCTURAL_KEYS = frozenset({"name", "cwd", "script", "args"})

# The two pm2 apps, by the name both files use.
APP_NAMES = ("vccircle-backend", "vccircle-frontend")


def _setup_sh_defaults() -> dict[str, str]:
    """`VAR="${VAR:-default}"` assignments in setup.sh, as {name: default}."""
    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            r'^(\w+)="\$\{\1:-([^}]*)\}"', SETUP_SH.read_text(), re.MULTILINE
        )
    }


def _resolve(value: str, defaults: dict[str, str], where: str) -> str:
    """Expand a `$VAR` / `${VAR}` reference through setup.sh's declared default."""
    m = re.fullmatch(r"\$\{(\w+)\}|\$(\w+)", value)
    if not m:
        return value
    var = m.group(1) or m.group(2)
    assert var in defaults, (
        f"{where} references ${var}, which setup.sh declares no default for, so the "
        "drift guard cannot resolve it to a value"
    )
    return defaults[var]


def _setup_sh_start_options(app_name: str) -> dict[str, str | None]:
    """The pm2 options setup.sh passes for `app_name`.

    Returns {flag: resolved value}, with value None for a flag that takes no
    value. Only the tokens before the bare `--` that separates pm2 options from
    the app argv are considered, so application arguments that also start with
    dashes (`--workers`, `--bind`, `start -p`) cannot be mistaken for options.
    """
    text = SETUP_SH.read_text()
    defaults = _setup_sh_defaults()

    start = re.search(
        r"pm2 start\b.*?--name\s+" + re.escape(app_name) + r"\b(?P<opts>.*?)--\s",
        text,
        re.DOTALL,
    )
    assert start is not None, f"no `pm2 start --name {app_name}` invocation in setup.sh"

    tokens = shlex.split(start.group("opts"))
    options: dict[str, str | None] = {}
    i = 0
    while i < len(tokens):
        if not tokens[i].startswith("--"):
            i += 1
            continue
        flag = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if nxt.startswith("--"):
            # A bare flag with no value, e.g. `--watch`.
            options[flag] = None
            i += 1
        else:
            options[flag] = _resolve(
                nxt, defaults, f"setup.sh's `pm2 start --name {app_name}`"
            )
            i += 2
    return options


def _ecosystem_options(app_name: str) -> dict[str, str]:
    """Every non-structural option `ecosystem.config.js` declares for `app_name`.

    App-level keys sit at six spaces of indentation; nested blocks (`env: {...}`)
    are indented further and are skipped. Values are unquoted and stripped so
    that trailing whitespace or an alternate quoting style in the JS file is not
    mistaken for a real difference.
    """
    block = re.search(
        r'\{\s*\n\s*name:\s*"'
        + re.escape(app_name)
        + r'"\s*,(?P<body>.*?)\n    \}',
        ECOSYSTEM_JS.read_text(),
        re.DOTALL,
    )
    assert block is not None, f"{app_name} is not defined in ecosystem.config.js"

    options: dict[str, str] = {}
    for line in block.group("body").splitlines():
        m = re.match(r" {6}(\w+):\s*(.*?)\s*,?\s*$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key in STRUCTURAL_KEYS or value.startswith("{"):
            continue
        options["--" + key.replace("_", "-")] = value.strip("\"'")
    return options


@pytest.mark.parametrize("app_name", APP_NAMES)
def test_setup_sh_passes_every_ecosystem_process_option(app_name: str) -> None:
    """No process option in ecosystem.config.js may be absent from setup.sh.

    This is the guard against the silent regression: an option present only in
    ecosystem.config.js is dropped from the running process the next time
    `./setup.sh services` re-registers it, with no error to notice.
    """
    declared = _ecosystem_options(app_name)
    assert declared, f"no process options parsed for {app_name} in ecosystem.config.js"

    passed = _setup_sh_start_options(app_name)
    missing = sorted(set(declared) - set(passed))
    assert not missing, (
        f"ecosystem.config.js declares {missing} for {app_name} but setup.sh's "
        f"`pm2 start --name {app_name}` does not pass them, so `./setup.sh services` "
        "would drop them from the running process"
    )


@pytest.mark.parametrize("app_name", APP_NAMES)
def test_setup_sh_process_options_match_ecosystem_values(app_name: str) -> None:
    """The option values must be equal, not merely both present.

    Comparing values is what makes this a real guard rather than a key-name
    grep: a renamed flag or a reordered argument line does not fail, but a
    drifted limit does. Valueless flags are compared by presence, since a bare
    `--watch` carries no value to disagree about.
    """
    declared = _ecosystem_options(app_name)
    passed = _setup_sh_start_options(app_name)

    mismatched = {
        flag: (declared[flag], passed[flag])
        for flag in declared
        if flag in passed and passed[flag] is not None and declared[flag] != passed[flag]
    }
    assert not mismatched, (
        f"{app_name} process options disagree between ecosystem.config.js and "
        f"setup.sh (ecosystem, setup.sh): {mismatched}"
    )


def test_both_startup_paths_bind_the_api_to_loopback() -> None:
    """The API bind must be 127.0.0.1 in both files, and they must agree.

    `setup.sh services` and `pm2 start ecosystem.config.js` are two ways to start
    the same API; a wildcard bind surviving in either one re-exposes :8001 on
    every interface of the host.
    """
    ecosystem_bind = re.search(r"--bind\s+([0-9.]+):(\d+)", ECOSYSTEM_JS.read_text())
    assert ecosystem_bind is not None, "ecosystem.config.js has no --bind"
    assert ecosystem_bind.group(1) == "127.0.0.1", (
        f"ecosystem.config.js binds the API to {ecosystem_bind.group(1)}; it must be "
        "127.0.0.1 so the API is not reachable off-host"
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
