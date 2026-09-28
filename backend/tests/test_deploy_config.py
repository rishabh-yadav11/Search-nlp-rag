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
`--watch`, which the backslash-continued `pm2 start` line leaves followed by
the `'\n'` token shlex emits for the continuation) are checked for presence
only, and a value that is a shell variable is resolved through setup.sh's own
`${VAR:-default}` declaration so that `--max-memory-restart "$API_MAX_MEMORY"`
compares equal to the literal `"5G"`.
"""
from __future__ import annotations

import re
import shlex
import sys
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
        if nxt.startswith("--") or not nxt.strip():
            # A bare flag with no value, e.g. `--watch`. The next token is
            # either the following `--flag` or the `'\n'` that shlex leaves
            # behind for the backslash line continuation the `pm2 start` line
            # uses; a whitespace-only token is not a value.
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


def _missing_options(
    declared: dict[str, str], passed: dict[str, str | None]
) -> list[str]:
    """Options `ecosystem.config.js` declares that setup.sh does not pass."""
    return sorted(set(declared) - set(passed))


def _mismatched_values(
    declared: dict[str, str], passed: dict[str, str | None]
) -> dict[str, tuple[str, str | None]]:
    """Declared options setup.sh passes, but with a different value.

    A valueless flag (value None on the setup.sh side) is not a mismatch: it
    carries no value to disagree about, so presence alone is the check.
    """
    return {
        flag: (declared[flag], passed[flag])
        for flag in declared
        if flag in passed and passed[flag] is not None and declared[flag] != passed[flag]
    }


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
    missing = _missing_options(declared, passed)
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

    mismatched = _mismatched_values(declared, passed)
    assert not mismatched, (
        f"{app_name} process options disagree between ecosystem.config.js and "
        f"setup.sh (ecosystem, setup.sh): {mismatched}"
    )


def _write_deploy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, setup_line: str, declared: str
) -> None:
    """Point the drift guard at a synthetic `setup.sh` + `ecosystem.config.js`.

    `setup_line` is the continuation-joined body of the app's `pm2 start` line
    (everything after `--name demo-app`), and `declared` is the app's option
    lines as they appear in `ecosystem.config.js`.
    """
    setup = tmp_path / "setup.sh"
    setup.write_text(
        '#!/usr/bin/env bash\n'
        'API_MAX_MEMORY="${API_MAX_MEMORY:-5G}"\n'
        "\n"
        "run_services() {\n"
        f"    pm2 start ./demo-app \\\n        --name demo-app \\\n{setup_line}"
        "        -- app.main:app)\n"
        "}\n"
    )
    ecosystem = tmp_path / "ecosystem.config.js"
    ecosystem.write_text(
        "module.exports = {\n"
        "  apps: [\n"
        "    {\n"
        '      name: "demo-app",\n'
        '      cwd: "/srv/demo",\n'
        '      script: "demo-app",\n'
        '      args: "app.main:app",\n'
        f"{declared}"
        "    },\n"
        "  ],\n"
        "};\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "SETUP_SH", setup)
    monkeypatch.setattr(sys.modules[__name__], "ECOSYSTEM_JS", ecosystem)


_WATCH_ECOSYSTEM = "      watch: true,\n"
_WATCH_SETUP = "        --watch \\\n"


def test_a_bare_flag_on_both_sides_is_compared_by_presence_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valueless flag declared and passed alike is agreement, not drift.

    The `pm2 start` lines are backslash-continued, so `shlex.split` emits a
    stray `'\n'` token after every flag. Reading that token as the flag's
    value made a consistent `--watch` pairing fail as
    `{'--watch': ('true', '\\n')}`, blocking a legitimate future change.
    """
    _write_deploy_files(
        tmp_path,
        monkeypatch,
        setup_line=_WATCH_SETUP,
        declared=_WATCH_ECOSYSTEM,
    )

    declared = _ecosystem_options("demo-app")
    passed = _setup_sh_start_options("demo-app")

    assert passed["--watch"] is None, (
        f"a bare `--watch` must parse as valueless, not with value {passed['--watch']!r}"
    )
    assert _missing_options(declared, passed) == []
    assert _mismatched_values(declared, passed) == {}


def test_a_bare_flag_declared_but_not_passed_is_still_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence-only comparison must not mean a bare flag is never checked.

    If `--watch` is declared in `ecosystem.config.js` but absent from the
    `pm2 start` line, `./setup.sh services` drops it from the running process,
    so the missing option must still be reported.
    """
    _write_deploy_files(
        tmp_path, monkeypatch, setup_line="", declared=_WATCH_ECOSYSTEM
    )

    assert _missing_options(
        _ecosystem_options("demo-app"), _setup_sh_start_options("demo-app")
    ) == ["--watch"]


def test_a_bare_flag_passed_but_not_declared_is_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is one-directional: extra setup.sh options are not reported.

    An option pm2 accepts on the CLI but the ecosystem file omits is not a
    silent-drop regression -- the process still gets it -- so it is out of
    scope here, and pinning that keeps the direction of the guard explicit.
    """
    _write_deploy_files(
        tmp_path, monkeypatch, setup_line=_WATCH_SETUP, declared=""
    )

    declared = _ecosystem_options("demo-app")
    passed = _setup_sh_start_options("demo-app")

    assert passed["--watch"] is None
    assert _missing_options(declared, passed) == []
    assert _mismatched_values(declared, passed) == {}


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


FRONTEND_PKG_JSON = REPO_ROOT / "frontend" / "package.json"


def _setup_sh_frontend_argv() -> list[str]:
    """The argv setup.sh hands `next` for the frontend, after the `--` separator.

    Every token is returned, not just the subcommand: the host is carried by a
    `-H` flag later in the argv, so dropping the tail would make the caller
    read an absent flag as Next's wildcard default.
    """
    start = re.search(
        r"pm2 start\b.*?--name\s+vccircle-frontend\b(?P<opts>.*?)--\s(?P<argv>[^)]*)\)",
        SETUP_SH.read_text(),
        re.DOTALL,
    )
    assert start is not None, "no `pm2 start --name vccircle-frontend` invocation in setup.sh"
    return shlex.split(start.group("argv"))


def _next_start_host(tokens: list[str]) -> str:
    """The address `next start` binds, from a `-H`/`--hostname` flag or Next's default."""
    for flag in ("-H", "--hostname"):
        if flag in tokens:
            return tokens[tokens.index(flag) + 1]
    # `next start` binds 0.0.0.0 when no hostname is given, so an absent flag
    # is the wildcard bind this issue is about, not an unspecified value.
    return "0.0.0.0"


def test_both_startup_paths_bind_the_frontend_to_loopback() -> None:
    """`next start` must be told 127.0.0.1 in every path that starts the frontend.

    The frontend is a production service behind nginx, so listening on the
    wildcard publishes the app shell, `/login`, `/signup` and `middleware.ts`
    on every interface of the host, bypassing TLS termination, the header and
    request-size limits, rate limiting and access logging. Next.js binds
    0.0.0.0 by default, so this is only true when the flag is actually passed.

    There are three such paths and each one is effective on its own: the pm2 app
    definition an operator starts by hand, the inline `pm2 start` line
    `./setup.sh services` re-registers the process from (it does NOT read
    ecosystem.config.js), and the `npm start` script a manual deploy runs.
    """
    ecosystem = re.search(
        r'"vccircle-frontend".*?args:\s*"([^"]+)"', ECOSYSTEM_JS.read_text(), re.DOTALL
    )
    assert ecosystem is not None, "vccircle-frontend declares no args in ecosystem.config.js"
    ecosystem_host = _next_start_host(shlex.split(ecosystem.group(1)))
    assert ecosystem_host == "127.0.0.1", (
        f"ecosystem.config.js binds the frontend to {ecosystem_host}; it must be "
        "127.0.0.1 so nginx on :80 is the only way in"
    )

    setup_host = _next_start_host(_setup_sh_frontend_argv())
    assert setup_host == "127.0.0.1", (
        f"setup.sh binds the frontend to {setup_host}; it must be 127.0.0.1, or "
        "`./setup.sh services` re-exposes :3000 on every interface"
    )

    script = FRONTEND_PKG_JSON.read_text(encoding="utf-8")
    pkg_start = re.search(r'"start":\s*"([^"]+)"', script)
    assert pkg_start is not None, "frontend/package.json has no start script"
    pkg_tokens = shlex.split(pkg_start.group(1))[1:]
    assert pkg_tokens[0] == "start", f"unexpected start script: {pkg_start.group(1)!r}"
    pkg_host = _next_start_host(pkg_tokens)
    assert pkg_host == "127.0.0.1", (
        f"npm start binds the frontend to {pkg_host}; it must be 127.0.0.1"
    )


def test_nginx_proxies_the_frontend_over_loopback() -> None:
    """The loopback bind is only safe because nginx already proxies to it.

    Binding Next.js to 127.0.0.1 while nginx proxies anywhere else takes the
    site down, so the precondition the bind change depends on is asserted here
    rather than left to whoever next edits the nginx template.
    """
    template = SETUP_SH.read_text()
    proxy_hosts = re.findall(r"proxy_pass\s+http://([0-9.]+|\$\w+):\$\{?(NEXT_PORT)\}?;", template)
    assert proxy_hosts, "the nginx template has no proxy_pass to the frontend port"

    defaults = _setup_sh_defaults()
    for host, _ in proxy_hosts:
        address = defaults[host[1:]] if host.startswith("$") else host
        assert address == "127.0.0.1", (
            f"nginx proxies the frontend to {address}, so binding Next.js to "
            "127.0.0.1 would take the site down"
        )
