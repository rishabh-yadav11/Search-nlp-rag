"""`setup.sh services` and `ecosystem.config.js` must define the same pm2 processes.

Both paths can start the API and they describe the same two pm2 apps, but the
mechanism by which they could drift changed. `./setup.sh services` no longer
re-registers the process from an inline `pm2 start` line: pm2's CLI has no
`--min-uptime` flag in any released version, so it starts from
`ecosystem.config.js` and EXPORTS the knobs to it. A hand-run
`pm2 start ecosystem.config.js` instead gets whatever the ecosystem file falls
back to.

The two definitions are therefore still independent -- setup.sh's
`${VAR:-default}` declarations versus the ecosystem file's own fallbacks -- and
they can still drift, in two ways that matter:

* a process option the ecosystem file declares that no exported knob drives, so
  `./setup.sh services` cannot influence it and a hand-run start applies a
  value nobody chose;
* a knob whose default in `setup.sh` differs from the fallback in
  `ecosystem.config.js`, so the two startup paths produce differently-shaped
  processes on the same host.

Both are compared here, option by option, and the option set is read from
`ecosystem.config.js` itself rather than from a hardcoded list, so an option
added there later is covered automatically. An earlier version of this file
enumerated only the three options that existed when it was written, which meant
adding a fourth option to the ecosystem file left the suite green -- the exact
drift it exists to catch.
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


def _setup_sh_exports() -> dict[str, str]:
    """The env vars `run_services` exports into `pm2 start ecosystem.config.js`.

    `{VAR: value}` for every `VAR="$VAR"` on that one invocation, with the value
    resolved through setup.sh's own `${VAR:-default}` declaration. Scoped to the
    invocation so an unrelated `VAR="$VAR"` elsewhere in the script cannot be
    mistaken for a knob the ecosystem file is meant to receive.
    """
    text = SETUP_SH.read_text()
    start = re.search(
        r'\(\s*cd\s+"\$SCRIPT_DIR"\s*&&(?P<exports>.*?)pm2 start\s+ecosystem\.config\.js',
        text,
        re.DOTALL,
    )
    assert start is not None, (
        "setup.sh does not start pm2 from ecosystem.config.js; if it went back to "
        "inline `pm2 start` lines, this guard is checking the wrong two paths"
    )

    defaults = _setup_sh_defaults()
    return {
        name: _resolve(f"${ref}", defaults, f"setup.sh's export of {name}")
        for name, ref in re.findall(r'(\w+)="\$(\w+)"', start.group("exports"))
        if name == ref
    }


def _ecosystem_knobs() -> dict[str, str]:
    """{const name: env var} for every knob ecosystem.config.js reads from the env."""
    text = ECOSYSTEM_JS.read_text()
    knobs: dict[str, str] = {}
    for const, var in re.findall(r"const\s+(\w+)\s*=\s*process\.env\.(\w+)", text):
        knobs[const] = var
    for const, var in re.findall(
        r"const\s+(\w+)\s*=\s*\w+\(\s*process\.env\.(\w+)\s*,", text
    ):
        knobs[const] = var
    return knobs


def _ecosystem_fallbacks() -> dict[str, str]:
    """{env var: the literal ecosystem.config.js falls back to when it is unset}.

    Only literals are recorded. A fallback that is a path expression
    (`path.resolve(__dirname)`) has no value to compare against a setup.sh
    default, and VCCIRCLE_ROOT is the one knob where that is correct: the
    file's own location IS the answer, which is the whole reason the services
    are started from the file.
    """
    text = ECOSYSTEM_JS.read_text()
    fallbacks: dict[str, str] = {}
    for var, rhs in re.findall(r"process\.env\.(\w+)\s*\|\|\s*([^\n;]+)", text):
        fallbacks[var] = rhs.strip().strip("\"'")
    for var, rhs in re.findall(r"\w+\(\s*process\.env\.(\w+)\s*,\s*([^,)]+)", text):
        fallbacks[var] = rhs.strip().strip("\"'")
    return fallbacks


def _setup_sh_start_options(app_name: str) -> dict[str, str | None]:
    """The value `./setup.sh services` actually starts `app_name` with.

    setup.sh starts from `ecosystem.config.js`, so an option's value on that
    path is the one its EXPORTED knob carries, resolved through setup.sh's own
    `${VAR:-default}` declaration so a `5G` here compares equal to the literal
    `"5G"` the ecosystem file falls back to. An option no exported knob drives
    is absent, which is the drift `_missing_options` reports.

    Read from the RAW declaration, not from `_ecosystem_options`: that one
    resolves each `const` to its fallback, and the NAME of the const is exactly
    what has to be looked up in the export list.
    """
    knobs = _ecosystem_knobs()
    exports = _setup_sh_exports()
    options: dict[str, str | None] = {}
    for flag, declared in _ecosystem_declared(app_name).items():
        var = knobs.get(declared)
        if var is not None and var in exports:
            options[flag] = exports[var]
    return options


def _ecosystem_declared(app_name: str) -> dict[str, str]:
    """Every non-structural option `ecosystem.config.js` declares, verbatim.

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
        options["--" + key.replace("_", "-")] = value.strip("\"'`")
    return options


def _ecosystem_options(app_name: str) -> dict[str, str]:
    """`_ecosystem_declared`, with each knob resolved to the value it falls back to.

    An option declared as a `const` resolves to that const's FALLBACK, i.e. the
    value a hand-run `pm2 start ecosystem.config.js` gets when the knob is unset.
    That is the value this guard compares against setup.sh's own default, so a
    `const` is never compared as the name of a variable. A const with no literal
    fallback (VCCIRCLE_ROOT, whose fallback is the file's own location) is left
    as the name, so it cannot be silently compared equal to an unrelated value.
    """
    knobs = _ecosystem_knobs()
    fallbacks = _ecosystem_fallbacks()
    resolved: dict[str, str] = {}
    for flag, declared in _ecosystem_declared(app_name).items():
        fallback = fallbacks.get(knobs.get(declared, ""))
        resolved[flag] = fallback if fallback else declared
    return resolved

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
def test_every_ecosystem_process_option_is_driven_by_a_knob_setup_sh_exports(
    app_name: str,
) -> None:
    """Every process option must be reachable from `./setup.sh services`.

    An option the ecosystem file declares that no exported knob drives is one
    `./setup.sh services` cannot influence, while a hand-run
    `pm2 start ecosystem.config.js` applies with whatever the file's fallback
    says. That is the same silent divergence this guard has always existed for,
    in the mechanism the startup path actually uses: a knob an operator sets and
    setup.sh exports is accepted and then ignored, which is worse than not
    offering it at all.
    """
    declared = _ecosystem_options(app_name)
    assert declared, f"no process options parsed for {app_name} in ecosystem.config.js"

    passed = _setup_sh_start_options(app_name)
    missing = _missing_options(declared, passed)
    assert not missing, (
        f"ecosystem.config.js declares {missing} for {app_name} but `./setup.sh "
        f"services` exports no knob that drives them, so a hand-run "
        f"`pm2 start ecosystem.config.js` and `./setup.sh services` would start "
        f"{app_name} with different options"
    )


@pytest.mark.parametrize("app_name", APP_NAMES)
def test_setup_sh_process_options_match_ecosystem_values(app_name: str) -> None:
    """The option values must be equal, not merely both present.

    Comparing values is what makes this a real guard rather than a key-name
    grep: a renamed knob does not fail, but a drifted limit does. setup.sh's
    `${VAR:-default}` and the ecosystem file's own fallback are two independent
    declarations of the same default, and a hand-run ecosystem start is exactly
    where the mismatch would show up.
    """
    declared = _ecosystem_options(app_name)
    passed = _setup_sh_start_options(app_name)

    mismatched = _mismatched_values(declared, passed)
    assert not mismatched, (
        f"{app_name} process options disagree between ecosystem.config.js and "
        f"setup.sh (ecosystem, setup.sh): {mismatched}"
    )


def _write_deploy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, defaults: str, exports: str, declared: str
) -> None:
    """Point the drift guard at a synthetic `setup.sh` + `ecosystem.config.js`.

    `defaults` is the body of setup.sh's `${VAR:-default}` declarations,
    `exports` the knobs it hands `pm2 start ecosystem.config.js`, and `declared`
    the app's option lines as they appear in `ecosystem.config.js`.
    """
    setup = tmp_path / "setup.sh"
    setup.write_text(
        "#!/usr/bin/env bash\n"
        f"{defaults}"
        "\n"
        "run_services() {\n"
        '    (cd "$SCRIPT_DIR" && \\\n'
        f"{exports}"
        "        pm2 start ecosystem.config.js)\n"
        "}\n"
    )
    ecosystem = tmp_path / "ecosystem.config.js"
    ecosystem.write_text(
        'const API_MAX_MEMORY = process.env.API_MAX_MEMORY || "5G";\n'
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


_DEFAULTS = 'API_MAX_MEMORY="${API_MAX_MEMORY:-5G}"\n'
_EXPORTS = '        API_MAX_MEMORY="$API_MAX_MEMORY" \\\n'
_DECLARED = "      max_memory_restart: API_MAX_MEMORY,\n"


def test_agreeing_options_are_not_reported_as_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline: a knob exported with the same default on both sides agrees.

    Without this, a comparison that reported drift for everything would pass
    every drift test below for entirely the wrong reason.
    """
    _write_deploy_files(
        tmp_path, monkeypatch, defaults=_DEFAULTS, exports=_EXPORTS, declared=_DECLARED
    )

    declared = _ecosystem_options("demo-app")
    passed = _setup_sh_start_options("demo-app")

    assert passed == {"--max-memory-restart": "5G"}, (
        f"the exported knob was not read: {passed!r}"
    )
    assert _missing_options(declared, passed) == []
    assert _mismatched_values(declared, passed) == {}


def test_a_knob_declared_but_not_exported_is_still_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An option no exported knob drives is reported, not skipped.

    setup.sh would start the app with the ecosystem file's fallback while
    believing it had applied its own value, and an operator's override would be
    accepted and discarded.
    """
    _write_deploy_files(
        tmp_path, monkeypatch, defaults=_DEFAULTS, exports="", declared=_DECLARED
    )

    assert _missing_options(
        _ecosystem_options("demo-app"), _setup_sh_start_options("demo-app")
    ) == ["--max-memory-restart"]


def test_a_default_that_drifted_is_reported_with_both_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale default must fail as loudly as a missing option.

    The option is present on both sides and only the numbers disagree, which is
    exactly the drift a presence-only check would pass.
    """
    _write_deploy_files(
        tmp_path,
        monkeypatch,
        defaults='API_MAX_MEMORY="${API_MAX_MEMORY:-8G}"\n',
        exports=_EXPORTS,
        declared=_DECLARED,
    )

    declared = _ecosystem_options("demo-app")
    passed = _setup_sh_start_options("demo-app")

    assert _missing_options(declared, passed) == []
    assert _mismatched_values(declared, passed) == {"--max-memory-restart": ("5G", "8G")}


def test_both_startup_paths_bind_the_api_to_loopback() -> None:
    """The API must be bound to loopback, and nginx must be the only way in.

    `./setup.sh services` and a hand-run `pm2 start ecosystem.config.js` are
    the two ways to start the API, and they now read the SAME line in
    `ecosystem.config.js`, so a wildcard bind cannot survive in one of them --
    but it can survive in the file itself, which is what this asserts. The port
    is resolved through the knob's fallback so a `${API_PORT}` template is
    compared to setup.sh's own `API_PORT` default rather than string-matched.
    """
    defaults = _setup_sh_defaults()
    ecosystem_bind = re.search(
        r"--bind\s+([0-9.]+|\$\{\w+\}):(\$\{\w+\}|\d+)", ECOSYSTEM_JS.read_text()
    )
    assert ecosystem_bind is not None, "ecosystem.config.js has no --bind"

    host = _resolve(ecosystem_bind.group(1), defaults, "ecosystem.config.js --bind host")
    assert host == "127.0.0.1", (
        f"ecosystem.config.js binds the API to {host}; it must be 127.0.0.1 so the "
        "API is not reachable off-host"
    )

    port = _resolve(ecosystem_bind.group(2), defaults, "ecosystem.config.js --bind port")
    assert port == defaults["API_PORT"], (
        f"ecosystem.config.js binds the API to port {port} but setup.sh's API_PORT "
        f"defaults to {defaults['API_PORT']}, so nginx would proxy to a port nothing "
        "is listening on"
    )

    # The bind is only safe because nginx proxies over loopback to that same
    # port; asserting it here is what stops the loopback bind from becoming the
    # reason the site is unreachable.
    proxies = re.findall(
        r"proxy_pass\s+http://([0-9.]+|\$\w+):\$\{?API_PORT\}?;", SETUP_SH.read_text()
    )
    assert proxies, "the nginx template has no proxy_pass to the API port"
    for proxy_host in proxies:
        address = defaults[proxy_host[1:]] if proxy_host.startswith("$") else proxy_host
        assert address == "127.0.0.1", (
            f"nginx proxies the API to {address}, so binding it to 127.0.0.1 would "
            "take the site down"
        )


FRONTEND_PKG_JSON = REPO_ROOT / "frontend" / "package.json"


def _ecosystem_frontend_argv() -> list[str]:
    """The argv `next start` is given, from the pm2 app definition.

    `./setup.sh services` and a hand-run `pm2 start ecosystem.config.js` both
    start the frontend from this one `args` string, so there is a single
    definition to assert on and a second path (the `npm start` script) to
    compare it against.

    Every token is returned, not just the subcommand: the host is carried by a
    `-H` flag later in the argv, so dropping the tail would make the caller read
    an absent flag as Next's wildcard default.
    """
    # `args` is a template literal in the ecosystem file (it interpolates
    # ${NEXT_PORT}), so the quoting is matched rather than assumed: a `"`-only
    # pattern silently finds nothing there, which reads as "declares no args".
    ecosystem = re.search(
        r'"vccircle-frontend".*?args:\s*(?P<q>["\'`])(?P<argv>[^"\'`]+)(?P=q)',
        ECOSYSTEM_JS.read_text(),
        re.DOTALL,
    )
    assert ecosystem is not None, "vccircle-frontend declares no args in ecosystem.config.js"
    return shlex.split(ecosystem.group("argv"))

def _next_start_flag(tokens: list[str], flag: str, default: str) -> str:
    """The value of `-H`/`--hostname` or `-p`/`--port`, or `default` if absent.

    A `$VAR` value is resolved through setup.sh's own `${VAR:-default}`
    declaration, so a correct config that factors the address out into a
    variable is recognised rather than reported as a regression.
    """
    for name in (flag, {"-H": "--hostname", "-p": "--port"}[flag]):
        if name in tokens:
            return _resolve(tokens[tokens.index(name) + 1], _setup_sh_defaults(), flag)
    return default


def _next_start_host(tokens: list[str]) -> str:
    """The address `next start` binds.

    `next start` binds all interfaces when no hostname is given, so an absent
    flag is the wildcard bind this issue is about, not an unspecified value.
    """
    return _next_start_flag(tokens, "-H", "0.0.0.0")


def test_both_startup_paths_bind_the_frontend_to_loopback() -> None:
    """`next start` must be told 127.0.0.1 in every path that starts the frontend.

    The frontend is a production service behind nginx, so listening on the
    wildcard publishes the app shell, `/login`, `/signup` and `middleware.ts`
    on every interface of the host, bypassing TLS termination, the header and
    request-size limits, rate limiting and access logging. Next.js binds
    0.0.0.0 by default, so this is only true when the flag is actually passed.

    There are two such paths and each one is effective on its own: the pm2 app
    definition in `ecosystem.config.js` (which is what `./setup.sh services`
    starts from as well as what an operator starts by hand), and the `npm start`
    script a manual deploy runs.
    """
    tokens = _ecosystem_frontend_argv()
    host = _next_start_host(tokens)
    assert host == "127.0.0.1", (
        f"ecosystem.config.js binds the frontend to {host}; it must be 127.0.0.1 so "
        "nginx on :80 is the only way in, and `./setup.sh services` would otherwise "
        "re-expose :3000 on every interface"
    )

    # A loopback bind on the wrong port is as broken as a wildcard bind on the
    # right one: nginx proxies to $NEXT_PORT, so a frontend on any other port is
    # unreachable through the site. `args` is a structural key, so the option
    # drift guard never sees it and this is the only check on the port.
    defaults = _setup_sh_defaults()
    port = _next_start_flag(tokens, "-p", "3000")
    assert port == defaults["NEXT_PORT"], (
        f"ecosystem.config.js starts the frontend on :{port} but setup.sh's "
        f"NEXT_PORT defaults to {defaults['NEXT_PORT']}; nginx proxies to "
        "$NEXT_PORT, so the frontend would be unreachable through the site"
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
