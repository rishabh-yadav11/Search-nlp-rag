"""Host-level infra hardening, asserted by running the real scripts and artifacts.

Nothing touches the network or a real /etc, cert store or pm2: sudo, curl and pm2
are stubs on PATH."""

from __future__ import annotations

import ast
import grp
import json
import os
import pwd
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SH = REPO_ROOT / "setup.sh"
ECOSYSTEM_JS = REPO_ROOT / "ecosystem.config.js"
LOGROTATE_TEMPLATE = REPO_ROOT / "deploy" / "logrotate.conf"
HEALTHCHECK_SH = REPO_ROOT / "deploy" / "healthcheck.sh"
BACKEND = REPO_ROOT / "backend"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node evaluates ecosystem.config.js")

# Ports differ from the shipped defaults, so a hardcoded port fails instead of passing by coincidence.
PUBLIC_PORT = "8080"
API_PORT = "18001"
NEXT_PORT = "13000"


def _bash_env(**over) -> dict[str, str]:
    """NGINX_* paths are fake: setup.sh reads the INSTALLED config to recover a TLS domain."""
    env = dict(os.environ)
    env.update(
        {
            "NGINX_TLS": "off",
            "LE_DOMAIN": "",
            "LE_ROOT": "/nonexistent-le-root-for-tests",
            "NGINX_CONF": "/nonexistent-nginx-conf-for-tests",
            "NGINX_LINK": "/nonexistent-nginx-link-for-tests",
            "PUBLIC_PORT": PUBLIC_PORT,
            "API_PORT": API_PORT,
            "NEXT_PORT": NEXT_PORT,
        }
    )
    env.update({k: str(v) for k, v in over.items()})
    return env


def _source(body: str, env: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess:
    """Sourcing runs no stage: setup.sh guards stages with a BASH_SOURCE[0] = $0 check."""
    return subprocess.run(
        ["bash", "-c", f'source "{SETUP_SH}"\n{body}\n'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )


def _stub(tmp_path: Path, name: str, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(body)
    path.chmod(0o755)


_RECORD = '#!/bin/sh\nprintf "%s %%s\\n" "$*" >> "$STUB_LOG"\n'


def _logrotate_stage(tmp_path: Path, **env_over) -> subprocess.CompletedProcess:
    """LOGROTATE_CONF points into the sandbox, so the real install writes here."""
    dest = tmp_path / "etc" / "logrotate.d" / "vccircle"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "stub.log"
    log.write_text("")

    # `sudo` passes through, so the stage really runs `install` on the sandboxed path.
    _stub(tmp_path, "sudo", _RECORD % "sudo" + 'exec "$@"\n')
    _stub(tmp_path, "logrotate", _RECORD % "logrotate" + "exit 0\n")

    env = _bash_env(
        STUB_LOG=str(log),
        LOGROTATE_CONF=str(dest),
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        **env_over,
    )
    return _source("run_logrotate", env, tmp_path)


def test_the_logrotate_stage_installs_a_real_file(tmp_path: Path):
    proc = _logrotate_stage(tmp_path)

    assert proc.returncode == 0, f"run_logrotate exited {proc.returncode}: {proc.stderr}"
    dest = tmp_path / "etc" / "logrotate.d" / "vccircle"
    assert dest.is_file(), (
        f"run_logrotate exited 0 but installed no {dest}; the logrotate config would "
        f"never reach /etc and the app logs would grow forever. stdout: {proc.stdout}"
    )
    mode = dest.stat().st_mode & 0o777
    assert mode == 0o644, (
        f"the installed config is mode {oct(mode)}; logrotate reads it as root, and "
        f"644 is the mode the install is supposed to promise"
    )
    assert dest.read_text().strip(), "the installed logrotate config is empty"


def test_the_installed_config_has_no_unsubstituted_template_values(tmp_path: Path):
    """`su` must come out activated: a policy naming no account rotates as root."""
    _logrotate_stage(tmp_path)
    installed = (tmp_path / "etc" / "logrotate.d" / "vccircle").read_text()

    assert "/path/to" not in installed, (
        f"the installed logrotate config still carries a template placeholder, so "
        f"it matches no files here:\n{installed}"
    )
    assert "deploy-user" not in installed, (
        f"the installed logrotate config still carries the template's placeholder "
        f"`su` account:\n{installed}"
    )
    for placeholder in re.findall(r"@[A-Z_]+@", installed):
        pytest.fail(f"an unsubstituted placeholder {placeholder} reached /etc:\n{installed}")


def _rendered_logrotate(tmp_path: Path, **env_over) -> str:
    proc = _source("render_logrotate_conf", _bash_env(**env_over), tmp_path)
    assert proc.returncode == 0, f"render_logrotate_conf failed: {proc.stderr}"
    return proc.stdout


def test_the_generated_logrotate_config_is_valid(tmp_path: Path):
    """logrotate rejects directives outside a stanza; a stanza header is absolute path globs."""
    config = _rendered_logrotate(tmp_path)

    assert config.strip(), "render_logrotate_conf produced nothing"
    assert config.count("{") == config.count("}"), f"unbalanced braces:\n{config}"

    depth, stanzas = 0, 0
    for line in config.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("{"):
            stanzas += 1
            assert depth == 0, (
                f"a stanza opens inside another stanza, which logrotate's parser "
                f"rejects: {line!r}"
            )
            for token in stripped[:-1].split():
                assert token.startswith("/"), (
                    f"stanza header {token!r} is not an absolute path; a relative or "
                    f"unsubstituted glob matches nothing and the logs grow forever"
                )
        elif stripped == "}":
            assert depth == 1, f"a closing brace with no open stanza: {line!r}"
        else:
            assert depth == 1, (
                f"directive {stripped!r} sits outside any stanza; logrotate rejects "
                f"the whole file when a directive has no block to belong to"
            )
        depth += stripped.count("{") - stripped.count("}")

    assert stanzas >= 1, f"no logrotate stanza in the generated config:\n{config}"
    assert depth == 0, f"unbalanced braces in the generated config:\n{config}"


def test_the_generated_config_rotates_the_directories_the_app_actually_writes(tmp_path: Path):
    config = _rendered_logrotate(tmp_path)

    logs = _source('printf "%s" "$LOGS"', _bash_env(), tmp_path)
    assert logs.returncode == 0
    logs_root = logs.stdout
    pm2_root = f"{os.environ['HOME']}/.pm2/logs"

    assert logs_root in config, (
        f"the generated config never rotates the app log directory {logs_root}:\n{config}"
    )
    assert pm2_root in config, (
        f"the generated config never rotates pm2's log directory {pm2_root}, so "
        f"~/.pm2/logs/*.log still grows forever:\n{config}"
    )


def test_the_generated_config_names_a_user_that_exists(tmp_path: Path):
    """An account not on this host fails the rotation, seen only in logrotate's own mail."""
    config = _rendered_logrotate(tmp_path)
    match = re.search(r"^\s*su\s+(\S+?)\s+(\S+?)\s*;?\s*$", config, re.MULTILINE)
    assert match, f"the generated config has no `su` directive:\n{config}"
    user, group = match.group(1), match.group(2)
    pwd.getpwnam(user)
    assert grp.getgrnam(group), f"the group {group!r} is not on this host"

    operator = subprocess.run(
        ["bash", "-c", "id -un"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert user == operator, (
        f"the config rotates as {user!r} but the operator running setup.sh is "
        f"{operator!r}; the rendered config belongs to a different host"
    )


def test_the_logrotate_template_the_stage_renders_exists():
    assert LOGROTATE_TEMPLATE.is_file(), (
        f"{LOGROTATE_TEMPLATE} does not exist, so there is nothing for the logrotate "
        f"stage to render and install"
    )
    text = LOGROTATE_TEMPLATE.read_text()
    assert "*.log" in text, f"{LOGROTATE_TEMPLATE} rotates no log files:\n{text}"
    # `su` ships commented: setup.sh's render stage activates it for this host
    assert re.search(r"^\s*#\s*su\s+\S+\s+\S+\s*;?\s*$", text, re.MULTILINE), (
        f"{LOGROTATE_TEMPLATE} has no `su` directive for the renderer to activate "
        f"with this host's user:\n{text}"
    )


def _perm_tree(tmp_path: Path, env_lines: str = "") -> Path:
    root = tmp_path / "repo"
    (root / "backend" / "data").mkdir(parents=True)
    (root / "backend" / "backups" / "col-20260101").mkdir(parents=True)
    (root / "backend" / ".env").write_text("SECRET=not-a-real-credential\n" + env_lines)
    (root / "backend" / "data" / "chat.db").write_text("chat")
    (root / "backend" / "data" / "auth.db").write_text("auth")
    # Fresh-checkout modes, so the test measures the change and not the umask.
    for path in (
        root / "backend" / ".env",
        root / "backend" / "data" / "chat.db",
        root / "backend" / "data" / "auth.db",
    ):
        path.chmod(0o644)
    (root / "backend" / "backups").chmod(0o755)
    return root


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _harden(root: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    return _source(f'harden_permissions "{root}"', _bash_env(), tmp_path)


def test_the_secrets_and_databases_are_left_owner_only(tmp_path: Path):
    root = _perm_tree(tmp_path)

    proc = _harden(root, tmp_path)
    assert proc.returncode == 0, f"harden_permissions failed: {proc.stderr}"

    for path in (
        root / "backend" / ".env",
        root / "backend" / "data" / "chat.db",
        root / "backend" / "data" / "auth.db",
    ):
        assert _mode(path) == 0o600, (
            f"{path.name} is mode {oct(_mode(path))} after harden_permissions; a file "
            f"holding credentials or user records must be readable only by its owner"
        )


def test_the_backup_directory_is_not_world_readable(tmp_path: Path):
    root = _perm_tree(tmp_path)

    _harden(root, tmp_path)

    mode = _mode(root / "backend" / "backups")
    assert mode == 0o700, (
        f"backend/backups is mode {oct(mode)}; a backup directory must not be "
        f"listable or readable by other accounts"
    )


def test_the_databases_are_found_where_the_application_puts_them(tmp_path: Path):
    """config.py resolves both DB paths against the backend cwd and the env can move them."""
    root = _perm_tree(tmp_path)
    (root / "backend" / "data" / "elsewhere").mkdir()
    moved_chat = root / "backend" / "data" / "elsewhere" / "chat.db"
    moved_chat.write_text("chat")
    moved_chat.chmod(0o644)
    moved_auth = tmp_path / "auth-moved.db"
    moved_auth.write_text("auth")
    moved_auth.chmod(0o644)
    (root / "backend" / ".env").write_text(
        (root / "backend" / ".env").read_text()
        + f"CHAT_DB_PATH=data/elsewhere/chat.db\nAUTH_DB_PATH={moved_auth}\n"
    )

    proc = _harden(root, tmp_path)
    assert proc.returncode == 0, f"harden_permissions failed: {proc.stderr}"

    assert _mode(moved_chat) == 0o600, (
        f"the database CHAT_DB_PATH points at is mode {oct(_mode(moved_chat))}; the "
        f"harden step ignored the override and tightened the default path instead"
    )
    assert _mode(moved_auth) == 0o600, (
        f"the database AUTH_DB_PATH points at is mode {oct(_mode(moved_auth))}; an "
        f"absolute override must be used as-is, not resolved against the backend dir"
    )


def test_a_database_that_does_not_exist_yet_is_not_an_error(tmp_path: Path):
    """A clean install has no database yet; failing there would break every first deploy."""
    root = _perm_tree(tmp_path)
    (root / "backend" / "data" / "chat.db").unlink()
    (root / "backend" / "data" / "auth.db").unlink()

    proc = _harden(root, tmp_path)

    assert proc.returncode == 0, (
        f"harden_permissions failed on a host with no databases yet: {proc.stderr}"
    )
    assert _mode(root / "backend" / ".env") == 0o600, (
        "the env file must still be tightened when the databases do not exist"
    )


def test_the_stages_that_create_these_files_call_the_hardening():
    script = SETUP_SH.read_text()

    def body(name: str) -> str:
        match = re.search(
            rf"^{name}\(\) \{{\n(?P<body>.*?)\n\}}", script, re.MULTILINE | re.DOTALL
        )
        assert match, f"could not find {name}() in {SETUP_SH}"
        return match.group("body")

    for stage, consequence in (
        ("run_backend", "the .env it just created stays world-readable"),
        ("run_services", "an upgraded host's existing databases stay world-readable"),
    ):
        assert re.search(r"^\s*harden_permissions\b", body(stage), re.MULTILINE), (
            f"{stage}() never calls harden_permissions, so {consequence}"
        )


def _image_defaults() -> dict[str, str]:
    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            r'^(QDRANT_IMAGE|REDIS_IMAGE)="\$\{\1:-([^}]*)\}"',
            SETUP_SH.read_text(),
            re.MULTILINE,
        )
    }


def _setup_sh_defaults() -> dict[str, str]:
    """Read from setup.sh rather than restated, so an assertion cannot drift from it."""
    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            r'^(\w+)="\$\{\1:-([^}]*)\}"', SETUP_SH.read_text(), re.MULTILINE
        )
    }


@pytest.mark.parametrize("var", ["QDRANT_IMAGE", "REDIS_IMAGE"])
def test_the_service_images_are_pinned_by_digest(var: str):
    """A tag is a name, not a pin: the same pull can return different bytes."""
    defaults = _image_defaults()
    assert var in defaults, f"setup.sh declares no default for {var}"
    image = defaults[var]

    assert "@sha256:" in image, (
        f"{var} defaults to {image!r}, which is a mutable tag. Pin it as "
        f"tag@sha256:<digest> the way QDRANT_IMAGE already is."
    )
    _repo, _, digest = image.partition("@")
    algorithm, _, hexdigest = digest.partition(":")
    assert algorithm == "sha256", f"{var} is pinned with {algorithm!r}, not sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", hexdigest), (
        f"the digest in {var} is {hexdigest!r}, which is not a sha256 digest"
    )


def test_the_documented_default_matches_the_pinned_one():
    """usage() is what an operator copies from; a stale line there undoes the pin."""
    match = re.search(
        r"usage\(\) \{\n\s*cat <<'EOF'\n(?P<text>.*?)\nEOF", SETUP_SH.read_text(), re.DOTALL
    )
    assert match, "could not find the usage() heredoc in setup.sh"
    text = match.group("text")

    for var, image in _image_defaults().items():
        assert "@sha256:" in image
        assert image in text or image.split("@")[0] in text, (
            f"usage() never mentions the {var} default ({image.split('@')[0]}); the "
            f"operator is reading a different image than the one that is pinned"
        )
    assert not re.search(r"redis:7-alpine(?![@\w-])", text), (
        "usage() still advertises the unpinned redis:7-alpine tag"
    )


_ECOSYSTEM_QUERY = (
    "const cfg = require('./ecosystem.config.js');"
    "process.stdout.write(JSON.stringify(cfg.apps.map(a => ({"
    "name: a.name, cwd: a.cwd, args: a.args, min_uptime: a.min_uptime}))))"
)


def _ecosystem_apps() -> list[dict]:
    """node evaluates the real file; a regex misreads template literals and arithmetic."""
    proc = subprocess.run(
        ["node", "-e", _ECOSYSTEM_QUERY],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"evaluating ecosystem.config.js failed: {proc.stderr}"
    return json.loads(proc.stdout)


@needs_node
def test_every_app_declares_a_min_uptime():
    """Without it a process dying seconds after every start is just a restarted one."""
    apps = _ecosystem_apps()
    assert {a["name"] for a in apps} == {"vccircle-backend", "vccircle-frontend"}

    for app in apps:
        value = app["min_uptime"]
        assert isinstance(value, int) and value > 0, (
            f"{app['name']} has min_uptime {value!r}; pm2 wants a positive number of "
            f"milliseconds, and a non-numeric one is a silent no-op"
        )


@needs_node
def test_min_uptime_is_actually_raised_above_pm2s_default():
    """pm2's own default is 1000ms, which a key-exists check accepts as readily as 30000."""
    declared = _setup_sh_defaults().get("MIN_UPTIME_MS")
    assert declared is not None and declared.isdigit(), (
        f"setup.sh declares no numeric MIN_UPTIME_MS default (got {declared!r}), so "
        f"the pm2 --min-uptime flag it passes has no fixed value to agree with"
    )
    expected = int(declared)
    pm2_default = 1000
    assert expected > pm2_default, (
        f"MIN_UPTIME_MS defaults to {expected}ms, which is not more than pm2's own "
        f"default of {pm2_default}ms; the option would be declared and inert"
    )
    for app in _ecosystem_apps():
        assert app["min_uptime"] == expected, (
            f"{app['name']} declares min_uptime {app['min_uptime']} but setup.sh "
            f"passes --min-uptime {expected}; the two must be the same number or "
            f"`./setup.sh services` registers a different process than this file does"
        )


@needs_node
def test_the_apps_run_from_this_checkout_rather_than_a_hardcoded_path():
    """pm2 resolves cwd before start, so a stale path fails only by restarting forever."""
    expected = {
        "vccircle-backend": str(REPO_ROOT / "backend"),
        "vccircle-frontend": str(REPO_ROOT / "frontend"),
    }
    for app in _ecosystem_apps():
        assert app["cwd"] == expected[app["name"]], (
            f"{app['name']} runs from {app['cwd']}, but this checkout is at "
            f"{REPO_ROOT}; the cwd must be resolved from the ecosystem file's own "
            f"location, or `pm2 start ecosystem.config.js` only works on one host"
        )
        assert Path(app["cwd"]).is_dir(), (
            f"{app['name']} runs from {app['cwd']}, which does not exist in this "
            f"checkout; pm2 would restart it forever with no such directory"
        )


@needs_node
def test_the_worker_count_follows_the_environment_and_survives_a_bad_one():

    def workers(value: str | None) -> str:
        env = dict(os.environ)
        env.pop("GUNICORN_WORKERS", None)
        if value is not None:
            env["GUNICORN_WORKERS"] = value
        proc = subprocess.run(
            [
                "node",
                "-e",
                (
                    "const a = require('./ecosystem.config.js').apps"
                    ".find(x => x.name === 'vccircle-backend');"
                    "process.stdout.write(a.args)"
                ),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, f"evaluating the backend app failed: {proc.stderr}"
        match = re.search(r"--workers\s+(\S+)", proc.stdout)
        assert match, f"no --workers in the backend args: {proc.stdout!r}"
        return match.group(1)

    assert workers(None) == "4", "the default worker count must stay 4"
    assert workers("7") == "7", "GUNICORN_WORKERS must actually take effect"
    # Number stringifies overflow as 1e+21, which gunicorn's int() rejects.
    for bad in ("abc", "0", "-2", "4; rm -rf /", "", "999999999999999999999", "1025"):
        got = workers(bad)
        assert got == "4", (
            f"GUNICORN_WORKERS={bad!r} produced --workers {got!r}; anything that is not "
            f"a plausible positive worker count must fall back to the default rather "
            f"than reach gunicorn"
        )
    assert workers("1024") == "1024", "a large but legitimate worker count must survive"


def _render_nginx(tmp_path: Path, mode: str, **over) -> str:
    proc = _source(f"render_nginx_config {mode}", _bash_env(**over), tmp_path)
    assert proc.returncode == 0, f"render_nginx_config {mode} failed: {proc.stderr}"
    return proc.stdout


def _server_blocks(config: str) -> list[str]:
    """Located by brace depth, so a nested location is never mistaken for a server."""
    blocks, current, depth = [], None, 0
    for line in config.splitlines():
        if current is None:
            if line.strip().startswith("server "):
                current, depth = [line], line.count("{") - line.count("}")
        else:
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                blocks.append("\n".join(current))
                current = None
    return blocks


def _server_block_on(config: str, port: str) -> str:
    matches = [
        b
        for b in _server_blocks(config)
        if re.search(rf"^\s*listen\s+{port}\b", b, re.MULTILINE)
    ]
    assert len(matches) == 1, f"expected exactly one server on {port}:\n{config}"
    return matches[0]


def _location_body(block: str, path: str) -> str:
    lines = block.splitlines()
    for at, line in enumerate(lines):
        opener = re.match(r"^\s*location\s+([^\n{]*)\{", line)
        if not opener:
            continue
        tokens = opener.group(1).split()
        if tokens and tokens[0] in ("^~", "=", "~", "~*"):
            tokens = tokens[1:]
        if not tokens or tokens[-1] != path:
            continue
        depth, body = line.count("{") - line.count("}"), [line]
        for following in lines[at + 1 :]:
            body.append(following)
            depth += following.count("{") - following.count("}")
            if depth == 0:
                return "\n".join(body)
    raise AssertionError(f"no location {path!r} in:\n{block}")


def _http_scope(config: str) -> str:
    """`limit_req_zone` is http-only; a zone in a server block fails `nginx -t`."""
    out, skip = [], 0
    for line in config.splitlines():
        if skip == 0 and line.strip().startswith("server "):
            skip = 1
        if skip == 0:
            out.append(line)
        skip += line.count("{") - line.count("}")
    return "\n".join(out)


def test_the_chat_stream_is_rate_limited_at_the_edge(tmp_path: Path):
    """The stream holds a worker up to 300s and an LLM call per turn; limit_req counts arrivals."""
    for mode in ("off", "on"):
        config = _render_nginx(tmp_path, mode)
        port = PUBLIC_PORT if mode == "off" else "443"
        chat = _location_body(_server_block_on(config, port), "/api/chat/")

        assert "limit_req " in chat, (
            f"the {mode} server proxies the chat stream with no limit_req:\n{chat}"
        )
        assert "nodelay" in chat, (
            f"without nodelay a burst of streams is rejected one at a time at the "
            f"configured rate instead of being admitted:\n{chat}"
        )
        assert re.search(r"limit_req_status\s+429\s*;", chat), (
            f"the chat limiter answers with nginx's default 503 rather than the 429 "
            f"the application's own limiter returns:\n{chat}"
        )
        assert "proxy_read_timeout 300s" in chat, (
            f"the chat location lost its 300s read timeout, so nginx would cut every "
            f"stream at its 60s default:\n{chat}"
        )


def test_the_rate_limiting_zone_is_declared_in_http_scope(tmp_path: Path):
    """`nginx -t` is the gate run_nginx rolls back on, so a bad zone takes the live site down."""
    config = _render_nginx(tmp_path, "off")
    scoped = _http_scope(config)

    zones = re.findall(r"limit_req_zone\s+([^;]+);", config)
    assert zones, f"the config declares no limit_req_zone, so no location can use one:\n{config}"
    for zone in zones:
        assert re.search(rf"limit_req_zone\s+{re.escape(zone)}\s*;", scoped), (
            f"the limit_req_zone {zone!r} is not at http scope; `limit_req_zone` is "
            f"only valid there, and nginx -t would reject the whole config"
        )
    assert re.search(r"limit_req_zone\s+\$binary_remote_addr\s+\S+", scoped), (
        f"the zone is not keyed on the client address:\n{config}"
    )


def test_no_path_the_application_limits_gets_a_second_limiter(tmp_path: Path):
    """Each application limit is tuned by a PUBLIC_* / AUTH_* knob; a second limiter makes it unreachable."""
    limited = _application_limited_paths()
    config = _render_nginx(tmp_path, "off")

    for block in _server_blocks(config):
        for path in limited:
            try:
                body = _location_body(block, path)
            except AssertionError:
                continue
            assert "limit_req" not in body, (
                f"{path} is limited by the application ({limited[path]}) and now also "
                f"by nginx, so that knob can never be reached:\n{body}"
            )


def _application_limited_paths() -> dict[str, str]:
    """Membership is a rate-limit call, not a naming convention, so a restated list would go stale."""
    limited: dict[str, str] = {}

    for source in ("main.py", "health.py", "auth.py"):
        text = (BACKEND / "app" / source).read_text()
        tree = ast.parse(text)
        prefix = _router_prefix(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                route = _route_path(decorator)
                if route is None:
                    continue
                knob = _public_rate_limit_knob(decorator) or _check_rate_limit_knob(node)
                if knob:
                    limited[prefix + route] = knob

    assert limited, (
        "no application rate limit was found in the backend source, so this file can "
        "no longer tell an overlapping edge limiter from a necessary one"
    )
    return limited


def _router_prefix(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for keyword in node.value.keywords:
                if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                    return str(keyword.value.value)
    return ""


def _route_path(decorator: ast.expr) -> str | None:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    if decorator.func.attr not in ("get", "post", "put", "patch", "delete"):
        return None
    if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
        return None
    path = decorator.args[0].value
    return path if isinstance(path, str) else None


def _public_rate_limit_knob(decorator: ast.expr) -> str | None:
    for node in ast.walk(decorator):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "public_rate_limit"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            return str(node.args[1].value)
    return None


def _check_rate_limit_knob(handler: ast.AST) -> str | None:
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_check_rate_limit"
        ):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "config"
                ):
                    return arg.attr
    return None


def test_the_limiter_rate_and_burst_are_configurable(tmp_path: Path):
    default = _render_nginx(tmp_path, "off")
    tuned = _render_nginx(
        tmp_path, "off", NGINX_CHAT_LIMIT_RATE="3r/m", NGINX_CHAT_LIMIT_BURST="2"
    )

    def rate(config: str) -> str:
        return re.search(r"limit_req_zone[^;]*\brate=(\S+?);", config).group(1)

    def burst(config: str) -> str:
        return re.search(r"limit_req\b[^;]*\bburst=(\d+)", _server_block_on(config, PUBLIC_PORT)).group(1)

    assert rate(tuned) == "3r/m", "NGINX_CHAT_LIMIT_RATE must reach the emitted zone"
    assert burst(tuned) == "2", "NGINX_CHAT_LIMIT_BURST must reach the emitted location"
    assert (rate(default), burst(default)) != ("3r/m", "2"), (
        "the rendered config ignored both knobs"
    )


def test_both_servers_carry_the_same_limiter(tmp_path: Path):
    """With TLS on, :80 is redirect-only and :443 proxies; limiting only one drops the protection silently."""
    plain = _render_nginx(tmp_path, "off")
    tls = _render_nginx(tmp_path, "on", LE_DOMAIN="example.test")

    off_chat = _location_body(_server_block_on(plain, PUBLIC_PORT), "/api/chat/")
    tls_chat = _location_body(_server_block_on(tls, "443"), "/api/chat/")

    for label, body in (("plain :80", off_chat), ("tls :443", tls_chat)):
        assert "limit_req " in body, f"the {label} server does not limit the chat stream:\n{body}"
    assert off_chat == tls_chat, (
        "the chat location differs between the plain and TLS servers, so the two "
        f"can drift apart:\n--- plain :80 ---\n{off_chat}\n--- tls :443 ---\n{tls_chat}"
    )


_CURL_STUB = """#!/bin/sh
# Records the probe, then answers from the table the test wrote. Deliberately
# faithful to the three shapes real curl produces, because the script branches on
# all of them and a stub that collapses them hides the branch under test:
#
#   * a status the server returned: printed on stdout, and -- when the caller
#     passed -f, which the strict backend probe does -- a non-zero exit (22) for
#     4xx/5xx, exactly as curl -f does;
#   * a refused connection: "000" printed and exit 7;
#   * a killed or timed-out curl: NOTHING printed and a non-zero exit.
printf 'curl %s\\n' "$*" >> "$STUB_LOG"
url=""
f=""
for arg in "$@"; do
    case "$arg" in
        http://*|https://*) url="$arg" ;;
        -f) f=1 ;;
    esac
done
while IFS='	' read -r code good; do
    [ -n "$good" ] || continue
    case "$url" in
        "$good"*)
            case "$code" in
                refused) printf '000'; exit 7 ;;
                killed) exit 28 ;;
            esac
            printf '%s' "$code"
            if [ -n "$f" ]; then
                case "$code" in
                    4*|5*) exit 22 ;;
                esac
            fi
            exit 0
            ;;
    esac
done < "$STUB_HEALTHY"
printf 'curl: (7) Failed to connect to %s port\\n' "$url" >&2
exit 7
"""

_PM2_STUB = _RECORD % "pm2" + "exit 0\n"
_SLEEP_STUB = "#!/bin/sh\nexit 0\n"


def _healthcheck(
    tmp_path: Path, *, healthy: dict[str, str] | set[str], frontend_base: str
) -> tuple[subprocess.CompletedProcess, str]:
    """`healthy` maps a URL prefix to the status curl reports; unlisted URLs fail to connect."""
    table = healthy if isinstance(healthy, dict) else {url: "200" for url in healthy}
    _stub(tmp_path, "curl", _CURL_STUB)
    _stub(tmp_path, "pm2", _PM2_STUB)
    _stub(tmp_path, "sleep", _SLEEP_STUB)

    answers = tmp_path / "healthy.tsv"
    answers.write_text("".join(f"{code}\t{url}\n" for url, code in sorted(table.items())))
    log = tmp_path / "stub.log"
    log.write_text("")
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "STUB_LOG": str(log),
            "STUB_HEALTHY": str(answers),
            "STUB_FAIL_URL": "",
            "BASE": f"http://localhost:{API_PORT}",
            "FRONTEND_BASE": frontend_base,
            "LOG": str(tmp_path / "healthcheck.log"),
            "HEALTHCHECK_WEBHOOK_URL": "",
            "HOME": str(tmp_path),
        }
    )
    proc = subprocess.run(
        ["bash", str(HEALTHCHECK_SH)], env=env, capture_output=True, text=True, check=False
    )
    return proc, log.read_text()


BACKEND_URL = f"http://localhost:{API_PORT}"
FRONTEND_URL = f"http://localhost:{NEXT_PORT}"


def test_a_healthy_host_is_left_alone(tmp_path: Path):
    """Cron mails on output, so a chatty healthy run trains the operator to ignore alerts."""
    proc, calls = _healthcheck(
        tmp_path, healthy={BACKEND_URL, FRONTEND_URL}, frontend_base=FRONTEND_URL
    )

    assert proc.returncode == 0, f"a healthy host reported failure: {proc.stderr}"
    assert proc.stdout == "", f"a healthy host wrote to stdout (cron mails on it): {proc.stdout!r}"
    assert "pm2" not in calls, f"a healthy host was restarted:\n{calls}"


def test_a_wedged_frontend_is_restarted_even_when_the_backend_is_fine(tmp_path: Path):
    """nginx proxies to a dead port and every page 502s while the backend-only check exits 0."""
    proc, calls = _healthcheck(
        tmp_path, healthy={BACKEND_URL}, frontend_base=FRONTEND_URL
    )

    assert proc.returncode != 0, (
        "a dead frontend was reported as a healthy host, so a site serving 502s "
        "everywhere looks fine to the operator"
    )
    assert "vccircle-frontend" in calls, f"the frontend was never restarted:\n{calls}"
    assert "vccircle-backend" not in calls, (
        f"a healthy backend was restarted because the frontend was down:\n{calls}"
    )


def test_both_services_down_is_reported_as_both(tmp_path: Path):
    proc, calls = _healthcheck(tmp_path, healthy=set(), frontend_base=FRONTEND_URL)

    assert proc.returncode != 0, "a host with both services down reported success"
    assert "vccircle-backend" in calls and "vccircle-frontend" in calls, (
        f"not every unhealthy service was restarted:\n{calls}"
    )
    message = (proc.stdout + proc.stderr).lower()
    assert "backend" in message and "frontend" in message, (
        f"the alert does not name both services, so the operator has to guess which "
        f"half of the site is down:\n{proc.stdout!r}"
    )


@pytest.mark.parametrize("status", ["301", "302", "404", "500"])
def test_a_frontend_that_answers_at_all_is_healthy(tmp_path: Path, status: str):
    """Next.js redirects or 404s `/` legitimately; a 500 still proves the process is bound."""
    proc, calls = _healthcheck(
        tmp_path,
        healthy={BACKEND_URL: "200", FRONTEND_URL: status},
        frontend_base=FRONTEND_URL,
    )

    assert proc.returncode == 0, (
        f"a frontend answering {status} was reported as down, so the monitor would "
        f"restart a serving process every cycle: {proc.stdout}{proc.stderr}"
    )
    assert "pm2" not in calls, f"a frontend answering {status} was restarted:\n{calls}"


def test_an_api_answering_500_is_restarted(tmp_path: Path):
    """/health tells a running-but-broken API from an absent one; the frontend has no such endpoint."""
    proc, calls = _healthcheck(
        tmp_path,
        healthy={BACKEND_URL: "500", FRONTEND_URL: "200"},
        frontend_base=FRONTEND_URL,
    )

    assert proc.returncode != 0, (
        "an API answering 500 on /health was reported as healthy; /health exists so "
        "that a running-but-broken API is distinguishable from an absent one"
    )
    assert "vccircle-backend" in calls, f"the broken API was never restarted:\n{calls}"
    assert "vccircle-frontend" not in calls, (
        f"a healthy frontend was restarted because the API was sick:\n{calls}"
    )


@pytest.mark.parametrize(
    ("probe_result", "what"),
    [
        ("refused", "a refused connection, which curl reports as 000"),
        ("killed", "a timeout, which prints no status at all and exits non-zero"),
    ],
)
def test_a_frontend_that_never_answers_is_restarted(
    tmp_path: Path, probe_result: str, what: str
):
    """A refusal prints 000 and exits 7, a timeout prints nothing; both mean down."""
    proc, calls = _healthcheck(
        tmp_path,
        healthy={BACKEND_URL: "200", FRONTEND_URL: probe_result},
        frontend_base=FRONTEND_URL,
    )

    assert proc.returncode != 0, (
        f"a frontend that produced {what} was reported as healthy"
    )
    assert "vccircle-frontend" in calls, (
        f"a frontend that produced {what} was never restarted:\n{calls}"
    )
