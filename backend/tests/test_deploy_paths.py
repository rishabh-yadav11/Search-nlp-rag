"""The shipped deploy configuration must not be pinned to one operator's home.

`ecosystem.config.js` is committed to the repo and started with
`pm2 start ecosystem.config.js`, so an absolute `cwd` pointing at one
developer's home makes every other checkout start nothing at all. The same
applies to `deploy/logrotate.conf`, which ships an installable copy of the
rotation policy. `backend/tests/test_deploy_config.py` guards the option
*values* these files agree on; this module guards the *paths* they point at.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECOSYSTEM_JS = REPO_ROOT / "ecosystem.config.js"
DEPLOY_DIR = REPO_ROOT / "deploy"

# The pm2 apps, in the order ecosystem.config.js declares them, mapped to the
# checkout directory each one must be resolved against.
APPS = (
    ("vccircle-backend", "backend"),
    ("vccircle-frontend", "frontend"),
)

# The literal that shipped before this guard existed, and the shape it was an
# instance of: any operator's home directory, not just that one username.
KNOWN_HOME_LITERAL = re.compile(r"home/ubuntu")
GENERIC_HOME = re.compile(r"/(?:home/[a-z_][a-z0-9_-]*|Users/[A-Za-z0-9_.-]+)/")

# A shipped `su <user> <group>` directive pinning one account. Separate from
# the two above because it names an identity rather than a path and has no
# leading slash to key off. The shipped placeholders are commented out, so they
# cannot match -- but an ACTIVE unedited `su deploy-user deploy-group` must
# fail, since that is the state an operator reaches by forgetting to substitute.
SU_DIRECTIVE = re.compile(r"^\s*su\s+\S+")

# A home-relative path to a `logs/` directory, e.g. `$HOME/search-nlp-rag/logs`.
# Not the same shape as GENERIC_HOME: it needs no leading `/home/<user>/` to
# match, so a script defaulting a log path to `$HOME/<fixed-name>/logs` resolves
# to a different directory on every other checkout -- writing, or rotating,
# nothing where the operator expects. Scoped to `logs/` deliberately: a plain
# `$HOME/.local/bin` PATH entry is legitimate, not a checkout pin.
HOME_RELATIVE_PIN = re.compile(r"\$(?:\{HOME\}|HOME)/[A-Za-z0-9_.-]+/logs/")

# A quoted string starting at the filesystem root, i.e. a path baked into the
# source instead of derived from the file's own location.
QUOTED_ABSOLUTE_PATH = re.compile(r"""["'`](/[^"'`\n]*)["'`]""")


def _shipped_deploy_files() -> list[Path]:
    """Every file this repo ships for deployment, ecosystem config included."""
    files = [ECOSYSTEM_JS, *(p for p in DEPLOY_DIR.rglob("*") if p.is_file())]
    return sorted(files)


def test_the_shipped_deploy_files_the_guard_scans_actually_exist() -> None:
    """Guard on the guard: the scan must not pass by scanning nothing.

    `_shipped_deploy_files()` is evaluated at import, so deleting or renaming
    `deploy/logrotate.conf` empties the parametrisation and every home-path test
    above would go green with the rotation policy simply absent.
    """
    expected_files = (
        ECOSYSTEM_JS,
        DEPLOY_DIR / "logrotate.conf",
        DEPLOY_DIR / "healthcheck.sh",
    )
    for expected in expected_files:
        assert expected.is_file(), f"{expected} is missing; the deploy guard is not scanning it"
    assert len(_shipped_deploy_files()) >= len(expected_files)


def _violations(pattern: re.Pattern[str], path: Path) -> list[tuple[int, str]]:
    """`pattern` matches in `path`, as 1-based (line number, line) pairs."""
    return [
        (number, line)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]


def _require_node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot evaluate ecosystem.config.js")
    return node


def _node_apps(node: str) -> list[dict[str, Any]]:
    """The `apps` array node actually builds by requiring the config file."""
    source = f"console.log(JSON.stringify(require({json.dumps(str(ECOSYSTEM_JS))})))"
    done = subprocess.run([node, "-e", source], capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"node failed to load ecosystem.config.js: {done.stderr}"
    apps = json.loads(done.stdout)["apps"]
    assert isinstance(apps, list)
    return apps


@pytest.mark.parametrize("path", _shipped_deploy_files(), ids=lambda p: p.name)
@pytest.mark.parametrize(
    ("pattern", "label"),
    [
        (KNOWN_HOME_LITERAL, "the previously shipped /home/ubuntu path"),
        (GENERIC_HOME, "an absolute per-user home directory"),
        (SU_DIRECTIVE, "a logrotate `su` directive pinning one operator's account"),
        (HOME_RELATIVE_PIN, "a $HOME-relative path pinned to one checkout's name"),
    ],
)
def test_no_hardcoded_home_directory_in_shipped_deploy_config(
    path: Path, pattern: re.Pattern[str], label: str
) -> None:
    """No shipped deploy file may pin a path to someone's home or checkout name.

    A checkout belonging to anyone else would rotate logs that do not exist, or
    start pm2 processes in a missing directory. The patterns cover the distinct
    spellings that defect takes: a literal absolute home, a home written as
    `$HOME/...`, and a logrotate `su` account.
    """
    found = _violations(pattern, path)
    assert found == [], (
        f"{path.name} hardcodes {label}: " + ", ".join(f"line {n}: {line.strip()}" for n, line in found)
    )


def test_ecosystem_config_paths_are_derived_not_literal() -> None:
    """`cwd` must resolve to this checkout, whatever directory the repo sits in."""
    text = ECOSYSTEM_JS.read_text(encoding="utf-8")

    assert "__dirname" in text, "ecosystem.config.js must derive paths from its own location"

    literals = QUOTED_ABSOLUTE_PATH.findall(text)
    assert literals == [], f"ecosystem.config.js bakes in absolute path(s): {literals}"

    apps = _node_apps(_require_node())
    by_name = {app["name"]: app for app in apps}
    for app_name, subdir in APPS:
        expected = REPO_ROOT / subdir
        assert expected.is_dir(), f"expected {expected} to exist in this checkout"
        assert by_name[app_name]["cwd"] == str(expected), (
            f"{app_name} cwd must resolve to this checkout's {subdir}/, not a fixed location"
        )


def test_ecosystem_config_is_loadable_by_node() -> None:
    """`pm2 start ecosystem.config.js` needs a file node can require and parse."""
    node = _require_node()

    done = subprocess.run(
        [node, "-e", f"require({json.dumps(str(ECOSYSTEM_JS))})"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"require() of ecosystem.config.js failed: {done.stderr}"

    names = [app["name"] for app in _node_apps(node)]
    assert names == [name for name, _ in APPS]
