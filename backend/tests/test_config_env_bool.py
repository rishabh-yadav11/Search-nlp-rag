from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import config as config_module
from app.config import _FALSE_SPELLINGS, _TRUE_SPELLINGS, _env_bool, _env_tristate

CONFIG_PY = Path(config_module.__file__).resolve()
BACKEND = CONFIG_PY.parent.parent

# Restated literally: deriving it from config would silently shrink these assertions.
FEATURE_TOGGLES = (
    "ENABLE_QUERY_EXPANSION",
    "ENABLE_ENTITY_BOOST",
    "ENABLE_WEAK_FALLBACK",
    "ENABLE_QUERY_FIX",
    "ENABLE_DIVERSITY",
    "ENABLE_CLICK_BOOST",
    "ENABLE_BODY_RESCUE",
    "ENABLE_RECOMMENDATIONS",
)

_CHILD_PROBE = (
    "import json\n"
    "from app.config import Config\n"
    "print('PROBE' + json.dumps([Config.ENABLE_DIVERSITY, Config.ENABLE_RECOMMENDATIONS]))\n"
)

TRUTHY_VARIANTS = ("1", "true", "TRUE", "True", "tRuE", "yes", "YES", "on", "ON", " true ", "  true  ", "1 ", " 1")
FALSY_VARIANTS = ("0", "false", "FALSE", "False", "no", "NO", "off", "OFF", " false ", " 0 ", " no  ")

JUNK = ("treu", "maybe", "enabled", "2", "-1", "y", "t", "truthy", "1 2", "yes-no", "null", "none")


@pytest.mark.parametrize("value", TRUTHY_VARIANTS)
def test_every_truthy_spelling_is_on(monkeypatch, value):
    monkeypatch.setenv("SOME_TOGGLE", value)
    assert _env_bool("SOME_TOGGLE", False) is True


@pytest.mark.parametrize("value", FALSY_VARIANTS)
def test_every_falsy_spelling_is_off(monkeypatch, value):
    monkeypatch.setenv("SOME_TOGGLE", value)
    assert _env_bool("SOME_TOGGLE", True) is False


def test_the_old_parser_mislabelled_these_as_off(monkeypatch):
    old_reads_off = [v for v in TRUTHY_VARIANTS if v.lower() not in ("1", "true", "yes")]
    assert old_reads_off, "no variant distinguishes the two parsers; the table is vacuous"
    for value in old_reads_off:
        monkeypatch.setenv("SOME_TOGGLE", value)
        assert _env_bool("SOME_TOGGLE", False) is True, value


@pytest.mark.parametrize("value", ["", " ", "   ", "\t", " \t "])
@pytest.mark.parametrize("default", [True, False])
def test_a_blank_value_is_off_and_not_the_default(monkeypatch, caplog, value, default):
    monkeypatch.setenv("SOME_TOGGLE", value)
    with caplog.at_level("WARNING", logger="app.config"):
        assert _env_bool("SOME_TOGGLE", default) is False, (
            f"{value!r} must read as OFF, not as the default {default}"
        )
    assert "SOME_TOGGLE" in caplog.text, "a blank toggle is indistinguishable from an intentional off; say so"


def test_a_blank_toggle_warns_rather_than_being_silent(monkeypatch, caplog):
    monkeypatch.setenv("SOME_TOGGLE", "")
    with caplog.at_level("WARNING", logger="app.config"):
        _env_bool("SOME_TOGGLE", True)
    assert "SOME_TOGGLE" in caplog.text


@pytest.mark.parametrize("default", [True, False])
def test_an_unset_toggle_takes_its_default(monkeypatch, default):
    monkeypatch.delenv("SOME_TOGGLE", raising=False)
    assert _env_bool("SOME_TOGGLE", default) is default


@pytest.mark.parametrize("value", JUNK)
def test_an_unrecognised_value_warns_and_takes_the_default(monkeypatch, caplog, value):
    monkeypatch.setenv("SOME_TOGGLE", value)
    with caplog.at_level("WARNING", logger="app.config"):
        assert _env_bool("SOME_TOGGLE", True) is True
    assert "SOME_TOGGLE" in caplog.text
    assert value in caplog.text or value.strip() == value, "the warning must show the rejected value"


@pytest.mark.parametrize("value", JUNK)
def test_no_input_ever_raises(monkeypatch, value):
    monkeypatch.setenv("SOME_TOGGLE", value)
    _env_bool("SOME_TOGGLE", True)
    _env_bool("SOME_TOGGLE", False)


def test_the_warning_names_the_spellings_that_work(monkeypatch, caplog):
    monkeypatch.setenv("SOME_TOGGLE", "treu")
    with caplog.at_level("WARNING", logger="app.config"):
        _env_bool("SOME_TOGGLE", True)
    for spelling in sorted(_TRUE_SPELLINGS | _FALSE_SPELLINGS):
        assert spelling in caplog.text, f"the warning should list {spelling!r}"


def test_the_two_readers_agree_on_every_spelling(monkeypatch):
    for spelling in sorted(_TRUE_SPELLINGS | _FALSE_SPELLINGS):
        for form in (spelling, spelling.upper(), f"  {spelling} "):
            monkeypatch.setenv("SOME_KNOB", form)
            assert _env_tristate("SOME_KNOB") is _env_bool("SOME_KNOB", True), form


def test_the_spellings_are_disjoint():
    assert not _TRUE_SPELLINGS & _FALSE_SPELLINGS


def _enable_assignments() -> dict[str, ast.expr]:
    tree = ast.parse(CONFIG_PY.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config"]
    assert len(classes) == 1, f"expected exactly one Config class, found {len(classes)}"
    found: dict[str, ast.expr] = {}
    for node in classes[0].body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("ENABLE_"):
            found[target.id] = node.value
    return found


def test_all_eight_toggles_exist():
    assert set(_enable_assignments()) == set(FEATURE_TOGGLES)


@pytest.mark.parametrize("name", FEATURE_TOGGLES)
def test_each_toggle_is_read_by_env_bool_with_a_true_default(name):
    """The guard is an AST walk, so only the exact ``_env_bool(NAME, True)`` call is accepted."""
    value = _enable_assignments()[name]
    assert isinstance(value, ast.Call), f"{name} is not a call; is it still parsed inline?"
    func = value.func
    assert isinstance(func, ast.Name) and func.id == "_env_bool", (
        f"{name} must be read by _env_bool, got {ast.dump(func)}"
    )
    args = [ast.literal_eval(a) for a in value.args]
    assert args == [name, True], f"{name} must be _env_bool({name!r}, True), got {args!r}"


def _is_getenv_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getenv"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    )


def test_no_env_value_is_membership_tested_inline():
    """The ban is the ``os.getenv(...) in (...)`` shape anywhere in the file, so a new call site cannot reintroduce the parse."""
    tree = ast.parse(CONFIG_PY.read_text())
    offenders = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.In) for op in node.ops)
        and _is_getenv_call(node.left)
    )
    assert not offenders, (
        f"an os.getenv() result is membership-tested at line(s) {offenders} in "
        f"{CONFIG_PY.name}; route it through _env_bool or _env_tristate"
    )


def _run_probe(value: str | None) -> tuple[list, str]:
    env = {k: v for k, v in os.environ.items() if k not in FEATURE_TOGGLES}
    if value is not None:
        # Both toggles: otherwise the probe cannot show them sharing one parser.
        env["ENABLE_DIVERSITY"] = value
        env["ENABLE_RECOMMENDATIONS"] = value
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROBE],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"importing app.config with ENABLE_DIVERSITY={value!r} failed: {proc.stderr}"
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE"))
    return json.loads(line[len("PROBE"):]), proc.stderr


@pytest.mark.parametrize(
    "value, expected",
    [
        ("TRUE", True),
        ("True", True),
        ("on", True),
        (" true ", True),
        ("1 ", True),
        ("FALSE", False),
        ("Off", False),
        (" 0 ", False),
        ("no", False),
        ("", False),
        ("   ", False),
        ("treu", True),
    ],
)
def test_a_deployed_value_reaches_the_config_attribute(value, expected):
    values, _ = _run_probe(value)
    assert values == [expected, expected], (
        f"ENABLE_DIVERSITY={value!r} produced {values!r}, expected {expected} for both toggles"
    )


def test_an_unset_toggle_is_on_in_a_real_process():
    values, _ = _run_probe(None)
    assert values == [True, True]


def test_a_typo_is_reported_to_the_operator_at_boot():
    _, stderr = _run_probe("treu")
    assert "ENABLE_DIVERSITY" in stderr, f"no warning reached stderr: {stderr!r}"


def test_a_blank_toggle_is_reported_to_the_operator_at_boot():
    _, stderr = _run_probe("")
    assert "ENABLE_DIVERSITY" in stderr, f"no warning reached stderr: {stderr!r}"
