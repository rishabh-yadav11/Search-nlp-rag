"""The ``ENABLE_*`` feature toggles must read the way an operator writes them.

Each toggle used to be parsed inline as
``os.getenv(name, "true").lower() in ("1", "true", "yes")``. That reads every
other spelling a person would plausibly type as OFF: ``TRUE``, ``True``,
``on``, ``" true "``, ``"1 "``. A retrieval feature then sits switched off with
nothing wrong visible anywhere -- the knob looks configured, ``.env.example``
says ``true``, the API answers as though the operator had asked for off, and
no log line exists to say so. The only way to notice is to already suspect it.

``config._env_bool`` fixes that by normalising case and surrounding space, and
by sharing one spelling set with ``_env_tristate`` so the same question has one
answer in this file. Three properties of that answer are pinned here, because
each of them was got wrong in an implementation that was written and then
reverted:

1. **A blank value stays OFF.** ``KEY=`` in a .env is how an operator clears a
   knob, python-dotenv writes it as an empty string, and ``.env.example`` ships
   all eight toggles as ``true`` -- so nothing in the repo hints that blank
   ever meant "off". Falling back to the default here would take a feature an
   operator had switched off and switch it back on behind their back.

2. **An unrecognised value does not stop the process.** Raising ``ValueError``
   at import turns a mistyped .env into a boot failure of the whole API, which
   is strictly worse than the ambiguity it removes, and it contradicts this
   module's own stated rule in ``_clamped_int``. The value is warned about and
   the default is used instead.

3. **A toggle only counts as switched when the ``Config`` attribute says so.**
   Testing the helper is not enough: the defect was eight call sites choosing
   not to call it, so the wiring is asserted from the AST and then exercised
   through a real ``import app.config`` in a real interpreter.
"""
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

# The eight toggles the issue names. Restated literally on purpose: if the set
# below were derived from config, deleting a knob would quietly shrink the
# assertions that cover it instead of failing.
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

# Probe run in a child interpreter. Reads the two attributes back through the
# real class body, so this covers the wiring as well as the parse.
_CHILD_PROBE = (
    "import json\n"
    "from app.config import Config\n"
    "print('PROBE' + json.dumps([Config.ENABLE_DIVERSITY, Config.ENABLE_RECOMMENDATIONS]))\n"
)

# Spellings an operator writes, and the old inline parser got wrong.
TRUTHY_VARIANTS = ("1", "true", "TRUE", "True", "tRuE", "yes", "YES", "on", "ON", " true ", "  true  ", "1 ", " 1")
FALSY_VARIANTS = ("0", "false", "FALSE", "False", "no", "NO", "off", "OFF", " false ", " 0 ", " no  ")

# Values that are not booleans at all. None may raise.
JUNK = ("treu", "maybe", "enabled", "2", "-1", "y", "t", "truthy", "1 2", "yes-no", "null", "none")


# --------------------------------------------------------------------------
# the parse itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", TRUTHY_VARIANTS)
def test_every_truthy_spelling_is_on(monkeypatch, value):
    monkeypatch.setenv("SOME_TOGGLE", value)
    assert _env_bool("SOME_TOGGLE", False) is True


@pytest.mark.parametrize("value", FALSY_VARIANTS)
def test_every_falsy_spelling_is_off(monkeypatch, value):
    monkeypatch.setenv("SOME_TOGGLE", value)
    assert _env_bool("SOME_TOGGLE", True) is False


def test_the_old_parser_mislabelled_these_as_off(monkeypatch):
    """The defect, stated as the old rule so it cannot be reintroduced.

    ``.lower() in ("1", "true", "yes")`` is what the toggles used to do, and it
    is the only thing the new parse has to disagree with. Every variant above
    that this asserts is OFF is a value an operator wrote meaning ON.
    """
    old_reads_off = [v for v in TRUTHY_VARIANTS if v.lower() not in ("1", "true", "yes")]
    assert old_reads_off, "no variant distinguishes the two parsers; the table is vacuous"
    for value in old_reads_off:
        monkeypatch.setenv("SOME_TOGGLE", value)
        assert _env_bool("SOME_TOGGLE", False) is True, value


# --------------------------------------------------------------------------
# hazard 1: a blank value keeps meaning off
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", " ", "   ", "\t", " \t "])
@pytest.mark.parametrize("default", [True, False])
def test_a_blank_value_is_off_and_not_the_default(monkeypatch, caplog, value, default):
    """``KEY=`` means off. It must never fall back to the shipped default.

    The default is True for all eight toggles and ``.env.example`` ships them
    as ``true``, so treating blank as "not configured" would re-enable a
    feature an operator switched off by emptying the line.
    """
    monkeypatch.setenv("SOME_TOGGLE", value)
    with caplog.at_level("WARNING", logger="app.config"):
        assert _env_bool("SOME_TOGGLE", default) is False, (
            f"{value!r} must read as OFF, not as the default {default}"
        )
    assert "SOME_TOGGLE" in caplog.text, "a blank toggle is indistinguishable from an intentional off; say so"


def test_a_blank_toggle_warns_rather_than_being_silent(monkeypatch, caplog):
    """A blank knob is far more likely to be a mistake than a decision.

    The value is deliberately unchanged from the old parser, so the only way an
    operator learns that clearing the line did not restore the default is a log
    line naming the key.
    """
    monkeypatch.setenv("SOME_TOGGLE", "")
    with caplog.at_level("WARNING", logger="app.config"):
        _env_bool("SOME_TOGGLE", True)
    assert "SOME_TOGGLE" in caplog.text


# --------------------------------------------------------------------------
# unset, and hazard 2: an unrecognised value must not stop the process
# --------------------------------------------------------------------------


@pytest.mark.parametrize("default", [True, False])
def test_an_unset_toggle_takes_its_default(monkeypatch, default):
    monkeypatch.delenv("SOME_TOGGLE", raising=False)
    assert _env_bool("SOME_TOGGLE", default) is default


@pytest.mark.parametrize("value", JUNK)
def test_an_unrecognised_value_warns_and_takes_the_default(monkeypatch, caplog, value):
    """Warn-and-default, agreeing with ``_clamped_int``'s documented rule.

    The default is the shipped value, and the WARNING names the key, the
    rejected value and the spellings that work, so a typo stays visible.
    """
    monkeypatch.setenv("SOME_TOGGLE", value)
    with caplog.at_level("WARNING", logger="app.config"):
        assert _env_bool("SOME_TOGGLE", True) is True
    assert "SOME_TOGGLE" in caplog.text
    assert value in caplog.text or value.strip() == value, "the warning must show the rejected value"


@pytest.mark.parametrize("value", JUNK)
def test_no_input_ever_raises(monkeypatch, value):
    """This module is imported at process start; nothing here may escape.

    A ``raise ValueError`` in this function is a boot failure of the whole API
    for any deployment whose .env already holds an unusual spelling.
    """
    monkeypatch.setenv("SOME_TOGGLE", value)
    _env_bool("SOME_TOGGLE", True)
    _env_bool("SOME_TOGGLE", False)


def test_the_warning_names_the_spellings_that_work(monkeypatch, caplog):
    """The operator needs the fix, not just the complaint."""
    monkeypatch.setenv("SOME_TOGGLE", "treu")
    with caplog.at_level("WARNING", logger="app.config"):
        _env_bool("SOME_TOGGLE", True)
    for spelling in sorted(_TRUE_SPELLINGS | _FALSE_SPELLINGS):
        assert spelling in caplog.text, f"the warning should list {spelling!r}"


# --------------------------------------------------------------------------
# one convention, not two
# --------------------------------------------------------------------------


def test_the_two_readers_agree_on_every_spelling(monkeypatch):
    """``_env_tristate`` and ``_env_bool`` must not drift into two conventions.

    A second answer to "is this value on" is how the toggles and
    AUTH_TRUST_X_FORWARDED_FOR came to disagree. Where the knob has a shipped
    default the two can still differ on an unrecognised value (None vs the
    default), which is deliberate: tristate has no default to fall back to and
    must never pick a side. For every recognised spelling they must agree.
    """
    for spelling in sorted(_TRUE_SPELLINGS | _FALSE_SPELLINGS):
        for form in (spelling, spelling.upper(), f"  {spelling} "):
            monkeypatch.setenv("SOME_KNOB", form)
            assert _env_tristate("SOME_KNOB") is _env_bool("SOME_KNOB", True), form


def test_the_spellings_are_disjoint():
    """A spelling in both sets would leave the answer to frozenset order."""
    assert not _TRUE_SPELLINGS & _FALSE_SPELLINGS


# --------------------------------------------------------------------------
# the wiring: the defect was eight call sites not calling the helper
# --------------------------------------------------------------------------


def _enable_assignments() -> dict[str, ast.expr]:
    """``ENABLE_*`` class attributes of ``Config``, mapped to their value node."""
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
    """A ninth toggle added with the old inline parse must fail here."""
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
    """Nobody may re-implement the parse at a new call site.

    The defect was ``os.getenv(...).lower() in (...)`` written out eight times,
    so the shape to ban is the membership test itself, not the eight call sites:
    a ninth knob added tomorrow would reintroduce it just as effectively.
    ``_env_bool`` and ``_env_tristate`` decide from the shared spelling sets and
    never ``in``-test a raw value, so any comparison of an ``os.getenv()``
    result is that parse returning.
    """
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
    """Import app.config in a child process with one toggle set to ``value``."""
    env = {k: v for k, v in os.environ.items() if k not in FEATURE_TOGGLES}
    if value is not None:
        # Both, so the probe shows the two knobs sharing one parser rather than
        # showing the second one sitting at its default.
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
        # The defect: every one of these read as OFF before.
        ("TRUE", True),
        ("True", True),
        ("on", True),
        (" true ", True),
        ("1 ", True),
        # Falsy spellings, now named rather than accidentally off.
        ("FALSE", False),
        ("Off", False),
        (" 0 ", False),
        ("no", False),
        # Hazard 1: blank stays off.
        ("", False),
        ("   ", False),
        # Hazard 2: a typo must not take the API down.
        ("treu", True),
    ],
)
def test_a_deployed_value_reaches_the_config_attribute(value, expected):
    values, _ = _run_probe(value)
    assert values == [expected, expected], (
        f"ENABLE_DIVERSITY={value!r} produced {values!r}, expected {expected} for both toggles"
    )


def test_an_unset_toggle_is_on_in_a_real_process():
    """What .env.example ships, and what the code default says, must agree."""
    values, _ = _run_probe(None)
    assert values == [True, True]


def test_a_typo_is_reported_to_the_operator_at_boot():
    """Warn-and-default is only better than silent if the warning is emitted."""
    _, stderr = _run_probe("treu")
    assert "ENABLE_DIVERSITY" in stderr, f"no warning reached stderr: {stderr!r}"


def test_a_blank_toggle_is_reported_to_the_operator_at_boot():
    _, stderr = _run_probe("")
    assert "ENABLE_DIVERSITY" in stderr, f"no warning reached stderr: {stderr!r}"
