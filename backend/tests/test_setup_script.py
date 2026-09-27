"""Tests for the AUTH_TRUST_X_FORWARDED_FOR migration in setup.sh.

setup.sh repairs an upgraded host's backend/.env in two steps: it appends the
shipped default when the key is missing, and warns when the operator has forced
the header to be trusted. The warning is a safety net -- a forced-true value
means X-Forwarded-For is trusted from any peer, so a client reaching :8001
directly can forge it to dodge a rate limit -- so it has to fire for every
spelling the application itself reads as forced-true, not just the literal
"true". Missing one means the operator gets the forgeable posture with no
warning at all.

The pattern is extracted from setup.sh and run through real grep, so these
tests cannot drift away from the script they are policing, and the expected
set is derived from config._env_tristate rather than restated here, so changing
the parser's truthy set turns this file red until setup.sh is updated too.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from app.config import _env_tristate

SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"

# The key the migration repairs, and the value it appends when absent.
FLAG = "AUTH_TRUST_X_FORWARDED_FOR"
SHIPPED_DEFAULT = "auto"

# Spellings grouped by what _env_tristate does with them. The truthy group is
# what the warning must cover; the others must stay silent.
TRUTHY_SPELLINGS = ("1", "true", "yes", "on")
FALSY_SPELLINGS = ("0", "false", "no", "off")
SILENT_SPELLINGS = ("auto", "", "maybe", "1 2", "yes-no")


def _guard_flags():
    """The -i/-E flags and the regex the warning branch of setup.sh greps with.

    Read out of the script rather than duplicated, so fixing the script is what
    makes these tests pass. Fails loudly if the branch moves or is renamed.
    """
    script = SETUP_SH.read_text()
    match = re.search(
        r"^\s*elif grep (?P<flags>-q[a-zA-Z]*) '(?P<pattern>[^']+)' \"\$ENV_FILE\"; then$",
        script,
        re.MULTILINE,
    )
    assert match, f"could not find the {FLAG} warning guard in {SETUP_SH}"
    assert FLAG in match.group("pattern"), "the guard must key on the flag it repairs"
    return match.group("flags"), match.group("pattern")


def _warns(env_line, flags, pattern):
    """Run the real grep against a one-line .env and report whether it matches."""
    if env_line is not None:
        contents = f"{env_line}\n"
    else:
        contents = "SOME_OTHER_SETTING=1\n"  # key absent
    proc = subprocess.run(
        ["grep", flags, "-e", pattern],
        input=contents,
        capture_output=True,
        text=True,
        check=False,
    )
    # grep exits 0 on match, 1 on no match, >1 on a usage error.
    assert proc.returncode in (0, 1), f"grep failed: {proc.returncode} {proc.stderr}"
    return proc.returncode == 0


@pytest.mark.parametrize("value", TRUTHY_SPELLINGS)
def test_warning_guard_matches_every_forced_true_spelling(value):
    """1/true/yes/on all force trust, so all must warn.

    This is the defect: matching only the literal "true" left an operator who
    wrote "1" or "yes" in the forgeable posture with nothing printed.
    """
    flags, pattern = _guard_flags()
    assert _warns(f"{FLAG}={value}", flags, pattern), f"forced-true {value!r} did not warn"


@pytest.mark.parametrize("value", ["TRUE", "True", "YES", "On", f"  {TRUTHY_SPELLINGS[1]}  "])
def test_warning_guard_is_case_and_whitespace_insensitive(value):
    """_env_tristate strips and lowercases before matching, so the guard must too."""
    flags, pattern = _guard_flags()
    assert _warns(f"{FLAG}={value}", flags, pattern), f"{value!r} did not warn"


@pytest.mark.parametrize("value", FALSY_SPELLINGS + SILENT_SPELLINGS)
def test_warning_guard_stays_silent_for_everything_else(value):
    """A falsy value is a deliberate local/direct choice and 'auto' is the new
    default; neither is the forgeable posture, so neither should nag."""
    flags, pattern = _guard_flags()
    assert not _warns(f"{FLAG}={value}", flags, pattern), f"{value!r} wrongly warned"


def test_warning_guard_stays_silent_when_the_key_is_absent():
    """An absent key is the case setup.sh appends to, not one it warns about."""
    flags, pattern = _guard_flags()
    assert not _warns(None, flags, pattern)


def test_guard_covers_exactly_the_spellings_the_parser_calls_true():
    """The guard and the parser must agree, or the warning covers the wrong set.

    Driven off _env_tristate itself: widen or narrow the parser's truthy set and
    this fails until setup.sh follows, which is the point -- the script and the
    application cannot each maintain their own idea of "forced true".
    """
    flags, pattern = _guard_flags()
    for candidate in TRUTHY_SPELLINGS + FALSY_SPELLINGS + SILENT_SPELLINGS:
        # Drive the real parser by putting the value in the environment.
        old = os.environ.get(FLAG)
        os.environ[FLAG] = candidate
        try:
            truthy = _env_tristate(FLAG) is True
        finally:
            if old is None:
                os.environ.pop(FLAG, None)
            else:
                os.environ[FLAG] = old
        assert _warns(f"{FLAG}={candidate}", flags, pattern) is truthy, (
            f"{candidate!r}: parser says truthy={truthy}, but the guard disagrees"
        )


def test_missing_key_is_appended_and_an_existing_value_is_never_rewritten():
    """The migration only ever appends, and only when the key is absent.

    Rewriting an operator's value would break the supported "proxy on another
    host" deployment, which is exactly what a forced-true setting is for.
    """
    lines = [line.strip() for line in SETUP_SH.read_text().splitlines()]
    append_line = f'echo "{FLAG}={SHIPPED_DEFAULT}" >> "$ENV_FILE"'
    assert append_line in lines, "setup.sh should append the shipped default when the key is absent"

    # The append must sit inside a presence check on the key, so it cannot run
    # for a value the operator already set.
    at = lines.index(append_line)
    opener = lines[at - 1]
    assert opener.startswith("if ! grep -q ") and f"^{FLAG}=" in opener, (
        f"the append must be guarded by a presence check on the key, got: {opener!r}"
    )

    # And nothing anywhere may rewrite an operator's value in place: silently
    # downgrading a forced-true would break the supported "proxy on another
    # host" deployment, which is exactly what a forced-true setting is for.
    # Only the guarded append may write to $ENV_FILE, and nothing may sed it in
    # place. (The warning lines mention the flag and >&2, so they are excluded
    # by requiring a write to $ENV_FILE or a sed.)
    writes = [
        line
        for line in lines
        if FLAG in line and ("sed -i" in line or ('"$ENV_FILE"' in line and ">>" in line))
    ]
    for line in writes:
        assert line == append_line, f"setup.sh must not rewrite the operator's value: {line!r}"


# --- the deploy gate must wait on readiness, not on the liveness stub (#279) ---


def test_the_backend_deploy_gate_waits_on_readiness_not_on_the_health_stub():
    """./setup.sh services declares the deploy done from what `wait_http` can
    reach. It used to wait on /health, which cannot fail: a backend holding the
    placeholder GEMINI_API_KEY from .env.example, a dead Qdrant client or
    unloaded models answered 200 there, so the script reported a successful
    deploy of a backend that could not answer a single chat question.

    /ready/deep is the readiness answer, uncached so a warm cache cannot pass
    the gate, unrated so the gate cannot be throttled into a false negative,
    and loopback-only -- which is what a deploy gate on this host is. (The
    watchdog that consumes the same endpoint is exercised for real, against
    stub binaries, in test_healthcheck_script.py; this one can only be read,
    because running setup.sh would build venvs and register pm2 processes.)
    """
    source = SETUP_SH.read_text()

    assert 'wait_http "http://localhost:$API_PORT/ready/deep"' in source
    assert 'wait_http "http://localhost:$API_PORT/health"' not in source
