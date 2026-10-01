"""backend/.env.example must ship the defaults app/config.py declares.

``app/config.py`` calls ``load_dotenv()`` at import, so a deployment reads
whatever is in ``backend/.env`` -- and ``.env`` is a copy of ``.env.example``.
Two knobs are governed by that file and disagreed with the code: CORS_ORIGINS
trusted a dead ``http://localhost:8000`` while the API serves 8001, and
AUTH_TRUST_X_FORWARDED_FOR read ``true`` in the template against a code default
of "auto". Both decide, unremarked, whether a client can forge the IP the rate
limiter buckets on.

Editing the template to match the code fixes today's file; the guard stops it
drifting again, stated as a property rather than a pair of literals: for every
key the template ships, setting that key to the template's value must leave
``Config`` exactly as it is with the key UNSET.

Both sides are read from files in a clean module, never from the ambient
process: ``load_dotenv`` is neutralised (it searches upward from the CALLING
file, so patching the CWD would not keep a real ``backend/.env`` out) and every
template key is removed from ``os.environ`` before the baseline load. Nothing
here skips: keys with a legitimate reason to differ are excluded, and each
exclusion is separately proved to still be needed.
"""
import importlib.util
from pathlib import Path

import dotenv
import pytest

from app import config as _config_module

BACKEND = Path(_config_module.__file__).resolve().parent.parent
ENV_EXAMPLE = BACKEND / ".env.example"
CONFIG_PY = Path(_config_module.__file__).resolve()

# Template keys that ship a value which is NOT the declared default, on
# purpose. Each needs a reason: an unexplained entry is the original defect
# one layer up. These are placeholders an operator MUST replace, and shipping
# them empty would look like a configured credential rather than an unset one.
PLACEHOLDER_VALUES: dict[str, str] = {
    "GEMINI_API_KEY": (
        "Ships the literal 'your_key_here' placeholder, which app/config.py's "
        "own key validation and app/health.py's readiness check both detect to "
        "refuse a boot. The declared default is the empty string, which would "
        "read as 'no key configured' instead of 'operator forgot to configure "
        "this'."
    ),
    "MYSQL_PASSWORD": (
        "Ships the literal 'changeme' placeholder. The declared default is the "
        "empty string, which would let a missing MySQL password look configured."
    ),
}

# Declared defaults that cannot be a constant, so no template value can be
# "the default" on every machine.
HOST_DERIVED: dict[str, str] = {
    "ALLOWED_HOSTS": (
        "Its default is derived from CORS_ORIGINS plus _machine_hosts() -- this "
        "box's hostname, bound addresses and default-route address. Any literal "
        "template value would be host-specific, so the template correctly ships "
        "it EMPTY ('derive from CORS_ORIGINS and this box') and there is nothing "
        "to compare. See test_api_surface_hardening.py for the derivation."
    ),
}

# Template keys that are not attributes of Config, and where they are read
# instead. A key listed here must say why; a typo in the template would
# otherwise be an operator setting nothing at all.
READ_ELSEWHERE: dict[str, str] = {
    "GEMINI_MODEL": (
        "Read as a nested fallback inside config.py (LLM_MODEL's own default) "
        "and directly by scripts/eval_scorer.py, so it is never a Config "
        "attribute of its own."
    ),
}

EXEMPT = set(PLACEHOLDER_VALUES) | set(HOST_DERIVED) | set(READ_ELSEWHERE)


def _template() -> dict[str, str]:
    """The key -> value pairs .env.example actually ships.

    Parsed with python-dotenv rather than a hand-rolled split so quoting and
    inline comments are handled the way the app's own loader handles them. The
    path is absolute, resolved from config.py, so the result does not depend on
    the working directory pytest was started in.
    """
    values = dotenv.dotenv_values(ENV_EXAMPLE)
    return {k: v for k, v in values.items() if v is not None}


def _load_config(tag: str):
    """A standalone Config class body executed in a throwaway module.

    ``Config`` reads the environment while its class body runs, so a reload is
    the only way to see a different environment. A plain importlib.reload of
    ``app.config`` would rebind ``app.config.config`` for every module that
    already did ``from app.config import config``, so a separate module name is
    used and the live singleton is never touched.
    """
    spec = importlib.util.spec_from_file_location(tag, CONFIG_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


def _baseline(monkeypatch):
    """Config as it reads with no template key set in the environment."""
    for key in _template():
        monkeypatch.delenv(key, raising=False)
    return _load_config("env_example_baseline_probe")


def _agreed_keys() -> list[str]:
    """Template keys the comparison below covers.

    Derived from the live Config and the template rather than hardcoded, so a
    key added to either file is picked up without editing this module. Keys
    left out are covered by the exclusion tests and by
    test_every_template_key_is_classified.
    """
    probe = _load_config("env_example_param_probe")
    return sorted(k for k in _template() if k not in EXEMPT and hasattr(probe, k))


@pytest.mark.parametrize("key", _agreed_keys())
def test_template_value_leaves_every_knob_at_its_declared_default(key, monkeypatch):
    """The template's value must produce the default, not merely look like it."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    value = _template()[key]

    default = getattr(_baseline(monkeypatch), key)
    monkeypatch.setenv(key, value)
    from_template = getattr(_load_config(f"env_example_{key}_probe"), key)

    assert from_template == default and type(from_template) is type(default), (
        f".env.example ships {key}={value!r}, which changes the value config.py "
        f"declares ({default!r}) into {from_template!r}. An operator copying the "
        f"template would deploy that instead of the default."
    )


def test_every_template_key_is_classified():
    """No key may fall through the parametrised case unclassified.

    A key that is neither a Config attribute nor a documented exclusion would
    otherwise drop out of the guard silently -- and a template key nothing
    reads is an operator setting a knob that does not exist.
    """
    values = _template()
    assert values, f"{ENV_EXAMPLE} parsed to no keys at all"
    probe = _load_config("env_example_coverage_probe")
    unclassified = [k for k in values if k not in EXEMPT and not hasattr(probe, k)]
    assert unclassified == [], (
        f"{unclassified} ship in .env.example but are neither Config attributes "
        f"nor documented in READ_ELSEWHERE/PLACEHOLDER_VALUES/HOST_DERIVED"
    )


@pytest.mark.parametrize("key", sorted(EXEMPT))
def test_an_exclusion_is_still_needed(key):
    """An exclusion that no longer applies must be removed, not left to rot.

    A stale entry would silently exempt that key from the guard above, so the
    next real drift on it would ship unnoticed. Requiring a non-empty reason
    also keeps an entry from silently becoming a licence to differ.
    """
    reason = {**PLACEHOLDER_VALUES, **HOST_DERIVED, **READ_ELSEWHERE}[key]
    assert reason.strip(), f"{key} is excluded with no stated reason"
    assert key in _template(), f"{key} is excluded but no longer ships in .env.example"


@pytest.mark.parametrize("key", sorted(PLACEHOLDER_VALUES))
def test_a_placeholder_exclusion_is_still_load_bearing(key, monkeypatch):
    """A placeholder must keep being a placeholder, i.e. still differ.

    If it ever equalled the declared default, the exclusion above would be
    hiding a key that is already in agreement -- and the operator-facing
    placeholder the health check looks for would be gone.
    """
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    default = getattr(_baseline(monkeypatch), key)
    monkeypatch.setenv(key, _template()[key])
    assert getattr(_load_config(f"placeholder_{key}_probe"), key) != default


def test_cors_origins_template_names_the_port_the_api_serves():
    """The dead :8000 origin is back, named as the operator needs to read it.

    The parametrised guard catches any divergence, but reports it as "this
    value changes the declared default".
    """
    origins = [o.strip() for o in _template()["CORS_ORIGINS"].split(",")]
    assert "http://localhost:8001" in origins, (
        "the API is served on 8001 (ecosystem.config.js binds 0.0.0.0:8001) and "
        "that is the origin a local dev client calls"
    )
    assert "http://localhost:8000" not in origins, (
        "nothing serves :8000, so a CORS entry for it trusts an origin that "
        "cannot exist"
    )


def test_xff_trust_ships_as_auto_not_a_forced_boolean():
    """AUTH_TRUST_X_FORWARDED_FOR must ship 'auto', the deliberate default.

    A forced true makes the rate limiter's client IP attacker-supplied for
    anyone reaching the API port directly; a forced false breaks the reference
    deploy, where nginx forwards from loopback. 'auto' (unset) resolves the
    header against the actual socket peer, correct for both topologies at once.
    """
    assert _template()["AUTH_TRUST_X_FORWARDED_FOR"].strip().lower() == "auto", (
        "AUTH_TRUST_X_FORWARDED_FOR must ship 'auto': trusting the header from "
        "any peer lets a direct client forge its rate-limit bucket, and refusing "
        "it from every peer breaks the loopback nginx proxy"
    )


def test_the_xff_default_is_documented_where_an_operator_will_look():
    """The decision is only deliberate if the template explains it."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("AUTH_TRUST_X_FORWARDED_FOR="))
    block = []
    for line in reversed(lines[:index]):
        if not line.startswith("#"):
            break
        block.append(line)
    explanation = "\n".join(reversed(block))
    assert explanation, "the shipped value has no comment above it"
    assert "X-Forwarded-For" in explanation and "loopback" in explanation, (
        "the comment above the shipped value must name the header and say what "
        "'auto' resolves against"
    )
