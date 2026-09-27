"""Regression tests for the clamped env knobs in `app.config` (#253).

The knobs guarded here are all *throughput* caps, and the interesting
assertion is never "the helper returned N" -- it is "the pipeline actually
observes N". A clamp that is applied in `config.py` but bypassed at a call
site is still a DoS, so the call-site tests drive the real
`_retrieval_leg` -> `hybrid_search` path with a hostile value in the
environment and assert on the number that actually reaches retrieval.
"""

import asyncio
import importlib
import os
from unittest.mock import patch

import pytest

import app.config as config_module
from app import main
from app.config import _clamped_int


@pytest.fixture
def effective_config():
    """Rebind `app.config` under a temporary environment.

    `app.config` computes its knobs at import time and every consumer did
    `from app.config import config`, so a plain `monkeypatch.setattr` would
    only prove the attribute is settable. Reloading the module under a patched
    environment re-runs the real parsing/clamping code. The original class
    object is put back on both the module and `app.main` afterwards so the
    rest of the suite sees exactly what it saw before.
    """
    saved_module_config = config_module.config
    saved_main_config = main.config

    class _Effective:
        def load(self, env):
            with patch.dict(os.environ, env):
                importlib.reload(config_module)
            fresh = config_module.config
            config_module.config = saved_module_config
            main.config = fresh
            return fresh

        def restore(self):
            main.config = saved_main_config
            config_module.config = saved_module_config

    eff = _Effective()
    yield eff
    eff.restore()


def _captured_hybrid_search_limit():
    """Run the real `_retrieval_leg` and return the top_k it hands to retrieval."""
    captured = {}

    async def fake_hybrid_search(query, top_k, qfilter=None, with_body=False):
        captured["top_k"] = top_k
        return []

    original = (main.expand_query, main.hybrid_search)
    main.expand_query = lambda q: q
    main.hybrid_search = fake_hybrid_search
    try:
        asyncio.run(main._retrieval_leg("funding deals", 1, None))
    finally:
        main.expand_query, main.hybrid_search = original
    return captured["top_k"]


# --- the helper itself -------------------------------------------------


def test_clamped_int_passes_in_range_values_through(monkeypatch):
    monkeypatch.setenv("PROBE_CANDIDATES", "17")
    assert _clamped_int("PROBE_CANDIDATES", 12, 5, 50) == 17


def test_clamped_int_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("PROBE_CANDIDATES", raising=False)
    assert _clamped_int("PROBE_CANDIDATES", 12, 5, 50) == 12


def test_clamped_int_clamps_above_max_and_logs_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("PROBE_CANDIDATES", "5000")
    with caplog.at_level("WARNING", logger="app.config"):
        assert _clamped_int("PROBE_CANDIDATES", 12, 5, 50) == 50
    assert any("PROBE_CANDIDATES" in r.getMessage() for r in caplog.records)


def test_clamped_int_clamps_below_min_and_logs_a_warning(monkeypatch, caplog):
    # 0 and a negative are the damaging low-side values: they leave no
    # candidates to rank at all rather than merely ranking fewer of them.
    for hostile in (0, -5):
        monkeypatch.setenv("PROBE_CANDIDATES", str(hostile))
        with caplog.at_level("WARNING", logger="app.config"):
            assert _clamped_int("PROBE_CANDIDATES", 12, 5, 50) == 5
    assert any("PROBE_CANDIDATES" in r.getMessage() for r in caplog.records)


def test_clamped_int_falls_back_to_default_on_non_integer(monkeypatch, caplog):
    monkeypatch.setenv("PROBE_CANDIDATES", "twelve")
    with caplog.at_level("WARNING", logger="app.config"):
        assert _clamped_int("PROBE_CANDIDATES", 12, 5, 50) == 12
    assert any("PROBE_CANDIDATES" in r.getMessage() for r in caplog.records)


def test_clamped_int_rejects_a_default_outside_the_bounds():
    # A default outside the bounds would make the effective value depend on
    # whether the operator set the variable at all. `ValueError` rather than
    # `AssertionError` on purpose: pytest runs with assertions enabled but the
    # shipped interpreter may be `python -O`, where an `assert` is stripped and
    # this guarantee would silently evaporate. See the `-O` test below.
    with pytest.raises(ValueError):
        _clamped_int("PROBE_CANDIDATES", 500, 5, 50)


def test_clamped_int_default_range_guard_survives_python_O():
    """The guard must not be an `assert`: `python -O` strips those, so the
    guarantee would hold under pytest and not in production. Subprocess both
    ways and require the ValueError in each."""
    import pathlib
    import subprocess
    import sys

    code = """
import sys
sys.path.insert(0, '.')
from app.config import _clamped_int
try:
    _clamped_int('NOPE', 500, 5, 50)
except ValueError:
    print('RAISED')
else:
    print('NO RAISE')
"""
    backend = pathlib.Path(__file__).resolve().parent.parent
    for flags in ([], ["-O"]):
        out = subprocess.run(
            [sys.executable, *flags, "-c", code],
            capture_output=True, text=True, cwd=str(backend), check=False,
        )
        assert "RAISED" in out.stdout, (
            f"default-range guard did not fire under flags={flags or ['(none)']}: "
            f"stdout={out.stdout!r} stderr={out.stderr[-300:]!r}"
        )


# --- RERANK_CANDIDATES at the call site --------------------------------


def test_rerank_candidates_above_max_reaches_call_site_clamped(effective_config):
    effective_config.load({"RERANK_CANDIDATES": "500"})
    assert main.config.RERANK_CANDIDATES == 50
    # Unclamped, 500 would have been handed straight to retrieval.
    assert _captured_hybrid_search_limit() == 50


def test_rerank_candidates_below_min_reaches_call_site_clamped(effective_config):
    effective_config.load({"RERANK_CANDIDATES": "0"})
    assert main.config.RERANK_CANDIDATES == 5
    # max(top_k=1, 0) == 1 without the clamp; the floor wins with it.
    assert _captured_hybrid_search_limit() == 5


# --- body-rescue and search knobs --------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"BODY_RESCUE_STEP": "0"}, 1),           # range(0, n, 0) raises ValueError
        ({"BODY_RESCUE_STEP": "-1"}, 1),
        ({"BODY_RESCUE_STEP": "99999"}, 1500),   # 116ms/body at step=1 on a 50K body
        ({"BODY_RESCUE_WINDOW": "0"}, 200),      # empty excerpt => rescue silently off
        ({"BODY_RESCUE_WINDOW": "999999"}, 8000),
        ({"BODY_RESCUE_MAX_CANDIDATES": "0"}, 1),
        ({"BODY_RESCUE_MAX_CANDIDATES": "5000"}, 50),
        ({"SEARCH_QUERY_MAX_CHARS": "1"}, 32),
        ({"SEARCH_QUERY_MAX_CHARS": "100000"}, 4000),
    ],
)
def test_body_rescue_and_search_knobs_are_clamped(env, expected):
    saved = config_module.config
    try:
        with patch.dict(os.environ, env):
            importlib.reload(config_module)
            fresh = config_module.config
        for name in env:
            assert getattr(fresh, name) == expected, name
    finally:
        config_module.config = saved
        importlib.reload(config_module)
        config_module.config = saved


def test_body_rescue_knobs_default_sane_on_a_clean_environment():
    cfg = config_module.config
    assert cfg.BODY_RESCUE_STEP >= 1
    assert cfg.BODY_RESCUE_WINDOW >= 200
    assert cfg.BODY_RESCUE_MAX_CANDIDATES >= 1
    assert cfg.SEARCH_QUERY_MAX_CHARS >= 32
