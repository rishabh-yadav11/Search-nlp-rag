"""Tests for the live USD→INR rate refresh (app/fx_rate.py).

The refresh path is a wrapper over a synchronous urllib call, so the tests pin
the fetch and parse edge cases by swapping in a fake fetch and asserting the
published rate and the fall-through on failure. The background loop's cadence is
already covered by config semantics (0 disables it).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import fx_rate
from app.config import config


def _monkeypatch_rate(monkeypatch, value):
    """Serve ``value`` from _fetch (a float, or None to simulate failure)."""
    monkeypatch.setattr(fx_rate, "_rate", config.INR_PER_USD)
    monkeypatch.setattr(fx_rate, "_fetch", lambda: value)


def test_refresh_publishes_fetched_rate(monkeypatch) -> None:
    _monkeypatch_rate(monkeypatch, 96.88)
    asyncio.run(fx_rate.refresh())
    assert fx_rate.rate_usd_inr() == 96.88


def test_refresh_keeps_fallback_on_fetch_failure(monkeypatch) -> None:
    _monkeypatch_rate(monkeypatch, None)
    asyncio.run(fx_rate.refresh())
    assert fx_rate.rate_usd_inr() == config.INR_PER_USD


def test_rate_usd_inr_starts_at_config_fallback() -> None:
    assert fx_rate.rate_usd_inr() == config.INR_PER_USD


def test_fetch_parses_feed_shape(monkeypatch) -> None:
    # A self-contained parse check via a fake urlopen: the real feed shape is
    # {"result":"success","rates":{"INR":96.88,...}} (see the module docstring).
    payload = {"result": "success", "rates": {"INR": 96.883664, "USD": 1.0}}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake_urlopen(url, timeout):
        assert url == config.FX_RATE_API_URL
        return _FakeResp()

    monkeypatch.setattr(fx_rate.urllib.request, "urlopen", fake_urlopen)
    assert fx_rate._fetch() == pytest.approx(96.883664)


def test_fetch_rejects_non_positive_rate(monkeypatch) -> None:
    payload = {"result": "success", "rates": {"INR": 0.0}}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(fx_rate.urllib.request, "urlopen", lambda url, timeout: _FakeResp())
    assert fx_rate._fetch() is None
