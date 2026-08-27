"""Tests for the daily LLM spend cap (app/cost_budget) and the facet helper.

The spend cap must fail closed (refuse LLM calls once the daily budget is
exhausted) while remaining best-effort when Redis is unreachable. The facet
helper is tested against a fake Qdrant HTTP layer so the endpoint logic is
exercised without a live server.
"""

import pytest

from app import cost_budget


def test_disabled_budget_never_blocks(monkeypatch):
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0)
    assert cost_budget._day_key()
    _run(cost_budget.assert_within_budget())  # must not raise


def test_assert_within_budget_raises_when_exhausted(monkeypatch):
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 1.0)
    monkeypatch.setattr(cost_budget, "spend_today", _async(lambda: 1.0))
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.assert_within_budget())


def test_assert_within_budget_allows_under_cap(monkeypatch):
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 1.0)
    monkeypatch.setattr(cost_budget, "spend_today", _async(lambda: 0.5))
    _run(cost_budget.assert_within_budget())  # must not raise


def test_assert_within_budget_redis_down_allows(monkeypatch):
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 1.0)
    monkeypatch.setattr(cost_budget, "_client", _raise_runtime_error_client)
    _run(cost_budget.assert_within_budget())  # never block on Redis outage


def test_spend_today_redis_down_returns_zero_and_warns(monkeypatch):
    """ERROR PATH — Redis down on read: fail open, assume no spend, warn once."""
    monkeypatch.setattr(cost_budget, "_client", _raise_runtime_error_client)
    warnings = []
    monkeypatch.setattr(cost_budget.logger, "warning", lambda *a, **k: warnings.append(a))
    assert _run(cost_budget.spend_today()) == 0.0
    assert len(warnings) == 1


def test_spend_today_reads_counter(monkeypatch):
    class Client:
        async def get(self, key):
            return "2.5"

    monkeypatch.setattr(cost_budget, "_client", lambda: Client())
    assert _run(cost_budget.spend_today()) == 2.5


def test_spend_today_zero_when_no_counter(monkeypatch):
    class Client:
        async def get(self, key):
            return None

    monkeypatch.setattr(cost_budget, "_client", lambda: Client())
    assert _run(cost_budget.spend_today()) == 0.0


def test_record_cost_skips_nonpositive(monkeypatch):
    calls = []
    monkeypatch.setattr(cost_budget, "_client", lambda: _fake_client(calls))
    _run(cost_budget.record_cost(0.0))
    _run(cost_budget.record_cost(-1.0))
    assert calls == []


def test_record_cost_redis_down_never_raises(monkeypatch):
    """ERROR PATH — Redis down on write: skip recording, warn once, no raise."""
    monkeypatch.setattr(cost_budget, "_client", _raise_runtime_error_client)
    warnings = []
    monkeypatch.setattr(cost_budget.logger, "warning", lambda *a, **k: warnings.append(a))
    _run(cost_budget.record_cost(0.5))  # must not raise
    assert len(warnings) == 1


def test_record_cost_converts_inr_to_usd(monkeypatch):
    """Regression: the day counter is compared against LLM_DAILY_BUDGET_USD, so
    INR cost from LLMResult.cost() must be converted to USD before incrementing."""
    calls = []
    monkeypatch.setattr(cost_budget, "_client", lambda: _fake_client(calls))
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 95.6)
    _run(cost_budget.record_cost(95.6))
    assert len(calls) == 1
    assert abs(calls[0][1] - 1.0) < 1e-9  # ₹95.6 -> $1.0


def test_to_usd_canonical_unit(monkeypatch):
    """Cost accounting is canonical in USD; to_usd converts the INR figure
    reported by LLMResult.cost() before it is stored/compared."""
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 95.6)
    assert abs(cost_budget.to_usd(95.6) - 1.0) < 1e-9
    assert abs(cost_budget.to_usd(0.0) - 0.0) < 1e-9


def test_close_resets_redis(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    redis = FakeRedis()
    monkeypatch.setattr(cost_budget, "_redis", redis)
    _run(cost_budget.close())
    assert redis.closed is True
    assert cost_budget._redis is None


def test_close_noop_when_no_redis(monkeypatch):
    monkeypatch.setattr(cost_budget, "_redis", None)
    _run(cost_budget.close())  # must not raise
    assert cost_budget._redis is None


def test_client_lazy_init_replaces_redis_db(monkeypatch):
    """_client builds REDIS_URL pointing at ANALYTICS_REDIS_DB and reuses the
    connection across calls."""
    created = []

    class FakeRedis:
        pass

    def fake_from_url(url, **kwargs):
        created.append((url, kwargs))
        return FakeRedis()

    monkeypatch.setattr(cost_budget, "_redis", None)
    monkeypatch.setattr(cost_budget.aioredis, "from_url", fake_from_url)
    monkeypatch.setattr(cost_budget.config, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(cost_budget.config, "ANALYTICS_REDIS_DB", 1)
    client = cost_budget._client()
    assert cost_budget._client() is client  # cached, not recreated
    assert len(created) == 1
    assert created[0][0] == "redis://localhost:6379/0"
    assert created[0][1]["db"] == 1
    assert created[0][1]["decode_responses"] is True


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _async(fn):
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def raise_runtime_error(*args, **kwargs):
    raise RuntimeError("redis down")


def _fake_client(calls):
    class Client:
        async def incrbyfloat(self, key, amount):
            calls.append((key, amount))
            return amount

    return Client()


def _raise_runtime_error_client(*args, **kwargs):
    class Client:
        async def get(self, key):
            raise RuntimeError("redis down")

        async def incrbyfloat(self, key, amount):
            raise RuntimeError("redis down")

    return Client()


def test_facet_values_from_fake_qdrant(monkeypatch):
    from app import main

    class _Point:
        def __init__(self, payload):
            self.payload = payload

    async def fake_scroll(**kwargs):
        return (
            [
                _Point({"industry_names": "Finance"}),
                _Point({"industry_names": "TMT"}),
                _Point({"industry_names": "General"}),
            ],
            None,
        )

    main.state["qdrant"] = type("FakeQdrant", (), {"scroll": staticmethod(fake_scroll)})()
    values = _run(main._facet_values("industry_names"))
    assert values == ["Finance", "General", "TMT"]
