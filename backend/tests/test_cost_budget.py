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


def test_record_cost_skips_nonpositive(monkeypatch):
    calls = []
    monkeypatch.setattr(cost_budget, "_client", lambda: _fake_client(calls))
    _run(cost_budget.record_cost(0.0))
    _run(cost_budget.record_cost(-1.0))
    assert calls == []


def test_record_cost_redis_down_never_raises(monkeypatch):
    monkeypatch.setattr(cost_budget, "_client", _raise_runtime_error_client)
    _run(cost_budget.record_cost(0.5))  # must not raise


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

    async def fake_request(**kwargs):
        return {
            "result": {
                "hits": [
                    {"value": "Finance", "count": 3},
                    {"value": "TMT", "count": 5},
                    {"value": "General", "count": 2},
                ]
            },
            "status": "ok",
        }

    class FakeApiClient:
        @staticmethod
        async def request(**kwargs):
            return await fake_request(**kwargs)

    class FakeCollectionsApi:
        api_client = FakeApiClient()

    class FakeHttp:
        collections_api = FakeCollectionsApi()

    main.state["qdrant"] = type("FakeQdrant", (), {"http": FakeHttp()})()
    values = _run(main._facet_values("industry_names"))
    assert values == ["Finance", "General", "TMT"]
