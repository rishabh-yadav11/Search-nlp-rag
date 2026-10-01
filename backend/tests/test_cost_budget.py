"""Daily LLM spend cap: reserve before a billed call, settle the real cost after,
refuse the call when the counter store is unreachable."""

import ast
import pathlib

import pytest
from _support import run_sync as _run

from app import config as _config_module
from app import cost_budget


def _declared_env_default(var):
    """The default shipped in ``os.getenv(var, default)``, read from source: the
    class-body call would read the ambient environment instead."""
    source = pathlib.Path(_config_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name != "Config":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                continue
            if not any(getattr(t, "id", None) == var for t in stmt.targets):
                continue
            call = stmt.value
            if isinstance(call.func, ast.Name) and len(call.args) == 1:
                call = call.args[0]
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "getenv":
                continue
            args = call.args
            if len(args) < 2 or getattr(args[0], "value", None) != var:
                continue
            default = args[1]
            if isinstance(default, ast.Constant) and isinstance(default.value, str):
                return default.value
    raise AssertionError(
        f"app/config.py no longer declares os.getenv({var!r}, <literal default>) "
        f"as Config.{var}; point this guard at the new declaration instead of "
        "letting the shipped default go unchecked"
    )


def test_shipped_default_cap_is_not_disabled():
    """Every other test monkeypatches the cap, so this is the only guard on the shipped default being fail-open."""
    assert float(_declared_env_default("LLM_DAILY_BUDGET_USD")) > 0.0


def test_env_example_agrees_with_the_shipped_budget_default():
    """Operators copy this template, so both sides are read from files and a local ``.env`` cannot make this pass."""
    declared = _declared_env_default("LLM_DAILY_BUDGET_USD")
    example = pathlib.Path(_config_module.__file__).resolve().parent.parent / ".env.example"
    values = [
        line.strip().partition("=")[2].strip()
        for line in example.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("LLM_DAILY_BUDGET_USD=")
    ]
    assert values == [declared]


def test_disabled_budget_reserve_is_a_noop(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0)
    assert cost_budget._day_key()
    assert _run(cost_budget.reserve(0.05)) == ""
    assert store.calls == []


def test_reserve_holds_then_blocks_and_rejection_creates_no_hold(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(3)]
    assert all(ids) and len(set(ids)) == 3
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    assert len(store.holds) == 3
    assert sum(store.holds.values()) == 150_000
    assert store.counter == 0


def test_concurrent_reserves_cannot_overspend(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    results = _gather(*(cost_budget.reserve(0.05) for _ in range(8)))
    admitted = [r for r in results if isinstance(r, str)]
    rejected = [r for r in results if isinstance(r, cost_budget.BudgetExceeded)]
    assert len(admitted) == 3
    assert len(rejected) == 5
    assert len(store.holds) == 3
    assert sum(store.holds.values()) <= 150_000


def test_reserve_defaults_to_configured_per_call_hold(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    monkeypatch.setattr(cost_budget.config, "LLM_CALL_RESERVE_USD", 0.05)
    _run(cost_budget.reserve())
    assert sum(store.holds.values()) == 50_000


def test_reserve_store_down_fails_closed(monkeypatch):
    store = _wire(monkeypatch, BrokenStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    with pytest.raises(cost_budget.BudgetUnavailable) as exc:
        _run(cost_budget.reserve(0.05))
    assert not isinstance(exc.value, cost_budget.BudgetExceeded)
    assert store.calls
    assert cost_budget._BUDGET_SCRIPT is None


def test_crashed_turn_stays_charged_when_its_hold_lapses(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.05)
    monkeypatch.setattr(cost_budget.config, "COST_RESERVATION_TTL_SECONDS", 900)
    crashed = _run(cost_budget.reserve(0.05))
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS + 901)
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    assert crashed not in store.holds
    assert crashed not in store.expires
    assert store.counter == 50_000
    assert store.accounted[crashed] == 50_000


def test_settle_after_a_crash_replaces_the_promoted_estimate(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    monkeypatch.setattr(cost_budget.config, "COST_RESERVATION_TTL_SECONDS", 900)
    slow = _run(cost_budget.reserve(0.05))
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS + 901)
    _run(cost_budget.reserve(0.01))  # any script call runs the sweep
    assert store.counter == 50_000
    assert store.accounted[slow] == 50_000
    _run(cost_budget.settle([slow], 0.07))
    assert store.counter == 70_000


def test_settling_the_same_reservation_ids_twice_charges_once(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(2)]
    _run(cost_budget.settle(ids, 0.12))
    assert store.counter == 120_000
    _run(cost_budget.settle(ids, 0.12))
    assert store.counter == 120_000
    _run(cost_budget.settle(ids, 0.30))
    assert store.counter == 120_000


def test_duplicate_reservation_ids_within_one_settle_charge_once(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    monkeypatch.setattr(cost_budget.config, "COST_RESERVATION_TTL_SECONDS", 900)
    live = [_run(cost_budget.reserve(0.05)) for _ in range(2)]
    _run(cost_budget.settle([live[0], live[0], live[1], live[1]], 0.12))
    assert store.counter == 120_000

    crashed = _run(cost_budget.reserve(0.05))
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS + 901)
    _run(cost_budget.reserve(0.01))  # any script call runs the sweep
    assert store.counter == 120_000 + 50_000
    _run(cost_budget.settle([crashed, crashed], 0.07))
    assert store.counter == 170_000 - 50_000 + 70_000


def test_release_never_refunds_a_charge_the_sweep_already_made(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.05)
    monkeypatch.setattr(cost_budget.config, "COST_RESERVATION_TTL_SECONDS", 900)
    crashed = _run(cost_budget.reserve(0.05))
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS + 901)
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    _run(cost_budget.release([crashed]))
    assert store.counter == 50_000


def test_zero_per_call_reserve_knob_cannot_switch_the_cap_off(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.000004)
    monkeypatch.setattr(cost_budget.config, "LLM_CALL_RESERVE_USD", 0.0)
    admitted = [_run(cost_budget.reserve(0.0)) for _ in range(4)]
    assert all(admitted)
    assert set(store.holds.values()) == {1}
    assert sum(store.holds.values()) == 4
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.0))


def test_live_hold_that_settles_normally_is_charged_exactly_once(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    first = _run(cost_budget.reserve(0.05))
    _run(cost_budget.settle([first], 0.05))
    assert store.counter == 50_000
    assert store.holds == {}
    assert all(_run(cost_budget.reserve(0.05)) for _ in range(2))
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))


def test_settle_writes_the_actual_cost_once_and_clears_holds(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(3)]
    _run(cost_budget.settle(ids, 0.12))
    assert store.counter == 120_000
    assert store.holds == {}
    assert store.expires == {}
    assert len(store.calls) == 4


def test_settle_records_a_call_that_overshot_its_reserve(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    ids = [_run(cost_budget.reserve(0.05))]
    _run(cost_budget.settle(ids, 0.20))
    assert store.counter == 200_000
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))


def test_settle_store_down_keeps_the_hold_counted(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(3)]
    broken = _wire(monkeypatch, BrokenStore())
    with pytest.raises(cost_budget.BudgetUnavailable):
        _run(cost_budget.settle(ids, 0.12))
    assert broken.calls
    _wire(monkeypatch, store)
    assert len(store.holds) == 3
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))


def test_settle_without_spend_releases_holds_only(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05))]
    _run(cost_budget.settle(ids, 0.0))
    assert store.counter == 0
    assert store.holds == {}
    assert _run(cost_budget.reserve(0.05))


def test_settle_records_spend_when_no_hold_was_taken(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    _run(cost_budget.settle([], 0.02))
    assert store.counter == 20_000


def test_no_holds_and_no_spend_never_touches_the_store(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    _run(cost_budget.settle([], 0.0))
    _run(cost_budget.release([]))
    assert store.calls == []


def test_release_drops_holds_without_touching_the_counter(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(2)]
    _run(cost_budget.settle(ids[:1], 0.03))
    _run(cost_budget.release(ids[1:]))
    assert store.counter == 30_000
    assert store.holds == {}
    assert _run(cost_budget.reserve(0.05))


def test_release_store_down_fails_closed(monkeypatch):
    _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05))]
    _wire(monkeypatch, BrokenStore())
    with pytest.raises(cost_budget.BudgetUnavailable):
        _run(cost_budget.release(ids))


def test_budget_reached_warns_once(monkeypatch):
    _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.05)
    monkeypatch.setattr(cost_budget, "_budget_reached_warned", False)
    warnings = []
    monkeypatch.setattr(cost_budget.logger, "warning", lambda *a, **k: warnings.append(a))
    _run(cost_budget.reserve(0.05))
    for _ in range(3):
        with pytest.raises(cost_budget.BudgetExceeded):
            _run(cost_budget.reserve(0.05))
    assert len(warnings) == 1


def test_script_is_registered_once_and_reregistered_after_failure(monkeypatch):
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    for _ in range(3):
        _run(cost_budget.reserve(0.05))
    assert store.registrations == 1
    _wire(monkeypatch, BrokenStore())
    with pytest.raises(cost_budget.BudgetUnavailable):
        _run(cost_budget.reserve(0.05))
    _wire(monkeypatch, store)
    _run(cost_budget.reserve(0.05))
    assert store.registrations == 2


def test_to_usd_canonical_unit(monkeypatch):
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 95.6)
    assert abs(cost_budget.to_usd(95.6) - 1.0) < 1e-9
    assert abs(cost_budget.to_usd(0.0) - 0.0) < 1e-9


def test_to_usd_falls_back_to_one_rate_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 0.0)
    monkeypatch.setattr(cost_budget, "_inr_fallback_warned", False)

    with caplog.at_level("WARNING", logger="cost_budget"):
        assert cost_budget.to_usd(42.0) == 42.0
        assert cost_budget.to_usd(0.0) == 0.0
        assert cost_budget.to_usd(7.5) == 7.5

    assert caplog.text.count("INR_PER_USD") == 1
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 100.0)
    caplog.clear()
    assert cost_budget.to_usd(100.0) == 1.0
    assert caplog.text == ""


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
    _run(cost_budget.close())
    assert cost_budget._redis is None


def test_client_lazy_init_replaces_redis_db(monkeypatch):
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
    assert cost_budget._client() is client
    assert len(created) == 1
    assert created[0][0] == "redis://localhost:6379/0"
    assert created[0][1]["db"] == 1
    assert created[0][1]["decode_responses"] is True


# The script takes "now" as an argument, so the sweep is driven by moving this clock, not by sleeping.
_BASE_TS = 1_700_000_000


class FakeBudgetStore:
    """In-memory model of the shipped ``_BUDGET_LUA``; both ship, so drift leaves
    these tests green while the real cap stops capping."""

    def __init__(self):
        self.counter = 0  # integer micro-USD actually spent
        self.holds = {}  # reservation id -> held micro-USD
        self.expires = {}  # reservation id -> epoch seconds it lapses at
        self.accounted = {}  # reservation id -> micros already charged for it
        self.calls = []  # (keys, args) of every script invocation
        self.registrations = 0

    def execute(self, keys, args):
        mode, now, hold_ttl, _counter_ttl, budget, amount, new_id = args[:7]
        ids = args[7:]
        now, budget, amount = int(now), int(budget), int(amount)
        self._sweep(now)
        if mode == "reserve":
            # Reserve-only floor: a zero hold would make the cap untrippable.
            amount = max(1, amount)
            total = self.counter + sum(self.holds.values()) + amount
            if budget > 0 and total > budget:
                return 1
            self.holds[new_id] = amount
            self.expires[new_id] = now + int(hold_ttl)
            return 0
        if mode == "settle":
            # Three ledger states, and collapsing any two loses money: absent ->
            # charge it; > 0 -> the sweep already charged the estimate, so replace
            # it; 0 -> an earlier settle already charged it, so charge nothing.
            promoted = 0
            all_accounted = bool(ids)
            seen = set()
            for rid in ids:
                if rid not in self.accounted:
                    all_accounted = False
                elif self.accounted[rid] > 0:
                    all_accounted = False
                    if rid not in seen:
                        promoted += self.accounted[rid]
                seen.add(rid)
                self.holds.pop(rid, None)
                self.expires.pop(rid, None)
            new_charge = 0 if all_accounted else amount
            self.counter = max(0, self.counter + new_charge - promoted)
            for rid in ids:
                self.accounted[rid] = 0
            return 0
        if mode == "release":
            # No refund: a hold the sweep promoted was already billed, so
            # releasing it must not make that spend free.
            for rid in ids:
                self.holds.pop(rid, None)
                self.expires.pop(rid, None)
            return 0
        raise AssertionError(f"unknown budget script mode {mode!r}")

    def _sweep(self, now):
        """Charge lapsed holds rather than freeing them, recording each as
        accounted so a late settle replaces the estimate."""
        for rid in [rid for rid, at in self.expires.items() if at <= now]:
            held = self.holds.pop(rid, 0)
            if held:
                self.counter += held
                self.accounted[rid] = held
            del self.expires[rid]


class BrokenStore:

    def __init__(self):
        self.calls = []
        self.registrations = 0

    def execute(self, keys, args):
        raise ConnectionError("redis down")


class _FakeRedis:

    def __init__(self, store):
        self._store = store

    def register_script(self, lua):
        self._store.registrations += 1
        store = self._store

        class _Script:
            async def __call__(self, keys=None, args=None, client=None):
                store.calls.append((list(keys), list(args)))
                return store.execute(keys, args)

        return _Script()


def _wire(monkeypatch, store):
    """Point the module at ``store``, with a fixed clock and no cached script handle."""
    monkeypatch.setattr(cost_budget, "_client", lambda: _FakeRedis(store))
    monkeypatch.setattr(cost_budget, "_BUDGET_SCRIPT", None)
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS)
    return store


def _gather(*coros):
    """Collect every outcome so one rejection does not cancel the rest."""
    import asyncio

    async def _all():
        return await asyncio.gather(*coros, return_exceptions=True)

    return asyncio.run(_all())


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

    class FakeQdrant:
        scroll = staticmethod(fake_scroll)

    monkeypatch.setattr(main, "AsyncQdrantClient", lambda *a, **k: FakeQdrant())
    main.state["qdrant"] = main.AsyncQdrantClient()
    values = _run(main._facet_values("industry_names"))
    assert values == ["Finance", "General", "TMT"]
