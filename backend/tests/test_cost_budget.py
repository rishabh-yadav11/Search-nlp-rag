"""Tests for the daily LLM spend cap (app/cost_budget) and the facet helper.

The cap is enforced by reserving against it BEFORE a billed LLM call and settling
the real cost afterwards, and it fails closed: an unreachable counter store
refuses the call instead of reading as "no spend". The store is modelled by
``FakeBudgetStore``, which implements the server-side contract of the budget
script so concurrency and crash behaviour are exercised without a live Redis.
"""

import ast
import pathlib

import pytest
from _support import run_sync as _run

from app import config as _config_module
from app import cost_budget


def _declared_env_default(var):
    """The literal default shipped in ``os.getenv(var, default)`` inside
    ``app/config.py``, read from the source rather than from the process.

    ``Config`` evaluates that call in its class body, so ``config.<VAR>`` is
    whatever the ambient environment says -- a developer's untracked, gitignored
    ``backend/.env`` included -- which made the shipped default unobservable and
    turned a source-level guard into an environment assertion. Reading the
    declaration keeps the guard on the code that ships, and cannot mutate the
    module 700+ other tests import.
    """
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
            # Unwrap the shipped float(...) conversion so the guard survives
            # dropping or adding it.
            if isinstance(call.func, ast.Name) and len(call.args) == 1:
                call = call.args[0]
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "getenv":
                continue
            # os.getenv(name, default): the default is the second positional arg
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
    """The cap must be ON by default, and nothing else pins it.

    Every other test sets LLM_DAILY_BUDGET_USD through monkeypatch, so the shipped
    value itself was unconstrained: restoring the fail-open default of 0 (= disabled)
    left the whole suite green. That default is the live-billing decision, so it is
    asserted here rather than left to a code comment."""
    assert float(_declared_env_default("LLM_DAILY_BUDGET_USD")) > 0.0


def test_env_example_agrees_with_the_shipped_budget_default():
    """``backend/.env.example`` is the template operators copy, so a default that
    survives review in the code but not in the template still ships a disabled cap
    to every new deployment. Both sides are read from files, so this stays
    independent of any local ``.env``."""
    declared = _declared_env_default("LLM_DAILY_BUDGET_USD")
    example = pathlib.Path(_config_module.__file__).resolve().parent.parent / ".env.example"
    values = [
        line.strip().partition("=")[2].strip()
        for line in example.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("LLM_DAILY_BUDGET_USD=")
    ]
    assert values == [declared]


def test_disabled_budget_reserve_is_a_noop(monkeypatch):
    """LLM_DAILY_BUDGET_USD <= 0 is the documented opt-out: reserve hands back
    an empty id and never opens a store call, so a disabled cap costs nothing
    and cannot fail."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0)
    assert cost_budget._day_key()
    assert _run(cost_budget.reserve(0.05)) == ""
    assert store.calls == []


def test_reserve_holds_then_blocks_and_rejection_creates_no_hold(monkeypatch):
    """A hold counts against the cap for the next caller, and a rejected
    reserve must not leave a hold behind (or rejections would leak budget)."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(3)]
    assert all(ids) and len(set(ids)) == 3
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    assert len(store.holds) == 3
    assert sum(store.holds.values()) == 150_000
    assert store.counter == 0  # holding is not spending


def test_concurrent_reserves_cannot_overspend(monkeypatch):
    """The old code read the counter, called the LLM, then wrote the cost at the
    end of the turn, so 8 turns reserving 0.05 against a 0.15 cap all saw
    "under budget" and all 8 were admitted. Holding the estimate before the call
    means the read-modify-write happens once, on the server, so exactly
    cap/estimate turns get in."""
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
    """With no estimate given, reserve takes the configured LLM_CALL_RESERVE_USD
    so the cap is enforced even on a call path that cannot price the request
    up front."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    monkeypatch.setattr(cost_budget.config, "LLM_CALL_RESERVE_USD", 0.05)
    _run(cost_budget.reserve())
    assert sum(store.holds.values()) == 50_000


def test_reserve_store_down_fails_closed(monkeypatch):
    """An unreadable counter says nothing about how much is left, so the only safe
    answer is to refuse the call. A fail-open read would admit spend from a down
    store, which is the case a cap exists to stop."""
    store = _wire(monkeypatch, BrokenStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    with pytest.raises(cost_budget.BudgetUnavailable) as exc:
        _run(cost_budget.reserve(0.05))
    # A distinct, catchable type: callers must be able to tell "out of money" from
    # "cannot tell".
    assert not isinstance(exc.value, cost_budget.BudgetExceeded)
    assert store.calls  # the store really was tried, then failed
    # The cached script handle is dropped so a restarted server re-registers.
    assert cost_budget._BUDGET_SCRIPT is None


def test_crashed_turn_stays_charged_when_its_hold_lapses(monkeypatch):
    """A turn that reserves and then dies never settles, and the call it covered
    was very likely billed. The sweep therefore CHARGES a lapsed hold to the
    counter: the hold stops occupying the holds table (so one crash cannot starve
    the cap until the day key rolls over) without ever making the crashed call
    free spend, which is what deleting the hold did."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.05)
    monkeypatch.setattr(cost_budget.config, "COST_RESERVATION_TTL_SECONDS", 900)
    crashed = _run(cost_budget.reserve(0.05))
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS + 901)
    # The day is NOT restored: the budget stays spent, so the next turn is
    # still refused rather than being handed the crashed call's money.
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))
    assert crashed not in store.holds
    assert crashed not in store.expires
    assert store.counter == 50_000
    assert store.accounted[crashed] == 50_000


def test_settle_after_a_crash_replaces_the_promoted_estimate(monkeypatch):
    """A turn slow enough that its hold lapses was already charged its
    ESTIMATE by the sweep. When it finally settles, the real cost must REPLACE
    that estimate rather than add to it: 50_000 - 50_000 + 70_000, not
    50_000 + 70_000, which would bill a phantom 0.05 the call never spent."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    monkeypatch.setattr(cost_budget.config, "COST_RESERVATION_TTL_SECONDS", 900)
    slow = _run(cost_budget.reserve(0.05))
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS + 901)
    _run(cost_budget.reserve(0.01))  # any script call runs the sweep
    assert store.counter == 50_000  # the lapsed hold was charged, not freed
    assert store.accounted[slow] == 50_000
    _run(cost_budget.settle([slow], 0.07))
    assert store.counter == 70_000


def test_settling_the_same_reservation_ids_twice_charges_once(monkeypatch):
    """settle is the one and only counter write for a turn. A retried or
    duplicated settle of ids that were already charged must not bill the turn
    again: the failure direction is safe but it can lock a deployment out of
    its budget for the rest of the day."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05)) for _ in range(2)]
    _run(cost_budget.settle(ids, 0.12))
    assert store.counter == 120_000
    _run(cost_budget.settle(ids, 0.12))
    assert store.counter == 120_000
    _run(cost_budget.settle(ids, 0.30))  # even a different amount charges 0
    assert store.counter == 120_000


def test_duplicate_reservation_ids_within_one_settle_charge_once(monkeypatch):
    """A caller that settles a repeated id must be charged once, and a
    repeated PROMOTED id must have its estimate refunded once, not once per
    occurrence (the ledger entry is read back only the first time)."""
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
    """release means "no billed call was made", but a hold the sweep already
    promoted WAS charged to the counter. Refunding it would make the crashed
    turn's cost free again, which is exactly what the ledger prevents."""
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
    """A non-positive LLM_CALL_RESERVE_USD would make every hold worth zero, so
    `total + amount > budget` could never trip and the cap silently admitted
    unlimited concurrent turns. The hold is floored at one micro-USD, so the
    budget still runs out -- at cap/1micro turns -- and no 0 hold is stored."""
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
    """The ordinary path: a hold taken and settled inside its TTL costs the
    actual amount once and leaves no hold behind. Two further 0.05 turns still
    fit in a 0.15 cap, which only holds if exactly 0.05 was charged."""
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
    assert store.counter == 120_000  # the actual cost, not the 0.15 held
    assert store.holds == {}
    assert store.expires == {}
    assert len(store.calls) == 4  # 3 reserves + the single settle write


def test_settle_records_a_call_that_overshot_its_reserve(monkeypatch):
    """Spend that already happened is recorded, not dropped: an over-cap
    counter blocks the NEXT call rather than pretending the call was cheap."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 0.15)
    ids = [_run(cost_budget.reserve(0.05))]
    _run(cost_budget.settle(ids, 0.20))
    assert store.counter == 200_000
    with pytest.raises(cost_budget.BudgetExceeded):
        _run(cost_budget.reserve(0.05))


def test_settle_store_down_keeps_the_hold_counted(monkeypatch):
    """If the settle write cannot land, the turn's spend must not vanish: the
    holds stay on the store, so the budget still reflects what was spent."""
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
    """A turn that reserved and then billed nothing (empty completion, early
    abort) gives the budget back without inventing spend."""
    store = _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05))]
    _run(cost_budget.settle(ids, 0.0))
    assert store.counter == 0
    assert store.holds == {}
    assert _run(cost_budget.reserve(0.05))  # budget is available again


def test_settle_records_spend_when_no_hold_was_taken(monkeypatch):
    """Cost incurred outside a hold (e.g. a retry the caller did not re-reserve
    for) still has to land on the counter, or it is free spend."""
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
    """A hold we cannot drop is still counted, so the caller must be able to
    see that the release did not happen rather than assume it did."""
    _wire(monkeypatch, FakeBudgetStore())
    monkeypatch.setattr(cost_budget.config, "LLM_DAILY_BUDGET_USD", 5.0)
    ids = [_run(cost_budget.reserve(0.05))]
    _wire(monkeypatch, BrokenStore())
    with pytest.raises(cost_budget.BudgetUnavailable):
        _run(cost_budget.release(ids))


def test_budget_reached_warns_once(monkeypatch):
    """The cap being hit is a normal operating state, not a per-request event:
    log it once so a busy hour does not turn the warning into the log."""
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
    """register_script compiles the Lua on the server; re-registering per call
    is waste, but a handle cached across a dead connection must be dropped."""
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
    """Cost accounting is canonical in USD; to_usd converts the INR figure
    reported by LLMResult.cost() before it is stored/compared."""
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 95.6)
    assert abs(cost_budget.to_usd(95.6) - 1.0) < 1e-9
    assert abs(cost_budget.to_usd(0.0) - 0.0) < 1e-9


def test_to_usd_falls_back_to_one_rate_and_warns_once(monkeypatch, caplog):
    """An unconfigured INR_PER_USD must not silently rescale every recorded cost.
    The module falls back to a 1.0 rate -- so the recorded USD is the raw INR
    figure rather than a wrong conversion -- and warns, but only ONCE: a
    misconfigured deploy would otherwise warn on every turn for the life of the
    process."""
    monkeypatch.setattr(cost_budget.config, "INR_PER_USD", 0.0)
    monkeypatch.setattr(cost_budget, "_inr_fallback_warned", False)

    with caplog.at_level("WARNING", logger="cost_budget"):
        assert cost_budget.to_usd(42.0) == 42.0
        assert cost_budget.to_usd(0.0) == 0.0
        assert cost_budget.to_usd(7.5) == 7.5

    assert caplog.text.count("INR_PER_USD") == 1
    # A configured rate must NOT warn -- the fallback is a misconfiguration path.
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


# Fixed clock for every test: the budget script takes "now" as an argument, so
# the expiry sweep is driven by moving this instead of sleeping.
_BASE_TS = 1_700_000_000


class FakeBudgetStore:
    """In-memory model of the Redis side of ``_BUDGET_LUA``.

    It implements the contract the script relies on rather than echoing the call:
    one atomic operation that (1) sweeps lapsed holds into real spend, (2) compares
    the counter plus every live hold against the cap, and (3) mutates -- with no
    await in the middle. That is what makes the concurrency test mean anything:
    the store refuses the 4th simultaneous 0.05 hold against a 0.15 cap exactly
    as Redis would.

    ``accounted`` is the fourth key: reservation id -> micros already charged on
    that id's behalf (0 once settled). The sweep writes a crashed hold's estimate
    there after charging it, settle reads it back to replace that estimate with
    the real cost, and its presence is what makes a repeated settle a no-op. The
    same scenarios run against the real Lua in ``test_budget_lua.py``, so the two
    cannot drift apart silently.
    """

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
            # The hold was never added to the counter (reserve only writes the
            # hash), so the actual cost is added whole -- discounting the hold
            # again here would lose the difference on every cheap call -- less
            # whatever the sweep already charged for these ids.
            #
            # Three ledger states, and collapsing any two of them loses money:
            # absent -> a live hold never charged (charge it); value > 0 -> the
            # sweep charged this id's estimate (refund it and charge the real
            # cost instead, a replacement not a sum); value == 0 -> an earlier
            # settle already charged it (charge nothing, which is what makes
            # settle idempotent).
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
            # Deliberately no refund: a hold the sweep promoted was already
            # billed, and releasing it must not make that spend free again.
            for rid in ids:
                self.holds.pop(rid, None)
                self.expires.pop(rid, None)
            return 0
        raise AssertionError(f"unknown budget script mode {mode!r}")

    def _sweep(self, now):
        """Charge every lapsed hold to the counter instead of freeing it: the
        call it covered was very likely billed, and a crash must not be free
        spend. The amount is recorded as already accounted for so a late
        settle of the same id replaces it with the real cost."""
        for rid in [rid for rid, at in self.expires.items() if at <= now]:
            held = self.holds.pop(rid, 0)
            if held:
                self.counter += held
                self.accounted[rid] = held
            del self.expires[rid]


class BrokenStore:
    """Redis that is unreachable: every script call raises."""

    def __init__(self):
        self.calls = []
        self.registrations = 0

    def execute(self, keys, args):
        raise ConnectionError("redis down")


class _FakeRedis:
    """Just enough of aioredis.Redis to serve the registered script."""

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
    """Point the module at ``store`` as its Redis, with a fixed clock, and drop
    any script handle cached against a previous client."""
    monkeypatch.setattr(cost_budget, "_client", lambda: _FakeRedis(store))
    monkeypatch.setattr(cost_budget, "_BUDGET_SCRIPT", None)
    monkeypatch.setattr(cost_budget, "_now_ts", lambda: _BASE_TS)
    return store


def _gather(*coros):
    """Run coroutines concurrently against one store, collecting the outcomes
    instead of letting the first failure cancel the rest."""
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
