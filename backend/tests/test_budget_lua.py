"""Runs the REAL ``_BUDGET_LUA`` source under lua5.1.

``test_cost_budget.py`` exercises the Python side of the spend cap against
``FakeBudgetStore``, which is only a *model* of what the server executes. If
the shipped Lua drifted from that model -- a renamed key, an inverted
condition, a missing write -- every Python test would still pass and the cap
would quietly stop capping. So the actual script text is written to disk, a
small harness binds ``KEYS``/``ARGV`` as Redis binds them over an in-memory
stub of the primitives the script uses, and the same scenarios are asserted
against the real thing.

One scenario per invocation (``lua5.1 harness.lua <name>``) so a failure names
itself, and every check prints the scenario it belongs to. Skipped, loudly,
where lua5.1 is unavailable (CI without it must not fail; it must also not
pretend to have covered the script).
"""

import re
import shutil
import subprocess
import textwrap

import pytest

from app import cost_budget

LUA = shutil.which("lua5.1")

pytestmark = pytest.mark.skipif(
    LUA is None, reason="lua5.1 not available; the shipped Lua script is not executed"
)

# The stub implements only what _BUDGET_LUA calls, and fails loudly on
# anything else so an unexpected new command shows up as an error rather than
# as a silently missing write. GET/HGET return a number-as-string and `false`
# for a missing hash field, exactly as a Redis reply reaches Lua.
HARNESS = """
local SCENARIO = arg[1]
local SCRIPT = assert(loadfile('budget.lua'))

local store, fails = nil, 0

-- Real key expiry. Redis deletes a key at the instant its TTL runs out, and
-- the first command issued after that deadline sees a MISSING key. Because
-- expiry is LAZY, that command is usually the very one that needed the data --
-- so with EXPIRE a no-op, no scenario could tell a container that outlives the
-- promotion window from one that dies the same instant the promotion becomes
-- possible, and a crash that is never charged looks identical to one that is.
--
-- The clock is the one the SCRIPT is handed (ARGV[2]), so a scenario picks the
-- instant a key dies by choosing `now`.
local function newstore() store = { c = 0, h = {}, z = {}, d = {}, exp = {}, now = 0 } end
local function hash(key) return key == 'd' and store.d or store.h end
local function holdcount() local n = 0; for _ in pairs(store.h) do n = n + 1 end; return n end
local function donecount() local n = 0; for _ in pairs(store.d) do n = n + 1 end; return n end

-- Drop every key whose TTL has run out. A rolled-over day counter reads as
-- zero, which is exactly how the script already treats a missing one.
local function expire_due()
  for k, at in pairs(store.exp) do
    if at and at <= store.now then
      store.exp[k] = nil
      if k == 'c' then store.c = 0 else store[k] = {} end
    end
  end
end

-- Real Redis REJECTS a non-positive TTL with an error instead of storing the
-- key forever. Without this, a scenario cannot tell a floored TTL from a
-- missing floor.
local function real_ttl(v, cmd)
  if not v or v < 1 then error("invalid expire time in '" .. cmd .. "' command") end
  return v
end

redis = {}

function redis.call(cmd, key, ...)
  expire_due()
  local a = { ... }
  if cmd == 'GET' then return tostring(store.c or 0) end
  if cmd == 'SET' then
    for i = 1, #a do
      if a[i] == 'EX' then store.exp[key] = store.now + real_ttl(tonumber(a[i + 1]), 'set') end
    end
    store.c = tonumber(a[1]) return 'OK'
  end
  if cmd == 'INCRBY' then store.c = (store.c or 0) + tonumber(a[1]) return store.c end
  if cmd == 'HGET' then return hash(key)[a[1]] or false end
  if cmd == 'HSET' then hash(key)[a[1]] = tonumber(a[2]) return 1 end
  if cmd == 'HDEL' then hash(key)[a[1]] = nil return 1 end
  if cmd == 'HVALS' then
    local out = {}
    for _, v in pairs(hash(key)) do out[#out + 1] = tostring(v) end
    return out
  end
  if cmd == 'ZADD' then store.z[a[2]] = tonumber(a[1]) return 1 end
  if cmd == 'ZREM' then store.z[a[1]] = nil return 1 end
  if cmd == 'ZRANGEBYSCORE' then
    local out = {}
    for m, s in pairs(store.z) do if s <= tonumber(a[2]) then out[#out + 1] = m end end
    table.sort(out)
    return out
  end
  if cmd == 'EXPIRE' then store.exp[key] = store.now + real_ttl(tonumber(a[1]), 'expire') return 1 end
  error('stub missing command: ' .. tostring(cmd))
end

-- run(mode, now, hold_ttl, budget, amount, new_id, ids, counter_ttl) -> the script's return value
local function run(mode, now, hold_ttl, budget, amount, new_id, ids, counter_ttl)
  KEYS = { 'c', 'h', 'z', 'd' }
  counter_ttl = counter_ttl or 604800
  local argv = { mode, tostring(now), tostring(hold_ttl), tostring(counter_ttl), tostring(budget), tostring(amount), new_id or '' }
  for _, id in ipairs(ids or {}) do argv[#argv + 1] = id end
  ARGV = argv
  store.now = tonumber(now)
  return SCRIPT()
end

local function check(ok, name, extra)
  if ok then
    print('  ok   ' .. name)
  else
    fails = fails + 1
    print('  FAIL ' .. name .. (extra and ('  <' .. tostring(extra) .. '>') or ''))
  end
end

local S = {}

-- The cap is measured against spend plus every LIVE hold, so 0.15 admits 3 of
-- 8 simultaneous 0.05 turns -- the check-then-act fix.
function S.reserve_caps_concurrent_turns()
  newstore()
  local admitted, rejected = 0, 0
  for i = 1, 8 do
    if run('reserve', 1000, 900, 150000, 50000, 'id' .. i) == 0 then admitted = admitted + 1
    else rejected = rejected + 1 end
  end
  check(admitted == 3, 'exactly 3 of 8 reserves admitted (got ' .. admitted .. ')')
  check(rejected == 5, '5 rejected (got ' .. rejected .. ')')
  check(holdcount() == 3, 'exactly 3 holds stored (got ' .. holdcount() .. ')')
  check(store.c == 0, 'holding is not spending (counter ' .. store.c .. ')')
end

function S.rejected_reserve_leaves_no_hold()
  newstore()
  run('reserve', 1000, 900, 60000, 50000, 'a')
  check(run('reserve', 1000, 900, 60000, 50000, 'b') == 1, 'second reserve rejected')
  check(holdcount() == 1, 'the rejected reserve left no hold (got ' .. holdcount() .. ')')
end

-- The ordinary path: the ACTUAL cost is written once, whole. A hold was never
-- added to the counter, so discounting it again would lose the difference on
-- every cheap call.
function S.settle_records_actual_once()
  newstore()
  run('reserve', 1000, 900, 150000, 50000, 'a')
  run('reserve', 1000, 900, 150000, 50000, 'b')
  run('settle', 1000, 900, 150000, 120000, '', { 'a', 'b' })
  check(store.c == 120000, 'counter == 0.12 actual, not 0.15 held (got ' .. store.c .. ')')
  check(holdcount() == 0, 'all holds cleared (got ' .. holdcount() .. ')')
end

function S.settle_overshoot_is_recorded()
  newstore()
  run('reserve', 1000, 900, 150000, 50000, 'a')
  run('settle', 1000, 900, 150000, 200000, '', { 'a' })
  check(store.c == 200000, 'counter carries the real 0.20 (got ' .. store.c .. ')')
  check(run('reserve', 1000, 900, 150000, 50000, 'next') == 1, 'the next call is refused')
end

-- A crashed turn never settles. The sweep must CHARGE its lapsed hold, not
-- hand the budget back: the call it covered was very likely billed, and a
-- crash is not free spend. The hold still leaves the holds table, so a crash
-- cannot starve the cap until the day key rolls over.
function S.crashed_hold_is_charged_when_it_lapses()
  newstore()
  run('reserve', 1000, 900, 60000, 50000, 'crash')
  check(run('reserve', 1000, 900, 60000, 50000, 'next') == 1, 'a second reserve is blocked by the hold')
  run('reserve', 1900, 900, 200000, 50000, 'later')  -- any call runs the sweep
  check(store.c == 50000, 'the lapsed hold was charged to the counter (got ' .. store.c .. ')')
  check(holdcount() == 1, 'the hold is gone and only the live one remains (got ' .. holdcount() .. ')')
  check(store.d['crash'] == 50000, 'the charge is on the ledger (got ' .. tostring(store.d['crash']) .. ')')
  -- With no hold outstanding, a further 0.05 turn is still refused against a
  -- 0.06 cap: the crashed call's money was charged, not handed back.
  run('release', 1900, 900, 200000, 0, '', { 'later' })
  check(run('reserve', 1900, 900, 60000, 50000, 'after') == 1, 'the day is NOT restored')
end

-- The promotion is an ESTIMATE. A slow call that settles afterwards must have
-- it replaced by the real cost -- 50000 - 50000 + 70000 -- and not added to
-- it, which would bill a phantom 0.05 the call never spent.
function S.settle_replaces_the_promoted_estimate()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'slow')
  run('reserve', 1900, 900, 500000, 1000, 'trigger')  -- any call runs the sweep
  check(store.c == 50000, 'estimate charged on lapse (got ' .. store.c .. ')')
  run('settle', 1900, 900, 500000, 70000, '', { 'slow' })
  check(store.c == 70000, 'promoted estimate replaced by the real 0.07 (got ' .. store.c .. ')')
  check(store.d['slow'] == 0, 'the id is left as a tombstone (got ' .. tostring(store.d['slow']) .. ')')
end

function S.settling_the_same_ids_twice_charges_once()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'a')
  run('reserve', 1000, 900, 500000, 50000, 'b')
  run('settle', 1000, 900, 500000, 120000, '', { 'a', 'b' })
  run('settle', 1000, 900, 500000, 120000, '', { 'a', 'b' })
  check(store.c == 120000, 'a repeated settle charges nothing (got ' .. store.c .. ')')
  run('settle', 1000, 900, 500000, 300000, '', { 'a' })
  check(store.c == 120000, 're-settling one of them charges nothing (got ' .. store.c .. ')')
end

function S.duplicate_ids_within_one_settle_charge_once()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'a')
  run('reserve', 1000, 900, 500000, 50000, 'b')
  run('settle', 1000, 900, 500000, 120000, '', { 'a', 'a', 'b', 'b' })
  check(store.c == 120000, 'a repeated live id is charged once (got ' .. store.c .. ')')

  run('reserve', 1000, 900, 500000, 50000, 'crash')
  run('reserve', 1900, 900, 500000, 1000, 'trigger')
  run('settle', 1900, 900, 500000, 70000, '', { 'crash', 'crash' })
  check(store.c == 170000 - 50000 + 70000, 'a repeated promoted id refunds once (got ' .. store.c .. ')')
end

-- release means "no billed call was made" -- but a hold the sweep already
-- promoted WAS charged, and refunding it here would make the crash free again.
function S.release_never_refunds_a_promoted_charge()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'crash')
  run('release', 1900, 900, 500000, 0, '', { 'crash' })
  check(store.c == 50000, 'the promoted charge survives a release (got ' .. store.c .. ')')
  check(holdcount() == 0, 'the hold is gone (got ' .. holdcount() .. ')')
end

function S.release_drops_a_live_hold()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'a')
  run('release', 1000, 900, 500000, 0, '', { 'a' })
  check(store.c == 0 and holdcount() == 0, 'counter left at 0 with no holds')
  check(donecount() == 0, 'a released hold is not recorded as charged')
end

-- A non-positive LLM_CALL_RESERVE_USD used to make every hold worth zero, so
-- `total + amount > budget` could never trip and the cap admitted unlimited
-- concurrent turns. The hold is floored at one micro-USD, and the floor is
-- reserve-only: a zero-cost settle must still be a no-op.
function S.zero_hold_is_floored_so_the_cap_still_bites()
  newstore()
  local admitted = 0
  for i = 1, 6 do
    if run('reserve', 1000, 900, 4, 0, 'z' .. i) == 0 then admitted = admitted + 1 end
  end
  check(admitted == 4, 'a 4 micro cap admits exactly 4 zero-cost turns (got ' .. admitted .. ')')
  local total = 0
  for _, v in pairs(store.h) do total = total + v end
  check(total == 4, 'every hold is worth 1 micro-USD (got ' .. total .. ')')
end

function S.zero_settle_is_a_noop()
  newstore()
  run('reserve', 1000, 900, 150000, 50000, 'a')
  run('settle', 1000, 900, 150000, 0, '', { 'a' })
  check(store.c == 0 and holdcount() == 0, 'a zero settle released without recording')
end

-- Cost incurred outside a hold (a retry the caller did not re-reserve for)
-- still has to land, or it is free spend.
function S.settle_with_no_ids_still_bills()
  newstore()
  run('settle', 1000, 900, 150000, 20000, '', {})
  check(store.c == 20000, 'spend with no reservation lands (got ' .. store.c .. ')')
  run('settle', 1000, 900, 150000, 20000, '', {})
  check(store.c == 40000, 'and is not treated as idempotent (got ' .. store.c .. ')')
end

function S.disabled_cap_admits_everything()
  newstore()
  local ok = true
  for i = 1, 5 do if run('reserve', 1000, 900, 0, 50000, 'd' .. i) ~= 0 then ok = false end end
  check(ok, 'cap disabled -> all 5 admitted')
end

-- A CRASH is not free spend -- and the promotion has to be REACHABLE, not just
-- arithmetically correct. Nothing here refreshes the holds containers between
-- the reserve and the sweep: the one later call is the first command the store
-- sees. A hold's score first satisfies `score <= now` exactly hold_ttl after
-- the reserve, which is also when containers expired at hold_ttl are gone. The
-- spend then evaporates and nothing reports it.
--
-- All three modes re-arm the container TTL, so a settle or a release in the
-- gap loses the charge just as a second reserve would; each is exercised below.
function S.crashed_hold_is_charged_without_a_keepalive()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'crash')
  check(store.c == 0, 'a live hold is not spend yet (got ' .. store.c .. ')')
  run('reserve', 1900, 900, 500000, 1000, 'later')
  check(store.c == 50000, 'the crashed turn is charged with no keepalive in between (got ' .. store.c .. ')')
  check(store.d['crash'] == 50000, 'the charge is on the ledger (got ' .. tostring(store.d['crash']) .. ')')
  -- Re-sweeping must not charge it a second time.
  run('reserve', 1900, 900, 500000, 1000, 'later2')
  check(store.c == 50000, 'a repeated sweep does not double charge (got ' .. store.c .. ')')
  -- The containers outlived the promotion instead of dying with it, so the
  -- live holds are still being counted against the cap.
  check(holdcount() == 2, 'the two live holds are still held (got ' .. holdcount() .. ')')
end

-- The SAME reachability property through the settle and release branches, which
-- also re-arm the container TTL. Under the old `EXPIRE ... hold_ttl` each of
-- these reset the containers' deadline to now + hold_ttl, so a hold created
-- just before one still died the instant it became sweepable. Sweeping at the
-- boundary `score == now` (1900, containers exp 2800) pins the near edge;
-- sweeping well past it (2799) pins the far edge, so a future change cannot
-- pass by luck at one instant.
function S.crashed_hold_survives_a_settle_or_release_in_the_gap()
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'crash')
  run('settle', 1000, 900, 500000, 0, '', { 'other' })   -- re-arms the TTLs
  run('reserve', 1900, 900, 500000, 1000, 'later')
  check(store.c == 50000, 'a settle in the gap does not lose the charge (got ' .. store.c .. ')')

  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'crash')
  run('release', 1000, 900, 500000, 0, '', { 'other' })  -- re-arms the TTLs
  run('reserve', 2799, 900, 500000, 1000, 'later')
  check(store.c == 50000, 'a release in the gap does not lose the charge (got ' .. store.c .. ')')
end

-- The TTL floors, one per scenario. Real Redis REJECTS `EXPIRE key 0` and
-- `SET key val EX 0` with an error, so a misconfigured
-- COST_RESERVATION_TTL_SECONDS=0 or COST_DAY_TTL_SECONDS=0 would make every
-- budget call fail closed and take chat down deployment-wide. Deleting either
-- guard has to fail a test that exercises THAT floor.
function S.zero_hold_ttl_is_floored()
  newstore()
  check(pcall(run, 'reserve', 1000, 0, 150000, 50000, 'ttl'), 'a zero hold_ttl does not error out')
  -- The floor is one second, so the hold is due at 1001 and is still
  -- promotable then: it was stored, not silently dropped.
  run('reserve', 1001, 0, 150000, 1000, 'later')
  check(store.c == 50000, 'the floored hold is charged when it lapses (got ' .. store.c .. ')')
end

function S.zero_counter_ttl_is_floored()
  newstore()
  -- The settle's own `SET KEYS[1] <total> EX counter_ttl` is the write that
  -- records the day's spend; a rejected TTL would lose the cost of every turn.
  check(pcall(run, 'settle', 1000, 900, 150000, 50000, '', { 'a' }, 0), 'a zero counter_ttl does not error out of a settle')
  check(store.c == 50000, 'the settle still records the cost (got ' .. store.c .. ')')
  -- The sweep re-arms the day counter with EXPIRE, which rejects 0 just the same.
  newstore()
  run('reserve', 1000, 900, 500000, 50000, 'crash')
  check(pcall(run, 'reserve', 1900, 900, 500000, 1000, 'later', {}, 0), 'a zero counter_ttl does not error out of the sweep')
  check(store.c == 50000, 'the promotion still lands (got ' .. store.c .. ')')
end

assert(SCENARIO, 'no scenario named on the command line')
print('scenario: ' .. SCENARIO)
assert(S[SCENARIO], 'unknown scenario: ' .. SCENARIO)
newstore()
S[SCENARIO]()
if fails == 0 then
  print('  -> ' .. SCENARIO .. ' OK')
  os.exit(0)
end
os.exit(1)
"""

SCENARIOS = [
    "reserve_caps_concurrent_turns",
    "rejected_reserve_leaves_no_hold",
    "settle_records_actual_once",
    "settle_overshoot_is_recorded",
    "crashed_hold_is_charged_when_it_lapses",
    "crashed_hold_is_charged_without_a_keepalive",
    "crashed_hold_survives_a_settle_or_release_in_the_gap",
    "settle_replaces_the_promoted_estimate",
    "settling_the_same_ids_twice_charges_once",
    "duplicate_ids_within_one_settle_charge_once",
    "release_never_refunds_a_promoted_charge",
    "release_drops_a_live_hold",
    "zero_hold_is_floored_so_the_cap_still_bites",
    "zero_settle_is_a_noop",
    "settle_with_no_ids_still_bills",
    "disabled_cap_admits_everything",
    "zero_hold_ttl_is_floored",
    "zero_counter_ttl_is_floored",
]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    """The shipped Lua text plus the harness, written next to each other so the
    harness can loadfile() the script exactly as Redis would receive it."""
    workdir = tmp_path_factory.mktemp("budget_lua")
    (workdir / "budget.lua").write_text(cost_budget._BUDGET_LUA, encoding="utf-8")
    (workdir / "harness.lua").write_text(textwrap.dedent(HARNESS).lstrip(), encoding="utf-8")
    return workdir


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_budget_lua_scenario(harness, scenario):
    """One scenario against the real script; the harness prints the name of any
    individual check that failed."""
    proc = subprocess.run(
        [LUA, "harness.lua", scenario],
        cwd=harness,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"{scenario} failed:\n{proc.stdout}\n{proc.stderr}"


def test_harness_defines_every_scenario_pytest_runs():
    """Guard the two lists against drifting apart: a scenario in SCENARIOS but
    missing from the Lua would fail on a Lua assert, and one defined in the Lua
    but missing from SCENARIOS would never run at all -- silently."""
    assert set(re.findall(r"^function S\.(\w+)\(\)", HARNESS, re.MULTILINE)) == set(SCENARIOS)
