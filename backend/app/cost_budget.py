"""Daily LLM spend cap that fails closed.

Tracks cumulative LLM cost for the current UTC day in Redis DB
ANALYTICS_REDIS_DB (same DB as analytics, so counters survive the query-cache
flush on deploys). When LLM_DAILY_BUDGET_USD > 0 and the running total plus
the outstanding holds would pass the cap, the call is refused instead of
racking up unbilled spend.

The cap is enforced as RESERVE / SETTLE around every billed LLM call:

* ``reserve(estimate_usd)`` holds the estimate against today's cap *before* the
  call runs and returns an opaque reservation id.
* ``settle(ids, actual_usd)`` drops those holds and writes the ACTUAL cost --
  the one and only counter write for the turn.
* ``release(ids)`` drops holds for a turn that made no billed call.

Holding before the call closes the check-then-act window: "read the counter, call
the LLM, write the counter at the end" lets N concurrent turns all read "under
budget" and overshoot by N times the per-call cost. Here the whole
read-modify-write happens once, inside a single Lua script, so at most
``cap / per-call reserve`` turns can be in flight at once.

Everything about the store fails CLOSED. An unreachable Redis raises
``BudgetUnavailable`` instead of quietly reading "no spend": a fail-open read
means a down, flushed or misconfigured counter admits unbounded spend, which is
precisely the case a cap exists to stop. A ``settle`` that cannot reach the
store raises too and leaves the holds in place, so nothing the turn did is
written off.

A hold that is never settled -- the worker crashed, the container was killed
mid-stream, the client disconnected -- expires after
COST_RESERVATION_TTL_SECONDS. Every script call sweeps expired holds before
doing anything else, and the sweep CHARGES a lapsed hold to the counter instead
of handing the budget back: a crashed turn's reserved estimate stays billed for
the rest of the UTC day. Deleting the hold instead would make every crash free
spend, and keeping it forever would let one crash starve the cap until the day
key rolls over.

Two consequences worth stating plainly:

* A hold that lapses while its turn is still running is charged its estimate,
  and the turn's later ``settle`` replaces that estimate with the real cost. The
  day total is therefore never under-counted; it can briefly over-count the
  estimate while the slow call is in flight.
* ``settle`` is idempotent: the first settle of a reservation id charges it, and
  a repeat charges nothing, so a retried settle cannot inflate the counter.
"""
import logging
import secrets
import time
from collections.abc import Sequence
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("cost_budget")

_redis = None

# Store the daily counter as integer micro-USD (1 USD == _COST_SCALE units)
# instead of a float. Many small increments otherwise drift in floating point;
# integer accumulation is exact and the comparison against the cap is done in
# the same units.
#
# The counter uses a DISTINCT key namespace (`llm:cost:micro:...`) from the
# legacy USD-valued `llm:cost:day:...` key: a pre-existing USD-valued key would
# otherwise be misread (divided by _COST_SCALE) and have micro-USD added to a
# USD value, corrupting that day's total until the key rolls.
_COST_SCALE = 1_000_000
_COST_KEY_PREFIX = "llm:cost:micro"

# ONE script serves all three operations. Doing the cap check and the mutation
# in a single server-side script is the whole point: any read from Python and a
# later write from Python is a TOCTOU window that lets concurrent turns
# overspend.
#
# KEYS[1] today's spent counter (integer micro-USD)
# KEYS[2] live holds hash: reservation id -> held micro-USD
# KEYS[3] holds expiry zset: reservation id -> epoch seconds it lapses at
# KEYS[4] holds-accounted hash: reservation id -> micro-USD already charged on
#        that id's behalf. Lapsed holds are promoted into the counter with
#        their amount recorded here, so a later settle REPLACES the estimate
#        instead of adding to it, and a repeated settle charges nothing.
#
# ARGV[1] mode: 'reserve' | 'settle' | 'release'
# ARGV[2] 'now' in epoch seconds (passed in, never redis.call('TIME'): the
#          clock has to be controllable from tests)
# ARGV[3] reservation TTL seconds (COST_RESERVATION_TTL_SECONDS)
# ARGV[4] counter TTL seconds (COST_DAY_TTL_SECONDS)
# ARGV[5] daily cap in micro-USD; <= 0 means the cap is disabled
# ARGV[6] amount in micro-USD: the estimate to hold, or the actual to settle
# ARGV[7] reservation id to create (reserve mode)
# ARGV[8..] reservation ids to drop (settle / release modes)
#
# Returns 1 when a reserve was rejected for exceeding the cap, 0 otherwise.
_BUDGET_LUA = """
local mode = ARGV[1]
local now = tonumber(ARGV[2])
local hold_ttl = tonumber(ARGV[3])
local counter_ttl = tonumber(ARGV[4])
local budget = tonumber(ARGV[5])
local amount = tonumber(ARGV[6])
local new_id = ARGV[7]
-- EXPIRE/SET ... EX reject a non-positive TTL with an error, which would turn a
-- misconfigured TTL into every call failing closed. Floor it at one second.
if hold_ttl < 1 then hold_ttl = 1 end
if counter_ttl < 1 then counter_ttl = 1 end

-- The holds hash and the holds-expiry zset must OUTLIVE the instant a hold first
-- becomes sweepable. A hold's zset score is `now + hold_ttl` and the sweep
-- predicate is `score <= now`, so the earliest moment a crashed call's promotion
-- is possible is exactly hold_ttl after the reserve -- precisely when containers
-- expired at hold_ttl are gone, and Redis expires them lazily on the first
-- command after the deadline, which is the sweep itself. Two TTLs of slack
-- leaves a full further hold_ttl window in which ANY budget call promotes the
-- lapsed hold.
local container_ttl = hold_ttl * 2

-- Sweep lapsed holds first, in every mode. A turn that reserved and then crashed
-- never settles, and an un-swept hold would keep eating budget until the day key
-- rolled over. A lapsed hold is PROMOTED, not freed: the call it covered was
-- very likely made and billed, so its estimate is charged to the counter and
-- recorded in KEYS[4] as already-accounted. Dropping the hold instead would hand
-- out the budget and make a crashed call free spend.
local lapsed = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for i = 1, #lapsed do
  local v = redis.call('HGET', KEYS[2], lapsed[i])
  if v then
    redis.call('INCRBY', KEYS[1], v)
    redis.call('HSET', KEYS[4], lapsed[i], v)
    -- INCRBY keeps an existing TTL, so a promotion cannot outlive the day.
    redis.call('EXPIRE', KEYS[1], counter_ttl)
  end
  redis.call('HDEL', KEYS[2], lapsed[i])
  redis.call('ZREM', KEYS[3], lapsed[i])
end

local counter = tonumber(redis.call('GET', KEYS[1]) or '0')

if mode == 'reserve' then
  -- Spend already incurred (counter) plus every live hold plus this estimate is
  -- what the cap is measured against. Reading the holds here, in the same script
  -- run that creates the hold, is what makes concurrent reserves safe.
  -- A zero (or negative) hold is floored at one micro-USD: `total + amount >
  -- budget` would otherwise be untrippable and a misconfigured
  -- LLM_CALL_RESERVE_USD would silently turn the cap into a no-op. This floor is
  -- deliberately reserve-only -- a zero-cost settle must still be a no-op.
  if amount < 1 then amount = 1 end
  local total = counter
  local held = redis.call('HVALS', KEYS[2])
  for i = 1, #held do
    total = total + tonumber(held[i])
  end
  if budget > 0 and total + amount > budget then
    return 1
  end
  redis.call('HSET', KEYS[2], new_id, amount)
  redis.call('ZADD', KEYS[3], now + hold_ttl, new_id)
  redis.call('EXPIRE', KEYS[2], container_ttl)
  redis.call('EXPIRE', KEYS[3], container_ttl)
  return 0
end

if mode == 'settle' then
  -- What the sweep already charged on these ids' behalf. It is refunded
  -- because the real cost is written here instead, so a hold that lapsed
  -- mid-call and settled afterwards costs its ACTUAL cost once, not the
  -- estimate plus the actual cost.
  --
  -- The ledger holds three distinguishable states, and collapsing any two of
  -- them loses money:
  --   absent       -> a live hold that was never charged: charge `amount`
  --   value > 0    -> the sweep charged this id's ESTIMATE: refund it and
  --                   charge the real cost instead (a replacement, not a sum)
  --   value == 0   -> a previous settle already charged it: charge nothing,
  --                   which is what makes settle idempotent
  -- `seen` keeps an id repeated inside ONE call from refunding its own
  -- promotion twice.
  local promoted = 0
  local all_accounted = (#ARGV >= 8)
  local seen = {}
  for i = 8, #ARGV do
    local raw = redis.call('HGET', KEYS[4], ARGV[i])
    if not raw then
      all_accounted = false
    else
      local acc = tonumber(raw)
      if acc > 0 then
        all_accounted = false
        if not seen[ARGV[i]] then promoted = promoted + acc end
      end
      seen[ARGV[i]] = true
    end
    redis.call('HDEL', KEYS[2], ARGV[i])
    redis.call('ZREM', KEYS[3], ARGV[i])
  end
  -- A hold is NOT spend and was never added to the counter -- reserve only
  -- HSET/ZADDs it -- so the real cost is added whole here. Subtracting the
  -- released hold would double-discount it and quietly lose the difference every
  -- time a call costs less than its estimate.
  --
  -- `amount` is what the call really cost and is deliberately allowed to exceed
  -- the held estimate: spend that already happened is recorded, not refused, and
  -- the resulting over-cap total blocks the NEXT call.
  --
  -- An EMPTY id list is never idempotent: nothing identifies what is being
  -- charged, so `amount` is written whole or spend incurred outside a hold would
  -- be lost.
  local new_charge = amount
  if all_accounted then new_charge = 0 end
  local next_counter = counter + new_charge - promoted
  if next_counter < 0 then next_counter = 0 end
  redis.call('SET', KEYS[1], next_counter, 'EX', counter_ttl)
  -- Tombstones: accounted, with nothing outstanding, so a later settle of the
  -- same ids charges nothing. No TTL here on purpose: they must survive past
  -- hold_ttl so a late settle cannot double charge.
  for i = 8, #ARGV do
    redis.call('HSET', KEYS[4], ARGV[i], 0)
  end
  -- The ledger is only as good as the day it belongs to; drop it with the day
  -- counter so tomorrow's reservations are never mistaken for today's.
  redis.call('EXPIRE', KEYS[4], counter_ttl)
  redis.call('EXPIRE', KEYS[2], container_ttl)
  redis.call('EXPIRE', KEYS[3], container_ttl)
  return 0
end

if mode == 'release' then
  -- Drops live holds and records nothing: no billed call was made. It
  -- deliberately does NOT refund KEYS[4] -- a promoted id was already charged to
  -- the counter by the sweep, and refunding it here would make a crashed turn's
  -- cost free again.
  for i = 8, #ARGV do
    redis.call('HDEL', KEYS[2], ARGV[i])
    redis.call('ZREM', KEYS[3], ARGV[i])
  end
  redis.call('EXPIRE', KEYS[2], container_ttl)
  redis.call('EXPIRE', KEYS[3], container_ttl)
  return 0
end

return -1
"""

# Registered once at first use and reused; register_script compiles the Lua
# source on the server, so re-registering on every call is wasteful.
_BUDGET_SCRIPT = None

_inr_fallback_warned = False
_budget_reached_warned = False


class BudgetExceeded(Exception):
    """Raised when the configured daily LLM spend cap is already exhausted."""


class BudgetUnavailable(Exception):
    """Raised when the spend counter store cannot be reached or read.

    Distinct from BudgetExceeded on purpose: an unreachable store says nothing
    about how much has been spent, and the only safe answer to "how much is
    left?" that we cannot prove is "refuse the call". Callers must treat it as
    a hard failure rather than an outage to be shrugged off."""


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        # Pin the DB explicitly so the daily cost counter never silently lands in
        # DB 0 (which a deploy FLUSHDB would wipe), disabling the budget
        # guardrail. The ``db`` kwarg overrides any db segment in REDIS_URL.
        _redis = aioredis.from_url(
            config.REDIS_URL,
            db=config.ANALYTICS_REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _day_key() -> str:
    return f"{_COST_KEY_PREFIX}:{datetime.now(UTC).strftime('%Y-%m-%d')}"


def _holds_key() -> str:
    return f"{_day_key()}:holds"


def _holds_expiry_key() -> str:
    return f"{_day_key()}:holds:exp"


def _holds_done_key() -> str:
    """Hash of reservation ids already charged to the counter (value 0 once settled)
    -- the ledger that makes a repeated settle idempotent."""
    return f"{_day_key()}:holds:done"


def _now_ts() -> int:
    """Current epoch seconds, as the script's clock. A module function (not an
    inline ``time.time()``) so the expiry sweep is testable: tests move the clock
    forward instead of sleeping."""
    return int(time.time())


def _to_micros(usd: float) -> int:
    """Convert a USD amount to integer micro-USD for exact counter storage.

    This is the single rounding point: the USD value is rounded to the nearest
    micro-USD here, so callers must not round again before this conversion.
    """
    return round(usd * _COST_SCALE)


def _budget_micros() -> int:
    """The daily cap expressed in the same integer micro-USD units as the counter."""
    return _to_micros(config.LLM_DAILY_BUDGET_USD)


async def _run_script(mode: str, amount_micros: int, ids: Sequence[str], new_id: str) -> int:
    """Run one _BUDGET_LUA invocation, converting any store failure to
    BudgetUnavailable. Returns the script's status code (1 == rejected)."""
    global _BUDGET_SCRIPT
    try:
        if _BUDGET_SCRIPT is None:
            _BUDGET_SCRIPT = _client().register_script(_BUDGET_LUA)
        return await _BUDGET_SCRIPT(
            keys=[_day_key(), _holds_key(), _holds_expiry_key(), _holds_done_key()],
            args=[
                mode,
                _now_ts(),
                config.COST_RESERVATION_TTL_SECONDS,
                config.COST_DAY_TTL_SECONDS,
                _budget_micros(),
                amount_micros,
                new_id,
                *ids,
            ],
        )
    except Exception as exc:
        # Drop the cached handle: a dead connection (or a server that restarted
        # and lost the script) must not be reused for the rest of the process.
        _BUDGET_SCRIPT = None
        raise BudgetUnavailable(f"daily cost counter unavailable: {exc}") from exc


async def reserve(estimate_usd: float = 0.0) -> str:
    """Hold ``estimate_usd`` against today's cap and return a reservation id.

    Call this immediately BEFORE a billed LLM call and keep the id until the turn
    ends: ``settle`` turns the hold into the real cost, ``release`` drops it when
    the call never happened. A non-positive ``estimate_usd`` falls back to the
    configured LLM_CALL_RESERVE_USD per-call hold. A hold is floored at one
    micro-USD: a zero hold would leave the cap's arithmetic unable to trip and
    silently admit every turn.

    Returns ``""`` (touching the store not at all) when the cap is disabled with
    LLM_DAILY_BUDGET_USD <= 0. Raises BudgetExceeded when the hold would pass the
    cap -- counting today's spend *and* every live hold, so concurrent turns
    contend for the same budget -- and BudgetUnavailable when the store cannot be
    reached.
    """
    if config.LLM_DAILY_BUDGET_USD <= 0:
        return ""
    amount = _to_micros(estimate_usd if estimate_usd > 0 else config.LLM_CALL_RESERVE_USD)
    reservation_id = secrets.token_hex(8)
    rejected = await _run_script("reserve", amount, (), reservation_id)
    if rejected:
        global _budget_reached_warned
        if not _budget_reached_warned:
            logger.warning(
                "LLM daily budget reached; refusing LLM call before it is made (fail closed)"
            )
            _budget_reached_warned = True
        raise BudgetExceeded("daily LLM spend cap exhausted")
    return reservation_id


async def settle(reservation_ids: Sequence[str], actual_usd: float) -> None:
    """Drop the given holds and record ``actual_usd`` as today's spend.

    This is the only counter write for a turn, and it happens once. ``actual_usd``
    may exceed what was reserved: already-incurred spend is recorded rather than
    dropped, which can put the counter over the cap and block the next call --
    the correct outcome. A non-positive ``actual_usd`` releases the holds without
    incrementing anything.

    Raises BudgetUnavailable if the store cannot be reached; the holds are then
    left in place (the script never ran), so the turn's cost is neither written
    off nor double counted once Redis returns.

    Settling the same ids again is a no-op: a reservation id is charged once, so
    a retried or duplicated settle cannot inflate the counter. A hold that lapsed
    while the turn was still running was already charged its estimate by the
    sweep, and this call replaces that estimate rather than adding to it.
    """
    ids = [rid for rid in reservation_ids if rid]
    amount = max(0, _to_micros(actual_usd))
    if not ids and amount == 0:
        return
    await _run_script("settle", amount, ids, "")


async def release(reservation_ids: Sequence[str]) -> None:
    """Drop the given holds without recording anything (no billed call was made)."""
    ids = [rid for rid in reservation_ids if rid]
    if not ids:
        return
    await _run_script("release", 0, ids, "")


def to_usd(cost_inr: float) -> float:
    """Convert an INR cost (as reported by ``LLMResult.cost()``) to USD.

    All cost accounting in this project is canonical in USD (the daily budget
    counter, analytics, and stored message costs), so callers convert at the
    recording boundary instead of mixing units. Falls back to a 1.0 rate if
    INR_PER_USD is unset (warning once, since that misconfiguration yields wrong
    costs).
    """
    global _inr_fallback_warned
    from app.fx_rate import rate_usd_inr

    rate = rate_usd_inr()
    if not rate:
        if not _inr_fallback_warned:
            logger.warning(
                "INR_PER_USD not configured; falling back to 1.0, so recorded USD costs will be wrong"
            )
            _inr_fallback_warned = True
        rate = 1.0
    return cost_inr / rate
