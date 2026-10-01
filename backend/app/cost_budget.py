"""Daily LLM spend cap that fails closed."""
import logging
import secrets
import time
from collections.abc import Sequence
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("cost_budget")

_redis = None

_COST_SCALE = 1_000_000
_COST_KEY_PREFIX = "llm:cost:micro"

# This Lua is the cap's source of truth -- tests execute it under lua5.1 -- so Python only marshals args into it.
# reserve/settle/sweep exists so a crashed turn cannot leak a hold forever: a lapsed hold is CHARGED, never refunded.
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

-- The holds hash and the holds-expiry zset must OUTLIVE the instant a hold
-- first becomes sweepable. A hold's zset score is `now + hold_ttl` and the
-- sweep predicate is `score <= now`, so the earliest moment the promotion of a
-- crashed call is POSSIBLE is exactly hold_ttl after the reserve -- which is
-- precisely when containers expired at hold_ttl are gone. Redis expires
-- lazily, on the first command issued after the deadline, and that command is
-- the sweep itself: the containers would be deleted in the same breath in
-- which the promotion became observable, and a crashed billed call's spend
-- would be silently lost. Two TTLs of slack leaves a full further hold_ttl
-- window in which ANY budget call promotes the lapsed hold.
local container_ttl = hold_ttl * 2

-- Sweep lapsed holds first, in every mode. A turn that reserved and then
-- crashed never settles, and an un-swept hold would keep eating budget until
-- the day key rolled over. A lapsed hold is PROMOTED, not freed: the call it
-- covered was very likely made and billed, so its estimate is charged to the
-- counter and recorded in KEYS[4] as already-accounted. Dropping the hold
-- instead would hand out the budget and make a crashed call free spend.
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
  -- Spend already incurred (counter) plus every live hold plus this estimate
  -- is what the cap is measured against. Reading the holds here, in the same
  -- script run that creates the hold, is what makes concurrent reserves safe.
  -- A zero (or negative) hold is floored at one micro-USD: `total + amount >
  -- budget` would otherwise be untrippable and a misconfigured
  -- LLM_CALL_RESERVE_USD would silently turn the cap into a no-op. This floor
  -- is deliberately reserve-only -- a zero-cost settle must still be a
  -- no-op -- and mirrors the TTL floors above.
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
  -- released hold here would double-discount it and quietly lose the
  -- difference every time a call costs less than its estimate.
  --
  -- `amount` is what the call really cost and is deliberately allowed to
  -- exceed the held estimate: spend that already happened is recorded, not
  -- refused, and the resulting over-cap total blocks the NEXT call.
  --
  -- An EMPTY id list is never idempotent: nothing identifies what is being
  -- charged, so `amount` is written whole or spend incurred outside a hold
  -- would be lost.
  local new_charge = amount
  if all_accounted then new_charge = 0 end
  local next_counter = counter + new_charge - promoted
  if next_counter < 0 then next_counter = 0 end
  redis.call('SET', KEYS[1], next_counter, 'EX', counter_ttl)
  -- Tombstones: accounted, with nothing outstanding. A later settle of the
  -- same ids sees them and charges nothing. No TTL here on purpose: they must
  -- survive past hold_ttl so a late settle cannot double charge.
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
  -- deliberately does NOT refund KEYS[4] -- a promoted id was already
  -- charged to the counter by the sweep, and refunding it here would make a
  -- crashed turn's cost free again.
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

_BUDGET_SCRIPT = None

_inr_fallback_warned = False
_budget_reached_warned = False


class BudgetExceeded(Exception):
    """Raised when the configured daily LLM spend cap is already exhausted."""


class BudgetUnavailable(Exception):
    """The counter store is unreachable or unreadable, so nothing is known about what was spent."""


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
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
    return f"{_day_key()}:holds:done"


def _now_ts() -> int:
    return int(time.time())


def _to_micros(usd: float) -> int:
    return round(usd * _COST_SCALE)


def _budget_micros() -> int:
    return _to_micros(config.LLM_DAILY_BUDGET_USD)


async def _run_script(mode: str, amount_micros: int, ids: Sequence[str], new_id: str) -> int:
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
        _BUDGET_SCRIPT = None
        raise BudgetUnavailable(f"daily cost counter unavailable: {exc}") from exc


async def reserve(estimate_usd: float = 0.0) -> str:
    """Hold ``estimate_usd`` before a billed call; raises BudgetExceeded if the hold would pass today's cap."""
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
    """Charge ``actual_usd`` to today and drop the holds; idempotent, so a repeated settle charges nothing."""
    ids = [rid for rid in reservation_ids if rid]
    amount = max(0, _to_micros(actual_usd))
    if not ids and amount == 0:
        return
    await _run_script("settle", amount, ids, "")


async def release(reservation_ids: Sequence[str]) -> None:
    ids = [rid for rid in reservation_ids if rid]
    if not ids:
        return
    await _run_script("release", 0, ids, "")


def to_usd(cost_inr: float) -> float:
    global _inr_fallback_warned
    rate = config.INR_PER_USD
    if not rate:
        if not _inr_fallback_warned:
            logger.warning(
                "INR_PER_USD not configured; falling back to 1.0, so recorded USD costs will be wrong"
            )
            _inr_fallback_warned = True
        rate = 1.0
    return cost_inr / rate
