"""Daily LLM spend cap that fails closed.

Tracks cumulative LLM cost for the current UTC day in Redis DB
ANALYTICS_REDIS_DB (same DB as analytics, so counters survive the query-cache
flush on deploys). When LLM_DAILY_BUDGET_USD > 0 and the running total exceeds
the cap, calls are refused (fail closed) instead of racking up unbilled spend.

Recording is best-effort: a Redis outage never raises into the request path —
we degrade to "over budget" only on a confirmed counter read, and skip recording
on write failure.
"""
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("cost_budget")

_redis = None

# Store the daily counter as integer micro-USD (1 USD == _COST_SCALE units)
# instead of a float. Many small INCRBYFLOAT calls otherwise drift in floating
# point; integer accumulation is exact and the comparison against the cap is
# done in the same units.
#
# The counter uses a DISTINCT key namespace (`llm:cost:micro:...`) from the
# legacy USD-valued `llm:cost:day:...` key. A pre-existing USD-valued key at
# deploy time would otherwise be misread (divided by _COST_SCALE) and have
# micro-USD added to a USD value, corrupting that day's total until the key
# rolls. The separate key never collides with any legacy USD key.
_COST_SCALE = 1_000_000
_COST_KEY_PREFIX = "llm:cost:micro"

# One Lua script makes the "increment + enforce cap + refresh TTL" sequence
# atomic on the server, so concurrent requests can't each read "under budget"
# and then all increment past LLM_DAILY_BUDGET_USD (a TOCTOU race). If the
# increment would exceed the cap it is rolled back (fail closed) and the
# script returns 1; otherwise 0.
_RECORD_LUA = """
local cur = redis.call('incrbyfloat', KEYS[1], ARGV[1])
redis.call('expire', KEYS[1], ARGV[2])
local budget = tonumber(ARGV[3])
if budget > 0 and tonumber(cur) > budget then
  redis.call('incrbyfloat', KEYS[1], -tonumber(ARGV[1]))
  return 1
end
return 0
"""

# Registered once at first use and reused; register_script compiles the Lua
# source on the server, so re-registering on every call is wasteful.
_RECORD_SCRIPT = None

_inr_fallback_warned = False
_budget_reached_warned = False


class BudgetExceeded(Exception):
    """Raised when the configured daily LLM spend cap is already exhausted."""


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        # Pin the DB explicitly so the daily cost counter never silently lands in
        # DB 0 (which a deploy FLUSHDB would wipe), disabling the budget guardrail.
        # The ``db`` kwarg overrides any db segment in REDIS_URL.
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


async def spend_today() -> float:
    """Cumulative LLM cost for today (USD). Returns 0 on Redis failure.

    The stored counter is integer micro-USD; it is divided back to USD here so
    the rest of the app keeps working in its canonical USD unit."""
    try:
        raw = await _client().get(_day_key())
        if not raw:
            return 0.0
        return float(raw) / _COST_SCALE
    except Exception as exc:
        logger.warning("cost budget Redis unavailable (%s); assuming no spend", exc)
        return 0.0


async def assert_within_budget() -> None:
    """Raise BudgetExceeded when today's spend already hits the cap.

    Disabled (cap <= 0) or unreadable Redis -> always allowed; we never block a
    request because the counter store is down, but we do fail closed on a
    confirmed over-budget read.
    """
    if config.LLM_DAILY_BUDGET_USD <= 0:
        return
    if await spend_today() >= config.LLM_DAILY_BUDGET_USD:
        raise BudgetExceeded()


async def record_cost(cost_inr: float) -> None:
    """Add ``cost_inr`` (INR, as reported by LLMResult.cost()) to today's
    running total, converted to USD (then integer micro-USD) so the counter is
    comparable to LLM_DAILY_BUDGET_USD. Best-effort, never raises.

    The increment, cap enforcement, and TTL refresh run in one atomic Lua
    script: if the new total would exceed the daily cap the increment is rolled
    back and the cost is not recorded (fail closed), so concurrent requests
    cannot collectively push spend past the budget."""
    if cost_inr <= 0:
        return
    try:
        amount = _to_micros(to_usd(cost_inr))
        if amount <= 0:
            return
        key = _day_key()
        global _RECORD_SCRIPT
        if _RECORD_SCRIPT is None:
            _RECORD_SCRIPT = _client().register_script(_RECORD_LUA)
        rejected = await _RECORD_SCRIPT(
            keys=[key],
            args=[amount, config.COST_DAY_TTL_SECONDS, _budget_micros()],
        )
        if rejected:
            global _budget_reached_warned
            if not _budget_reached_warned:
                logger.warning(
                    "LLM daily budget reached; further spend not recorded (fail closed)"
                )
                _budget_reached_warned = True
    except Exception as exc:
        logger.warning("cost budget Redis unavailable (%s); spend not recorded", exc)


def _to_micros(usd: float) -> int:
    """Convert a USD amount to integer micro-USD for exact counter storage.

    This is the single rounding point: the USD value is rounded to the nearest
    micro-USD here, so callers must not round again before this conversion."""
    return int(round(usd * _COST_SCALE))


def _budget_micros() -> int:
    """The daily cap expressed in the same integer micro-USD units as the counter."""
    return _to_micros(config.LLM_DAILY_BUDGET_USD)


def to_usd(cost_inr: float) -> float:
    """Convert an INR cost (as reported by ``LLMResult.cost()``) to USD.

    All cost accounting in this project is canonical in USD (the daily budget
    counter, analytics, and stored message costs), so callers convert at the
    recording boundary instead of mixing units. Falls back to a 1.0 rate if
    INR_PER_USD is unset (and warns once, since that misconfiguration yields
    wrong costs)."""
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
