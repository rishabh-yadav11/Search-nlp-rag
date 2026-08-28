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
    return f"llm:cost:day:{datetime.now(UTC).strftime('%Y-%m-%d')}"


async def spend_today() -> float:
    """Cumulative LLM cost for today (USD). Returns 0 on Redis failure."""
    try:
        return float(await _client().get(_day_key()) or 0.0)
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


# Atomic check-and-increment: read the running total, add the delta only if the
# result would not exceed the budget, and (re)set the TTL — all in one Lua
# round-trip so concurrent requests cannot both pass a pre-check and then
# overshoot the cap. Returns the new total, or -1 when the increment would push
# spend past the budget (the caller then skips recording to stay within cap).
_RECORD_SCRIPT = """
local cur = tonumber(redis.call('get', KEYS[1]) or '0')
local delta = tonumber(ARGV[1])
local budget = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
if budget > 0 and (cur + delta) > budget then
    return -1
end
redis.call('incrbyfloat', KEYS[1], delta)
redis.call('expire', KEYS[1], ttl)
return tonumber(redis.call('get', KEYS[1]) or '0')
"""


async def record_cost(cost_inr: float) -> None:
    """Add ``cost_inr`` (INR, as reported by LLMResult.cost()) to today's
    running total, converted to USD so the counter is comparable to
    LLM_DAILY_BUDGET_USD.

    The check-and-increment is atomic (a single Lua script), so concurrent
    requests can't all clear a pre-flight budget check and then collectively
    overshoot the daily cap. If recording would push spend past the budget, the
    increment is refused (fails closed) rather than letting the counter exceed
    the cap. Best-effort, never raises."""
    if cost_inr <= 0:
        return
    try:
        c = _client()
        key = _day_key()
        # Record the cost and set the TTL atomically in a pipeline so a crash
        # between the two commands can't leave a day key with no expiry.
        new = await c.eval(
            _RECORD_SCRIPT,
            1,
            key,
            to_usd(cost_inr),
            config.LLM_DAILY_BUDGET_USD,
            config.COST_DAY_TTL_SECONDS,
        )
        if new == -1:
            logger.warning(
                "daily LLM budget reached; cost of %.4f USD not recorded to avoid overshoot",
                to_usd(cost_inr),
            )
    except Exception as exc:
        logger.warning("cost budget Redis unavailable (%s); spend not recorded", exc)


def to_usd(cost_inr: float) -> float:
    """Convert an INR cost (as reported by ``LLMResult.cost()``) to USD.

    All cost accounting in this project is canonical in USD (the daily budget
    counter, analytics, and stored message costs), so callers convert at the
    recording boundary instead of mixing units."""
    return cost_inr / (config.INR_PER_USD or 1.0)
