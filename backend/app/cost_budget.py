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


async def record_cost(cost_inr: float) -> None:
    """Add ``cost_inr`` (INR, as reported by LLMResult.cost()) to today's
    running total, converted to USD so the counter is comparable to
    LLM_DAILY_BUDGET_USD. Best-effort, never raises."""
    if cost_inr <= 0:
        return
    try:
        await _client().incrbyfloat(_day_key(), cost_inr / (config.INR_PER_USD or 1.0))
    except Exception as exc:
        logger.warning("cost budget Redis unavailable (%s); spend not recorded", exc)
