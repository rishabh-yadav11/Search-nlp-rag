"""Self-hosted, cookie-free search analytics.

Aggregates live in Redis DB ``ANALYTICS_REDIS_DB`` (default 1) so they survive
the query-cache flush (``FLUSHDB`` on DB 0 during deploys) and are shared
across gunicorn workers. No user identifiers, no client-side scripts and no
cookie banner are involved: every event is derived server-side from the request
itself plus an anonymous ``/analytics/click`` beacon from the frontend.

Recording is best-effort: a Redis outage never raises into the request path,
it only logs a warning once and stops recording until Redis returns.
"""
import logging
import re
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("analytics")

_redis = None
_warned = False


def _base_url(db: int) -> str:
    """REDIS_URL with its database index replaced by ``db``."""
    return re.sub(r"/\d+$", f"/{db}", config.REDIS_URL)


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            _base_url(config.ANALYTICS_REDIS_DB),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis


def _degraded(exc: Exception) -> None:
    global _warned
    if not _warned:
        logger.warning("analytics Redis unavailable (%s); recording paused", exc)
        _warned = True


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def record_search(
    query: str,
    result_count: int,
    weak: bool,
    cached: bool,
    latency_ms: float,
    filtered: bool,
) -> None:
    """Count one /search event and its outcome. Never raises."""
    try:
        p = _client().pipeline()
        p.incr("analytics:search:total")
        p.incr(f"analytics:search:day:{_today()}")
        p.incr("analytics:search:latency:sum", int(latency_ms))
        p.incr("analytics:search:latency:count")
        p.incr("analytics:search:cached" if cached else "analytics:search:uncached")
        p.zincrby("analytics:top_queries", 1, query)
        if filtered:
            p.incr("analytics:search:filtered")
        if result_count == 0:
            p.incr("analytics:search:zero_results")
        elif weak:
            p.incr("analytics:search:weak")
        await p.execute()
    except Exception as exc:
        _degraded(exc)


async def record_ask(query: str, outcome: str, cached: bool, cost: float = 0.0) -> None:
    """Count one /ask event. ``outcome`` is one of answered/fallback/none/error."""
    try:
        p = _client().pipeline()
        p.incr("analytics:ask:total")
        p.incr(f"analytics:ask:day:{_today()}")
        p.incr(f"analytics:ask:outcome:{outcome}")
        p.incr("analytics:ask:cached" if cached else "analytics:ask:uncached")
        p.zincrby("analytics:top_asks", 1, query)
        if cost:
            p.incrbyfloat("analytics:ask:cost", cost)
        await p.execute()
    except Exception as exc:
        _degraded(exc)


async def record_click(query: str, position: int) -> None:
    """Count one result click from the frontend beacon. Never raises."""
    try:
        p = _client().pipeline()
        p.incr("analytics:click:total")
        p.incr(f"analytics:click:pos:{position}")
        p.zincrby("analytics:click_top_queries", 1, query)
        await p.execute()
    except Exception as exc:
        _degraded(exc)


def _i(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _f(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pct(part: int, total: int) -> float:
    return round(100.0 * part / total, 2) if total else 0.0


async def summary() -> dict:
    """Aggregated metrics since the analytics DB was last cleared. Never raises."""
    try:
        c = _client()
        day = f"analytics:search:day:{_today()}"
        keys = [
            "analytics:search:total",
            day,
            "analytics:search:zero_results",
            "analytics:search:weak",
            "analytics:search:filtered",
            "analytics:search:latency:sum",
            "analytics:search:latency:count",
            "analytics:ask:total",
            "analytics:click:total",
        ]
        vals = await c.mget(keys)
        (
            search_total,
            search_today,
            zero,
            weak,
            filtered,
            lat_sum,
            lat_count,
            ask_total,
            click_total,
        ) = vals
        cached = await c.get("analytics:search:cached")
        ask_cost = await c.get("analytics:ask:cost")

        top_queries = await c.zrevrange("analytics:top_queries", 0, 19, withscores=True)
        top_asks = await c.zrevrange("analytics:top_asks", 0, 9, withscores=True)
        click_positions = await c.zrevrange("analytics:click_top_queries", 0, 9, withscores=True)

        pos_keys = [f"analytics:click:pos:{i}" for i in range(1, 11)]
        pos_vals = await c.mget(pos_keys)

        total = _i(search_total)
        return {
            "searches_total": total,
            "searches_today": _i(search_today),
            "zero_result_rate": _pct(_i(zero), total),
            "weak_result_rate": _pct(_i(weak), total),
            "filtered_rate": _pct(_i(filtered), total),
            "cache_hit_rate": _pct(_i(cached), total),
            "avg_latency_ms": round(_f(lat_sum) / _i(lat_count), 1) if _i(lat_count) else 0.0,
            "asks_total": _i(ask_total),
            "ask_cost": _f(ask_cost),
            "clicks_total": _i(click_total),
            "top_queries": [[q, _i(s)] for q, s in top_queries],
            "top_asks": [[q, _i(s)] for q, s in top_asks],
            "click_positions": {str(i): _i(v) for i, v in zip(range(1, 11), pos_vals)},
            "click_top_queries": [[q, _i(s)] for q, s in click_positions],
        }
    except Exception as exc:
        _degraded(exc)
        return {"error": "analytics unavailable", "detail": str(exc)}
