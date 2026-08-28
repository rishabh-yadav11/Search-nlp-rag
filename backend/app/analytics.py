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
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("analytics")

# Click positions are bucketed 1..CLICK_POSITION_MAX in the summary view, so an
# unauthenticated beacon can only poison within this range (never create
# arbitrarily-named ``analytics:click:pos:{n}`` keys).
CLICK_POSITION_MIN = 1
CLICK_POSITION_MAX = 10

# Sorted-set reads are paginated in batches of this size when we need a true
# sum of all member scores (the top-50 window otherwise undercounts).
_ZSUM_BATCH = 200

_redis = None
_warned = False


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        # Pin the DB explicitly so counters never silently land in DB 0 (which a
        # deploy FLUSHDB would wipe). The ``db`` kwarg overrides any db segment in
        # REDIS_URL, so this is safe whether or not the URL carries a db index.
        _redis = aioredis.from_url(
            config.REDIS_URL,
            db=config.ANALYTICS_REDIS_DB,
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


def _click_query_key(q: str) -> str:
    # Normalize identically on the read and write paths so a stored key is
    # always retrievable. The beacon is unauthenticated, so bound the query
    # length before it becomes a Redis key (unbounded key size = unbounded
    # memory growth); truncate rather than hash to keep the key human-readable
    # in diagnostics. NOTE: two distinct queries sharing a 256-char prefix
    # collide into one aggregated key; that is acceptable for top-query
    # analytics, where the signal is intentionally coarse.
    q = (q or "").strip()[: config.CLICK_QUERY_MAX_LEN]
    return f"analytics:query_click:{q}"


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


async def record_click(query: str, position: int, article_id: int | None = None) -> None:
    """Count one result click from the frontend beacon. Never raises.

    Also tallies per-query per-article clicks (keyed ``analytics:query_click:{q}``
    as a sorted set of {article_id: count}) so the click-boost layer can learn
    which results users actually open for a query.
    """
    try:
        # Defensive: the beacon is unauthenticated, so an attacker could send an
        # arbitrarily long query. Bound it before it becomes a sorted-set member
        # (unbounded member size = unbounded memory growth). Keep the key stable
        # by truncating rather than hashing.
        query = (query or "").strip()[: config.CLICK_QUERY_MAX_LEN]
        # Clamp position into the valid display range so a poisoned beacon cannot
        # create arbitrary ``analytics:click:pos:{n}`` keys. Position 0 or
        # negative collapses to the first slot; values above the max cap at the
        # last tracked slot.
        try:
            pos = int(position)
        except (TypeError, ValueError):
            pos = CLICK_POSITION_MIN
        pos = max(CLICK_POSITION_MIN, min(CLICK_POSITION_MAX, pos))
        # Validate the article id before it becomes a sorted-set member; an
        # invalid id is skipped so it can't poison the per-query click signal.
        q_article_id = None
        if article_id is not None:
            try:
                q_article_id = int(article_id)
            except (TypeError, ValueError):
                q_article_id = None
        p = _client().pipeline()
        p.incr("analytics:click:total")
        p.incr(f"analytics:click:pos:{pos}")
        p.zincrby("analytics:click_top_queries", 1, query)
        if q_article_id is not None:
            qkey = _click_query_key(query)
            p.zincrby(qkey, 1, str(q_article_id))
            # Expire the per-query set so distinct-query sets don't accumulate
            # forever; refreshed on each click.
            p.expire(qkey, config.CLICK_QUERY_TTL_SECONDS)
        await p.execute()
    except Exception as exc:
        _degraded(exc)


async def click_signals(query: str) -> dict | None:
    """Per-query click signal for the click-boost layer, or None when the query
    has too little click volume to act on. Returns ``{"total": int, "by_id": {id: count}}``."""
    try:
        c = _client()
        key = _click_query_key(query)
        raw = await c.zrevrange(key, 0, 50, withscores=True)
        if not raw:
            return None
        # "total clicks" is the sum of per-article click counts across ALL
        # members, not just the top-50 window returned above (which would
        # undercount once a query has more than 50 clicked articles). Paginate
        # the full set so the reported total is accurate.
        total = 0
        offset = 0
        while True:
            chunk = await c.zrevrange(
                key, offset, offset + _ZSUM_BATCH - 1, withscores=True
            )
            if not chunk:
                break
            total += sum(int(count) for _article, count in chunk)
            if len(chunk) < _ZSUM_BATCH:
                break
            offset += _ZSUM_BATCH
        if total < config.CLICK_BOOST_MIN_CLICKS:
            return None
        return {
            "total": total,
            # Guard the article-id cast: a poisoned/garbage member is skipped
            # rather than raising, so one bad beacon can't break the signal.
            "by_id": {
                int(article_id): int(count)
                for article_id, count in raw
                if count >= 1
                and _safe_int(article_id) is not None
            },
        }
    except Exception as exc:
        _degraded(exc)
        return None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
            click_total,
        ) = vals
        cached = await c.get("analytics:search:cached")

        top_queries = await c.zrevrange("analytics:top_queries", 0, 19, withscores=True)
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
            "clicks_total": _i(click_total),
            "top_queries": [[q, _i(s)] for q, s in top_queries],
            "click_positions": {str(i): _i(v) for i, v in zip(range(1, 11), pos_vals)},
            "click_top_queries": [[q, _i(s)] for q, s in click_positions],
        }
    except Exception as exc:
        _degraded(exc)
        logger.exception("analytics summary failed; returning generic error to client")
        return {"error": "analytics unavailable"}
