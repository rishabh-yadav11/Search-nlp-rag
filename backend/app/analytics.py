
import hashlib
import hmac
import logging
import re
import secrets
from datetime import UTC, datetime, timedelta

import redis.asyncio as aioredis

from app.config import config
from app.degraded import DegradedLatch
from app.input_hygiene import normalize_text

logger = logging.getLogger("analytics")


class AnalyticsUnavailableError(RuntimeError):
    pass

_latch = DegradedLatch(logger, "analytics Redis")




CLICK_POSITION_MIN = 1
CLICK_POSITION_MAX = 10




TOP_QUERIES_N = 20
TOP_CLICKED_QUERIES_N = 10



_ZSUM_BATCH = 200







LATENCY_BUCKETS = (
    ("<10", 10),
    ("10-25", 25),
    ("25-50", 50),
    ("50-100", 100),
    ("100-250", 250),
    ("250-500", 500),
    ("500-1000", 1000),
    ("1000-2500", 2500),
)
_LATENCY_OVERFLOW_LABEL = ">2500"


_LATENCY_PERCENTILES = (50, 90, 95, 99)


TOP_QUERIES_TODAY_N = 20


_INTENT_CLASSES = ("timed", "category", "flashback", "recency", "other")







CLICK_SIGNAL_KEY_PREFIX = "analytics:query_click:"





QUERY_DIGEST_PREFIX = "q1:"
QUERY_DIGEST_HEX_LEN = 32
_DIGEST_RE = re.compile(r"^q1:[0-9a-f]{32}$")






LEGACY_ROW_OVERFETCH = 5




QUERY_DIGEST_KEY_REDIS_KEY = "analytics:query_digest_key"









_QUERY_DIGEST_KEY: str | None = None

_redis = None





_digest_warned = False



_legacy_scrubbed = False


async def _digest_key(c) -> str | None:

    global _QUERY_DIGEST_KEY
    if _QUERY_DIGEST_KEY is not None:
        return _QUERY_DIGEST_KEY
    configured = (getattr(config, "ANALYTICS_QUERY_KEY", "") or "").strip()
    if configured:
        _QUERY_DIGEST_KEY = configured
        return _QUERY_DIGEST_KEY
    try:
        stored = await c.get(QUERY_DIGEST_KEY_REDIS_KEY)
        if not stored:



            await c.set(QUERY_DIGEST_KEY_REDIS_KEY, secrets.token_hex(32), nx=True)
            stored = await c.get(QUERY_DIGEST_KEY_REDIS_KEY)
        if not stored:
            return None
        _QUERY_DIGEST_KEY = stored
        return _QUERY_DIGEST_KEY
    except Exception as exc:
        global _digest_warned
        if not _digest_warned:
            logger.warning(
                "analytics digest key unavailable (%s); query-keyed counts not "
                "recorded (total/latency/cache counters still are)",
                exc,
            )
            _digest_warned = True
        return None


def query_digest(query: str, key: str) -> str:

    normalized = _normalise_query(query)
    mac = hmac.new(key.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256)
    return QUERY_DIGEST_PREFIX + mac.hexdigest()[:QUERY_DIGEST_HEX_LEN]


async def _scrub_legacy_members(c, key: str) -> int:

    try:
        rows = await c.zrange(key, 0, -1)
    except Exception as exc:


        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
        return 0
    stale = [m for m in rows if not _is_digest(m)]
    if not stale:
        return 0
    try:
        await c.zrem(key, *stale)
    except Exception as exc:
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
        return 0
    logger.info(
        "removed %d pre-upgrade verbatim query members from %s (issue #348)",
        len(stale),
        key,
    )
    return len(stale)


async def _scrub_legacy_once(c) -> None:

    global _legacy_scrubbed
    if _legacy_scrubbed:
        return
    await _scrub_legacy_members(c, "analytics:top_queries")
    await _scrub_legacy_members(c, "analytics:click_top_queries")


    _legacy_scrubbed = True


def _is_digest(member) -> bool:

    return isinstance(member, str) and _DIGEST_RE.match(member) is not None


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
    global _redis, _QUERY_DIGEST_KEY, _digest_warned, _legacy_scrubbed







    _QUERY_DIGEST_KEY = None
    _digest_warned = False
    _legacy_scrubbed = False
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _normalise_query(q: str) -> str:

    return normalize_text(q or "")[: config.CLICK_QUERY_MAX_LEN].casefold()


def _click_query_key(q: str, digest_key: str) -> str:








    return f"{CLICK_SIGNAL_KEY_PREFIX}{query_digest(q, digest_key)}"


async def _claim_click_signal(client_ip: str | None, query: str, article_id: int) -> tuple[bool, str | None]:

    window = config.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS
    if window <= 0 or not client_ip:


        return True, None
    digest = hashlib.sha256(f"{client_ip}\x00{query}\x00{article_id}".encode()).hexdigest()
    key = f"analytics:click:seen:{digest}"
    try:



        return (True, key) if await _client().set(key, 1, nx=True, ex=window) else (False, None)
    except Exception:
        logger.warning("click-signal dedupe unavailable; dropping ranking signal", exc_info=True)
        return False, None


async def _release_click_signal(key: str | None) -> None:

    if key is None:
        return
    try:
        await _client().delete(key)
    except Exception:
        logger.warning("could not release click-signal claim", exc_info=True)


def _latency_bucket_label(latency_ms: float) -> str:

    lat = float(latency_ms) if latency_ms else 0.0
    for label, upper in LATENCY_BUCKETS:
        if lat < upper:
            return label
    return _LATENCY_OVERFLOW_LABEL


def _hour_label(dt) -> str:

    return dt.strftime("%Y-%m-%dT%H")


def _last_hours(n: int) -> list[datetime]:

    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    return [now - timedelta(hours=i) for i in reversed(range(n))]


def _percentile_from_buckets(buckets: list[tuple[int, str]]) -> dict[str, float]:

    total = sum(b for b, _ in buckets)
    if total <= 0:
        return {f"p{p}": 0.0 for p in _LATENCY_PERCENTILES}


    mids = []
    prev_upper = 0.0
    for label, upper in LATENCY_BUCKETS:
        mids.append(round((prev_upper + upper) / 2, 1))
        prev_upper = upper
    overflow_upper = LATENCY_BUCKETS[-1][1]
    mids.append(round(overflow_upper * 2, 1))

    cum = []
    acc = 0
    for b, _ in buckets:
        acc += b
        cum.append(acc)
    out: dict[str, float] = {}
    for p in _LATENCY_PERCENTILES:
        target = (p / 100.0) * total
        idx = 0
        while idx < len(cum) and cum[idx] < target:
            idx += 1
        if idx >= len(cum):
            out[f"p{p}"] = round(mids[-1], 2)
            continue
        out[f"p{p}"] = round(mids[idx], 2)
    return out


async def record_search(
    query: str,
    result_count: int,
    weak: bool,
    cached: bool,
    latency_ms: float,
    filtered: bool,
    *,
    intent: str | None = None,
    session_id: str = "",
) -> None:

    try:
        c = _client()




        digest_key = await _digest_key(c)
        p = c.pipeline()
        p.incr("analytics:search:total")
        p.incr(f"analytics:search:day:{_today()}")
        p.incr("analytics:search:latency:sum", int(latency_ms))
        p.incr("analytics:search:latency:count")
        p.incr("analytics:search:cached" if cached else "analytics:search:uncached")



        p.incr(f"analytics:search:latency_bucket:{_latency_bucket_label(latency_ms)}")
        p.incr(f"analytics:search:hour:{_hour_label(datetime.now(UTC))}")
        if intent:
            key = f"analytics:search:intent:{intent}"
            p.incr(key)
            p.expire(key, config.CLICK_QUERY_TTL_SECONDS)
        if digest_key is not None:

            digest = query_digest(query, digest_key)
            p.zincrby("analytics:top_queries", 1, digest)


            p.expire("analytics:top_queries", config.CLICK_QUERY_TTL_SECONDS)


            today_key = f"analytics:top_queries:{_today()}"
            p.zincrby(today_key, 1, digest)
            p.expire(today_key, config.TOP_QUERIES_TODAY_TTL_DAYS * 86400)



            sid = (session_id or "").strip()
            if sid:
                sess_key = f"analytics:session:{sid}"
                p.zincrby(sess_key, datetime.now(UTC).timestamp(), digest)
                p.expire(sess_key, config.ANALYTICS_SESSION_TTL_HOURS * 3600)
        if filtered:
            p.incr("analytics:search:filtered")
        if result_count == 0:
            p.incr("analytics:search:zero_results")
        elif weak:
            p.incr("analytics:search:weak")
        await p.execute()
    except Exception as exc:
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
    else:




        _latch.log_recovered()


async def record_click(
    query: str,
    position: int,
    article_id: int | None = None,
    client_ip: str | None = None,
    *,
    session_id: str = "",
) -> None:

    claim_key = None
    try:
        c = _client()









        raw_query = query or ""
        canonical = _normalise_query(raw_query)












        digest_key = await _digest_key(c)




        try:
            pos = int(position)
        except (TypeError, ValueError):
            pos = CLICK_POSITION_MIN
        pos = max(CLICK_POSITION_MIN, min(CLICK_POSITION_MAX, pos))


        q_article_id = None
        if article_id is not None:
            try:
                q_article_id = int(article_id)
            except (TypeError, ValueError):
                q_article_id = None
        p = c.pipeline()
        p.incr("analytics:click:total")
        p.incr(f"analytics:click:pos:{pos}")
        sid = (session_id or "").strip()
        if sid:
            clicks_key = f"analytics:session:{sid}:clicks"
            p.incr(clicks_key)
            p.expire(clicks_key, config.ANALYTICS_SESSION_TTL_HOURS * 3600)
        if digest_key is not None:
            p.zincrby("analytics:click_top_queries", 1, query_digest(raw_query, digest_key))


            p.expire("analytics:click_top_queries", config.CLICK_QUERY_TTL_SECONDS)
            if q_article_id is not None:


                qkey = _click_query_key(raw_query, digest_key)
                claimed, claim_key = await _claim_click_signal(client_ip, canonical, q_article_id)
                if claimed:
                    p.zincrby(qkey, 1, str(q_article_id))

                    p.expire(qkey, config.CLICK_QUERY_TTL_SECONDS)
        await p.execute()
        claim_key = None
    except Exception as exc:


        await _release_click_signal(claim_key)
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
    else:
        _latch.log_recovered()


async def click_signals(query: str) -> dict | None:

    try:
        c = _client()
        digest_key = await _digest_key(c)
        if digest_key is None:




            return None
        key = _click_query_key(query, digest_key)




        total = 0
        by_id: dict[int, int] = {}
        offset = 0
        while True:
            chunk = await c.zrevrange(
                key, offset, offset + _ZSUM_BATCH - 1, withscores=True
            )
            if not chunk:
                break
            for article_id, count in chunk:
                count_i = int(count)
                if count_i < 1:
                    continue


                aid = _safe_int(article_id)
                if aid is None:
                    continue
                total += count_i
                by_id[aid] = by_id.get(aid, 0) + count_i
            if len(chunk) < _ZSUM_BATCH:
                break
            offset += _ZSUM_BATCH
        if not by_id:
            _latch.log_recovered()
            return None
        if total < config.CLICK_BOOST_MIN_CLICKS:
            _latch.log_recovered()
            return None
        _latch.log_recovered()
        return {"total": total, "by_id": by_id}
    except Exception as exc:
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
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


def _digest_rows(rows, limit: int) -> list:

    kept = [[m, _i(s)] for m, s in rows if _is_digest(m)]
    return kept[:limit]


async def summary() -> dict:

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


        bucket_keys = [f"analytics:search:latency_bucket:{label}" for label, _ in
                       LATENCY_BUCKETS] + [f"analytics:search:latency_bucket:{_LATENCY_OVERFLOW_LABEL}"]
        bucket_labels = [label for label, _ in LATENCY_BUCKETS] + [_LATENCY_OVERFLOW_LABEL]
        bucket_vals = await c.mget(bucket_keys)

        bucket_rows = [(int(v or 0), lbl) for v, lbl in zip(bucket_vals, bucket_labels)]

        hour_keys = [f"analytics:search:hour:{_hour_label(dt)}" for dt in _last_hours(24)]
        hour_vals = await c.mget(hour_keys)
        hourly_volume = [[_hour_label(dt), int(v or 0)] for dt, v in zip(_last_hours(24), hour_vals)]



        intent_key = lambda label: f"analytics:search:intent:{label}"
        intent_vals = await c.mget([intent_key(l) for l in _INTENT_CLASSES])
        intent_rows = {l: int(v or 0) for l, v in zip(_INTENT_CLASSES, intent_vals)}

        top_today = await c.zrevrange(
            f"analytics:top_queries:{_today()}", 0, TOP_QUERIES_TODAY_N - 1, withscores=True
        )
        top_queries_today = _digest_rows(top_today, TOP_QUERIES_TODAY_N)




        await _scrub_legacy_once(c)



        top_queries = await c.zrevrange(
            "analytics:top_queries",
            0,
            (TOP_QUERIES_N * LEGACY_ROW_OVERFETCH) - 1,
            withscores=True,
        )
        click_top_queries = await c.zrevrange(
            "analytics:click_top_queries",
            0,
            (TOP_CLICKED_QUERIES_N * LEGACY_ROW_OVERFETCH) - 1,
            withscores=True,
        )




        positions = range(CLICK_POSITION_MIN, CLICK_POSITION_MAX + 1)
        pos_vals = await c.mget([f"analytics:click:pos:{i}" for i in positions])

        total = _i(search_total)
        _latch.log_recovered()
        return {
            "searches_total": total,
            "searches_today": _i(search_today),
            "zero_result_rate": _pct(_i(zero), total),
            "weak_result_rate": _pct(_i(weak), total),
            "filtered_rate": _pct(_i(filtered), total),
            "cache_hit_rate": _pct(_i(cached), total),
            "avg_latency_ms": round(_f(lat_sum) / _i(lat_count), 1) if _i(lat_count) else 0.0,
            "clicks_total": _i(click_total),
            "top_queries": _digest_rows(top_queries, TOP_QUERIES_N),
            "click_positions": {str(i): _i(v) for i, v in zip(positions, pos_vals)},
            "click_top_queries": _digest_rows(click_top_queries, TOP_CLICKED_QUERIES_N),
            "latency": _percentile_from_buckets(bucket_rows),
            "hourly_volume": hourly_volume,
            "intent_counts": intent_rows,
            "top_queries_today": top_queries_today,
        }
    except Exception as exc:
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
        logger.exception("analytics summary failed; analytics store unavailable")
        raise AnalyticsUnavailableError("analytics unavailable") from exc
