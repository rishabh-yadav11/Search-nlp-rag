"""Best-effort, cookie-free analytics in a Redis DB a deploy FLUSHDB never wipes; never raises into the request path."""
import hashlib
import hmac
import logging
import re
import secrets
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config
from app.degraded import DegradedLatch
from app.input_hygiene import normalize_text

logger = logging.getLogger("analytics")


class AnalyticsUnavailableError(RuntimeError):
    """The store was unreadable — distinct from a real report whose counters are all zero."""

_latch = DegradedLatch(logger, "analytics Redis")

CLICK_POSITION_MIN = 1
CLICK_POSITION_MAX = 10

# ``zrevrange``'s end index is INCLUSIVE, so each is read as ``0, N - 1``.
TOP_QUERIES_N = 20
TOP_CLICKED_QUERIES_N = 10

# Page size for summing ALL members of a sorted set; a top-50 window undercounts past 50.
_ZSUM_BATCH = 200

CLICK_SIGNAL_KEY_PREFIX = "analytics:query_click:"

# Self-describing: a member without the prefix is verbatim text, dropped on read.
QUERY_DIGEST_PREFIX = "q1:"
QUERY_DIGEST_HEX_LEN = 32
_DIGEST_RE = re.compile(r"^q1:[0-9a-f]{32}$")

LEGACY_ROW_OVERFETCH = 5

QUERY_DIGEST_KEY_REDIS_KEY = "analytics:query_digest_key"

# Keyed, not a bare hash: short queries are reversible offline by anyone who can read the
# dashboard, so the key is random, unlogged, and must be stable across workers/restarts.
_QUERY_DIGEST_KEY: str | None = None

_redis = None
# Separate from ``_latch``: a digest-key failure is not an outage and must not silence it.
_digest_warned = False
_legacy_scrubbed = False


async def _digest_key(c) -> str | None:
    """The digest secret, or None if Redis cannot supply it; callers then skip only the query-keyed fields."""
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
            # nx + re-read: concurrent workers adopt the first writer's key, so a lost race is not an error.
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
    """Keyed digest a query is stored and reported under, so user query text never reaches Redis."""
    normalized = _normalise_query(query)
    mac = hmac.new(key.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256)
    return QUERY_DIGEST_PREFIX + mac.hexdigest()[:QUERY_DIGEST_HEX_LEN]


async def _scrub_legacy_members(c, key: str) -> int:
    """Delete, don't filter on read: every write re-arms the key's TTL, so a legacy member never lapses."""
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
        # Pin the DB: the ``db`` kwarg overrides any db segment in REDIS_URL.
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
    # The cached key came from the client being closed, so it cannot vouch for a reconnected one.
    _QUERY_DIGEST_KEY = None
    _digest_warned = False
    _legacy_scrubbed = False
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _normalise_query(q: str) -> str:
    """NFKC/control-char-normalised (no NUL/CRLF from the beacon reaches Redis), bounded, casefolded."""
    return normalize_text(q or "")[: config.CLICK_QUERY_MAX_LEN].casefold()


def _click_query_key(q: str, digest_key: str) -> str:
    # Callers pass the RAW query: normalising twice is not a no-op (the bound can land on a
    # space that a second pass strips).
    return f"{CLICK_SIGNAL_KEY_PREFIX}{query_digest(q, digest_key)}"


async def _claim_click_signal(client_ip: str | None, query: str, article_id: int) -> tuple[bool, str | None]:
    """One click per client per (query, article): the anonymous beacon makes repeats a forgery vector. Fails CLOSED."""
    window = config.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS
    if window <= 0 or not client_ip:
        return True, None
    digest = hashlib.sha256(f"{client_ip}\x00{query}\x00{article_id}".encode()).hexdigest()
    key = f"analytics:click:seen:{digest}"
    try:
        # Only a claim we won may be released: a loser returning the key would delete the winner's.
        return (True, key) if await _client().set(key, 1, nx=True, ex=window) else (False, None)
    except Exception:
        logger.warning("click-signal dedupe unavailable; dropping ranking signal", exc_info=True)
        return False, None


async def _release_click_signal(key: str | None) -> None:
    """Give a spent claim back so a Redis failure between claim and write does not silence this client's window."""
    if key is None:
        return
    try:
        await _client().delete(key)
    except Exception:
        logger.warning("could not release click-signal claim", exc_info=True)


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
        c = _client()
        digest_key = await _digest_key(c)
        p = c.pipeline()
        p.incr("analytics:search:total")
        p.incr(f"analytics:search:day:{_today()}")
        p.incr("analytics:search:latency:sum", int(latency_ms))
        p.incr("analytics:search:latency:count")
        p.incr("analytics:search:cached" if cached else "analytics:search:uncached")
        if digest_key is not None:
            p.zincrby("analytics:top_queries", 1, query_digest(query, digest_key))
            p.expire("analytics:top_queries", config.CLICK_QUERY_TTL_SECONDS)
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
) -> None:
    """Never raises; ``article_id`` must be pre-validated against the collection, else it mints an unmatchable boost."""
    claim_key = None
    try:
        c = _client()
        # ``canonical`` feeds the dedupe digest only; ``_click_query_key`` applies the digest
        # itself, and normalising twice is not a no-op.
        raw_query = query or ""
        canonical = _normalise_query(raw_query)
        # KNOWN LIMITATION: /search boosts on the typo-corrected query while the beacon posts
        # the raw one, so typo'd votes land under a key the ranking path never reads.
        digest_key = await _digest_key(c)
        # Clamp: an unauthenticated beacon must not mint arbitrary ``pos:{n}`` keys.
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
        claim_key = None  # the vote landed, so the claim is now genuinely spent
    except Exception as exc:
        await _release_click_signal(claim_key)
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
    else:
        _latch.log_recovered()


async def click_signals(query: str) -> dict | None:
    """Click-boost signal for a query, or None when its click volume is too low to act on."""
    try:
        c = _client()
        digest_key = await _digest_key(c)
        if digest_key is None:
            # Addressing the set with the raw query would reintroduce the leak: no signal instead.
            return None
        key = _click_query_key(query, digest_key)
        # Sum every member, paginated: the click-boost layer relies on sum(by_id.values()) == total.
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
    """Aggregated metrics since the analytics DB was last cleared; raises AnalyticsUnavailableError if unreadable."""
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

        # Scrub before reading: filtering at the read boundary alone would leave the corpus in Redis and its backups.
        await _scrub_legacy_once(c)
        # Overfetch because _digest_rows drops legacy rows that could swallow the window; end index is INCLUSIVE.
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

        # Derived from the write path's clamp bounds so read and write cannot desynchronize.
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
        }
    except Exception as exc:
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
        logger.exception("analytics summary failed; analytics store unavailable")
        raise AnalyticsUnavailableError("analytics unavailable") from exc
