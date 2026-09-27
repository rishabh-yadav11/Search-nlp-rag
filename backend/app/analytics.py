"""Self-hosted, cookie-free search analytics.

Aggregates live in Redis DB ``ANALYTICS_REDIS_DB`` (default 1) so they survive
the query-cache flush (``FLUSHDB`` on DB 0 during deploys) and are shared
across gunicorn workers. No user identifiers, no client-side scripts and no
cookie banner are involved: every event is derived server-side from the request
itself plus an anonymous ``/analytics/click`` beacon from the frontend.

Recording is best-effort: a Redis outage never raises into the request path,
it only logs a warning once and stops recording until Redis returns.
"""
import hashlib
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("analytics")


class AnalyticsUnavailableError(RuntimeError):
    """The analytics store could not be read.

    Raised by :func:`summary` instead of returning an error-shaped payload, so
    the HTTP layer can answer 503. Returning ``{"error": ...}`` as a 200 was
    indistinguishable from a report whose counters are legitimately all zero.
    """


# Click positions are bucketed 1..CLICK_POSITION_MAX in the summary view, so an
# unauthenticated beacon can only poison within this range (never create
# arbitrarily-named ``analytics:click:pos:{n}`` keys).
CLICK_POSITION_MIN = 1
CLICK_POSITION_MAX = 10

# Window sizes for the two "top N" lists in ``summary()``. Redis ``zrevrange``
# takes an INCLUSIVE end index, so each is read as ``0, N - 1`` at the call site
# (writing a bare ``0, N`` would silently return N + 1 items).
TOP_QUERIES_N = 20
TOP_CLICKED_QUERIES_N = 10

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


def _normalise_query(q: str) -> str:
    """Canonical form of a query for the click-boost key: casefolded, with every
    run of whitespace collapsed to a single space, then length-bounded.

    Applied on BOTH the write path (``record_click``) and the read path
    (``click_signals``), so a stored key is always retrievable. Without it the
    unauthenticated beacon mints a separate boost key per spelling of one
    logical query (``"ola ipo"``, ``"OLA  IPO "``, ``"ola  IPO"``), which both
    splits the real signal and lets a client grow Redis with keys nothing ever
    reads. Truncate rather than hash to keep the key human-readable.
    """
    return " ".join((q or "").split())[: config.CLICK_QUERY_MAX_LEN].casefold()


def _click_query_key(q: str) -> str:
    # Normalize identically on the read and write paths so a stored key is
    # always retrievable. The beacon is unauthenticated, so bound the query
    # length before it becomes a Redis key (unbounded key size = unbounded
    # memory growth); truncate rather than hash to keep the key human-readable
    # in diagnostics. NOTE: two distinct queries sharing a 256-char prefix
    # collide into one aggregated key; that is acceptable for top-query
    # analytics, where the signal is intentionally coarse.
    return f"analytics:query_click:{_normalise_query(q)}"


async def _claim_click_signal(client_ip: str | None, query: str, article_id: int) -> tuple[bool, str | None]:
    """Whether ``client_ip`` may contribute a click to (query, article_id) now,
    plus the claim key to release if the tally it unlocks does not land.

    True the first time within the dedupe window, False for every repeat. The
    beacon is anonymous, so the only thing separating a real click from a forged
    one is where it came from. Counting every beacon verbatim lets a single host
    cross ``CLICK_BOOST_MIN_ARTICLE_CLICKS`` in a handful of requests and boost
    an article of its choosing, poisoning the ranking every other user sees. One
    click per client per (query, article) keeps the signal meaningful --
    re-opening the same result carries no new ranking information -- while
    requiring genuinely distinct clients to reach the threshold.

    The claim key is a digest, so no query text or client IP is recoverable
    from it, and it carries a TTL so the dedupe set cannot grow unbounded. Fails
    CLOSED on a Redis error: a dropped click only slows the learning signal
    down, whereas a skipped claim re-opens the forging hole.
    """
    window = config.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS
    if window <= 0 or not client_ip:
        # No window configured, or a caller with no client to attribute the
        # click to: there is nothing to deduplicate against.
        return True, None
    digest = hashlib.sha256(f"{client_ip}\x00{query}\x00{article_id}".encode()).hexdigest()
    key = f"analytics:click:seen:{digest}"
    try:
        return bool(await _client().set(key, 1, nx=True, ex=window)), key
    except Exception:
        logger.warning("click-signal dedupe unavailable; dropping ranking signal", exc_info=True)
        return False, None


async def _release_click_signal(key: str | None) -> None:
    """Give a spent claim back when the tally it unlocked never landed, so a
    transient Redis failure between the claim and the write does not silence a
    real user's click for the rest of the window. Never raises."""
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
        # Bound the stored query before it becomes a sorted-set member; an
        # unauthenticated caller could otherwise grow Redis without limit.
        # Truncate rather than hash to keep it human-readable (mirrors the
        # click beacon's truncation).
        query = (query or "").strip()[: config.CLICK_QUERY_MAX_LEN]
        p = _client().pipeline()
        p.incr("analytics:search:total")
        p.incr(f"analytics:search:day:{_today()}")
        p.incr("analytics:search:latency:sum", int(latency_ms))
        p.incr("analytics:search:latency:count")
        p.incr("analytics:search:cached" if cached else "analytics:search:uncached")
        p.zincrby("analytics:top_queries", 1, query)
        # Expire the aggregate so an idle deployment's top_queries key (and its
        # unbounded distinct-query members) cannot accumulate forever; refreshed
        # on every search, mirroring the click beacon's per-query TTL.
        p.expire("analytics:top_queries", config.CLICK_QUERY_TTL_SECONDS)
        if filtered:
            p.incr("analytics:search:filtered")
        if result_count == 0:
            p.incr("analytics:search:zero_results")
        elif weak:
            p.incr("analytics:search:weak")
        await p.execute()
    except Exception as exc:
        _degraded(exc)


async def record_click(
    query: str,
    position: int,
    article_id: int | None = None,
    client_ip: str | None = None,
) -> None:
    """Count one result click from the frontend beacon. Never raises.

    Also tallies per-query per-article clicks (keyed ``analytics:query_click:{q}``
    as a sorted set of {article_id: count}) so the click-boost layer can learn
    which results users actually open for a query.

    ``client_ip`` is the resolved client address. It gates only the ranking
    signal (one click per client per query/article per window, see
    ``_claim_click_signal``); the raw click counters and position buckets are
    recorded either way, so the admin-facing analytics are unaffected by the
    dedupe. ``article_id`` is expected to have been checked against the
    collection by the caller -- recording an id that is not in the index would
    mint a boost record nothing can ever match.
    """
    claim_key = None
    try:
        # Defensive: the beacon is unauthenticated, so an attacker could send an
        # arbitrarily long query. Bound it before it becomes a sorted-set member
        # (unbounded member size = unbounded memory growth). Keep the key stable
        # by truncating rather than hashing. The human-facing top-queries
        # aggregate keeps the client's original casing; only the boost key is
        # canonicalised (see ``_click_query_key``).
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
        # Expire the aggregate set too, so distinct-query growth from the
        # unauthenticated beacon doesn't accumulate forever; refreshed on each click.
        p.expire("analytics:click_top_queries", config.CLICK_QUERY_TTL_SECONDS)
        if q_article_id is not None:
            qkey = _click_query_key(query)
            claimed, claim_key = await _claim_click_signal(client_ip, _normalise_query(query), q_article_id)
            if claimed:
                p.zincrby(qkey, 1, str(q_article_id))
                # Expire the per-query set so distinct-query sets don't accumulate
                # forever; refreshed on each click.
                p.expire(qkey, config.CLICK_QUERY_TTL_SECONDS)
        await p.execute()
        claim_key = None  # the vote landed, so the claim is now genuinely spent
    except Exception as exc:
        # A claim that unlocked a tally which never landed would otherwise
        # silence this client for the whole window; give it back.
        await _release_click_signal(claim_key)
        _degraded(exc)


async def click_signals(query: str) -> dict | None:
    """Per-query click signal for the click-boost layer, or None when the query
    has too little click volume to act on. Returns ``{"total": int, "by_id": {id: count}}``."""
    try:
        c = _client()
        key = _click_query_key(query)
        # "total clicks" and the per-article breakdown are built from ALL members
        # of the sorted set, paginated in batches. Using only a top-50 window
        # would undercount once a query has more than 50 clicked articles and
        # break the invariant ``sum(by_id.values()) == total`` that the click-boost
        # layer relies on. Paginate the full set so the reported total and
        # breakdown stay consistent.
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
                # Guard the article-id cast: a poisoned/garbage member is skipped
                # rather than raising, so one bad beacon can't break the signal.
                aid = _safe_int(article_id)
                if aid is None:
                    continue
                total += count_i
                by_id[aid] = by_id.get(aid, 0) + count_i
            if len(chunk) < _ZSUM_BATCH:
                break
            offset += _ZSUM_BATCH
        if not by_id:
            return None
        if total < config.CLICK_BOOST_MIN_CLICKS:
            return None
        return {"total": total, "by_id": by_id}
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
    """Aggregated metrics since the analytics DB was last cleared.

    Raises :class:`AnalyticsUnavailableError` if the analytics Redis cannot be
    read; callers must surface that as a failed request rather than as data.
    """
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

        top_queries = await c.zrevrange("analytics:top_queries", 0, TOP_QUERIES_N - 1, withscores=True)
        click_top_queries = await c.zrevrange(
            "analytics:click_top_queries", 0, TOP_CLICKED_QUERIES_N - 1, withscores=True
        )

        # Read exactly the buckets ``record_click`` can write. Derived from the
        # same CLICK_POSITION_MIN/MAX the write path clamps to, so raising or
        # lowering the bound cannot desynchronize recording from reporting.
        positions = range(CLICK_POSITION_MIN, CLICK_POSITION_MAX + 1)
        pos_vals = await c.mget([f"analytics:click:pos:{i}" for i in positions])

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
            "click_positions": {str(i): _i(v) for i, v in zip(positions, pos_vals)},
            "click_top_queries": [[q, _i(s)] for q, s in click_top_queries],
        }
    except Exception as exc:
        _degraded(exc)
        logger.exception("analytics summary failed; analytics store unavailable")
        raise AnalyticsUnavailableError("analytics unavailable") from exc
