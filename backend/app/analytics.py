"""Self-hosted, cookie-free search analytics.

Aggregates live in Redis DB ``ANALYTICS_REDIS_DB`` (default 1) so they survive
the query-cache flush (``FLUSHDB`` on DB 0 during deploys) and are shared
across gunicorn workers. No user identifiers, no client-side scripts and no
cookie banner are involved: every event is derived server-side from the request
itself plus an anonymous ``/analytics/click`` beacon from the frontend.

Recording is best-effort: a Redis outage never raises into the request path.
It warns once per outage, logs a single "recovered" line when Redis returns,
then re-arms so the next outage is announced too.
"""
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
    """The analytics store could not be read.

    Raised by :func:`summary` instead of returning an error-shaped payload, so the
    HTTP layer can answer 503: a 200 carrying ``{"error": ...}`` was
    indistinguishable from a report whose counters are legitimately all zero.
    """

_latch = DegradedLatch(logger, "analytics Redis")

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

# Prefix of the per-query click-signal sorted set (see ``_click_query_key``).
# Named, not spelled inline where it is built, because the ops report in
# ``scripts/click_boost_report.py`` has to SCAN for these keys: a report holding
# its own copy of the prefix would silently find nothing if the prefix changed
# and would report the click boost as inert -- the one answer that must never be
# wrong.
CLICK_SIGNAL_KEY_PREFIX = "analytics:query_click:"

# Shape of a query digest as stored in Redis and returned by ``summary()``.
# The prefix makes a digest self-describing (a member that does not match it is
# a verbatim query, and is dropped on read) and gives the truncation a
# delimiter that no user query can forge into looking like a different member.
QUERY_DIGEST_PREFIX = "q1:"
QUERY_DIGEST_HEX_LEN = 32
_DIGEST_RE = re.compile(r"^q1:[0-9a-f]{32}$")

# How much extra to read from each top-query set before dropping non-digest
# members (see ``_digest_rows``). Legacy verbatim rows are dropped on read, so a
# window read of exactly N could be entirely legacy and report an empty list.
# Bounded rather than "read everything" because these sets can hold many distinct
# queries and the response stays a top-N report.
LEGACY_ROW_OVERFETCH = 5

# Where the auto-generated digest key is persisted. It lives in the analytics
# Redis (not a file) so it inherits the same backup/flush lifecycle as the data
# it keys, and it is never returned over HTTP or written to a log.
QUERY_DIGEST_KEY_REDIS_KEY = "analytics:query_digest_key"

# The secret mixed into every query digest. Digests must be KEYED, not a bare
# hash: a short natural-language query has a small enough dictionary that an
# unkeyed digest is reversible offline by anyone who can read the admin
# dashboard, and a hardcoded default would be no better since it ships in this
# repository. The key is random and lives in the analytics Redis (see
# ``_digest_key``); it must be stable across the gunicorn workers and across
# restarts, or one user's query hashes differently per worker and its counts
# split four ways.
_QUERY_DIGEST_KEY: str | None = None

_redis = None
# The digest-key warning is separate from the ``_latch`` outage latch on purpose.
# A digest-key failure is NOT a Redis outage -- recording continues without the
# query-keyed fields -- so it must not consume the outage latch, or a transient
# key hiccup would silence the far more important "recording paused" alert for a
# real outage later on.
_digest_warned = False
# Set once the pre-upgrade verbatim members have been deleted from both
# top-query sets, so the scrub runs once per process rather than per request.
# Reset by close() alongside the digest key.
_legacy_scrubbed = False


async def _digest_key(c) -> str | None:
    """The secret mixed into every query digest, or None if it cannot be read.

    ``ANALYTICS_QUERY_KEY`` wins when set, so an operator can pin the digest
    namespace across a Redis rebuild. Otherwise the key is generated once and
    persisted in the analytics Redis, shared by every worker and surviving a
    restart with no operator action and no public default.

    None means a Redis hiccup costs the query-keyed fields only, never the
    counters: callers skip the top-query member this time and the search still
    counts.
    """
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
            # SET NX so concurrent workers cannot each mint a different key for
            # the same deployment: the first writer wins and the rest adopt it.
            # A lost race is not an error -- the winner's value is authoritative.
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
    """The opaque identifier a query is stored and reported under.

    Search queries are user-authored content aggregated BY TEXT, so the top-N
    lists would be a systematically harvested corpus of what every user typed,
    readable from the store itself. A keyed digest closes that: the text never
    reaches Redis, so no reader of these aggregates can hand it back. Redacting
    only in ``summary()`` would leave the corpus in the store for the next
    reader.

    Truncated to 128 bits: same-input stability is what makes the counts
    aggregate and lets ``click_signals`` find the signal for the query in hand,
    while the truncation keeps the stored member bounded without a length cap.

    The input is canonicalised with ``_normalise_query`` first -- the SAME
    canonical form the click-signal claim uses. That is what keeps one logical
    query a single key across spellings: hashing raw text would give
    ``"ola ipo"``, ``"OLA IPO"`` and ``"Ola   IPO"`` three different digests,
    stranding the signal under keys the ranking path never looks up and
    splitting one query's counts three ways.
    """
    normalized = _normalise_query(query)
    mac = hmac.new(key.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256)
    return QUERY_DIGEST_PREFIX + mac.hexdigest()[:QUERY_DIGEST_HEX_LEN]


async def _scrub_legacy_members(c, key: str) -> int:
    """Delete pre-upgrade verbatim members from a top-query set. Returns the count.

    Hiding them at the read boundary is not enough: the write path re-issues
    EXPIRE on the whole key on every event, so a legacy member's TTL is re-armed
    continuously and the verbatim corpus would sit in the analytics Redis -- and
    its backups -- indefinitely on any deployment still taking searches.

    Guarded to once per process per key (``_legacy_scrubbed``) because after the
    first pass there is nothing left to delete, and a fresh install never has
    anything to begin with.
    """
    try:
        rows = await c.zrange(key, 0, -1)
    except Exception as exc:
        # Best-effort: never let the scrub break a read. If it fails, the
        # read-side filter still withholds the text from the response.
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
    """One-time scrub of both top-query sets, on the first summary() read."""
    global _legacy_scrubbed
    if _legacy_scrubbed:
        return
    await _scrub_legacy_members(c, "analytics:top_queries")
    await _scrub_legacy_members(c, "analytics:click_top_queries")
    # Set even if a delete failed: the read-side filter withholds the text on
    # every read regardless, so re-scanning per request buys nothing.
    _legacy_scrubbed = True


def _is_digest(member) -> bool:
    """True when a stored member is one of our digests rather than raw text."""
    return isinstance(member, str) and _DIGEST_RE.match(member) is not None


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


async def close() -> None:
    global _redis, _QUERY_DIGEST_KEY, _digest_warned, _legacy_scrubbed
    # The cached key goes too: it was read through the client being closed, so
    # keeping it would let a reconnected client keep signing digests with a key
    # the new store was never checked against. The digest warning is reset for
    # the same reason the key is -- a warning already emitted against a store
    # that is gone would otherwise be suppressed against the one replacing it.
    # The scrub flag too, for the same reason: a reconnected store may be the
    # one that actually needs the legacy members deleted.
    _QUERY_DIGEST_KEY = None
    _digest_warned = False
    _legacy_scrubbed = False
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _normalise_query(q: str) -> str:
    """Canonical form of a query for the click-signal dedupe claim: NFKC-normalised
    with control characters removed and whitespace collapsed, then length-bounded,
    then casefolded.

    Applied to the claim input on the write path (``record_click``) so every
    spelling of one logical query claims the same vote. The per-query click KEY
    is a keyed digest (see ``_click_query_key``), so no user text is stored under
    any spelling and this form needs no read-path counterpart to stay
    retrievable.

    The NFKC/control-character pass is ``input_hygiene.normalize_text``, the same
    canonical form the search cache keys use, so a full-width or NUL-bearing
    spelling cannot open a second key and no NUL/CRLF from an unauthenticated
    beacon can reach Redis at all. Casefolding stays on top: for the boost signal
    "OLA" and "ola" are the same question. The bound is applied before the
    casefold, so a fold that expands ("ss" from a sharp s) can leave the result a
    character or two over ``CLICK_QUERY_MAX_LEN``; the claim input is a message,
    not a key, and key bounding is handled by ``_click_query_key``.
    """
    return normalize_text(q or "")[: config.CLICK_QUERY_MAX_LEN].casefold()


def _click_query_key(q: str, digest_key: str) -> str:
    # Normalize identically on the read and write paths so a stored key is always
    # retrievable; ``query_digest`` is that shared normalization. Both call sites
    # hand this function the RAW query and let it normalize internally, so read
    # and write cannot drift into two different orders of bounding the input.
    #
    # A digest keeps the key retrievable -- ``click_signals`` hashes the query in
    # hand the same way -- without the query text ever being a key and without a
    # length cap, since a digest is fixed-width.
    return f"{CLICK_SIGNAL_KEY_PREFIX}{query_digest(q, digest_key)}"


async def _claim_click_signal(client_ip: str | None, query: str, article_id: int) -> tuple[bool, str | None]:
    """Whether ``client_ip`` may contribute a click to (query, article_id) now,
    plus the claim key to release if the tally it unlocks does not land.

    True the first time within the dedupe window, False for every repeat. The
    beacon is anonymous, so the only thing separating a real click from a forged
    one is where it came from: counting every beacon verbatim lets a single host
    cross ``CLICK_BOOST_MIN_ARTICLE_CLICKS`` in a handful of requests and boost
    an article of its choosing, poisoning the ranking every other user sees. One
    click per client per (query, article) keeps the signal meaningful while
    requiring genuinely distinct clients to reach the threshold.

    The claim key is a digest, so no query text or client IP is recoverable from
    it, and it carries a TTL so the dedupe set cannot grow unbounded. Fails
    CLOSED on a Redis error: a dropped click only slows the learning signal down,
    whereas a skipped claim re-opens the forging hole.
    """
    window = config.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS
    if window <= 0 or not client_ip:
        # No window configured, or a caller with no client to attribute the
        # click to: there is nothing to deduplicate against.
        return True, None
    digest = hashlib.sha256(f"{client_ip}\x00{query}\x00{article_id}".encode()).hexdigest()
    key = f"analytics:click:seen:{digest}"
    try:
        # Only a claim we actually won may be released later: returning the key
        # on a lost claim would let a client that merely *lost* the race delete
        # the winner's claim if the write then failed, and vote twice.
        return (True, key) if await _client().set(key, 1, nx=True, ex=window) else (False, None)
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
        c = _client()
        # Counters first and unconditionally: they are the part of the report with
        # no user-authored text in it, so a digest-key failure (below) must cost
        # only the top-query member. Resolved before the pipeline is built, so the
        # member is hashed from the same text on every path.
        digest_key = await _digest_key(c)
        p = c.pipeline()
        p.incr("analytics:search:total")
        p.incr(f"analytics:search:day:{_today()}")
        p.incr("analytics:search:latency:sum", int(latency_ms))
        p.incr("analytics:search:latency:count")
        p.incr("analytics:search:cached" if cached else "analytics:search:uncached")
        if digest_key is not None:
            # The member is a digest, so the verbatim text never enters the store.
            p.zincrby("analytics:top_queries", 1, query_digest(query, digest_key))
            # Expire the aggregate so an idle deployment's top_queries key (and
            # its unbounded distinct-member set) cannot accumulate forever.
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
        # Warn on the transition into the outage, log one "recovered" line when
        # it ends, then re-arm so a later outage is announced rather than
        # swallowed. Volume is rate-limited by elapsed time, not by request
        # count; see app/degraded.py.
        _latch.log_recovered()


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

    ``client_ip`` gates only the ranking signal (one click per client per
    query/article per window, see ``_claim_click_signal``); the raw click
    counters and position buckets are recorded either way, so the admin-facing
    analytics are unaffected by the dedupe. ``article_id`` is expected to have
    been checked against the collection by the caller -- recording an id that is
    not in the index would mint a boost record nothing can ever match.
    """
    claim_key = None
    try:
        c = _client()
        # Two forms of the query, because the two consumers need different things.
        #
        # ``canonical`` feeds the click-signal dedupe digest, so every spelling of
        # one logical query claims the same vote. It must NOT be passed to
        # ``_click_query_key``: that applies ``query_digest`` itself, and
        # normalizing twice is not a no-op -- the length bound can land on a space,
        # which a second pass strips. The read path (``click_signals``) hands a
        # raw query to ``_click_query_key``, so the write path does too: both keys
        # then come from the same call on the same input, structurally.
        raw_query = query or ""
        canonical = _normalise_query(raw_query)
        # KNOWN LIMITATION: /search boosts on the typo-corrected query
        # (apply_click_boost(q_fixed, ...)), while the beacon posts the raw one.
        # With no vocab artifact built, fix_query is a documented no-op and the
        # two agree; an operator who HAS built the vocab will see typo'd traffic's
        # votes stranded under the raw key. Left alone: it is signal loss, not
        # write amplification, and routing the beacon through fix_query would put
        # a query-correction dependency in the analytics write path, where a raise
        # costs the whole click.
        # The aggregate members are NOT written from any of these forms: they
        # carry a keyed digest, so neither the casing nor the text reaches Redis.
        # As in record_search: counters are unconditional, the query-keyed fields
        # are skipped if the digest key cannot be resolved.
        digest_key = await _digest_key(c)
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
        p = c.pipeline()
        p.incr("analytics:click:total")
        p.incr(f"analytics:click:pos:{pos}")
        if digest_key is not None:
            p.zincrby("analytics:click_top_queries", 1, query_digest(raw_query, digest_key))
            # Expire the aggregate set so distinct-member growth from the
            # unauthenticated beacon cannot accumulate forever.
            p.expire("analytics:click_top_queries", config.CLICK_QUERY_TTL_SECONDS)
            if q_article_id is not None:
                # The per-query set is keyed by the same digest as the top-query
                # member, so the Redis keyspace holds no user text either.
                qkey = _click_query_key(raw_query, digest_key)
                claimed, claim_key = await _claim_click_signal(client_ip, canonical, q_article_id)
                if claimed:
                    p.zincrby(qkey, 1, str(q_article_id))
                    # Expire the per-query set for the same reason.
                    p.expire(qkey, config.CLICK_QUERY_TTL_SECONDS)
        await p.execute()
        claim_key = None  # the vote landed, so the claim is now genuinely spent
    except Exception as exc:
        # A claim that unlocked a tally which never landed would otherwise
        # silence this client for the whole window; give it back.
        await _release_click_signal(claim_key)
        _latch.warn_degraded("analytics Redis unavailable (%s); recording paused", exc)
    else:
        _latch.log_recovered()


async def click_signals(query: str) -> dict | None:
    """Per-query click signal for the click-boost layer, or None when the query
    has too little click volume to act on. Returns ``{"total": int, "by_id": {id: count}}``."""
    try:
        c = _client()
        digest_key = await _digest_key(c)
        if digest_key is None:
            # Without the key the per-query set cannot be addressed, and the
            # alternative -- falling back to the raw query -- would reintroduce
            # exactly the leak this path removed. No signal, rather than a
            # wrong-shaped lookup.
            return None
        key = _click_query_key(query, digest_key)
        # "total clicks" and the per-article breakdown are built from ALL members
        # of the sorted set, paginated in batches: a top-50 window would undercount
        # once a query has more than 50 clicked articles and break the invariant
        # ``sum(by_id.values()) == total`` that the click-boost layer relies on.
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
    """Top-`limit` digest rows from an over-fetched window, newest score first.

    The write path stores digests, so in steady state every member passes and this
    is a pure format check. It exists for verbatim-query members recorded before
    the write path switched to digests, and drops them rather than returning
    them, because the only alternative is handing back the very text being
    removed.

    This WITHHOLDS, it does not REMOVE: the write path re-arms the whole key's
    TTL on every event, so a legacy member never lapses on a live deployment.
    ``_scrub_legacy_once`` is what actually deletes them; this filter is what
    makes the response correct in the meantime.

    The caller over-fetches (``LEGACY_ROW_OVERFETCH``) precisely because this
    drops rows: reading only N members and filtering afterwards would let a
    cluster of legacy rows eat the whole window.
    """
    kept = [[m, _i(s)] for m, s in rows if _is_digest(m)]
    return kept[:limit]


async def summary() -> dict:
    """Aggregated metrics since the analytics DB was last cleared.

    The two top-N lists carry an opaque per-query DIGEST, never the query text:
    search queries are user-authored content aggregated by text, so returning
    them made this a cross-user read of everyone's search history. See
    :func:`query_digest` for why the text is kept out of the store entirely.

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

        # Delete any pre-upgrade verbatim members before reading the window, so
        # the corpus is gone from the store and not merely withheld from this
        # response. Best-effort; the filter below still applies either way.
        await _scrub_legacy_once(c)
        # Read a wider window than we report, because ``_digest_rows`` drops
        # pre-upgrade verbatim members. zrevrange's end index is INCLUSIVE, hence
        # the -1.
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

        # Read exactly the buckets ``record_click`` can write. Derived from the
        # same CLICK_POSITION_MIN/MAX the write path clamps to, so raising or
        # lowering the bound cannot desynchronize recording from reporting.
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
