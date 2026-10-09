"""User interaction tracking and personalized profile generation.

Records article interactions (clicks, reads, views) per user and builds a
time-decayed preference profile stored in Redis. The profile consists of an
aggregated dense embedding vector and top-category affinity scores, used by
the recommender engine for personalized recommendations.

Cold-start: when no interaction history exists, recommend() falls back to
latest top stories across diverse industries.
"""
import json
import logging
import math
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import redis.asyncio as aioredis

from app.config import config
from app.degraded import DegradedLatch

logger = logging.getLogger(__name__)

_latches = {
    op: DegradedLatch(logger, f"user profile Redis ({op})")
    for op in (
        "record_interaction",
        "get_user_interactions",
        "get_user_profile_vector",
        "build_user_profile",
        "get_user_profile_categories",
        "invalidate_user_profile",
        "get_trending_articles",
    )
}


# Interaction types are a CLOSED set. ``article:interactions:{id}`` is a Redis
# hash whose FIELD NAME is the interaction type, so every distinct type a caller
# supplies mints a new field that only the key TTL ever expires. The set is a
# deliberate superset of what callers send today ('view', 'click', 'read'), so no
# working client breaks, while anything outside the set is refused.
class InteractionType(StrEnum):
    VIEW = "view"
    CLICK = "click"
    READ = "read"


_INTERACTION_TYPE_VALUES = frozenset(t.value for t in InteractionType)


class InteractionResult(StrEnum):
    """Why one ``record_interaction`` call did or did not write.

    Every decline mode is distinct because the caller must answer each one
    truthfully: reporting a rejected interaction type or a spent quota as
    "unknown article" would be a false statement about a real article.
    """

    RECORDED = "recorded"
    INVALID_TYPE = "invalid_type"
    UNKNOWN_ARTICLE = "unknown_article"
    CAP_REACHED = "cap_reached"
    UNAVAILABLE = "unavailable"


class UnknownArticleError(ValueError):
    """Raised when an interaction names an article that is not in the index."""


def _coerce_interaction_type(value: str) -> str:
    """Return the canonical interaction type, or raise for anything unknown.

    Last line of defence at the write layer: the HTTP model validates the field
    too, but this value chooses a Redis hash field, so an unrecognised one must
    never reach HINCRBY even if a future or internal caller skips the model. The
    value is MATCHED against the enum, not merely length-capped -- a cap alone
    would still admit an unbounded number of distinct fields.
    """
    normalised = (value or "").strip().lower()
    if normalised not in _INTERACTION_TYPE_VALUES:
        raise ValueError(f"unknown interaction type: {value!r}")
    return normalised


# How long a CONFIRMED article is remembered, so a reader clicking the same
# article repeatedly does not re-query the index on every event.
_ARTICLE_EXISTS_TTL_SECONDS = 300
_ARTICLE_EXISTS_KEY = "user_profile:article_exists"


async def _require_known_article(client: aioredis.Redis, article_id: int) -> None:
    """Raise UnknownArticleError unless article_id is a real indexed article.

    Every distinct id would otherwise mint an ``article:interactions:{id}`` hash
    plus a per-user detail key that outlives the request by
    USER_INTERACTION_TTL_DAYS, so an integer loop turns into unbounded key growth.

    Only a CONFIRMED article is cached: caching a rejection would key it to the
    caller-chosen id, so the flood this check exists to stop would still grow
    the keyspace, and an index blip would be latched as "absent" for the whole
    TTL. An unreachable index therefore propagates (reported as UNAVAILABLE)
    instead of being mistaken for a negative answer.
    """
    key = f"{_ARTICLE_EXISTS_KEY}:{article_id}"
    if await client.get(key) == "1":
        return

    from app.main import state  # lazy import avoids a startup cycle

    points = await state["qdrant"].retrieve(
        collection_name=config.QDRANT_COLLECTION,
        ids=[article_id],
        with_payload=False,
        with_vectors=False,
    )
    if not points:
        raise UnknownArticleError(article_id)
    await client.set(key, "1", ex=_ARTICLE_EXISTS_TTL_SECONDS)


async def _has_interaction_slot(client: aioredis.Redis, user_id: str, article_id: int) -> bool:
    """Whether this user may mint one more distinct article-interaction key.

    The user's existing ``user:interactions:{user_id}`` sorted set already holds
    exactly the distinct article ids they have interacted with, so it doubles as
    the ledger instead of introducing a second counter with its own TTL. A
    re-interaction with a known article is always allowed: it rewrites existing
    keys and mints nothing new.
    """
    cap = config.USER_MAX_DISTINCT_INTERACTIONS
    if cap <= 0:
        return True
    key = f"user:interactions:{user_id}"
    pipe = client.pipeline()
    pipe.zcard(key)
    pipe.zscore(key, str(article_id))
    distinct, already_seen = await pipe.execute()
    if already_seen is not None:
        return True
    return int(distinct) < cap

# Redis DB for user profiles (separate from the analytics DB so a deploy flush
# of the query cache does not empty it).
_PROFILE_REDIS_DB = config.USER_PROFILE_REDIS_DB

# Lightweight cached client reuse so repeated calls share one socket-pooled
# instance rather than re-creating connections on every call. Calls can arrive
# before app startup sets it, so fall back to creating a short-lived client.
_redis_client_instance: aioredis.Redis | None = None

# TTL constants. ``_INTERACTION_SET_TTL_DAYS`` keeps raw interactions long-term
# for profile building; the profile vector and categories are recomputed
# periodically as new signals arrive.
_INTERACTION_SET_TTL_DAYS = 365
_PROFILE_VECTOR_TTL_HOURS = 6
_CATEGORIES_TTL_HOURS = 6

# Number of interaction records to consider for profile building (most recent N)
_PROFILE_MAX_INTERACTIONS = 50

# Trending index: a sorted set of article_id -> total interaction count, kept in
# step by record_interaction so the trending read path ranks candidates without
# walking the keyspace. Scores are always read back from the per-article
# counters, so the index only decides which candidates get hydrated -- a score
# that drifts can cost a candidate slot, never a wrong number. The index and its
# ready marker share the counter TTL and are refreshed together, so they fall out
# of step only if Redis evicts one of them.
_TRENDING_INDEX_KEY = "trending:article_scores"
_TRENDING_INDEX_READY_KEY = "trending:article_scores:ready"
_TRENDING_CACHE_TTL_SECONDS = 3600
_TRENDING_RANK_BATCH = 50      # candidates hydrated per pipelined HGETALL round trip
_TRENDING_SCAN_COUNT = 500     # only used by the one-time index bootstrap


def _redis_client() -> aioredis.Redis:
    global _redis_client_instance
    if _redis_client_instance is None:
        _redis_client_instance = aioredis.from_url(
            config.REDIS_URL,
            db=_PROFILE_REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client_instance


async def record_interaction(
    user_id: str,
    article_id: int,
    interaction_type: str = InteractionType.CLICK,
    dwell_time_ms: int | None = None,
    feed_type: str = "",
    session_id: str = "",
) -> InteractionResult:
    """Record a user-article interaction in Redis, reporting why if it did not.

    Stores a ``user:interactions:{user_id}`` sorted set scored by timestamp, a
    per-article interaction detail hash for dwell-time analysis, and a
    per-interaction-kind article counter for trending.

    ``feed_type`` (``personalized|trending|latest``) and ``session_id`` are
    beacon context: feed_type is persisted in the interaction detail hash so the
    feed a click came from stays recoverable, and session_id is stored alongside
    so a search->click->interaction chain can be joined server-side. Both are
    opaque and bounded.

    Rejects an unknown ``interaction_type``, an ``article_id`` that is not in the
    article index, and a user who has already interacted with
    USER_MAX_DISTINCT_INTERACTIONS distinct articles. Each is checked BEFORE the
    pipeline is built, so a declined call writes nothing at all.
    """
    # Validate the kind before anything is queued: this value becomes a Redis
    # hash FIELD name, so an unchecked string is an unbounded field mint.
    try:
        kind = _coerce_interaction_type(interaction_type)
    except ValueError:
        logger.warning("Rejected interaction with unknown type %r", interaction_type)
        return InteractionResult.INVALID_TYPE

    client = _redis_client()
    try:
        await _require_known_article(client, article_id)
    except UnknownArticleError:
        logger.warning("Rejected interaction for unknown article %s", article_id)
        return InteractionResult.UNKNOWN_ARTICLE
    except Exception as exc:  # noqa: BLE001
        # The index could not be reached, so the id is unverified. That is NOT
        # the same answer as "not indexed" and must not be reported as one.
        logger.warning("Interaction article check unavailable: %s", exc)
        return InteractionResult.UNAVAILABLE

    try:
        if not await _has_interaction_slot(client, user_id, article_id):
            logger.warning("User %s hit the distinct-interaction cap", user_id)
            return InteractionResult.CAP_REACHED
        now = datetime.now(UTC).timestamp()
        article_key = f"article:interactions:{article_id}"
        pipe = client.pipeline()

        # Queued first: pipeline results come back in command order, so results[0]
        # is this HINCRBY's post-increment value. A value of 1 means the article's
        # counters are brand new, so the trending index still holds this
        # article's pre-expiry score and must be re-seeded, not incremented.
        pipe.hincrby(article_key, kind, 1)
        pipe.hset(article_key, "last_timestamp", str(now))
        pipe.expire(article_key, config.USER_INTERACTION_TTL_DAYS * 86400)

        pipe.zadd(f"user:interactions:{user_id}", {str(article_id): now})
        pipe.expire(f"user:interactions:{user_id}", _INTERACTION_SET_TTL_DAYS * 86400)

        detail_key = f"user:interaction_detail:{user_id}:{article_id}"
        pipe.hset(detail_key, mapping={
            "type": kind,
            "timestamp": str(now),
            "dwell_time_ms": str(dwell_time_ms or 0),
            "feed_type": (feed_type or "")[:64],
            "session_id": (session_id or "")[:128],
        })
        pipe.expire(detail_key, config.USER_INTERACTION_TTL_DAYS * 86400)

        # Advance the trending index in the same transaction, so trending never
        # has to scan the keyspace to discover this article.
        pipe.zincrby(_TRENDING_INDEX_KEY, 1, str(article_id))
        pipe.expire(_TRENDING_INDEX_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)
        pipe.expire(_TRENDING_INDEX_READY_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)

        # Derived data is only valid for the interaction snapshot it was built
        # from. Invalidate it in the same Redis transaction as the new signal.
        pipe.delete(
            f"user:profile_vector:{user_id}",
            f"user:categories:{user_id}",
        )

        results = await pipe.execute()
        reseed_needed = results[0] == 1
    except Exception as exc:  # noqa: BLE001
        # Warn on the transition into the outage, log one "recovered" line when
        # it ends, then re-arm so the next outage is announced again. Volume
        # is rate-limited by elapsed time, not by request count. See
        # app/degraded.py.
        _latches["record_interaction"].warn_degraded(
            "Failed to record user interaction: %s", exc
        )
        return InteractionResult.UNAVAILABLE
    # The interaction is durably recorded, so Redis answered: a real recovery
    # observation, which is what re-arms the latch for the next outage.
    _latches["record_interaction"].log_recovered()

    if reseed_needed:
        # Deliberately outside the guard above: the interaction is already durably
        # recorded at this point, so a failed index repair must not turn a true
        # RECORDED into a false UNAVAILABLE. The stale index entry is corrected by
        # the next interaction regardless.
        try:
            await _reseed_trending_index(client, article_id, article_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to re-seed trending index: %s", exc)
    return InteractionResult.RECORDED


async def _reseed_trending_index(client: aioredis.Redis, article_id: int, article_key: str) -> None:
    """Re-seed the trending index when an article's counters start from scratch.

    The counters expired (or never existed) while the index kept the article's
    pre-expiry score, so overwrite it with the counters' real total instead of
    incrementing the stale one. If a concurrent write lands in the gap between
    the read and the write, the index is left one behind rather than one ahead:
    scores are read back from the counters on every trending read, and the next
    interaction increments the index to the correct value.
    """
    total = _article_total(await client.hgetall(article_key))
    if total <= 0:
        return
    pipe = client.pipeline()
    pipe.zadd(_TRENDING_INDEX_KEY, {str(article_id): float(total)})
    pipe.expire(_TRENDING_INDEX_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)
    await pipe.execute()


async def get_user_interactions(user_id: str, limit: int = _PROFILE_MAX_INTERACTIONS) -> list[tuple[int, float]]:
    """Recent user interactions as (article_id, timestamp) tuples, newest first."""
    try:
        client = _redis_client()
        # zrevrange returns members in descending score order (most recent first)
        items = await client.zrevrange(
            f"user:interactions:{user_id}",
            0,
            limit - 1,
            withscores=True,
        )
        _latches["get_user_interactions"].log_recovered()
        return [(int(article_id), float(ts)) for article_id, ts in items]
    except Exception as exc:  # noqa: BLE001
        _latches["get_user_interactions"].warn_degraded(
            "Failed to get user interactions: %s", exc
        )
        return []


async def get_user_profile_vector(user_id: str) -> list[float] | None:
    """Return the cached or newly-derived preference vector, if available.

    Redis stores the vector as a JSON string so its dimension and ordered values
    survive round trips. A missing or malformed value is a cold start.
    """
    try:
        client = _redis_client()
        raw_vector = await client.get(f"user:profile_vector:{user_id}")
        if raw_vector is None:
            _latches["get_user_profile_vector"].log_recovered()
            return await build_user_profile(user_id)
        if isinstance(raw_vector, bytes):
            raw_vector = raw_vector.decode("utf-8")
        values = json.loads(raw_vector)
        if not isinstance(values, list):
            raise TypeError("profile vector is not a JSON array")
        vector = [float(value) for value in values]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("profile vector contains invalid values")
        _latches["get_user_profile_vector"].log_recovered()
        return vector
    except Exception as exc:  # noqa: BLE001
        _latches["get_user_profile_vector"].warn_degraded(
            "Failed to get user profile vector; using cold start: %s", exc
        )
        return None


async def build_user_profile(user_id: str) -> list[float] | None:
    """Build and cache a deterministic profile from recent article interactions.

    Article vectors and category payloads are read together from Qdrant. Any
    storage or lookup failure leaves the user in the documented cold-start
    path rather than serving an incomplete personalized profile.
    """
    try:
        interactions = await get_user_interactions(user_id)
        if not interactions:
            _latches["build_user_profile"].log_recovered()
            return None

        from app.main import state  # lazy import avoids a startup cycle

        articles = await state["qdrant"].retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=[article_id for article_id, _ in interactions],
            with_payload=["industry_names", "dealtype_names"],
            with_vectors=["dense"],
        )
        by_id = {int(article.id): article for article in articles}
        now = datetime.now(UTC).timestamp()
        weighted_values: list[tuple[list[float], float]] = []
        affinities: dict[str, float] = {}

        for article_id, timestamp in interactions:
            article = by_id.get(article_id)
            if article is None:
                continue
            raw_vector = article.vector.get("dense") if isinstance(article.vector, dict) else article.vector
            if not isinstance(raw_vector, (list, tuple)):
                continue
            vector = [float(value) for value in raw_vector]
            if not vector or not all(math.isfinite(value) for value in vector):
                continue
            # Newer signals have higher influence while preserving determinism.
            # The 30-day time constant is deliberate and matches the one in
            # recommender._calculate_recency_score; it is a fixed constant rather
            # than a config knob, so a stale environment variable cannot
            # desynchronise the two decays.
            weight = math.exp(-max(0.0, now - timestamp) / (30 * 86400))
            weighted_values.append((vector, weight))
            payload = article.payload or {}
            for key in ("industry_names", "dealtype_names"):
                categories = payload.get(key, [])
                if not isinstance(categories, list):
                    categories = [categories]
                for category in categories:
                    if isinstance(category, str) and category:
                        affinities[category] = affinities.get(category, 0.0) + weight

        if not weighted_values:
            _latches["build_user_profile"].log_recovered()
            return None
        dimension = len(weighted_values[0][0])
        if any(len(vector) != dimension for vector, _ in weighted_values):
            raise ValueError("article vectors have inconsistent dimensions")
        total_weight = sum(weight for _, weight in weighted_values)
        profile = [sum(vector[index] * weight for vector, weight in weighted_values) / total_weight for index in range(dimension)]

        client = _redis_client()
        pipe = client.pipeline()
        vector_key = f"user:profile_vector:{user_id}"
        category_key = f"user:categories:{user_id}"
        pipe.set(vector_key, json.dumps(profile), ex=_PROFILE_VECTOR_TTL_HOURS * 3600)
        pipe.delete(category_key)
        if affinities:
            pipe.zadd(category_key, affinities)
            pipe.expire(category_key, _CATEGORIES_TTL_HOURS * 3600)
        await pipe.execute()
        _latches["build_user_profile"].log_recovered()
        return profile
    except Exception as exc:  # noqa: BLE001
        _latches["build_user_profile"].warn_degraded(
            "Failed to build user profile; using cold start: %s", exc
        )
        return None

async def get_user_profile_categories(user_id: str) -> list[tuple[str, float]]:
    """Get top affinity categories for a user from Redis cache, as
    (category, score) tuples sorted by score descending."""
    try:
        client = _redis_client()
        items = await client.zrevrange(
            f"user:categories:{user_id}",
            0,
            -1,
            withscores=True,
        )
        _latches["get_user_profile_categories"].log_recovered()
        return [(str(cat), float(score)) for cat, score in items]
    except Exception as exc:  # noqa: BLE001
        _latches["get_user_profile_categories"].warn_degraded(
            "Failed to get user categories: %s", exc
        )
        return []


async def invalidate_user_profile(user_id: str) -> None:
    """Clear cached user profile to force recomputation on next request."""
    try:
        client = _redis_client()
        await client.delete(
            f"user:profile_vector:{user_id}",
            f"user:categories:{user_id}",
        )
    except Exception as exc:  # noqa: BLE001
        _latches["invalidate_user_profile"].warn_degraded(
            "Failed to invalidate user profile: %s", exc
        )
    else:
        _latches["invalidate_user_profile"].log_recovered()


def _article_total(counts: dict) -> int:
    """Total interactions for an article from its per-article counter hash.

    Summed by NAME against the known interaction kinds, ignoring bookkeeping
    fields such as ``last_timestamp``. A name-blind sum over every digit-valued
    field would credit junk fields, so any kind minted before the write-side enum
    landed would keep inflating a chosen article's trending score for the full
    USER_INTERACTION_TTL_DAYS.
    """
    return sum(
        int(counts[kind_name])
        for kind_name in _INTERACTION_TYPE_VALUES
        if kind_name in counts and counts[kind_name].isdigit()
    )


async def _ensure_trending_index(client: aioredis.Redis) -> None:
    """Seed the trending index from the per-article counters if it is missing.

    Installs with `article:interactions:*` hashes but no sorted set behind them
    get one single scan, after which every read is served from the index. The
    ready marker is written only here, never by the write path, so an install
    that takes interactions before its first trending read still gets seeded. It
    is written even when no counters are found, and carries the index's TTL so
    the two cannot outlive each other.
    """
    if await client.exists(_TRENDING_INDEX_READY_KEY):
        return

    cursor = 0
    while True:
        cursor, keys = await client.scan(
            cursor, match="article:interactions:*", count=_TRENDING_SCAN_COUNT
        )
        keys = [k for k in keys if k.startswith("article:interactions:")]
        if keys:
            pipe = client.pipeline()
            for key in keys:
                pipe.hgetall(key)
            counts_list = await pipe.execute()
            seeds: dict[str, float] = {}
            for key, counts in zip(keys, counts_list, strict=True):
                total = _article_total(counts or {})
                if total > 0:
                    seeds[key.rsplit(":", 1)[-1]] = float(total)
            if seeds:
                pipe = client.pipeline()
                pipe.zadd(_TRENDING_INDEX_KEY, seeds)
                pipe.expire(_TRENDING_INDEX_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)
                await pipe.execute()
        if cursor == 0:
            break

    await client.set(
        _TRENDING_INDEX_READY_KEY, "1", ex=config.USER_INTERACTION_TTL_DAYS * 86400
    )


async def get_trending_articles(limit: int = 10) -> list[dict]:
    """Get trending articles based on click velocity over recent window.

    Ranks candidates from the incrementally maintained trending index and
    hydrates their scores with one pipelined HGETALL per rank batch, so the
    result is the same as a full scan without walking the keyspace.
    """
    try:
        client = _redis_client()
        window_start = datetime.now(UTC) - timedelta(days=config.TRENDING_VELOCITY_WINDOW_DAYS)
        window_key = f"trending:window:{window_start.strftime('%Y-%m-%d')}"

        cached = await client.get(window_key)
        if cached:
            _latches["get_trending_articles"].log_recovered()
            return json.loads(cached)

        await _ensure_trending_index(client)

        # Walk the index in bounded rank batches: an indexed article whose
        # counters have since expired must be skipped, and the index is ranked by
        # a score that may have moved on since.
        article_scores: dict[str, float] = {}
        rank = 0
        while len(article_scores) < limit:
            ranked = await client.zrevrange(
                _TRENDING_INDEX_KEY, rank, rank + _TRENDING_RANK_BATCH - 1
            )
            if not ranked:
                break
            # One pipelined HGETALL per rank batch instead of a round trip per
            # key, which is the whole point of the index.
            pipe = client.pipeline()
            for article_id in ranked:
                pipe.hgetall(f"article:interactions:{article_id}")
            counts_list = await pipe.execute()
            for article_id, counts in zip(ranked, counts_list, strict=True):
                total = _article_total(counts or {})
                if total > 0:
                    article_scores[article_id] = float(total)
            rank += len(ranked)

        # article_id breaks ties, so the order does not depend on where the
        # keyspace scan happened to start
        sorted_articles = sorted(
            article_scores.items(), key=lambda x: (-x[1], int(x[0]))
        )[:limit]
        result = [{"article_id": int(aid), "score": score} for aid, score in sorted_articles]

        if result:
            await client.set(window_key, json.dumps(result), ex=_TRENDING_CACHE_TTL_SECONDS)

        _latches["get_trending_articles"].log_recovered()
        return result
    except Exception as exc:  # noqa: BLE001
        _latches["get_trending_articles"].warn_degraded(
            "Failed to get trending articles: %s", exc
        )
        return []
