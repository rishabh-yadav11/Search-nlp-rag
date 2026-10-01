"""Per-user interaction tracking and profile generation; every failure path degrades to the recommender's cold-start fallback."""
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


# Closed set: the type becomes a Redis hash FIELD name, so any other value would mint a field that only the key TTL ever expires.
class InteractionType(StrEnum):
    VIEW = "view"
    CLICK = "click"
    READ = "read"


_INTERACTION_TYPE_VALUES = frozenset(t.value for t in InteractionType)


class InteractionResult(StrEnum):
    """Why one ``record_interaction`` call did or did not write; the modes are distinct because the caller must report each truthfully."""

    RECORDED = "recorded"
    INVALID_TYPE = "invalid_type"
    UNKNOWN_ARTICLE = "unknown_article"
    CAP_REACHED = "cap_reached"
    UNAVAILABLE = "unavailable"


class UnknownArticleError(ValueError):
    """Raised when an interaction names an article that is not in the index."""


def _coerce_interaction_type(value: str) -> str:
    """Last line of defence: the value becomes a Redis hash field name, so it is matched against the closed enum rather than length-capped."""
    normalised = (value or "").strip().lower()
    if normalised not in _INTERACTION_TYPE_VALUES:
        raise ValueError(f"unknown interaction type: {value!r}")
    return normalised


# How long a CONFIRMED article is remembered, so a repeat click does not re-query the index.
_ARTICLE_EXISTS_TTL_SECONDS = 300
_ARTICLE_EXISTS_KEY = "user_profile:article_exists"


async def _require_known_article(client: aioredis.Redis, article_id: int) -> None:
    """Only a CONFIRMED article is cached: caching a rejection would key it to the caller-chosen id, so an unreachable index propagates rather than being cached as absent."""
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
    """The user's sorted set already holds their distinct article ids, so it is the ledger; re-interacting with a known article mints nothing."""
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

# Redis DB for user profiles (separate from analytics DB to survive deploy flushes).
_PROFILE_REDIS_DB = config.USER_PROFILE_REDIS_DB

_redis_client_instance: aioredis.Redis | None = None

_INTERACTION_SET_TTL_DAYS = 365
_PROFILE_VECTOR_TTL_HOURS = 6
_CATEGORIES_TTL_HOURS = 6

_PROFILE_MAX_INTERACTIONS = 50

# Scores are always re-read from the per-article counters, so a drifted index entry can cost a candidate slot, never produce a wrong number.
_TRENDING_INDEX_KEY = "trending:article_scores"
_TRENDING_INDEX_READY_KEY = "trending:article_scores:ready"
_TRENDING_CACHE_TTL_SECONDS = 3600
_TRENDING_RANK_BATCH = 50
_TRENDING_SCAN_COUNT = 500


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
) -> InteractionResult:
    """Record an interaction; every rejection is decided before the pipeline is queued, so a declined call writes nothing."""
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
        # An unreachable index means unverified, not "not indexed"; the two must not share an answer.
        logger.warning("Interaction article check unavailable: %s", exc)
        return InteractionResult.UNAVAILABLE

    try:
        if not await _has_interaction_slot(client, user_id, article_id):
            logger.warning("User %s hit the distinct-interaction cap", user_id)
            return InteractionResult.CAP_REACHED
        now = datetime.now(UTC).timestamp()
        article_key = f"article:interactions:{article_id}"
        pipe = client.pipeline()

        # Queued first: pipeline results come back in command order, so results[0] is this HINCRBY's post-increment value.
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
        })
        pipe.expire(detail_key, config.USER_INTERACTION_TTL_DAYS * 86400)

        # Advanced in the same transaction so trending never has to scan the keyspace.
        pipe.zincrby(_TRENDING_INDEX_KEY, 1, str(article_id))
        pipe.expire(_TRENDING_INDEX_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)
        pipe.expire(_TRENDING_INDEX_READY_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)

        # Derived data is only valid for the snapshot it was built from, so it is invalidated with the new signal.
        pipe.delete(
            f"user:profile_vector:{user_id}",
            f"user:categories:{user_id}",
        )

        results = await pipe.execute()
        # A post-increment of 1 means the counters were just (re)created and the index still holds the pre-expiry score: re-seed, do not increment.
        reseed_needed = results[0] == 1
    except Exception as exc:  # noqa: BLE001
        # Latched, not recomputed per failure: warn once into the outage, log "recovered" when it ends, then re-arm (elapsed-time rate limit; see app/degraded.py).
        _latches["record_interaction"].warn_degraded(
            "Failed to record user interaction: %s", exc
        )
        return InteractionResult.UNAVAILABLE
    _latches["record_interaction"].log_recovered()

    if reseed_needed:
        # Deliberately outside the guard: the interaction is already durable, so a failed index repair must not turn RECORDED into UNAVAILABLE.
        try:
            await _reseed_trending_index(client, article_id, article_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to re-seed trending index: %s", exc)
    return InteractionResult.RECORDED


async def _reseed_trending_index(client: aioredis.Redis, article_id: int, article_key: str) -> None:
    """Overwrite the index entry with the counters' real total once those counters have been recreated at zero."""
    total = _article_total(await client.hgetall(article_key))
    if total <= 0:
        return
    pipe = client.pipeline()
    pipe.zadd(_TRENDING_INDEX_KEY, {str(article_id): float(total)})
    pipe.expire(_TRENDING_INDEX_KEY, config.USER_INTERACTION_TTL_DAYS * 86400)
    await pipe.execute()


async def get_user_interactions(user_id: str, limit: int = _PROFILE_MAX_INTERACTIONS) -> list[tuple[int, float]]:
    """Recent user interactions as (article_id, timestamp) tuples, most recent first."""
    try:
        client = _redis_client()
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
    """Cached JSON vector; a non-finite or wrong-dimension stored value raises rather than being coerced, so a corrupt cache falls back to cold start."""
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
            # Decay is applied to the stored timestamps on every rebuild; the 30-day constant mirrors recommender._calculate_recency_score, so a config knob cannot desynchronise the two.
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
    """Top affinity categories as (category, score), highest score first."""
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
    """Summed by name against the known kinds: a name-blind sum credits junk fields and inflates trending scores for the full interaction TTL."""
    return sum(
        int(counts[kind_name])
        for kind_name in _INTERACTION_TYPE_VALUES
        if kind_name in counts and counts[kind_name].isdigit()
    )


async def _ensure_trending_index(client: aioredis.Redis) -> None:
    """The ready marker is written only here, so an install that takes interactions before its first trending read is still seeded."""
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
    try:
        client = _redis_client()
        window_start = datetime.now(UTC) - timedelta(days=config.TRENDING_VELOCITY_WINDOW_DAYS)
        window_key = f"trending:window:{window_start.strftime('%Y-%m-%d')}"

        cached = await client.get(window_key)
        if cached:
            _latches["get_trending_articles"].log_recovered()
            return json.loads(cached)

        await _ensure_trending_index(client)

        # Bounded rank batches: an indexed article whose counters have since expired must be skipped.
        article_scores: dict[str, float] = {}
        rank = 0
        while len(article_scores) < limit:
            ranked = await client.zrevrange(
                _TRENDING_INDEX_KEY, rank, rank + _TRENDING_RANK_BATCH - 1
            )
            if not ranked:
                break
            pipe = client.pipeline()
            for article_id in ranked:
                pipe.hgetall(f"article:interactions:{article_id}")
            counts_list = await pipe.execute()
            for article_id, counts in zip(ranked, counts_list, strict=True):
                total = _article_total(counts or {})
                if total > 0:
                    article_scores[article_id] = float(total)
            rank += len(ranked)

        # article_id breaks ties, so the order no longer depends on where the scan started
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
