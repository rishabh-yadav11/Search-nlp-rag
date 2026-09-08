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
import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger(__name__)

# Redis DB for user profiles (separate from analytics DB to survive deploy flushes).
_PROFILE_REDIS_DB = config.USER_PROFILE_REDIS_DB

# Lightweight cached client reuse so repeated calls share one socket-pooled
# instance rather than re-creating connections on every call. Calls can arrive
# before app startup sets it, so fall back to creating a short-lived client.
_redis_client_instance: aioredis.Redis | None = None

# TTL constants
_INTERACTION_SET_TTL_DAYS = 365  # keep raw interactions long-term for profile building
_PROFILE_VECTOR_TTL_HOURS = 6    # recompute profile periodically as new signals arrive
_CATEGORIES_TTL_HOURS = 6

# Number of interaction records to consider for profile building (most recent N)
_PROFILE_MAX_INTERACTIONS = 50


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
    interaction_type: str = "click",
    dwell_time_ms: int | None = None,
) -> None:
    """Record a user-article interaction in Redis.

    Stores:
      - A sorted set of interactions: user:interactions:{user_id} -> article_id scored by timestamp
      - Individual article interaction details for dwell-time analysis
    """
    try:
        client = _redis_client()
        now = datetime.now(UTC).timestamp()
        pipe = client.pipeline()

        # Add to user's interaction history (sorted set, score = timestamp)
        pipe.zadd(f"user:interactions:{user_id}", {str(article_id): now})
        pipe.expire(f"user:interactions:{user_id}", _INTERACTION_SET_TTL_DAYS * 86400)

        # Record interaction type for potential future dwell-time analysis
        detail_key = f"user:interaction_detail:{user_id}:{article_id}"
        pipe.hset(detail_key, mapping={
            "type": interaction_type,
            "timestamp": str(now),
            "dwell_time_ms": str(dwell_time_ms or 0),
        })
        pipe.expire(detail_key, config.USER_INTERACTION_TTL_DAYS * 86400)

        # Update article-level interaction counts (for future popularity scoring)
        article_key = f"article:interactions:{article_id}"
        pipe.hincrby(article_key, interaction_type, 1)
        pipe.hset(article_key, "last_timestamp", str(now))
        pipe.expire(article_key, config.USER_INTERACTION_TTL_DAYS * 86400)

        # Derived data is only valid for the interaction snapshot it was built
        # from. Invalidate it in the same Redis transaction as the new signal.
        pipe.delete(
            f"user:profile_vector:{user_id}",
            f"user:categories:{user_id}",
        )

        await pipe.execute()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record user interaction: %s", exc)


async def get_user_interactions(user_id: str, limit: int = _PROFILE_MAX_INTERACTIONS) -> list[tuple[int, float]]:
    """Get recent user interactions sorted by recency.

    Returns list of (article_id, timestamp) tuples.
    """
    try:
        client = _redis_client()
        # zrevrange returns members in descending score order (most recent first)
        items = await client.zrevrange(
            f"user:interactions:{user_id}",
            0,
            limit - 1,
            withscores=True,
        )
        return [(int(article_id), float(ts)) for article_id, ts in items]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get user interactions: %s", exc)
        return []


async def get_user_profile_vector(user_id: str) -> list[float] | None:
    """Return the cached or newly-derived preference vector, if available.

    Redis stores the vector as a JSON string so its dimension and ordered
    values survive round trips. A missing or malformed value is a cold start.
    """
    try:
        client = _redis_client()
        raw_vector = await client.get(f"user:profile_vector:{user_id}")
        if raw_vector is None:
            return await build_user_profile(user_id)
        if isinstance(raw_vector, bytes):
            raw_vector = raw_vector.decode("utf-8")
        values = json.loads(raw_vector)
        if not isinstance(values, list):
            raise ValueError("profile vector is not a JSON array")
        vector = [float(value) for value in values]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("profile vector contains invalid values")
        return vector
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get user profile vector; using cold start: %s", exc)
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
        return profile
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build user profile; using cold start: %s", exc)
        return None

async def get_user_profile_categories(user_id: str) -> list[tuple[str, float]]:
    """Get top affinity categories for a user from Redis cache.

    Returns list of (category, score) tuples sorted by score descending.
    """
    try:
        client = _redis_client()
        items = await client.zrevrange(
            f"user:categories:{user_id}",
            0,
            -1,
            withscores=True,
        )
        return [(str(cat), float(score)) for cat, score in items]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get user categories: %s", exc)
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
        logger.warning("Failed to invalidate user profile: %s", exc)


async def get_trending_articles(limit: int = 10) -> list[dict]:
    """Get trending articles based on click velocity over recent window.

    Queries Redis for article interaction counts in the trending window.
    Returns list of {article_id, score} dicts sorted by popularity.
    """
    try:
        client = _redis_client()
        # Get articles with most interactions in the trending window
        window_start = datetime.now(UTC) - timedelta(days=config.TRENDING_VELOCITY_WINDOW_DAYS)
        window_key = f"trending:window:{window_start.strftime('%Y-%m-%d')}"

        # Check if we have a cached trending set for this window
        cached = await client.get(window_key)
        if cached:
            return json.loads(cached)

        # Query individual article scores
        article_scores: dict[str, float] = {}
        batch_size = 100
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match="article:interactions:*", count=batch_size)
            if not keys:
                break
            for key in keys:
                article_id = key.split(":")[-1]
                if not key.startswith("article:interactions:"):
                    continue
                counts = await client.hgetall(key)
                total = sum(int(v) for v in counts.values() if v.isdigit())
                if total > 0:
                    article_scores[article_id] = float(total)
            if cursor == 0:
                break

        # Sort by score and return top articles
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        result = [{"article_id": int(aid), "score": score} for aid, score in sorted_articles]

        # Cache for the window duration
        if result:
            await client.set(window_key, json.dumps(result), ex=3600)

        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to get trending articles: %s", exc)
        return []


async def get_interaction_history(user_id: str) -> list[dict]:
    """Get user's interaction history for profile building.

    Returns list of interaction records with article metadata.
    """
    interactions = await get_user_interactions(user_id)
    if not interactions:
        return []

    # Build a lookup of article metadata from Redis or Qdrant
    # For now, return basic interaction data; metadata can be enriched later
    return [
        {"article_id": aid, "timestamp": ts, "recency": (datetime.now(UTC).timestamp() - ts)}
        for aid, ts in interactions
    ]
