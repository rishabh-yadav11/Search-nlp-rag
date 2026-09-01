"""Hybrid recommendation engine for similar articles and personalized feeds.

Provides:
  1. Similar articles for a given article (item-based collaborative filtering via
     dense vector similarity in Qdrant)
  2. Personalized recommendations for a user (user-based via aggregated profile
     vector + category affinity)
  3. Trending/popular articles (click-velocity based, Redis-backed)
  4. Latest top stories (cold-start fallback)

Architecture:
  - Candidate generators fetch articles from different sources
  - A hybrid scorer combines signals with configurable weights
  - Post-processing applies diversity and exclusion filters
"""
import asyncio
import copy
import logging
import math
import re
from datetime import UTC, datetime

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    ScoredPoint,
)

from app.config import config
from app.rerank_boost import extract_entities
from app.user_profile import (
    get_trending_articles,
    get_user_interactions,
    get_user_profile_categories,
)

logger = logging.getLogger(__name__)

# Redis keys for cached recommendations
SIMILAR_ARTICLES_TTL_SECONDS = 3600  # 1 hour
USER_RECOMMENDATIONS_TTL_SECONDS = 1800  # 30 minutes


async def get_similar_articles(
    article_id: int | str,
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    same_category: bool = False,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    """Get articles similar to the given article using dense vector similarity.

    Args:
        article_id: The article ID to find similar articles for
        limit: Maximum number of similar articles to return
        same_category: If True, filter to same industry/dealtype
        exclude_ids: Articles to exclude from results

    Returns:
        List of article dicts with 'id', 'title', 'score', etc.
    """
    if not config.ENABLE_RECOMMENDATIONS:
        return []

    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        # Build filter to exclude the source article and any specified IDs
        must_not = []
        if article_id:
            must_not.append(FieldCondition(key="id", match={"value": int(article_id)}))
        for eid in exclude_ids:
            try:
                must_not.append(FieldCondition(key="id", match={"value": int(eid)}))
            except (ValueError, TypeError):
                pass

        # Optionally filter to same category
        qfilter = None
        if same_category:
            # Fetch the source article's categories first
            source_result = await client.retrieve(
                collection_name=config.QDRANT_COLLECTION,
                point_id=int(article_id),
                with_payload=["industry_names", "dealtype_names"],
            )
            if source_result:
                payload = source_result[0].payload or {}
                industry = payload.get("industry_names")
                dealtype = payload.get("dealtype_names")

                conditions = []
                if industry:
                    conditions.append(FieldCondition(
                        key="industry_names",
                        match=MatchAny(any=industry if isinstance(industry, list) else [industry])
                    ))
                if dealtype:
                    conditions.append(FieldCondition(
                        key="dealtype_names",
                        match=MatchAny(any=dealtype if isinstance(dealtype, list) else [dealtype])
                    ))

                if conditions:
                    qfilter = Filter(must=conditions, must_not=must_not)
                else:
                    qfilter = Filter(must_not=must_not)
            else:
                qfilter = Filter(must_not=must_not)
        else:
            qfilter = Filter(must_not=must_not) if must_not else None

        # Query using the article's own vector as a query point (point_id query)
        result = await client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=int(article_id),  # Use point ID as query for nearest neighbors
            using="dense",  # Collection uses a named 'dense' vector
            query_filter=qfilter,
            limit=limit * 3,  # Fetch more to apply filters/post-processing
            with_payload=True,
            with_vectors=False,
        )

        return _format_articles(result.points, exclude_ids=[int(article_id)] + [int(eid) for eid in exclude_ids if isinstance(eid, str) and eid.isdigit()])

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error getting similar articles for %s: %s", article_id, exc)
        return []


async def get_personalized_recommendations(
    user_id: str,
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    """Get personalized recommendations for a user.

    Uses:
      1. User's interaction history to build preference profile
      2. Dense vector similarity to find related articles
      3. Category affinity matching
      4. Recency boost for fresh content
      5. Popularity signal from Redis

    Falls back to latest top stories if no user history exists.

    Args:
        user_id: The user ID to get recommendations for
        limit: Maximum number of recommendations
        exclude_ids: Articles to exclude (e.g., already viewed)

    Returns:
        List of article dicts with scores and metadata
    """
    if not config.ENABLE_RECOMMENDATIONS:
        return []

    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        # Get user's interaction history
        interactions = await get_user_interactions(user_id)
        categories = await get_user_profile_categories(user_id)

        if not interactions:
            # Cold start: return latest diverse articles
            logger.info("Cold start for user %s, returning latest articles", user_id)
            return await _get_latest_top_stories(limit, exclude_ids)

        # Build filter to exclude already-interacted articles
        must_not = [FieldCondition(key="id", match=MatchAny(any=[int(eid) for eid in exclude_ids if isinstance(eid, (int, str)) and str(eid).isdigit()]))]
        # Also exclude recently interacted articles
        recent_article_ids = [aid for aid, _ in interactions[:10]]  # Last 10 interactions
        if recent_article_ids:
            must_not.append(FieldCondition(key="id", match=MatchAny(any=recent_article_ids)))

        qfilter = Filter(must_not=must_not) if must_not else None

        # Get user's top categories
        top_categories = categories[:3] if categories else []
        category_filter = None
        if top_categories:
            # Build a filter for top industries
            industry_conditions = []
            for cat, _ in top_categories:
                if "industry" in cat.lower():
                    industry_conditions.append(FieldCondition(
                        key="industry_names",
                        match=MatchAny(any=[cat])
                    ))
            if industry_conditions:
                category_filter = Filter(must=industry_conditions)

        # Fetch candidate articles using different strategies
        candidates: dict[int | str, dict] = {}

        # 1. Vector similarity from recent interactions (semantic)
        async def _vector_candidates():
            """Get candidates from vector similarity to recent interactions."""
            results = []
            for article_id, _ in interactions[:5]:  # Use last 5 interactions
                try:
                    pts = await client.query_points(
                        collection_name=config.QDRANT_COLLECTION,
                        query=int(article_id),
                        using="dense",  # Collection uses a named 'dense' vector
                        query_filter=qfilter,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                    results.extend(pts.points)
                except Exception:  # noqa: BLE001, S112
                    continue
            return results

        # 2. Category-based candidates
        async def _category_candidates():
            """Get candidates matching user's top categories."""
            if not category_filter:
                return []
            try:
                pts = await client.query_points(
                    collection_name=config.QDRANT_COLLECTION,
                    query_filter=category_filter,
                    limit=limit * 2,
                    with_payload=True,
                    with_vectors=False,
                )
                return pts.points
            except Exception:  # noqa: BLE001
                return []

        # 3. Trending candidates
        async def _trending_candidates():
            """Get trending articles."""
            try:
                trending = await get_trending_articles(limit)
                if not trending:
                    return []
                ids = [t["article_id"] for t in trending]
                pts, _ = await client.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    limit=len(ids) * 5,
                    offset=None,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=qfilter,
                )
                # Match by ID, preserving the trending score order
                id_set = set(ids)
                return [p for p in pts if isinstance(p.id, int) and p.id in id_set][:len(ids)]
            except Exception:  # noqa: BLE001
                return []

        # Fetch all candidate sources in parallel
        vector_results, category_results, trending_results = await asyncio.gather(
            _vector_candidates(),
            _category_candidates(),
            _trending_candidates(),
        )

        # Score and blend candidates
        now = datetime.now(UTC)
        for point in vector_results:
            pid = point.id
            if pid in candidates:
                continue
            payload = point.payload or {}
            published = payload.get("published_date", "")
            score = _calculate_recency_score(published, now)
            candidates[pid] = {
                "point": point,
                "payload": payload,
                "semantic_score": score,
                "category_score": 0.0,
                "trending_score": 0.0,
            }

        for point in category_results:
            pid = point.id
            if pid in candidates:
                continue
            payload = point.payload or {}
            published = payload.get("published_date", "")
            score = _calculate_recency_score(published, now)
            candidates[pid] = {
                "point": point,
                "payload": payload,
                "semantic_score": 0.0,
                "category_score": score,
                "trending_score": 0.0,
            }

        for point in trending_results:
            pid = point.id
            if pid in candidates:
                continue
            payload = point.payload or {}
            published = payload.get("published_date", "")
            score = _calculate_recency_score(published, now)
            candidates[pid] = {
                "point": point,
                "payload": payload,
                "semantic_score": 0.0,
                "category_score": 0.0,
                "trending_score": score,
            }

        # Apply hybrid scoring
        scored = []
        for pid, data in candidates.items():
            final_score = (
                config.RECOMMEND_SIMILARITY_WEIGHT * data["semantic_score"] +
                config.RECOMMEND_CATEGORY_WEIGHT * data["category_score"] +
                config.RECOMMEND_RECENCY_WEIGHT * _calculate_recency_score(
                    data["payload"].get("published_date", ""), now
                ) +
                config.RECOMMEND_POPULARITY_WEIGHT * data["trending_score"]
            )
            scored.append({
                **data,
                "final_score": final_score,
                "point_id": pid,
            })

        # Sort by final score and return top results
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_candidates = scored[:limit * 2]

        return _format_articles(
            [c["point"] for c in top_candidates],
            exclude_ids=[int(eid) for eid in exclude_ids if isinstance(eid, (int, str)) and str(eid).isdigit()] + recent_article_ids
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error getting personalized recommendations for %s: %s", user_id, exc)
        return []


async def get_trending_feed(
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    """Get trending/popular articles based on click velocity.

    Args:
        limit: Maximum number of articles
        exclude_ids: Articles to exclude

    Returns:
        List of article dicts sorted by trending score
    """
    if not config.ENABLE_RECOMMENDATIONS:
        return []

    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        # Get trending data from Redis
        trending = await get_trending_articles(limit * 2)
        if not trending:
            # Fallback to latest articles
            return await _get_latest_top_stories(limit, exclude_ids)

        # Fetch full article details from Qdrant by their point IDs.
        ids = [t["article_id"] for t in trending]
        points = await client.retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

        score_lookup = {t["article_id"]: t["score"] for t in trending}
        result = []
        for point in points:
            pid = point.id
            if pid in exclude_ids or pid not in score_lookup:
                continue
            payload = point.payload or {}
            result.append({
                "id": pid,
                "title": payload.get("title", ""),
                "url": payload.get("url", ""),
                "published_date": payload.get("published_date"),
                "category": payload.get("category"),
                "summary": payload.get("summary", ""),
                "author_names": payload.get("author_names", []),
                "industry_names": payload.get("industry_names", []),
                "dealtype_names": payload.get("dealtype_names", []),
                "score": score_lookup[pid],
            })

        result.sort(key=lambda article: article["score"], reverse=True)
        return result[:limit]

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error getting trending feed: %s", exc)
        return []


async def get_latest_top_stories(
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    """Get latest top stories for cold-start fallback.

    Returns diverse articles from recent time period.
    """
    return await _get_latest_top_stories(limit, exclude_ids)


async def _get_latest_top_stories(
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    """Internal: Get latest top stories with diversity."""
    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        # Build filter
        qfilter = None
        if exclude_ids:
            qfilter = Filter(must_not=[
                FieldCondition(key="id", match=MatchAny(any=[int(eid) for eid in exclude_ids if isinstance(eid, (int, str)) and str(eid).isdigit()]))
            ])

        # Get recent articles (last 30 days)
        pts, _ = await client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=limit * 3,
            with_payload=True,
            with_vectors=False,
            scroll_filter=qfilter,
        )

        # Sort by published date and take most recent
        now = datetime.now(UTC)
        scored = []
        for point in pts:
            payload = point.payload or {}
            published = payload.get("published_date", "")
            recency = _calculate_recency_score(published, now)
            scored.append((point, recency))

        scored.sort(key=lambda x: x[1], reverse=True)
        return _format_articles([p for p, _ in scored[:limit * 2]], exclude_ids=exclude_ids)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error getting latest top stories: %s", exc)
        return []


def _calculate_recency_score(published_date: str, now: datetime) -> float:
    """Calculate recency score for an article.

    Returns a value between 0 and 1, where 1 is very recent and 0 is old.
    """
    if not published_date:
        return 0.5  # Default middle score for missing dates

    try:
        if published_date.endswith("Z"):
            published_date = published_date[:-1] + "+00:00"
        pub_dt = datetime.fromisoformat(published_date)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=UTC)
        age_days = (now - pub_dt).total_seconds() / 86400
        # Exponential decay: half-life of 30 days
        return math.exp(-age_days / 30)
    except (ValueError, TypeError):
        return 0.5


def _format_articles(
    points: list[ScoredPoint],
    exclude_ids: list[int] | None = None,
) -> list[dict]:
    """Format Qdrant points into article dicts."""
    exclude_ids = exclude_ids or []
    results = []

    for point in points:
        pid = point.id
        if pid in exclude_ids:
            continue

        payload = point.payload or {}
        if not payload:
            continue

        results.append({
            "id": pid,
            "title": payload.get("title", ""),
            "url": payload.get("url", ""),
            "published_date": payload.get("published_date"),
            "category": payload.get("category"),
            "summary": payload.get("summary", ""),
            "body": payload.get("body", ""),
            "author_names": payload.get("author_names", []) or [],
            "industry_names": payload.get("industry_names", []) or [],
            "dealtype_names": payload.get("dealtype_names", []) or [],
            "score": point.score if hasattr(point, 'score') else 0.0,
        })

    return results


# Module-level state reference (set during app startup from main.state)
state: dict = {}


# Acquisition relation-direction reranking. When a query names a company in a
# specific acquisition role (target vs buyer), results where that company plays
# the WRONG role (e.g. X as the acquirer in a "who acquired X?" query) are
# demoted and results where it plays the RIGHT role are promoted. This keeps
# "who acquired X?" from surfacing articles where X itself did the buying.


def _entity_acquisition_role(text: str, entity: str) -> bool | None:
    """Whether ``entity`` is the acquirer (True) or the acquired/target (False)
    in ``text``, or None when the text states no clear relation.

    "X acquired Y" / "X bought Y" -> X is the acquirer (True). "Y acquired X" /
    "X was acquired by Y" -> X is the target (False). Target (passive) forms are
    tested first so the passive "X was acquired by Y" is not mistaken for an
    active acquirer mention.
    """
    e = re.escape(entity)
    if re.search(
        rf"\b{e}\b[^.?!]*?\b(?:was|were|is|are|been|be)\b[^.?!]*?\b"
        rf"(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b[^.?!]*?\bby\b",
        text, re.IGNORECASE,
    ):
        return False
    if re.search(
        rf"\b{e}\b[^.?!]*?\b(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b[^.?!]*?\bby\b(?=\s+[A-Z])",
        text, re.IGNORECASE,
    ):
        return False
    if re.search(
        rf"\b(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b[^.?!]*?\b{e}\b",
        text, re.IGNORECASE,
    ):
        return False
    if re.search(
        rf"\b{e}\b[^.?!]*?\b(acquir\w+|bought|buyout|take\s*over|took\s*over|takeover)\b",
        text, re.IGNORECASE,
    ):
        return True
    return None


def rerank_acquisition_relation(query: str, results: list, direction: str | None = None) -> list:
    """Re-rank ``results`` by acquisition relation direction.

    ``direction`` comes from ``query_intent.acquisition_relation``: ``'target'``
    means the query's company was acquired (so articles where it is the buyer are
    demoted), ``'buyer'`` means it did the acquiring (so articles where it is the
    target are demoted). Returns an unchanged copy when ``direction`` is None or
    no entity can be identified. Inputs are never mutated.
    """
    if not direction:
        return list(results)
    entities = extract_entities(query)
    if not entities:
        return list(results)

    PROMOTE, DEMOTE = 1.30, 0.70
    scored: list[tuple[float, object]] = []
    for r in results:
        title = getattr(r, "title", "") or ""
        summary = getattr(r, "summary", "") or ""
        text = f"{title}. {summary}"
        role = None
        for e in entities:
            role = _entity_acquisition_role(text, e)
            if role is not None:
                break
        score = getattr(r, "score", None)
        if score is None:
            new_score = 0.0
        elif role is None:
            new_score = score
        elif direction == "target" and role is False:
            new_score = score * PROMOTE
        elif direction == "target" and role is True:
            new_score = score * DEMOTE
        elif direction == "buyer" and role is True:
            new_score = score * PROMOTE
        elif direction == "buyer" and role is False:
            new_score = score * DEMOTE
        else:
            new_score = score
        clone = copy.copy(r)
        clone.score = new_score
        scored.append((new_score, clone))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [clone for _, clone in scored]

