"""Hybrid recommendation engine: similar articles, personalized feeds, trending feed."""
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

SIMILAR_ARTICLES_TTL_SECONDS = 3600
USER_RECOMMENDATIONS_TTL_SECONDS = 1800

# Mirrors main._PAYLOAD_FIELDS minus the large `body` field, which only chat fetches.
_RECOMMEND_PAYLOAD_FIELDS = [
    "title",
    "url",
    "published_date",
    "category",
    "summary",
    "author_names",
    "industry_names",
    "dealtype_names",
]


def _candidate_pool(limit: int, *, over: int = 1) -> int:
    """Candidate pool for a page of ``limit``: deliberately wider, since scoring keeps only the best ``limit * 2`` and formatting then drops excluded and empty ids."""
    return max(limit * over, config.RECOMMEND_CANDIDATES_LIMIT)


async def get_similar_articles(
    article_id: int | str,
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    same_category: bool = False,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    """Dense-vector neighbours of ``article_id``, excluding it and ``exclude_ids``, optionally restricted to the same industry/dealtype."""
    if not config.ENABLE_RECOMMENDATIONS:
        return []

    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        must_not = []
        if article_id:
            must_not.append(FieldCondition(key="id", match={"value": int(article_id)}))
        for eid in exclude_ids:
            try:
                must_not.append(FieldCondition(key="id", match={"value": int(eid)}))
            except (ValueError, TypeError):
                pass

        qfilter = None
        if same_category:
            source_result = await client.retrieve(
                collection_name=config.QDRANT_COLLECTION,
                point_id=int(article_id),
                with_payload=["industry_names", "dealtype_names"],
            )
            if source_result:
                payload = source_result[0].payload or {}
                industry = payload.get("industry_names")
                dealtype = payload.get("dealtype_names")

                # A mistyped key here yields an empty related list for every user while the suite stays green.
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

        # Qdrant treats a point ID as a query vector and returns its nearest neighbours.
        result = await client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=int(article_id),
            using="dense",  # Collection uses a named 'dense' vector
            query_filter=qfilter,
            # 3x, not _candidate_pool: nothing below truncates, so the fetch width IS the response width.
            limit=limit * 3,
            with_payload=_RECOMMEND_PAYLOAD_FIELDS,
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
    """Personalized recommendations blending interaction history, vector similarity, category affinity, recency and trending score."""
    if not config.ENABLE_RECOMMENDATIONS:
        return []

    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        interactions = await get_user_interactions(user_id)
        categories = await get_user_profile_categories(user_id)

        if not interactions:
            # Cold start: with no history there is no signal, so serve the general feed rather than nothing.
            logger.info("Cold start for user %s, returning latest articles", user_id)
            return await _get_latest_top_stories(limit, exclude_ids)

        must_not = [FieldCondition(key="id", match=MatchAny(any=[int(eid) for eid in exclude_ids if isinstance(eid, (int, str)) and str(eid).isdigit()]))]
        recent_article_ids = [aid for aid, _ in interactions[:10]]
        if recent_article_ids:
            must_not.append(FieldCondition(key="id", match=MatchAny(any=recent_article_ids)))

        qfilter = Filter(must_not=must_not) if must_not else None

        top_categories = categories[:3] if categories else []
        category_filter = None
        if top_categories:
            industry_conditions = []
            for cat, _ in top_categories:
                if "industry" in cat.lower():
                    industry_conditions.append(FieldCondition(
                        key="industry_names",
                        match=MatchAny(any=[cat])
                    ))
            if industry_conditions:
                category_filter = Filter(must=industry_conditions)

        candidates: dict[int | str, dict] = {}

        async def _vector_candidates():
            results = []
            for article_id, _ in interactions[:5]:
                try:
                    pts = await client.query_points(
                        collection_name=config.QDRANT_COLLECTION,
                        query=int(article_id),
                        using="dense",
                        query_filter=qfilter,
                        limit=_candidate_pool(limit),
                        with_payload=_RECOMMEND_PAYLOAD_FIELDS,
                        with_vectors=False,
                    )
                    results.extend(pts.points)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Error getting vector candidates for user %s article %s: %s",
                        user_id, article_id, exc,
                    )
                    continue
            return results

        async def _category_candidates():
            if not category_filter:
                return []
            try:
                pts = await client.query_points(
                    collection_name=config.QDRANT_COLLECTION,
                    query_filter=category_filter,
                    limit=_candidate_pool(limit),
                    with_payload=_RECOMMEND_PAYLOAD_FIELDS,
                    with_vectors=False,
                )
                return pts.points
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error getting category candidates for user %s: %s", user_id, exc)
                return []

        async def _trending_candidates():
            try:
                # Bare limit, not _candidate_pool: the trending cache key carries no limit, so a wider pool buys no recall.
                trending = await get_trending_articles(limit)
                if not trending:
                    return []
                ids = [t["article_id"] for t in trending]
                pts, _ = await client.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    limit=len(ids) * 5,
                    offset=None,
                    with_payload=_RECOMMEND_PAYLOAD_FIELDS,
                    with_vectors=False,
                    scroll_filter=qfilter,
                )
                id_set = set(ids)
                return [p for p in pts if isinstance(p.id, int) and p.id in id_set][:len(ids)]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error getting trending candidates for user %s: %s", user_id, exc)
                return []

        vector_results, category_results, trending_results = await asyncio.gather(
            _vector_candidates(),
            _category_candidates(),
            _trending_candidates(),
        )

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
    """Trending/popular articles by click velocity, as article dicts sorted by trending score."""
    if not config.ENABLE_RECOMMENDATIONS:
        return []

    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        # 2x, not _candidate_pool: the feed returns result[:limit], so a wider pool only fetches payloads that get trimmed.
        trending = await get_trending_articles(limit * 2)
        if not trending:
            return await _get_latest_top_stories(limit, exclude_ids)

        ids = [t["article_id"] for t in trending]
        points = await client.retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=ids,
            with_payload=_RECOMMEND_PAYLOAD_FIELDS,
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


async def _get_latest_top_stories(
    limit: int = config.RECOMMEND_DEFAULT_LIMIT,
    exclude_ids: list[int | str] | None = None,
) -> list[dict]:
    try:
        client = state["qdrant"]
        exclude_ids = exclude_ids or []

        qfilter = None
        if exclude_ids:
            qfilter = Filter(must_not=[
                FieldCondition(key="id", match=MatchAny(any=[int(eid) for eid in exclude_ids if isinstance(eid, (int, str)) and str(eid).isdigit()]))
            ])

        pts, _ = await client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=_candidate_pool(limit, over=3),
            with_payload=_RECOMMEND_PAYLOAD_FIELDS,
            with_vectors=False,
            scroll_filter=qfilter,
        )

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
    """Recency score for an article, between 0 (old) and 1 (very recent)."""
    if not published_date:
        return 0.5

    try:
        if published_date.endswith("Z"):
            published_date = published_date[:-1] + "+00:00"
        pub_dt = datetime.fromisoformat(published_date)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=UTC)
        age_days = (now - pub_dt).total_seconds() / 86400
        return math.exp(-age_days / 30)
    except (ValueError, TypeError):
        return 0.5


def _format_articles(
    points: list[ScoredPoint],
    exclude_ids: list[int] | None = None,
) -> list[dict]:
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
            "author_names": payload.get("author_names", []) or [],
            "industry_names": payload.get("industry_names", []) or [],
            "dealtype_names": payload.get("dealtype_names", []) or [],
            "score": point.score if hasattr(point, 'score') else 0.0,
        })

    return results


# Populated from main.state during startup.
state: dict = {}


def _entity_acquisition_role(text: str, entity: str) -> bool | None:
    """Whether ``entity`` is the acquirer (True) or the target (False) in ``text``, or None when unstated; passive forms are matched first so "X was acquired by Y" is not read as X buying."""
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
    """Re-rank ``results`` so the queried entity's acquisition role matches ``direction``; returns an unchanged copy when ``direction`` is None."""
    if not direction:
        return list(results)
    entities = extract_entities(query)
    if not entities:
        return list(results)

    # PROMOTE the role the query asked for and DEMOTE the opposite; transposing this matrix silently inverts every rerank.
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

