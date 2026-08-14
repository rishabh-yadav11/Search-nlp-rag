import asyncio
import math
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
    SparseVector,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.answer_fallback import date_label, fallback_answer, search_note, weak_results
from app.config import config
from app.health import router as health_router
from app.llm import LLMUnavailableError, generate_answer
from app.query_expand import expand_query
from app.query_intent import extract_list_topic, extract_year_range, rewrite_year_in_review, top_k_hint
from app.redis_cache import cache
from app.rerank_boost import apply_entity_boost

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    state["sparse_model"] = SparseTextEmbedding(config.SPARSE_MODEL)
    state["reranker"] = CrossEncoder(config.RERANK_MODEL, device="cpu")
    state["qdrant"] = AsyncQdrantClient(url=config.QDRANT_URL, timeout=30)
    state["llm"] = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL) if config.GROQ_API_KEY else None
    yield
    await state["qdrant"].close()
    await cache.close()


app = FastAPI(title="VCCircle New Search", lifespan=lifespan)

# POC-only: wide open for local frontend on :3000. Restrict origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health_router)


class SourceArticle(BaseModel):
    id: int
    title: str
    url: str
    published_date: str | None = None
    category: str | None = None
    summary: str = ""
    body: str = ""
    author_names: list[str] = []
    industry_names: list[str] = []
    dealtype_names: list[str] = []
    score: float


class SourceSummary(BaseModel):
    """Public DTO for search/ask results. Exposes a short `summary` excerpt for
    editors; the full article `body` is never included in the response."""

    id: int
    title: str
    url: str
    published_date: str | None = None
    category: str | None = None
    summary: str = ""
    author_names: list[str] = []
    industry_names: list[str] = []
    dealtype_names: list[str] = []
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SourceSummary]
    cached: bool
    latency_ms: float
    note: str | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceSummary]
    cached: bool
    latency_ms: float
    note: str | None = None


def to_summary(a: SourceArticle) -> SourceSummary:
    return SourceSummary(
        id=a.id,
        title=a.title,
        url=a.url,
        published_date=a.published_date,
        category=a.category,
        summary=a.summary,
        score=a.score,
        author_names=a.author_names,
        industry_names=a.industry_names,
        dealtype_names=a.dealtype_names,
    )


def _parse_date(s: str) -> datetime | None:
    """'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', or RFC3339 -> aware datetime (UTC)."""
    if not s:
        return None
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T", 1))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def build_facet_filter(
    industry: str | None,
    dealtype: str | None,
    author: str | None,
    from_date: str | None,
    to_date: str | None,
) -> Filter | None:
    """Qdrant filter for the faceted search params, or None when unfiltered."""
    conditions = []
    for key, raw in (("industry_names", industry), ("dealtype_names", dealtype), ("author_names", author)):
        if raw:
            values = [v.strip() for v in raw.split(",") if v.strip()]
            if values:
                conditions.append(FieldCondition(key=key, match=MatchAny(any=values)))
    if from_date:
        dt = _parse_date(from_date)
        if dt is None:
            raise HTTPException(status_code=400, detail=f"invalid from_date: {from_date!r}")
        conditions.append(FieldCondition(key="published_date", range=DatetimeRange(gte=dt.isoformat())))
    if to_date:
        dt = _parse_date(to_date)
        if dt is None:
            raise HTTPException(status_code=400, detail=f"invalid to_date: {to_date!r}")
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        conditions.append(FieldCondition(key="published_date", range=DatetimeRange(lte=end.isoformat())))
    if not conditions:
        return None
    return Filter(must=conditions)


def facet_cache_token(
    industry: str | None,
    dealtype: str | None,
    author: str | None,
    from_date: str | None,
    to_date: str | None,
) -> str:
    return f"{industry or ''}|{dealtype or ''}|{author or ''}|{from_date or ''}|{to_date or ''}"


def _effective_intent(
    q: str,
    from_date: str | None,
    to_date: str | None,
) -> tuple[str, str | None, str | None]:
    """Rewrite year-in-review queries for retrieval and derive an auto date
    filter from the query's year intent. Explicit user dates always win."""
    retrieval_q, _ = rewrite_year_in_review(q)
    if from_date or to_date:
        return retrieval_q, from_date, to_date
    rng = extract_year_range(q)
    if rng:
        return retrieval_q, rng[0], rng[1]
    return retrieval_q, from_date, to_date


def _merge_results(*groups: list[SourceArticle]) -> list[SourceArticle]:
    """Concatenate and dedupe by id, keeping the highest score for each id.

    Used to combine the raw RRF-candidate sets from the Flashback and bare-topic
    retrieval legs *before* a single cross-encoder rerank against the original
    query, so scores remain comparable across legs."""
    best: dict[int, SourceArticle] = {}
    for group in groups:
        for a in group:
            prev = best.get(a.id)
            if prev is None or a.score > prev.score:
                best[a.id] = a
    return list(best.values())


def _retrieval_queries(q: str) -> list[str]:
    """Queries to run for a user query. For year-in-review intents this is the
    Flashback-rewritten query PLUS the bare topic (year-filtered) so niche
    topics that have no dedicated Flashback article still surface their specific
    articles (e.g. 'venture debt providers', 'unicorns created'). Otherwise a
    single query."""
    flashback, changed = rewrite_year_in_review(q)
    if not changed:
        return [q]
    topic = extract_list_topic(q) or q
    # Dedupe identical entries (e.g. query that's already 'Flashback Y topic').
    return list(dict.fromkeys([flashback, topic]))


async def hybrid_search(
    query: str,
    top_k: int,
    min_dense_score: float | None = None,
    qfilter: Filter | None = None,
) -> list[SourceArticle]:
    # Sentence-transformers and fastembed are CPU/sync-bound: run them off the
    # event loop so the async handlers stay responsive under load.
    dense_vec = (await asyncio.to_thread(partial(state["model"].encode, query, normalize_embeddings=True))).tolist()
    sparse_emb = next(await asyncio.to_thread(state["sparse_model"].embed, [query]))
    sparse_vec = SparseVector(indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist())

    dense_prefetch = Prefetch(
        query=dense_vec,
        using="dense",
        limit=top_k * 4,
        score_threshold=min_dense_score if min_dense_score is not None else None,
    )
    sparse_prefetch = Prefetch(query=sparse_vec, using="sparse", limit=top_k * 4)

    result = await state["qdrant"].query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[dense_prefetch, sparse_prefetch],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
    )

    return [
        SourceArticle(
            id=p.id,
            title=p.payload.get("title", ""),
            url=p.payload.get("url", ""),
            published_date=p.payload.get("published_date"),
            category=p.payload.get("category"),
            summary=p.payload.get("summary", ""),
            body=p.payload.get("body", ""),
            author_names=p.payload.get("author_names") or [],
            industry_names=p.payload.get("industry_names") or [],
            dealtype_names=p.payload.get("dealtype_names") or [],
            score=p.score,
        )
        for p in result.points
    ]


async def rerank(query: str, results: list[SourceArticle]) -> list[SourceArticle]:
    """Cross-encoder rerank of RRF candidates, in place. Rewrites score with a
    sigmoid-normalized relevance score (0-1) so both ordering and the score the
    frontend shows reflect reranked relevance."""
    if len(results) <= 1:
        return results
    pairs = [(query, f"{a.title}. {a.summary or ''}".strip()) for a in results]
    logits = await asyncio.to_thread(state["reranker"].predict, pairs)
    for a, s in zip(results, logits):
        a.score = float(1 / (1 + math.exp(-s)))
    results.sort(key=lambda a: a.score, reverse=True)
    return results


def _recency_multiplier(published_date: str | None) -> float:
    """1 - STRENGTH * (1 - exp(-age_days / DECAY)); no boost for missing dates."""
    dt = _parse_date(published_date)
    if dt is None:
        return 1.0
    age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
    return 1.0 - config.RECENCY_STRENGTH * (1.0 - math.exp(-age_days / config.RECENCY_DECAY_DAYS))


def sort_results(results: list[SourceArticle]) -> list[SourceArticle]:
    """Recency-tempered relevance first, recency second: blended score desc,
    then published_date desc (missing dates last)."""
    results.sort(
        key=lambda a: (a.score * _recency_multiplier(a.published_date), a.published_date or ""),
        reverse=True,
    )
    return results


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(config.TOP_K, ge=1, le=50),
    industry: str | None = Query(None),
    dealtype: str | None = Query(None),
    author: str | None = Query(None),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
):
    start = time.perf_counter()
    retrieval_q, eff_from, eff_to = _effective_intent(q, from_date, to_date)
    if config.ENABLE_QUERY_EXPANSION and "flashback" not in retrieval_q.lower():
        retrieval_q = expand_query(retrieval_q)
    eff_top_k = min(max(top_k, top_k_hint(q) or 0), 50)
    cache_key = f"search:{retrieval_q}:{eff_top_k}:{facet_cache_token(industry, dealtype, author, eff_from, eff_to)}"
    cached_results = await cache.get(cache_key)
    if cached_results is not None:
        summaries = [SourceSummary.model_validate(d) for d in cached_results]
        return SearchResponse(
            query=q,
            results=summaries,
            cached=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            note=search_note([s.score for s in summaries], date_label(eff_from, eff_to)),
        )

    qfilter = build_facet_filter(industry, dealtype, author, eff_from, eff_to)
    groups = []
    for rq in _retrieval_queries(q):
        if config.ENABLE_QUERY_EXPANSION and "flashback" not in rq.lower():
            rq = expand_query(rq)
        candidates = await hybrid_search(rq, max(eff_top_k, config.RERANK_CANDIDATES), qfilter=qfilter)
        groups.append(candidates)
    reranked = await rerank(q, _merge_results(*groups))
    if config.ENABLE_ENTITY_BOOST:
        reranked = apply_entity_boost(q, reranked)
    results = sort_results(reranked)[:eff_top_k]
    await cache.set(cache_key, [to_summary(r).model_dump() for r in results])
    return SearchResponse(
        query=q,
        results=[to_summary(r) for r in results],
        cached=False,
        latency_ms=(time.perf_counter() - start) * 1000,
        note=search_note([r.score for r in results], date_label(eff_from, eff_to)),
    )


ANSWER_PROMPT = """You are answering a question using ONLY the numbered articles below. \
Cite the article number(s) for every factual claim, like [1] or [2][3]. \
If the question asks for a list or ranking (e.g. "top N deals", "top articles", "which companies/funds"), \
extract and enumerate every matching item that appears in the articles, with citations. \
Do not refuse because the list is long; list as many as the articles mention, and say how many \
were found if fewer than the requested number. \
If the articles genuinely contain no information relevant to the question, say so plainly instead of guessing.

Articles:
{context}

Question: {question}

Answer (with inline [n] citations):"""


def source_context(s: SourceArticle, idx: int) -> str:
    meta = s.published_date or "n/a"
    if s.author_names:
        meta += f" | Authors: {', '.join(s.author_names)}"
    if s.industry_names:
        meta += f" | Industry: {', '.join(s.industry_names)}"
    if s.dealtype_names:
        meta += f" | Dealtype: {', '.join(s.dealtype_names)}"
    parts = [f"[{idx}] {s.title} ({meta})"]
    if s.summary:
        parts.append(s.summary)
    if s.body:
        parts.append(s.body[:1500])
    return "\n".join(parts)


@app.get("/ask", response_model=AskResponse)
async def ask(
    q: str = Query(..., min_length=1),
    top_k: int = Query(config.TOP_K, ge=1, le=20),
    industry: str | None = Query(None),
    dealtype: str | None = Query(None),
    author: str | None = Query(None),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
):
    if state["llm"] is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    start = time.perf_counter()
    retrieval_q, eff_from, eff_to = _effective_intent(q, from_date, to_date)
    if config.ENABLE_QUERY_EXPANSION and "flashback" not in retrieval_q.lower():
        retrieval_q = expand_query(retrieval_q)
    eff_top_k = min(max(top_k, top_k_hint(q) or 0), 20)
    cache_key = f"ask:{retrieval_q}:{eff_top_k}:{facet_cache_token(industry, dealtype, author, eff_from, eff_to)}"
    cached = await cache.get(cache_key)
    if cached is not None:
        sources = [SourceSummary.model_validate(s) for s in cached["sources"]]
        return AskResponse(
            query=q,
            answer=cached["answer"],
            sources=sources,
            cached=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            note=cached.get("note"),
        )

    qfilter = build_facet_filter(industry, dealtype, author, eff_from, eff_to)
    groups = []
    for rq in _retrieval_queries(q):
        if config.ENABLE_QUERY_EXPANSION and "flashback" not in rq.lower():
            rq = expand_query(rq)
        candidates = await hybrid_search(rq, max(eff_top_k, config.RERANK_CANDIDATES), qfilter=qfilter)
        groups.append(candidates)
    reranked = await rerank(q, _merge_results(*groups))
    if config.ENABLE_ENTITY_BOOST:
        reranked = apply_entity_boost(q, reranked)
    sources = [s for s in sort_results(reranked) if s.score >= config.ASK_MIN_SCORE][:eff_top_k]

    note = search_note([s.score for s in sources], date_label(eff_from, eff_to)) if config.ENABLE_WEAK_FALLBACK else None

    if not sources:
        answer = "No sufficiently relevant articles were found for this query."
        await cache.set(cache_key, {"answer": answer, "sources": [], "note": note})
        return AskResponse(query=q, answer=answer, sources=[], cached=False,
                            latency_ms=(time.perf_counter() - start) * 1000, note=note)

    if config.ENABLE_WEAK_FALLBACK and weak_results(q, [s.score for s in sources]):
        answer = fallback_answer(q, len(sources), date_label(eff_from, eff_to))
        await cache.set(cache_key, {"answer": answer, "sources": [to_summary(s).model_dump() for s in sources], "note": note})
        return AskResponse(query=q, answer=answer, sources=[to_summary(s) for s in sources], cached=False,
                            latency_ms=(time.perf_counter() - start) * 1000, note=note)

    context = "\n\n".join(source_context(s, i + 1) for i, s in enumerate(sources))
    prompt = ANSWER_PROMPT.format(context=context, question=q)

    try:
        answer = await generate_answer(state["llm"], prompt, config.LLM_MODEL)
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail={"error": "LLM temporarily unavailable", "detail": "The language model could not be reached; please retry shortly."},
        )

    await cache.set(cache_key, {"answer": answer, "sources": [to_summary(s).model_dump() for s in sources], "note": note})
    return AskResponse(query=q, answer=answer, sources=[to_summary(s) for s in sources], cached=False,
                        latency_ms=(time.perf_counter() - start) * 1000, note=note)


FACETS_CACHE_KEY = "facets:v1"
FACETS_LIMIT = 200


@app.get("/facets")
async def facets():
    """Distinct industry_names and dealtype_names values across the collection,
    used for filter autocomplete. Cached in Redis (small controlled vocab)."""
    cached = await cache.get(FACETS_CACHE_KEY)
    if cached is not None:
        return cached

    industries: set[str] = set()
    dealtypes: set[str] = set()
    offset = None
    while True:
        points, offset = await state["qdrant"].scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            for v in pl.get("industry_names") or []:
                if isinstance(v, str) and v.strip():
                    industries.add(v.strip())
            for v in pl.get("dealtype_names") or []:
                if isinstance(v, str) and v.strip():
                    dealtypes.add(v.strip())
        if offset is None or not points:
            break

    result = {
        "industry": sorted(industries)[:FACETS_LIMIT],
        "dealtype": sorted(dealtypes)[:FACETS_LIMIT],
    }
    await cache.set(FACETS_CACHE_KEY, result)
    return result
