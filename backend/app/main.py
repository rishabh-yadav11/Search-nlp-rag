import asyncio
import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
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
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Import config FIRST so its OMP/MKL thread caps land before torch/onnxruntime import.
from app import auth as auth_module
from app import chat as chat_module
from app.analytics import AnalyticsUnavailableError, record_click, record_search
from app.analytics import close as close_analytics
from app.analytics import summary as analytics_data
from app.answer_fallback import date_label, weak_results_note
from app.auth import _client_ip, public_rate_limit, require_auth, require_permission, user_rate_limit
from app.chat import ChatAnalyticsUnavailableError
from app.click_boost import apply_click_boost
from app.close_guard import DEFAULT_CLOSE_TIMEOUT, close_quietly
from app.config import config, ensure_data_paths_ready
from app.cost_budget import close as close_cost_budget
from app.diversity import diversify
from app.encoders import DenseEncoder
from app.health import close_redis as health_module_close_redis
from app.health import router as health_router
from app.health import warn_if_llm_key_unusable
from app.input_hygiene import build_cache_key, normalize_text, split_facet_values
from app.logging_config import configure_logging
from app.observability import (
    RequestIdMiddleware,
    attach_request_id_filter,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.query_expand import expand_query
from app.query_fix import fix_query, init_fixer
from app.query_intent import (
    acquisition_relation,
    extract_list_topic,
    extract_recency_range,
    extract_year_range,
    is_recency_intent,
    normalize_word_numbers,
    range_query_topic,
    rewrite_year_in_review,
    strip_recency_intent,
    strip_recency_window,
    suggested_top_k,
)
from app.query_intent import (
    extract_content_type as _classify_content_type,
)
from app.recommender import (
    SIMILAR_ARTICLES_TTL_SECONDS,
    USER_RECOMMENDATIONS_TTL_SECONDS,
    get_personalized_recommendations,
    get_similar_articles,
    get_trending_feed,
    rerank_acquisition_relation,
)
from app.redis_cache import cache
from app.rerank_boost import apply_entity_boost
from app.reranker import Reranker
from app.user_profile import (
    InteractionResult,
    InteractionType,
    get_user_interactions,
    invalidate_user_profile,
    record_interaction,
)

# Uvicorn's worker leaves the root logger at WARNING with no handlers, so every module logger
# drops its INFO records until this runs at import.
configure_logging()

state = {}

# Per worker, so concurrent requests don't thrash the torch/onnxruntime thread pools.
inference_lock = asyncio.Lock()

logger = logging.getLogger(__name__)

# Live facet vocabularies from Qdrant (normalized value -> original-cased value), published by
# rebinding the global only once a full replacement has loaded, so an extractor never reads a
# map being mutated in place. Extraction only emits values present here: an unknown filter
# value matches nothing and breaks the query.
_DEALTYPE_FACETS: dict[str, str] = {}
_INDUSTRY_FACETS: dict[str, str] = {}
_CONTENT_TYPE_FACETS: dict[str, str] = {}

# Natural-language synonyms -> facet keyword, resolved against the live vocabulary.
_DEALTYPE_ALIASES: dict[str, str] = {
    "funding": "venture capital",
    "fundraise": "venture capital",
    "fund raise": "venture capital",
    "seed": "venture capital",
    "raised": "venture capital",
    "raising": "venture capital",
    "capital": "venture capital",
    "venture": "venture capital",
    "vc": "venture capital",
    "startup funding": "venture capital",
    "private equity": "private equity",
    "pe": "private equity",
    "m&a": "m&a",
    "merger": "m&a",
    "mergers": "m&a",
    "acquisition": "m&a",
    "acquisitions": "m&a",
    "acquire": "m&a",
    "acquired": "m&a",
    "buyout": "m&a",
    "takeover": "m&a",
    "credit": "credit",
    "investment banking": "investment banking",
    "markets": "markets",
}

_INDUSTRY_ALIASES: dict[str, str] = {
    "fintech": "finance",
    "financial technology": "finance",
    "healthtech": "healthcare",
    "edtech": "education",
    "ecommerce": "retail",
    "e-commerce": "retail",
    "e commerce": "retail",
    "saas": "technology",
    "software": "technology",
    "cleantech": "cleantech",
    "media": "media & entertainment",
    "telecom": "telecom",
    "real estate": "real estate",
    "manufacturing": "manufacturing",
    "consumer": "consumer",
    "retail": "retail",
    "technology": "technology",
    "healthcare": "healthcare",
    "education": "education",
    "finance": "finance",
}

def _resolve_facet(query: str, aliases: dict[str, str], facets: dict[str, str]) -> str | None:
    """Return the real facet value reachable via a whole-word alias, else None.

    Only curated aliases are matched, never raw facet labels, so a common-word
    label like 'People' cannot be triggered by ordinary text.
    """
    q = query.lower()
    # Longest alias first, so 'capital' cannot shadow 'venture capital'.
    for alias, kw in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            if kw in facets:
                return facets[kw]
            # Tightest (shortest) facet wins, so the choice is not dict-order bound.
            candidates = [orig for norm, orig in facets.items() if kw in norm]
            if candidates:
                return min(candidates, key=lambda o: (len(o), o.lower()))
    return None


def extract_dealtype(query: str) -> str | None:
    return _resolve_facet(query, _DEALTYPE_ALIASES, _DEALTYPE_FACETS)


def extract_industry(query: str) -> str | None:
    return _resolve_facet(query, _INDUSTRY_ALIASES, _INDUSTRY_FACETS)


def extract_content_type(query: str) -> str | None:
    kw = _classify_content_type(query)
    if kw is None:
        return None
    facets = _CONTENT_TYPE_FACETS
    if kw in facets:
        return facets[kw]
    # Tightest (shortest) facet wins, so the choice is not dict-order bound.
    candidates = [orig for norm, orig in facets.items() if kw in norm]
    if candidates:
        return min(candidates, key=lambda o: (len(o), o.lower()))
    return None


async def _load_facet_maps() -> None:
    """Load and atomically publish live facet maps; a failed load keeps the prior vocabulary."""
    global _DEALTYPE_FACETS, _INDUSTRY_FACETS, _CONTENT_TYPE_FACETS

    for name, key in (
        ("_DEALTYPE_FACETS", "dealtype_names"),
        ("_INDUSTRY_FACETS", "industry_names"),
        ("_CONTENT_TYPE_FACETS", "content_type"),
    ):
        try:
            values = await _facet_values(key)
            facets = {value.strip().lower(): value for value in values}
        except Exception as exc:
            # A missing map and an unreachable Qdrant both leave it empty; only the log tells them apart.
            logger.warning(
                "facet map load failed for %s (%s: %s); continuing without it",
                key,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            continue
        if name == "_DEALTYPE_FACETS":
            _DEALTYPE_FACETS = facets
        elif name == "_INDUSTRY_FACETS":
            _INDUSTRY_FACETS = facets
        else:
            _CONTENT_TYPE_FACETS = facets


# Per-step teardown budget: all ten timing out costs 20s of the 30s graceful_timeout.
_TEARDOWN_CLOSE_TIMEOUT = DEFAULT_CLOSE_TIMEOUT


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before any store opens: a bad data path yields an empty SQLite file and silent 404s.
    ensure_data_paths_ready(config)
    logger.info(
        "data locations: CHAT_DB_PATH=%s AUTH_DB_PATH=%s QUERY_FIX_VOCAB_PATH=%s",
        config.CHAT_DB_PATH,
        config.AUTH_DB_PATH,
        config.QUERY_FIX_VOCAB_PATH,
    )

    state["model"] = DenseEncoder(config.EMBED_MODEL, config.EMBED_DEVICE, config.TORCH_THREADS)
    state["sparse_model"] = SparseTextEmbedding(config.SPARSE_MODEL)
    state["reranker"] = Reranker(config.RERANK_MODEL, backend=config.RERANK_BACKEND)
    state["qdrant"] = AsyncQdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=30)
    await _load_facet_maps()
    # Names the real cause (missing/placeholder GEMINI_API_KEY) without killing boot.
    warn_if_llm_key_unusable()
    state["llm"] = AsyncOpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL) if config.GEMINI_API_KEY else None

    chat_store = chat_module.ChatStore(config.CHAT_DB_PATH)
    await chat_store.connect()
    chat_module.store = chat_store
    state["chat_retention"] = asyncio.create_task(chat_module.retention_loop())

    auth_store = auth_module.AuthStore(config.AUTH_DB_PATH)
    await auth_store.connect()
    auth_module.store = auth_store
    await auth_module.bootstrap_admin()
    await auth_module.report_legacy_password_hashes()
    state["auth_token_purge"] = asyncio.create_task(auth_module.token_purge_loop())

    init_fixer(
        config.ENABLE_QUERY_FIX,
        config.QUERY_FIX_VOCAB_PATH,
        max_edit=config.QUERY_FIX_MAX_EDIT,
        min_count=config.QUERY_FIX_MIN_COUNT,
        min_token_len=config.QUERY_FIX_MIN_TOKEN_LEN,
    )

    from app import recommender
    recommender.state = state

    yield

    # Each step goes through close_guard, so one raise or hang cannot strand the rest.
    teardown_steps = (
        ("chat retention task", lambda: _cancel_and_wait("chat retention task", state["chat_retention"])),
        ("chat store", chat_store.close),
        ("auth token purge task", lambda: _cancel_and_wait("auth token purge task", state["auth_token_purge"])),
        ("auth store", auth_store.close),
        ("qdrant client", state["qdrant"].close),
        ("cache client", lambda: cache.close()),
        ("analytics redis", close_analytics),
        ("cost budget redis", close_cost_budget),
        ("auth rate-limit redis", auth_module.close_rate_redis),
        ("readiness redis", health_module_close_redis),
    )
    for resource_name, close_step in teardown_steps:
        await close_quietly(
            resource_name,
            close_step,
            timeout=_TEARDOWN_CLOSE_TIMEOUT,
            suppress=Exception,
            log=logger,
        )


async def _cancel_and_wait(resource_name: str, task: asyncio.Task) -> None:
    """Cancel a background loop and wait for it to unwind.

    ``asyncio.wait`` rather than ``wait_for``: a task that swallows its
    ``CancelledError`` never completes, so ``wait_for`` blocks forever instead of
    timing out. The inner budget must stay strictly under the caller's
    _TEARDOWN_CLOSE_TIMEOUT -- equal, and the outer wait_for always wins and
    misreports the step.
    """
    task.cancel()
    done, _pending = await asyncio.wait({task}, timeout=_TEARDOWN_CLOSE_TIMEOUT / 2)
    if not done:
        logger.warning("Background task for %s ignored cancellation; abandoning it", resource_name)
    elif not task.cancelled() and (exc := task.exception()) is not None:
        # asyncio.wait does not re-raise, so this is the only trace of a loop that died.
        logger.warning("Background task for %s failed during shutdown", resource_name, exc_info=exc)


app = FastAPI(
    title="VCCircle New Search",
    lifespan=lifespan,
    # Off in every environment: these three routes are unauthenticated reconnaissance of the
    # route list, its models, and which routes sit behind which dependency.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    # Required now that the credential is a cookie; safe only because allow_origins is an
    # explicit list, never "*".
    allow_credentials=True,
)
app.add_middleware(
    # The allow-list comes from config and is validated there, never empty or "*".
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS,
)

# Outermost (add_middleware inserts at 0) so even a rejected request is correlated.
app.add_middleware(RequestIdMiddleware)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
# Must stay after configure_logging(): installed_handler() is None until then.
attach_request_id_filter()

logger.warning("TrustedHost allowed hosts: %s", ", ".join(config.ALLOWED_HOSTS))

app.include_router(health_router)
app.include_router(auth_module.router)
app.include_router(chat_module.router)


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
    tag_names: list[str] = []
    content_type: str | None = None
    score: float


class SourceSummary(BaseModel):
    """Public result DTO: carries `summary`, never the full article `body`."""

    id: int
    title: str
    url: str
    published_date: str | None = None
    category: str | None = None
    summary: str = ""
    author_names: list[str] = []
    industry_names: list[str] = []
    dealtype_names: list[str] = []
    tag_names: list[str] = []
    content_type: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SourceSummary]
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
        tag_names=a.tag_names,
        content_type=a.content_type,
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
    content_type: str | None = None,
    tag: str | None = None,
) -> Filter | None:
    """Qdrant filter for the faceted params, or None when unfiltered.

    Every facet path funnels through here, so the value caps cannot be sidestepped
    by arriving through a different caller.
    """
    conditions = []
    for key, field, raw in (
        ("industry_names", "industry", industry),
        ("dealtype_names", "dealtype", dealtype),
        ("author_names", "author", author),
        ("content_type", "content_type", content_type),
        ("tag_names", "tag", tag),
    ):
        values = split_facet_values(field, raw)
        if values:
            conditions.append(FieldCondition(key=key, match=MatchAny(any=values)))
    if from_date:
        dt = _parse_date(from_date)
        if dt is None:
            # Static message: echoing caller input would make this 400 a reflection point.
            raise HTTPException(status_code=400, detail="invalid from_date")
        conditions.append(FieldCondition(key="published_date", range=DatetimeRange(gte=dt.isoformat())))
    if to_date:
        dt = _parse_date(to_date)
        if dt is None:
            raise HTTPException(status_code=400, detail="invalid to_date")
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
    content_type: str | None = None,
    tag: str | None = None,
) -> str:
    """Cache-key fragment for the faceted params.

    Length-prefixed, not delimiter-joined: a pipe inside a value used to let two
    different filters produce one token and serve each other's cached results.
    """
    return build_cache_key(
        *(normalize_text(value or "") for value in
          (industry, dealtype, author, from_date, to_date, content_type, tag))
    )


def _effective_intent(
    q: str,
    from_date: str | None,
    to_date: str | None,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Rewrite the query for retrieval, derive an auto date filter from its date
    intent, and extract implied category facets. Explicit user dates always win.

    Returns (retrieval_q, eff_from, eff_to, dealtype, industry).
    """
    q = normalize_word_numbers(q)
    retrieval_q, _ = rewrite_year_in_review(q)
    dealtype = extract_dealtype(q)
    industry = extract_industry(q)
    if from_date or to_date:
        return retrieval_q, from_date, to_date, dealtype, industry
    rng = extract_year_range(q)
    if rng:
        cleaned = range_query_topic(q)
        if cleaned:
            retrieval_q = cleaned
        return retrieval_q, rng[0], rng[1], dealtype, industry
    # Window and soft recency words are stripped: the date filter already scopes them.
    rng = extract_recency_range(q)
    if rng:
        cleaned = strip_recency_intent(strip_recency_window(q))
        if cleaned:
            retrieval_q = cleaned
        return retrieval_q, rng[0], rng[1], dealtype, industry
    if is_recency_intent(q):
        cleaned = strip_recency_intent(retrieval_q)
        if cleaned:
            retrieval_q = cleaned
    return retrieval_q, from_date, to_date, dealtype, industry


def _merge_results(*groups: list[SourceArticle]) -> list[SourceArticle]:
    """Concatenate and dedupe by id, keeping the highest score.

    Legs merge before a single rerank so their scores stay comparable. A body-less
    entry never overrides a body-bearing one for the same id.
    """
    best: dict[int, SourceArticle] = {}
    for group in groups:
        for a in group:
            prev = best.get(a.id)
            if prev is None:
                best[a.id] = a
                continue
            prev_has_body = bool(prev.body)
            a_has_body = bool(a.body)
            if a_has_body and not prev_has_body:
                best[a.id] = a
            elif prev_has_body and not a_has_body:
                continue
            elif a.score > prev.score:
                best[a.id] = a
    return list(best.values())


# The 'Flashback <year>' prefix _effective_intent emits for a year-in-review intent.
_FLASHBACK_PREFIX_RE = re.compile(r"^\s*flashback\s+(?:19|20)\d{2}\b\s*", re.IGNORECASE)


def _retrieval_queries(q: str) -> list[str]:
    """Retrieval legs for a query: the Flashback rewrite plus the bare topic for
    year-in-review intents, the bare topic for month-scoped ones, else the query."""
    flashback, changed = rewrite_year_in_review(q)
    if changed:
        topic = extract_list_topic(q) or q
        return list(dict.fromkeys([flashback, topic]))
    # Called with the query _effective_intent already rewrote, so the 'top N ... in
    # <year>' cue is gone; re-detect the emitted prefix to re-emit the topic leg.
    m = _FLASHBACK_PREFIX_RE.match(q)
    if m:
        topic = q[m.end():].strip()
        if topic:
            return list(dict.fromkeys([q, topic]))
    scoped = range_query_topic(q)
    if scoped:
        return [scoped]
    return [q]


# `body` is excluded: it is large and only chat needs it, fetched via _attach_bodies.
_PAYLOAD_FIELDS = [
    "title",
    "url",
    "published_date",
    "category",
    "summary",
    "author_names",
    "industry_names",
    "dealtype_names",
    "tag_names",
    "content_type",
]


def _embed_sparse(model, text: str):
    """Sparse-embed one query, consuming fastembed's lazy generator off the loop."""
    return next(iter(model.embed([text])))


async def hybrid_search(
    query: str,
    top_k: int,
    qfilter: Filter | None = None,
    with_body: bool = False,
) -> list[SourceArticle]:
    # CPU/sync-bound: off the event loop, serialized so concurrent requests don't thrash the
    # inference thread pools. The (dense, sparse) pair is deterministic and independent of
    # the qfilter, so it is cached per embedding model. Clamp before the key is built so key
    # and embedded text describe one string; expand_query only grows, so an in-limit caller
    # can arrive over the limit again.
    query = query[: config.RETRIEVAL_QUERY_MAX_CHARS]
    # Normalised here so the key and the embedded text cannot drift apart. Only shrinks the
    # text, so it stays after the clamp.
    query = normalize_text(query)
    # Same two bounds as the search:/retrieve: keys: the query segment is digested past
    # CACHE_KEY_QUERY_MAX_CHARS and the whole key is length-prefixed and bounded.
    vec_key = build_cache_key(config.EMBED_MODEL, config.SPARSE_MODEL,
                              _cache_key_component(query), namespace="vec")
    vec = await cache.get(vec_key)
    if vec is None:
        async with inference_lock:
            dense_vec = (await asyncio.to_thread(state["model"].encode, query)).tolist()
            sparse_emb = await asyncio.to_thread(_embed_sparse, state["sparse_model"], query)
        vec = {
            "dense": dense_vec,
            "si": sparse_emb.indices.tolist(),
            "sv": sparse_emb.values.tolist(),
        }
        await cache.set(vec_key, vec, ttl=config.VECTOR_CACHE_TTL_SECONDS)
    sparse_vec = SparseVector(indices=vec["si"], values=vec["sv"])

    dense_prefetch = Prefetch(
        query=vec["dense"],
        using="dense",
        limit=top_k * 4,
    )
    sparse_prefetch = Prefetch(query=sparse_vec, using="sparse", limit=top_k * 4)

    result = await state["qdrant"].query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[dense_prefetch, sparse_prefetch],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=qfilter,
        limit=top_k,
        with_payload=True if with_body else _PAYLOAD_FIELDS,
    )

    return [
        SourceArticle(
            id=p.id,
            title=payload.get("title", ""),
            url=payload.get("url", ""),
            published_date=payload.get("published_date"),
            category=payload.get("category"),
            summary=payload.get("summary", ""),
            body=payload.get("body", ""),
            author_names=payload.get("author_names") or [],
            industry_names=payload.get("industry_names") or [],
            dealtype_names=payload.get("dealtype_names") or [],
            tag_names=payload.get("tag_names") or [],
            content_type=payload.get("content_type") or None,
            score=p.score,
        )
        for p in result.points
        # A malformed point with no payload would displace a real result.
        if (payload := p.payload)
    ]


async def rerank(query: str, results: list[SourceArticle]) -> list[SourceArticle]:
    """Cross-encoder rerank of RRF candidates, in place, rescaling score to a
    sigmoid-normalized 0-1 relevance."""
    if len(results) <= 1:
        return results
    # The cross-encoder tokenizes both sides, so the query needs the same bound
    # the dense encoder gets.
    query = query[: config.RETRIEVAL_QUERY_MAX_CHARS]
    pairs = [(query, f"{a.title}. {a.summary or ''}".strip()) for a in results]
    async with inference_lock:
        logits = await asyncio.to_thread(state["reranker"].predict, pairs)
    for a, s in zip(results, logits):
        a.score = float(1 / (1 + math.exp(-s)))
    results.sort(key=lambda a: a.score, reverse=True)
    return results


_STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "could", "did", "do", "does", "for", "from", "had",
        "has", "have", "how", "in", "into", "is", "it", "its", "may",
        "might", "must", "no", "not", "of", "on", "or", "over", "said",
        "say", "says", "should", "so", "than", "that", "the", "their",
        "them", "then", "there", "these", "they", "this", "those", "to",
        "under", "was", "we", "were", "what", "when", "where", "which",
        "who", "whose", "why", "will", "with", "would", "you", "your",
    )
)


def _query_content_tokens(query: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS and len(w) > 1}


def _effective_step(positions: int, step: int, max_windows: int | None) -> int:
    """The stride to scan ``positions`` window starts with.

    Clamping the stride alone does not bound the work (step=1 over a 50K body
    scores 48,501 windows), so a configured ``max_windows`` budget widens it
    instead. ``None`` or <= 0 means no budget. A budget of 1 degenerates to the
    single window at start=0.
    """
    step = max(1, step)
    if max_windows and max_windows > 0 and -(-positions // step) > max_windows:
        step = max(1, -(-positions // max_windows))
    return step


def _best_body_window(
    body: str, tokens: set[str], win: int, step: int, max_windows: int | None = None
) -> str:
    """The body region with the most distinct query tokens, in original casing.

    ``max_windows`` bounds the work by window COUNT rather than by the stride;
    see ``_effective_step``.
    """
    if not tokens or len(body) <= win:
        return body
    low = body.lower()
    positions = len(body) - win + 1
    step = _effective_step(positions, step, max_windows)


    best_score, best_start = -1, 0
    for start in range(0, positions, step):
        score = sum(1 for t in tokens if t in low[start:start + win])
        if score > best_score:
            best_score, best_start = score, start
    tail = low[-win:]
    if sum(1 for t in tokens if t in tail) > best_score:
        return body[-win:]
    return body[best_start:best_start + win]


async def body_rescue(query: str, articles: list[SourceArticle]) -> list[SourceArticle]:
    """Chat-only rerank rescue for deep-body matches, in place.

    When the top score is weak the title+summary cross-encoder is unreliable, so
    each candidate is re-scored against its best body window and kept as
    max(baseline, body).

    Gated on ENABLE_BODY_RESCUE here rather than only at the call sites: this is
    where the cost is paid, so this is where the guard belongs.
    """
    if not articles:
        return articles
    if not config.ENABLE_BODY_RESCUE:
        return articles
    if max((a.score for a in articles), default=0.0) >= config.BODY_RESCUE_THRESHOLD:
        return articles
    # Clamp here too: the one in rerank() is a local and cannot reach this call site.
    query = query[: config.RETRIEVAL_QUERY_MAX_CHARS]
    tokens = _query_content_tokens(query)
    if not tokens:
        return articles
    candidates: list[tuple[int, int, tuple[str, str]]] = []
    for i, a in enumerate(articles):
        if not a.body:
            continue
        win = _best_body_window(
            a.body, tokens,
            config.BODY_RESCUE_WINDOW, config.BODY_RESCUE_STEP,
            config.BODY_RESCUE_MAX_WINDOWS,
        )
        # Distinct query tokens in the best window: what the second pass gets to work with.
        low_win = win.lower()
        overlap = sum(1 for t in tokens if t in low_win)
        candidates.append((overlap, i, (query, f"{a.title}. {a.summary or ''}. {win}".strip())))
    if not candidates:
        return articles
    # Shortlist by body-window overlap, NOT by score or list order: a weak title+summary score
    # is the reason the rescue exists, so ranking by it would drop the matches it surfaces.
    candidates.sort(key=lambda c: (-c[0], c[1]))
    kept = candidates[: config.BODY_RESCUE_MAX_CANDIDATES]
    pairs = [c[2] for c in kept]
    indices = [c[1] for c in kept]
    async with inference_lock:
        logits = await asyncio.to_thread(state["reranker"].predict, pairs)
    for i, logit in zip(indices, logits):
        articles[i].score = max(articles[i].score, float(1 / (1 + math.exp(-logit))))
    articles.sort(key=lambda a: a.score, reverse=True)
    return articles


def _recency_multiplier(
    published_date: str | None,
    recency_strength: float = config.RECENCY_STRENGTH,
    recency_decay_days: float = config.RECENCY_DECAY_DAYS,
) -> float:
    """1 - STRENGTH * (1 - exp(-age_days / DECAY)); no boost for missing dates."""
    dt = _parse_date(published_date)
    if dt is None:
        return 1.0
    age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
    return 1.0 - recency_strength * (1.0 - math.exp(-age_days / recency_decay_days))


def _tz_stripped_pub(published_date: str | None) -> str:
    """Strip a trailing tz offset so string comparison orders by wall clock.

    Records indexed before the UTC-shift fix carry a '+00:00' suffix and new ones
    do not; only a TRAILING offset is removed, so an embedded '+' is left intact.
    """
    if not published_date:
        return ""
    return _TZ_RE.sub("", published_date)


# Trailing ISO-8601 offset ('Z' or +/-HH:MM / +/-HHMM), stripped by _tz_stripped_pub.
_TZ_RE = re.compile(r"(?:[zZ]|[+-]\d{2}:?\d{2})$")


def sort_results(
    results: list[SourceArticle],
    recency_boost: bool = False,
) -> list[SourceArticle]:
    """Recency-tempered relevance first, published_date second (missing dates last).

    ``recency_boost`` weights freshness far more heavily for a recency-intent query."""
    strength = config.RECENCY_BOOST_STRENGTH if recency_boost else config.RECENCY_STRENGTH
    decay = config.RECENCY_BOOST_DECAY_DAYS if recency_boost else config.RECENCY_DECAY_DAYS
    results.sort(
        key=lambda a: (
            a.score * _recency_multiplier(a.published_date, strength, decay),
            _tz_stripped_pub(a.published_date),
        ),
        reverse=True,
    )
    return results


def _filter_token(qfilter: Filter | None) -> str:
    if qfilter is None:
        return ""
    return json.dumps(qfilter.model_dump(), sort_keys=True, default=str)


def _cache_key_component(value: str) -> str:
    """Bounded, deterministic cache-key fragment for a request-supplied string.

    Past the bound it becomes an ``h:``-prefixed sha256, not a truncation:
    truncating would collide distinct long values onto one key and serve one
    query's or one filter's results for another's. Short values keep their exact
    previous key, so existing entries still hit.
    """
    if len(value) <= config.CACHE_KEY_QUERY_MAX_CHARS:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"h:{digest}"


# Every input that can change what a cached result contains; missing one is a stale
# answer for a whole TTL. A test keeps this list in step with the knobs it exercises.
_RETRIEVAL_CONFIG_INPUTS: tuple[tuple[str, str], ...] = (
    ("qdrant_url", "QDRANT_URL"),
    ("qdrant_collection", "QDRANT_COLLECTION"),
    ("embed_model", "EMBED_MODEL"),
    ("sparse_model", "SPARSE_MODEL"),
    # A different device shifts the dense floats enough to reorder the RRF fusion.
    ("embed_device", "EMBED_DEVICE"),
    # Only "torch" ships today, kept so a second backend invalidates; over-invalidating is cheap.
    ("rerank_backend", "RERANK_BACKEND"),
    ("rerank_model", "RERANK_MODEL"),
    ("rerank_candidates", "RERANK_CANDIDATES"),
    ("recency_strength", "RECENCY_STRENGTH"),
    ("recency_decay_days", "RECENCY_DECAY_DAYS"),
    # The recency-intent branch of the blend; retuning either invalidates both entries.
    ("recency_boost_strength", "RECENCY_BOOST_STRENGTH"),
    ("recency_boost_decay_days", "RECENCY_BOOST_DECAY_DAYS"),
    ("enable_query_expansion", "ENABLE_QUERY_EXPANSION"),
    ("enable_entity_boost", "ENABLE_ENTITY_BOOST"),
    ("enable_click_boost", "ENABLE_CLICK_BOOST"),
    ("click_boost_min_clicks", "CLICK_BOOST_MIN_CLICKS"),
    ("click_boost_min_article_clicks", "CLICK_BOOST_MIN_ARTICLE_CLICKS"),
    ("click_boost_min_share", "CLICK_BOOST_MIN_SHARE"),
    ("click_boost_mult", "CLICK_BOOST_MULT"),
    ("click_query_max_len", "CLICK_QUERY_MAX_LEN"),
    # Decides which recorded clicks a query can be boosted by at all.
    ("redis_url", "REDIS_URL"),
    ("analytics_redis_db", "ANALYTICS_REDIS_DB"),
    ("enable_diversity", "ENABLE_DIVERSITY"),
    ("diversity_lambda", "DIVERSITY_LAMBDA"),
    ("diversity_sim_threshold", "DIVERSITY_SIM_THRESHOLD"),
    ("ask_min_score", "ASK_MIN_SCORE"),
)


def retrieval_config_fingerprint() -> str:
    """Short, stable digest of every config value that can change what a cached
    result *contains*, so a redeploy that flips a toggle invalidates its own entries.
    """
    values = {name: getattr(config, attr) for name, attr in _RETRIEVAL_CONFIG_INPUTS}
    canonical = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def retrieve_cache_key(q: str, top_k: int, qfilter: Filter | None) -> str:
    """Cache key for a ``retrieve_and_rerank`` result set.

    The single place the ``retrieve:`` key is composed. Two bounds compose here and
    neither replaces the other: ``_cache_key_component`` bounds the query segment
    by digest (so two over-long queries still key apart) and ``build_cache_key``
    length-prefixes the parts so the whole key is injective and bounded. The
    filter JSON is long and can contain the old ``:`` join's delimiters, which is
    why both are needed.
    """
    return build_cache_key(
        _cache_key_component(q), top_k,
        _filter_token(qfilter), retrieval_config_fingerprint(),
        namespace="retrieve",
    )


def search_cache_key(retrieval_q: str, eff_top_k: int, facets: str) -> str:
    """Cache key for a /search summary page.

    Same two composing bounds as :func:`retrieve_cache_key`; ``facets`` is an
    already length-prefixed token, passed through as one opaque part.
    """
    return build_cache_key(
        _cache_key_component(retrieval_q), eff_top_k, facets,
        retrieval_config_fingerprint(),
        namespace="search",
    )


# Sentinel so a prefetched miss (None) is distinguishable from "no prefetch happened".
_NO_PREFETCH = object()


async def _attach_bodies(articles: list[SourceArticle]) -> None:
    """Fetch the article `body` payloads for a set of articles in one Qdrant call."""
    ids = [a.id for a in articles]
    if not ids:
        return
    resp = await state["qdrant"].retrieve(
        collection_name=config.QDRANT_COLLECTION,
        ids=ids,
        with_payload=["body"],
    )
    # Qdrant may key points by int or str; both sides are str so a str id keeps its body.
    bodies = {str(p.id): (p.payload or {}).get("body", "") for p in resp}
    for a in articles:
        a.body = bodies.get(str(a.id), "")


async def _retrieval_leg(
    rq: str,
    top_k: int,
    qfilter: Filter | None,
) -> list[SourceArticle]:
    if config.ENABLE_QUERY_EXPANSION and "flashback" not in rq.lower():
        rq = expand_query(rq)
    return await hybrid_search(rq, max(top_k, config.RERANK_CANDIDATES), qfilter=qfilter)


async def retrieve_and_rerank(
    q: str,
    top_k: int,
    qfilter: Filter | None,
    need_body: bool = False,
    prefetched: object = _NO_PREFETCH,
) -> list[SourceArticle]:
    """Run every retrieval leg, merge RRF candidates, rerank, apply the entity boost.

    A non-empty reranked set is cached in Redis; empty sets never are. Bodies are
    not cached, so ``need_body`` re-fetches them from Qdrant.

    ``prefetched`` is a value the caller already read for this exact key (``None``
    meaning "already looked up, and it was a miss"), saving one Redis round trip.
    """
    # One normalised spelling drives the key, the legs and the boosts, so equivalent
    # spellings share a cache entry and no raw control character reaches the key.
    q = normalize_text(fix_query(q)[0])
    # Recency intent weights freshness; a rolling window is already scoped by the date filter.
    recency_boost = is_recency_intent(q)
    # Two composing bounds: _cache_key_component bounds the query, build_cache_key the key.
    cache_key = retrieve_cache_key(q, top_k, qfilter)
    cached = await cache.get(cache_key) if prefetched is _NO_PREFETCH else prefetched
    if cached is not None:
        articles = [SourceArticle.model_validate(d) for d in cached]
        if need_body and articles:
            await _attach_bodies(articles)
        return articles

    queries = _retrieval_queries(q)
    groups = await asyncio.gather(*(_retrieval_leg(rq, top_k, qfilter) for rq in queries))
    reranked = await rerank(range_query_topic(q) or q, _merge_results(*groups))
    if config.ENABLE_ENTITY_BOOST:
        reranked = apply_entity_boost(range_query_topic(q) or q, reranked)
    direction = acquisition_relation(q)
    if direction:
        reranked = rerank_acquisition_relation(q, reranked, direction)
    reranked = sort_results(reranked, recency_boost)
    if need_body:
        await _attach_bodies(reranked)
    # Bodies are excluded: large, and chat re-fetches them on a hit (_attach_bodies).
    # Empty sets are never cached: replaying a transient failure as "no results" for the
    # whole TTL is the bug where a date-filtered query returned nothing.
    if reranked:
        await cache.set(cache_key, [a.model_dump(exclude={"body"}) for a in reranked])
    return reranked


async def retrieve_with_auto_facet_fallback(
    retrieval_q: str,
    top_k: int,
    *,
    industry: str | None,
    dealtype: str | None,
    author: str | None,
    content_type: str | None = None,
    tag: str | None = None,
    eff_from: str | None,
    eff_to: str | None,
    auto_industry: str | None,
    auto_dealtype: str | None,
    auto_content_type: str | None = None,
    need_body: bool = False,
    prefetched: object = _NO_PREFETCH,
) -> tuple[list[SourceArticle], str | None, str | None, str | None]:
    """Retrieve with the effective (explicit-or-auto) facets, falling back to dropping
    an *auto* facet that zeroes out an otherwise valid query.

    Returns ``(results, final_industry, final_dealtype, final_content_type)``
    reflecting any fallback, so callers cache and note what was actually retrieved.

    Only an empty (or lone below-gate) set triggers the retry and only auto facets are
    dropped: an explicit facet and the date window always stay, so a genuinely empty
    corpus still reports "no results". All auto facets drop together -- each is a
    relaxation of the same semantic guess. ``tag`` has no auto counterpart, so it is
    never relaxed.
    """

    eff_industry = industry or auto_industry
    eff_dealtype = dealtype or auto_dealtype
    eff_content_type = content_type or auto_content_type
    qfilter = build_facet_filter(
        eff_industry, eff_dealtype, author, eff_from, eff_to, eff_content_type, tag
    )
    # The prefetch is for THIS filter only; the relaxed retry keys on a different one.
    if prefetched is _NO_PREFETCH:
        results = await retrieve_and_rerank(retrieval_q, top_k, qfilter, need_body=need_body)
    else:
        results = await retrieve_and_rerank(
            retrieval_q, top_k, qfilter, need_body=need_body, prefetched=prefetched
        )
    # One below-gate hit left by a mis-applied auto facet is a dead set too, so relax for it.
    lone_weak_hit = len(results) == 1 and results[0].score < config.ASK_MIN_SCORE
    if (results and not lone_weak_hit) or not (auto_industry or auto_dealtype or auto_content_type):
        results = await _temporal_date_fallback(
            results, top_k, eff_from, eff_to, eff_industry, eff_dealtype, author, tag, need_body
        )
        return results, eff_industry, eff_dealtype, eff_content_type
    # An auto facet zeroed the set: drop the auto ones (never an explicit one) and retry once.
    relaxed_industry = eff_industry if not (auto_industry and industry is None) else None
    relaxed_dealtype = eff_dealtype if not (auto_dealtype and dealtype is None) else None
    relaxed_content_type = eff_content_type if not (auto_content_type and content_type is None) else None
    relaxed = build_facet_filter(
        relaxed_industry, relaxed_dealtype, author, eff_from, eff_to, relaxed_content_type, tag
    )
    if relaxed == qfilter:
        results = await _temporal_date_fallback(
            results, top_k, eff_from, eff_to, eff_industry, eff_dealtype, author, tag, need_body
        )
        return results, eff_industry, eff_dealtype, eff_content_type
    relaxed_results = await retrieve_and_rerank(retrieval_q, top_k, relaxed, need_body=need_body)
    relaxed_results = await _temporal_date_fallback(
        relaxed_results, top_k, eff_from, eff_to, relaxed_industry, relaxed_dealtype, author, tag,
        need_body,
    )
    return relaxed_results, relaxed_industry, relaxed_dealtype, relaxed_content_type


# A temporal query's lexical signal can be too weak to pass the relevance gate ('what
# happened in May 2021'); below this many hits, the window's recency-sorted articles fill it.
_TEMPORAL_FALLBACK_MIN = 3


async def _temporal_date_fallback(
    results: list[SourceArticle],
    top_k: int,
    from_date: str | None,
    to_date: str | None,
    industry: str | None,
    dealtype: str | None,
    author: str | None,
    tag: str | None,
    need_body: bool,
) -> list[SourceArticle]:
    """Fill a date-scoped query's gap with the window's most recent articles.

    A temporal query's date window IS the intent, so recency within it is a valid
    relevance signal. Returns ``results`` unchanged with no window, enough hits, or
    an empty window.
    """
    if not (from_date or to_date):
        return results
    strong = [r for r in results if r.score >= config.ASK_MIN_SCORE]
    if len(strong) >= min(_TEMPORAL_FALLBACK_MIN, top_k):
        return results
    date_only = await retrieve_by_date_window(
        top_k, from_date, to_date,
        industry=industry, dealtype=dealtype, author=author, tag=tag, need_body=need_body,
    )
    if not date_only:
        return results
    merged = _merge_results(results, date_only)
    return sort_results(merged)[:top_k]


async def retrieve_by_date_window(
    top_k: int,
    from_date: str | None,
    to_date: str | None,
    *,
    industry: str | None = None,
    dealtype: str | None = None,
    author: str | None = None,
    tag: str | None = None,
    need_body: bool = False,
) -> list[SourceArticle]:
    """Most recent articles in [from_date, to_date], ignoring the lexical query.

    Narrowed by the same facets the lexical leg was given, so filling a gap cannot
    widen the caller's filter. Scored by recency so they clear the relevance gate.
    """
    qfilter = build_facet_filter(industry, dealtype, author, from_date, to_date, None, tag)
    if qfilter is None:
        return []
    # published_date is a DATETIME payload index, so order_by picks the window's top_k by
    # recency without paging the whole window into memory.
    points, _ = await state["qdrant"].scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=qfilter,
        limit=top_k,
        offset=None,
        order_by={"key": "published_date", "direction": "desc"},
        with_payload=_PAYLOAD_FIELDS,
        with_vectors=False,
    )
    if not points:
        return []
    articles = [
        SourceArticle(
            id=p.id,
            title=(payload := p.payload or {}).get("title", ""),
            url=payload.get("url", ""),
            published_date=payload.get("published_date"),
            category=payload.get("category"),
            summary=payload.get("summary", ""),
            body="",
            author_names=payload.get("author_names") or [],
            industry_names=payload.get("industry_names") or [],
            dealtype_names=payload.get("dealtype_names") or [],
            # A recency-agnostic base: sort_results applies the multiplier once, so fillers
            # and lexical hits share one scale. The floor sits below the lexical band, so a
            # filler never outranks a real match.
            score=config.DATE_FILLER_SCORE,
        )
        for p in points
    ]
    if need_body:
        await _attach_bodies(articles)
    articles.sort(
        key=lambda a: (_tz_stripped_pub(a.published_date),),
        reverse=True,
    )
    return articles[:top_k]


@app.get(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(public_rate_limit("search", "PUBLIC_SEARCH_RATE_PER_MIN"))],
)
async def search(
    # max_length rejects with 422 rather than truncating, which would answer a query the
    # caller never asked.
    q: str = Query(..., min_length=1, max_length=config.SEARCH_QUERY_MAX_CHARS),
    top_k: int = Query(config.TOP_K, ge=1, le=50),
    industry: str | None = Query(None),
    dealtype: str | None = Query(None),
    author: str | None = Query(None),
    content_type: str | None = Query(None),
    tag: str | None = Query(None),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
):
    start = time.perf_counter()
    # One normalised spelling drives the key, retrieval text, analytics and the response. A
    # control-character-only query normalises to empty, so reject it here.
    q = normalize_text(q)
    if not q:
        raise HTTPException(status_code=400, detail="empty query")
    # The cache key below is built from these, so an oversized facet must be refused first.
    for facet_field, facet_raw in (("industry", industry), ("dealtype", dealtype),
                                   ("author", author), ("content_type", content_type),
                                   ("tag", tag)):
        split_facet_values(facet_field, facet_raw)
    q_fixed, _ = fix_query(q)
    retrieval_q, eff_from, eff_to, auto_dealtype, auto_industry = _effective_intent(q_fixed, from_date, to_date)
    auto_content_type = extract_content_type(q_fixed)
    # Auto facets fill in only when the caller passed none, and the explicit values are
    # kept aside so the fallback can tell a never-dropped facet from a droppable one.
    explicit_industry, explicit_dealtype, explicit_content_type = industry, dealtype, content_type
    dealtype = dealtype or auto_dealtype
    industry = industry or auto_industry
    content_type = content_type or auto_content_type
    # Query expansion happens exactly once inside _retrieval_leg (chat shares it), so
    # expanding here too would make /search diverge from the chat pipeline.
    eff_top_k = min(max(top_k, suggested_top_k(q) or 0), 50)
    # Same two composing bounds as the retrieve: key.
    cache_key = search_cache_key(
        retrieval_q, eff_top_k,
        facet_cache_token(industry, dealtype, author, eff_from, eff_to, content_type, tag),
    )
    filtered = any((industry, dealtype, author, content_type, tag, from_date, to_date))
    # Two entries are needed (this summary page, and the retrieval set under it) and are read
    # in one MGET; the prefetch key must be derived exactly as retrieve_and_rerank derives
    # its own, or the lookup is wasted.
    prefetch_key = retrieve_cache_key(
        fix_query(retrieval_q)[0], eff_top_k,
        build_facet_filter(industry, dealtype, author, eff_from, eff_to, content_type, tag),
    )
    cached_results, cached_articles = await cache.get_many([cache_key, prefetch_key])
    if cached_results is not None:
        summaries = [SourceSummary.model_validate(d) for d in cached_results]
        note = weak_results_note([s.score for s in summaries], date_label(eff_from, eff_to))
        await record_search(q, len(summaries), bool(note), cached=True,
                            latency_ms=(time.perf_counter() - start) * 1000, filtered=filtered)
        return SearchResponse(query=q, results=summaries, cached=True,
                              latency_ms=(time.perf_counter() - start) * 1000, note=note)

    # Rerank on the retrieval query (date words stripped), as chat does; the raw phrase
    # dilutes the cross-encoder. The effective facets may still relax below.
    reranked, final_industry, final_dealtype, final_content_type = await retrieve_with_auto_facet_fallback(
        retrieval_q, eff_top_k,
        industry=explicit_industry, dealtype=explicit_dealtype, author=author,
        content_type=explicit_content_type, tag=tag,
        eff_from=eff_from, eff_to=eff_to,
        auto_industry=auto_industry, auto_dealtype=auto_dealtype,
        auto_content_type=auto_content_type, prefetched=cached_articles,
    )
    if config.ENABLE_CLICK_BOOST:
        reranked = await apply_click_boost(q_fixed, reranked)
    if config.ENABLE_DIVERSITY:
        reranked = diversify(reranked, eff_top_k, lam=config.DIVERSITY_LAMBDA,
                             sim_thresh=config.DIVERSITY_SIM_THRESHOLD)
    results = reranked[:eff_top_k]
    note = weak_results_note([r.score for r in results], date_label(eff_from, eff_to))
    # Empty sets are never cached (a transient miss replayed as "no results" for the whole
    # TTL), and neither are relaxed-fallback sets: their key still names the auto facet and
    # would collide with an explicit-facet request. `tag` is absent on purpose: no auto
    # counterpart exists to relax it into.
    fell_back = final_industry != industry or final_dealtype != dealtype or final_content_type != content_type
    if results and not fell_back:
        await cache.set(cache_key, [to_summary(r).model_dump() for r in results])
    # Computed from the effective facets so the hit and miss paths report the same semantics.
    await record_search(q, len(results), bool(note), cached=False,
                        latency_ms=(time.perf_counter() - start) * 1000, filtered=filtered)
    return SearchResponse(query=q, results=[to_summary(r) for r in results], cached=False,
                          latency_ms=(time.perf_counter() - start) * 1000, note=note)


# So the model can tell the cap cut the text, not that the article ended.
BODY_TRUNCATION_NOTE = "\n[... body truncated ...]"


def source_context(s: SourceArticle, idx: int, body_limit: int | None = None) -> str:
    """Pack an article's metadata + summary + capped body into a numbered context block.

    The cap is CHAT_BODY_CHAR_LIMIT, or ``body_limit`` when the caller budgets a total."""
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
        # An explicit ``body_limit`` always wins, even 0; only None falls back to the default.
        limit = config.CHAT_BODY_CHAR_LIMIT if body_limit is None else max(0, int(body_limit))
        if limit and len(s.body) > limit:
            parts.append(s.body[:limit] + BODY_TRUNCATION_NOTE)
        elif limit:
            parts.append(s.body)
    return "\n".join(parts)


# v2: a v1 entry has no ``tags`` and would serve an empty list until its own TTL expired.
FACETS_CACHE_KEY = "facets:v2"
FACETS_LIMIT = 200
# Its own constant because it is applied to a FINISHED frequency ranking, not a
# truncated alphabetical scan, whose survivors depend on the collection's scroll order.
TAGS_FACET_LIMIT = 200
# Counted in points consumed, not round trips: a capped scan stops wherever the
# collection's scroll order puts it, so the check cannot be once per page.
FACET_CAP_CHECK_EVERY = 256
# Page size, not FACETS_LIMIT, sets the round-trip count; widening it cannot change results.
FACET_SCROLL_PAGE = 1024

# The in-flight /facets miss: /facets is unauthenticated and fetched on every page load, so
# without this a cold-cache burst starts one full-collection walk per caller. The scope is
# this process, so gunicorn still pays one walk per worker. The done callback releases the
# entry, so a failure or a dead loop retries instead of wedging.
_facet_scan_task: asyncio.Task | None = None


def _release_facet_scan(task: asyncio.Task) -> None:
    """Drop the in-flight entry once the scan settles.

    Identity-checked, so a failed scan is retryable rather than wedging the key.
    """
    global _facet_scan_task
    if _facet_scan_task is task:
        _facet_scan_task = None


async def _facets_uncached() -> dict[str, list[str]]:
    """The facet vocabularies, scanned concurrently, then cached.

    return_exceptions keeps every sibling outcome retrieved and stops a late tag failure
    being masked by the first; a non-tag failure still propagates, and nothing is cached
    on the way out. A failed ``tags`` walk degrades to an empty list -- the tag input is
    free text, so an empty suggestion list still filters -- rather than 500-ing the
    endpoint.
    """
    industry, dealtype, tags = await asyncio.gather(
        _facet_values("industry_names"),
        _facet_values("dealtype_names"),
        _top_facet_values("tag_names", TAGS_FACET_LIMIT),
        return_exceptions=True,
    )
    for outcome in (industry, dealtype):
        if isinstance(outcome, BaseException):
            raise outcome
    if isinstance(tags, BaseException):
        logger.warning(
            "tag facet scan failed (%s: %s); serving industry/dealtype without tag suggestions",
            type(tags).__name__,
            tags,
            exc_info=tags,
        )
        tags = []
    result = {"industry": industry, "dealtype": dealtype, "tags": tags}
    await cache.set(FACETS_CACHE_KEY, result)
    return result


async def _facets_single_flight() -> dict[str, list[str]]:
    """``_facets_uncached`` once, shared by every caller that misses together.

    Waiters ``shield`` the task so a disconnecting request cannot abort the scan the
    others are on. One "exception in shielded future" ERROR per shared scan that
    genuinely failed is the accepted cost.
    """
    global _facet_scan_task
    running = _facet_scan_task
    # A task carries its loop, so an entry left by a closed loop is not reusable.
    if running is not None and not running.done() and running.get_loop() is asyncio.get_running_loop():
        return await asyncio.shield(running)
    task = asyncio.create_task(_facets_uncached())
    _facet_scan_task = task
    task.add_done_callback(_release_facet_scan)
    return await asyncio.shield(task)


def _facet_point_values(payload: dict, key: str) -> list[str]:
    """The non-empty string values one point contributes to ``key``.

    A non-string is dropped, not stringified, so a malformed payload cannot put ``'42'``
    into a vocabulary a user can then select.
    """
    value = payload.get(key)
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item]
    return []



async def _facet_values(key: str) -> list[str]:
    """Distinct payload values for ``key``, paged via the public ``scroll`` API.

    qdrant-client 1.11 exposes no stable public facet method, so this pages the
    collection rather than reaching into the client's private HTTP internals. The cap
    is intentional: reaching it logs a warning instead of masking a runaway vocabulary.
    """
    values: set[str] = set()
    next_offset = None
    truncated = False
    while True:
        pts, next_offset = await state["qdrant"].scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=FACET_SCROLL_PAGE,
            with_payload=[key],
            with_vectors=False,
            offset=next_offset,
        )
        for start in range(0, len(pts), FACET_CAP_CHECK_EVERY):
            for p in pts[start : start + FACET_CAP_CHECK_EVERY]:
                values.update(_facet_point_values(p.payload or {}, key))
            if len(values) >= FACETS_LIMIT:
                truncated = True
                break
        if truncated or next_offset is None or not pts:
            # `not pts` guards a client that returns an empty page without clearing the offset.
            break
    # Stopped by the cap rather than by exhaustion: the data is truncated by design.
    if len(values) >= FACETS_LIMIT:
        logger.warning("facet %s hit FACETS_LIMIT=%d; results truncated", key, FACETS_LIMIT)
    return sorted(values)[:FACETS_LIMIT]


async def _top_facet_values(key: str, limit: int) -> list[str]:
    """The ``limit`` most frequent values for ``key``, most frequent first.

    A tag is free text, not a controlled vocabulary, so the top N cannot be known without
    counting every value on every point: a full walk with no early exit. Frequency order
    with an alphabetical tie-break keeps the cached payload byte-stable.
    """
    counts: Counter[str] = Counter()
    next_offset = None
    while True:
        pts, next_offset = await state["qdrant"].scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=FACET_SCROLL_PAGE,
            with_payload=[key],
            with_vectors=False,
            offset=next_offset,
        )
        for p in pts:
            counts.update(_facet_point_values(p.payload or {}, key))
        if next_offset is None or not pts:
            # Same `not pts` guard as _facet_values: an empty page without a cleared offset loops forever.
            break
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [value for value, _count in ranked[:limit]]


@app.get(
    "/facets",
    dependencies=[Depends(public_rate_limit("facets", "PUBLIC_FACETS_RATE_PER_MIN"))],
)
async def facets():
    """Filter vocabularies for autocomplete, cached in Redis.

    ``tags`` is the ``TAGS_FACET_LIMIT`` most frequent values, ranked before truncation.
    A miss is single-flight, so callers that miss together share one scan and one write.
    """
    cached = await cache.get(FACETS_CACHE_KEY)
    if cached is not None:
        return cached
    return await _facets_single_flight()


class ClickEvent(BaseModel):
    query: str = ""
    position: int = 0
    id: int | None = None


async def _article_in_index(article_id: int | None) -> bool:
    """True when ``article_id`` is a point in the Qdrant collection.

    The beacon's ``id`` is caller-supplied, so without this an unauthenticated caller
    mints a ranking vote for any integer. Fails CLOSED: an unconfirmable article must
    not become a vote, though the raw counters are still recorded.
    """
    if article_id is None:
        return False
    client = state.get("qdrant")
    if client is None:
        return False
    try:
        points = await client.retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=[article_id],
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)
    except Exception:
        logger.warning("click-beacon article existence check failed for id %s", article_id, exc_info=True)
        return False


@app.post(
    "/analytics/click",
    dependencies=[Depends(public_rate_limit("click", "PUBLIC_CLICK_RATE_PER_MIN"))],
)
async def analytics_click(event: ClickEvent, request: Request):
    """Anonymous result-click beacon; returns no data, so it stays unauthenticated.

    A session gate would drop every logged-out visitor's clicks. Amplification is closed
    from the other side instead: a per-IP limit, a per-client dedupe, and the index check.
    """
    article_id = event.id if await _article_in_index(event.id) else None
    if event.id is not None and article_id is None:
        logger.info("click beacon id %s is not in the collection; recorded without a ranking vote", event.id)
    await record_click(event.query, event.position, article_id, client_ip=_client_ip(request))
    return {"ok": True}


def _analytics_unavailable(message: str) -> JSONResponse:
    """503 for an analytics feed whose store could not be read.

    Status and body must agree: a 200 carrying an error is indistinguishable from a
    genuinely all-zero report.
    """
    return JSONResponse(
        status_code=503,
        content={"error": message, "detail": "the analytics store could not be read"},
    )


@app.get("/analytics/summary")
async def get_analytics_summary(
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("analytics:read")),
):
    """Aggregated search/click metrics. Admin-only.

    Query lists carry an opaque digest, never the text: aggregating by text would make
    this a cross-user read of everyone's search history. An unreadable store answers 503.
    """
    # Best-effort: a deployment with no chat store still serves the summary rather than 500.
    try:
        store = chat_module._require_store()
        await store.record_admin_audit(request.state.user_id, "analytics.summary.read")
    except Exception:
        logger.exception("admin audit write failed for analytics.summary.read")
    try:
        return await analytics_data()
    except AnalyticsUnavailableError as exc:
        return _analytics_unavailable(str(exc))


@app.get("/analytics/chat")
async def get_analytics_chat(
    request: Request,
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("analytics:read")),
):
    """Cross-user chat usage aggregates. Admin-only.

    No user-authored text is ever included; an unreadable store answers 503.
    """
    store = chat_module._require_store()
    try:
        await store.record_admin_audit(request.state.user_id, "analytics.chat.read")
    except Exception:
        logger.exception("admin audit write failed for analytics.chat.read")
    try:
        return await store.global_stats()
    except ChatAnalyticsUnavailableError as exc:
        return _analytics_unavailable(str(exc))


# A hash VALUE, not a field or a key, so it does not drive key growth.
MAX_DWELL_TIME_MS = 24 * 60 * 60 * 1000


class InteractionEvent(BaseModel):
    """User article interaction event.

    ``interaction_type`` is a closed enum because it becomes a Redis hash FIELD, and
    ``article_id`` is index-verified before any key is written.
    """
    # le caps at int64: a larger id is a client-side error, which would turn a reject into a 500.
    article_id: int = Field(..., ge=1, le=2**63 - 1, description="Indexed article id")
    interaction_type: InteractionType = InteractionType.CLICK
    dwell_time_ms: int | None = Field(None, ge=0, le=MAX_DWELL_TIME_MS)


class SimilarArticlesResponse(BaseModel):
    article_id: int | str
    similar_articles: list[dict]
    limit: int
    cached: bool = False


# Pre-deploy entries still carry the full article body and are served verbatim, so the version
# is what stops them being replayed for the rest of their TTL.
RECOMMEND_CACHE_VERSION = "v2"
# One search view renders one similar list per result, so this bounds how many of
# those per-id queries one request can ask for at once.
SIMILAR_BATCH_MAX_IDS = 20


class SimilarArticlesBatchRequest(BaseModel):
    article_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=SIMILAR_BATCH_MAX_IDS,
        description="Indexed article ids",
    )
    limit: int = Field(config.RECOMMEND_DEFAULT_LIMIT, ge=1, le=20)
    same_category: bool = False


class SimilarArticlesGroup(BaseModel):
    article_id: int
    similar_articles: list[dict]
    cached: bool = False


class SimilarArticlesBatchResponse(BaseModel):
    results: list[SimilarArticlesGroup]


class RecommendationsResponse(BaseModel):
    user_id: str
    recommendations: list[dict]
    limit: int
    cold_start: bool = False
    cached: bool = False


class TrendingResponse(BaseModel):
    articles: list[dict]
    limit: int
    window_days: int

# limit is bounded, so a user's whole for-you cache is exactly this range of knowable keys
# and one DEL replaces a prefix SCAN of the entire keyspace. Keep these bounds in sync with
# the Query in get_for_you, or entries are silently left behind.
FOR_YOU_MIN_LIMIT = 1
FOR_YOU_MAX_LIMIT = 20


def _for_you_cache_key(user_id: str, limit: int) -> str:
    """The one spelling of a /recommend/for-you key, so writer and invalidator cannot drift."""
    return f"recommend:for-you:{user_id}:{RECOMMEND_CACHE_VERSION}:{limit}"


def _for_you_cache_keys(user_id: str) -> list[str]:
    return [
        _for_you_cache_key(user_id, limit)
        for limit in range(FOR_YOU_MIN_LIMIT, FOR_YOU_MAX_LIMIT + 1)
    ]


@app.post(
    "/recommend/interaction",
    dependencies=[
        # Both axes: one shared NAT address defeats the per-IP bucket, and one rotating
        # account defeats nothing once the per-account bucket is present.
        Depends(public_rate_limit("interaction", "PUBLIC_INTERACTION_RATE_PER_MIN")),
        Depends(user_rate_limit("interaction", "INTERACTION_USER_RATE_PER_MIN")),
    ],
)
async def record_user_interaction(
    event: InteractionEvent,
    request: Request,
    _auth: None = Depends(require_auth),
):
    """Record a user-article interaction. Authenticated callers only.

    Fail-closed on a limiter-Redis outage (503): no health check probes this route,
    so failing open would re-open the amplification the limits close.
    """
    user_id = request.state.user_id
    result = await record_interaction(
        user_id=user_id,
        article_id=event.article_id,
        interaction_type=event.interaction_type,
        dwell_time_ms=event.dwell_time_ms,
    )
    if result is InteractionResult.INVALID_TYPE:
        raise HTTPException(status_code=422, detail="Unknown interaction_type")
    if result is InteractionResult.UNKNOWN_ARTICLE:
        raise HTTPException(status_code=404, detail="Unknown article")
    if result is InteractionResult.CAP_REACHED:
        # The article is real; this ACCOUNT hit its cap, so 404 would be false.
        raise HTTPException(
            status_code=429,
            detail="Interaction limit reached for this account",
            headers={"Retry-After": str(config.PUBLIC_RATE_WINDOW_SECONDS)},
        )
    if result is InteractionResult.UNAVAILABLE:
        # Redis or the index could not answer: "unknown article" would be a lie and the write did not happen.
        raise HTTPException(status_code=503, detail="Interaction store unavailable")
    await invalidate_user_profile(user_id)
    await cache.delete_keys(_for_you_cache_keys(user_id))
    return {"status": "ok", "article_id": event.article_id}


@app.get("/recommend/similar/{article_id}", response_model=SimilarArticlesResponse)
async def get_similar(
    article_id: int,
    limit: int = Query(config.RECOMMEND_DEFAULT_LIMIT, ge=1, le=20),
    same_category: bool = Query(False),
    _auth: None = Depends(require_auth),
):
    """Articles similar to one, by dense similarity with optional category filtering."""
    cached_key = _similar_cache_key(article_id, limit, same_category)
    cached = await cache.get(cached_key)
    if cached:
        return SimilarArticlesResponse(
            article_id=article_id,
            similar_articles=cached,
            limit=limit,
            cached=True,
        )

    articles = await get_similar_articles(
        article_id=article_id,
        limit=limit,
        same_category=same_category,
    )
    if articles:
        await cache.set(cached_key, articles, ttl=SIMILAR_ARTICLES_TTL_SECONDS)

    return SimilarArticlesResponse(
        article_id=article_id,
        similar_articles=articles,
        limit=limit,
        cached=False,
    )


def _similar_cache_key(article_id: int, limit: int, same_category: bool) -> str:
    """Cache key for one article's similar list.

    Shared with the batched route so the two spellings cannot drift. The version is
    part of it because pre-deploy entries still carry the full article body.
    """
    return f"recommend:similar:{RECOMMEND_CACHE_VERSION}:{article_id}:{limit}:{same_category}"


@app.post("/recommend/similar/batch", response_model=SimilarArticlesBatchResponse)
async def get_similar_batch(
    body: SimilarArticlesBatchRequest,
    _auth: None = Depends(require_auth),
):
    """The per-article route's answer for a whole view at once, read with one MGET.

    Ids are answered in request order; an article with no similar rows is an empty list.
    """
    # A repeated id costs one Qdrant query and one response group.
    article_ids = list(dict.fromkeys(body.article_ids))
    cached = await cache.get_many(
        [
            _similar_cache_key(article_id, body.limit, body.same_category)
            for article_id in article_ids
        ]
    )

    groups: dict[int, SimilarArticlesGroup] = {}
    missing: list[int] = []
    for article_id, value in zip(article_ids, cached):
        if value:
            groups[article_id] = SimilarArticlesGroup(
                article_id=article_id, similar_articles=value, cached=True
            )
        else:
            missing.append(article_id)

    if missing:
        computed = await asyncio.gather(
            *(
                get_similar_articles(
                    article_id=article_id,
                    limit=body.limit,
                    same_category=body.same_category,
                )
                for article_id in missing
            )
        )
        # Same rule as the per-article route: an empty result is a transient miss, not cacheable.
        await asyncio.gather(
            *(
                cache.set(
                    _similar_cache_key(article_id, body.limit, body.same_category),
                    articles,
                    ttl=SIMILAR_ARTICLES_TTL_SECONDS,
                )
                for article_id, articles in zip(missing, computed)
                if articles
            )
        )
        for article_id, articles in zip(missing, computed):
            groups[article_id] = SimilarArticlesGroup(
                article_id=article_id, similar_articles=articles, cached=False
            )

    return SimilarArticlesBatchResponse(
        results=[groups[article_id] for article_id in article_ids]
    )


@app.get("/recommend/for-you", response_model=RecommendationsResponse)
async def get_for_you(
    limit: int = Query(config.RECOMMEND_DEFAULT_LIMIT, ge=FOR_YOU_MIN_LIMIT, le=FOR_YOU_MAX_LIMIT),
    _auth: None = Depends(require_auth),
    request: Request = None,
):
    """Personalized recommendations; no interaction history sets ``cold_start``."""
    user_id = request.state.user_id

    cached_key = _for_you_cache_key(user_id, limit)
    cached = await cache.get(cached_key)
    if cached:
        return RecommendationsResponse(
            user_id=user_id,
            recommendations=cached,
            limit=limit,
            cached=True,
        )

    interactions = await get_user_interactions(user_id)
    exclude_ids = [aid for aid, _ in interactions[:20]]

    articles = await get_personalized_recommendations(
        user_id=user_id,
        limit=limit,
        exclude_ids=exclude_ids,
    )

    cold_start = len(interactions) == 0
    if articles:
        await cache.set(cached_key, articles, ttl=USER_RECOMMENDATIONS_TTL_SECONDS)

    return RecommendationsResponse(
        user_id=user_id,
        recommendations=articles,
        limit=limit,
        cold_start=cold_start,
        cached=False,
    )


@app.get("/recommend/trending", response_model=TrendingResponse)
async def get_trending(
    limit: int = Query(config.RECOMMEND_DEFAULT_LIMIT, ge=1, le=20),
    _auth: None = Depends(require_auth),
):
    """Trending articles by click velocity over the configured window."""
    cached_key = f"recommend:trending:{RECOMMEND_CACHE_VERSION}:{limit}"
    cached = await cache.get(cached_key)
    if cached:
        return TrendingResponse(
            articles=cached,
            limit=limit,
            window_days=config.TRENDING_VELOCITY_WINDOW_DAYS,
        )

    articles = await get_trending_feed(limit=limit)
    if articles:
        await cache.set(cached_key, articles, ttl=SIMILAR_ARTICLES_TTL_SECONDS)

    return TrendingResponse(
        articles=articles,
        limit=limit,
        window_days=config.TRENDING_VELOCITY_WINDOW_DAYS,
    )
