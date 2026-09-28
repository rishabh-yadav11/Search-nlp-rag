import asyncio
import hashlib
import json
import logging
import math
import re
import time
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

# Import config FIRST so the OMP/MKL thread caps in app.config are set before
# any inference library (torch/onnxruntime) is imported below.
from app import auth as auth_module
from app import chat as chat_module
from app.analytics import AnalyticsUnavailableError, record_click, record_search
from app.analytics import close as close_analytics
from app.analytics import summary as analytics_data
from app.answer_fallback import date_label, weak_results_note
from app.auth import public_rate_limit, require_auth, require_permission, user_rate_limit
from app.chat import ChatAnalyticsUnavailableError
from app.click_boost import apply_click_boost
from app.config import config, ensure_data_paths_ready
from app.cost_budget import close as close_cost_budget
from app.diversity import diversify
from app.encoders import DenseEncoder
from app.health import close_redis as health_module_close_redis
from app.health import router as health_router
from app.health import warn_if_llm_key_unusable
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

# Uvicorn's worker leaves the root logger at WARNING with no handlers, so every
# module logger inherits WARNING and drops its INFO records (#293). Done at
# import -- before the lifespan and before any request is served -- so boot
# events and the background purge loops are emitted under the real startup path.
configure_logging()

state = {}

# Serializes CPU-bound inference (dense encode, sparse embed, rerank) per
# worker so concurrent requests don't contend for the CPU and thrash torch /
# onnxruntime thread pools. Async I/O (Qdrant/Redis) is unaffected.
inference_lock = asyncio.Lock()

logger = logging.getLogger(__name__)

# Live facet vocabularies loaded from Qdrant (normalized lowercased value ->
# original-cased value). Each vocabulary is published by rebinding its module
# global only after its complete replacement has loaded, so extractors see an
# old complete map, a new complete map, or the initial empty map—never a map
# being mutated in place. Category extraction only ever emits a value present in
# these maps, so a natural-language query like 'funding news' maps to the real
# 'Venture Capital' facet instead of guessing a label that would match nothing
# (an unknown filter value returns zero results and breaks the query). The actual
# labels (e.g. 'Venture Capital', 'M&A', 'Finance') come from the live index, not
# from hard-coded assumptions.
_DEALTYPE_FACETS: dict[str, str] = {}
_INDUSTRY_FACETS: dict[str, str] = {}
# Live `content_type` vocabulary (article/interview/video in the corpus), loaded
# from Qdrant so content-type intent resolves only to real values.
_CONTENT_TYPE_FACETS: dict[str, str] = {}

# Natural-language synonyms -> the facet keyword used to resolve against the live
# vocabulary. Whole-word matched against the query; the keyword is then looked up
# (exact, then substring) in the facet map. Resolution only succeeds when a real
# facet value exists, so an unknown corpus degrades to no filter (current
# behavior) rather than emitting a bogus value. Synonyms are mapped to the REAL
# dealtype labels the corpus uses (funding rounds -> 'Venture Capital', not a
# mythical 'Funding' facet).
_DEALTYPE_ALIASES: dict[str, str] = {
    # Venture capital / funding rounds
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
    # Private equity
    "private equity": "private equity",
    "pe": "private equity",
    # M&A / consolidation
    "m&a": "m&a",
    "merger": "m&a",
    "mergers": "m&a",
    "acquisition": "m&a",
    "acquisitions": "m&a",
    "acquire": "m&a",
    "acquired": "m&a",
    "buyout": "m&a",
    "takeover": "m&a",
    # Other real deal-type labels
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
    """Return a real facet value (original casing) reachable via a synonym alias
    in ``query`` (whole word), else None.

    ``facets`` maps normalized -> original facet value; ``aliases`` maps a synonym
    phrase -> the keyword to look up in ``facets``. Only explicit, curated
    synonyms are matched (never raw facet labels), so common-word facet labels
    like 'People' or 'General' can't be accidentally triggered by ordinary text.

    Aliases are tried longest-first so a longer phrase (e.g. 'venture capital')
    wins over a shorter word it embeds (e.g. 'capital'). For a matched keyword, an
    exact normalized facet name is preferred; only if no exact facet exists do we
    fall back to a substring match, choosing the tightest (shortest) candidate so
    the resolution is deterministic rather than an artifact of dict insertion
    order."""
    q = query.lower()
    # Longest alias first: a specific multi-word synonym must beat a shorter word
    # it embeds (otherwise 'capital' could shadow 'venture capital').
    for alias, kw in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", q):
            # Exact keyword match against the normalized facet vocabulary wins.
            if kw in facets:
                return facets[kw]
            # Substring fallback: prefer the tightest (shortest) normalized facet
            # containing the keyword so the choice is deterministic.
            candidates = [orig for norm, orig in facets.items() if kw in norm]
            if candidates:
                return min(candidates, key=lambda o: (len(o), o.lower()))
    return None


def extract_dealtype(query: str) -> str | None:
    """Map a natural-language query to a real ``dealtype_names`` facet value
    (e.g. 'funding news' -> 'Venture Capital', 'merger news' -> 'M&A'), or None."""
    return _resolve_facet(query, _DEALTYPE_ALIASES, _DEALTYPE_FACETS)


def extract_industry(query: str) -> str | None:
    """Map a natural-language query to a real ``industry_names`` facet value
    (e.g. 'fintech funding' -> 'Finance'), or None."""
    return _resolve_facet(query, _INDUSTRY_ALIASES, _INDUSTRY_FACETS)


def extract_content_type(query: str) -> str | None:
    """Map a content-type intent modifier to a real ``content_type`` facet value
    (e.g. 'interviews with X' -> 'interview', 'founders of Y' -> 'founder'), or
    None. The bare modifier is classified in query_intent and promoted here to a
    real facet value from the live ``content_type`` vocabulary (exact, then
    substring), so an unknown corpus degrades to no filter (current behavior)."""
    kw = _classify_content_type(query)
    if kw is None:
        return None
    facets = _CONTENT_TYPE_FACETS
    if kw in facets:
        return facets[kw]
    # Substring fallback: prefer the tightest (shortest) normalized facet value
    # containing the keyword so the choice is deterministic, not dict-order bound.
    candidates = [orig for norm, orig in facets.items() if kw in norm]
    if candidates:
        return min(candidates, key=lambda o: (len(o), o.lower()))
    return None


async def _load_facet_maps() -> None:
    """Load and atomically publish live facet maps from Qdrant.

    A failed load retains the prior complete vocabulary; at startup that prior
    vocabulary is empty, preserving degraded no-filter behavior.
    """
    global _DEALTYPE_FACETS, _INDUSTRY_FACETS, _CONTENT_TYPE_FACETS

    for name, key in (
        ("_DEALTYPE_FACETS", "dealtype_names"),
        ("_INDUSTRY_FACETS", "industry_names"),
        ("_CONTENT_TYPE_FACETS", "content_type"),
    ):
        try:
            values = await _facet_values(key)
            facets = {value.strip().lower(): value for value in values}
        except Exception:  # noqa: BLE001 - degraded mode, never crash startup
            logger.warning("facet map load failed for %s", key)
            continue
        if name == "_DEALTYPE_FACETS":
            _DEALTYPE_FACETS = facets
        elif name == "_INDUSTRY_FACETS":
            _INDUSTRY_FACETS = facets
        else:
            _CONTENT_TYPE_FACETS = facets


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First, before any store is opened or any request can be served: prove the
    # configured data locations are real and writable. Without this, an
    # unreachable or read-only data directory yields a running app backed by a
    # BRAND-NEW empty SQLite file, and every session 404s with no error logged.
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
    state["qdrant"] = AsyncQdrantClient(url=config.QDRANT_URL, timeout=30)
    await _load_facet_maps()
    # Names the real cause (missing / placeholder / malformed GEMINI_API_KEY) in
    # the log at boot, without taking the process down with it (#279).
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
    state["auth_token_purge"] = asyncio.create_task(auth_module.token_purge_loop())

    init_fixer(
        config.ENABLE_QUERY_FIX,
        config.QUERY_FIX_VOCAB_PATH,
        max_edit=config.QUERY_FIX_MAX_EDIT,
        min_count=config.QUERY_FIX_MIN_COUNT,
        min_token_len=config.QUERY_FIX_MIN_TOKEN_LEN,
    )

    # Connect recommender engine to shared state
    from app import recommender
    recommender.state = state

    yield
    state["chat_retention"].cancel()
    await asyncio.gather(state["chat_retention"], return_exceptions=True)
    await chat_store.close()
    state["auth_token_purge"].cancel()
    await asyncio.gather(state["auth_token_purge"], return_exceptions=True)
    await auth_store.close()
    await state["qdrant"].close()
    await cache.close()
    await close_analytics()
    await close_cost_budget()
    await auth_module.close_rate_redis()
    await health_module_close_redis()


app = FastAPI(
    title="VCCircle New Search",
    lifespan=lifespan,
    # No interactive docs and no published schema, in any environment: those
    # three routes hand any unauthenticated caller the complete route list, the
    # request/response models (including the mass-assignable UserPatchIn) and
    # which routes sit behind which dependency — the reconnaissance step for
    # probing /users and the admin management endpoints.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(
    # Reject requests whose Host is not one this deployment answers to. The
    # allow-list comes from config (derived from CORS_ORIGINS by default) and
    # is validated there, so it can never be silently empty or wildcarded.
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS,
)

# Outermost of the user middlewares (add_middleware inserts at index 0) so every
# request, including one rejected by TrustedHost/CORS below it, is correlated.
app.add_middleware(RequestIdMiddleware)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
# #293's configure_logging() above owns the app's single root handler; this only
# adds the per-request id filter to that handler, so a record it renders carries
# the id. Installing a second handler here would write every app line twice, and
# it must stay after that call -- installed_handler() is None until then.
attach_request_id_filter()

# A host that is missing from ALLOWED_HOSTS answers 400 to every request, which
# looks like a broken app rather than a config mistake. Log the effective list
# at import (before anything can fail) so the cause is visible in the worker
# logs straight away. WARNING, not INFO: this is the one clue during such an
# outage, and it has to survive an operator who raised LOG_LEVEL to cut volume.
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
    content_type: str | None = None
    score: float


class SourceSummary(BaseModel):
    """Public DTO for search/chat results. Exposes a short `summary` excerpt for
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
) -> Filter | None:
    """Qdrant filter for the faceted search params, or None when unfiltered."""
    conditions = []
    for key, raw in (
        ("industry_names", industry),
        ("dealtype_names", dealtype),
        ("author_names", author),
        ("content_type", content_type),
    ):
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
    content_type: str | None = None,
) -> str:
    return f"{industry or ''}|{dealtype or ''}|{author or ''}|{from_date or ''}|{to_date or ''}|{content_type or ''}"


def _effective_intent(
    q: str,
    from_date: str | None,
    to_date: str | None,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Rewrite the query for retrieval, derive an auto date filter from the
    query's date intent, and extract any category facets (dealtype/industry) the
    natural-language query implies. Explicit user dates always win.

    Month-scoped queries (e.g. 'top pharma deals of month january 2025') use the
    bare topic as the retrieval query (the date filter scopes the month), so the
    noisy 'top/of/month/year' words don't dilute the embedding match. The same
    applies to quarter, fiscal-year, and year-span queries.

    Returns (retrieval_q, eff_from, eff_to, dealtype, industry). The dealtype/
    industry are looked up against the live facet vocabulary and are None when the
    query implies no category (or the facet maps are empty). Date words (months,
    years, quarters) are stripped from the retrieval query because the date filter
    already scopes the window; the natural phrasing (e.g. 'funding news') is kept
    so the embedding/rerank match stays strong, while the facet filter (when one
    resolves) still scopes results. The content-type modifier (interviews/
    founders/etc.) is derived separately by extract_content_type so the facet maps
    can stay cold-start-safe and the call sites control its fallback."""
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
    # Rolling recency windows ('this week', 'today') resolve to a recent date
    # range so the filter drops old evergreen articles; the window words are
    # stripped from the retrieval query because the date filter already scopes it.
    # Soft recency terms ('latest', 'recent') are also stripped so they don't
    # dilute the embedding match — but the recency intent itself is still detected
    # on the original query (is_recency_intent) for ranking weight.
    rng = extract_recency_range(q)
    if rng:
        cleaned = strip_recency_intent(strip_recency_window(q))
        if cleaned:
            retrieval_q = cleaned
        return retrieval_q, rng[0], rng[1], dealtype, industry
    # Soft recency-intent queries (no hard window) still have their recency terms
    # removed from the retrieval text for the same relevance reason, while the
    # ranking weight derived from is_recency_intent is preserved.
    if is_recency_intent(q):
        cleaned = strip_recency_intent(retrieval_q)
        if cleaned:
            retrieval_q = cleaned
    return retrieval_q, from_date, to_date, dealtype, industry


def _merge_results(*groups: list[SourceArticle]) -> list[SourceArticle]:
    """Concatenate and dedupe by id, keeping the highest score for each id.

    Used to combine the raw RRF-candidate sets from the Flashback and bare-topic
    retrieval legs *before* a single cross-encoder rerank against the original
    query, so scores remain comparable across legs.

    A body-less entry is never allowed to override a body-bearing entry sharing
    the same id (e.g. a date-only fallback filler must not shadow a lexical hit
    that carries the article body chat needs). When one side has a body and the
    other does not, the body-bearing entry wins regardless of score; otherwise
    the higher score wins."""
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


# The leading 'Flashback <year>' prefix _effective_intent emits for a year-in-
# review intent ('top N <topic> in <year>'), used to recover that intent when
# _retrieval_queries is called with the already-rewritten query.
_FLASHBACK_PREFIX_RE = re.compile(r"^\s*flashback\s+(?:19|20)\d{2}\b\s*", re.IGNORECASE)


def _retrieval_queries(q: str) -> list[str]:
    """Queries to run for a user query. For year-in-review intents this is the
    Flashback-rewritten query PLUS the bare topic (year-filtered) so niche
    topics that have no dedicated Flashback article still surface their specific
    articles (e.g. 'venture debt providers', 'unicorns created'). For
    month-scoped queries the bare topic is used directly (date filter scopes the
    month). Otherwise a single query."""
    flashback, changed = rewrite_year_in_review(q)
    if changed:
        topic = extract_list_topic(q) or q
        return list(dict.fromkeys([flashback, topic]))
    # In the real pipeline this is called with the query _effective_intent has
    # already rewritten (e.g. 'Flashback 2025 unicorns created'), so the
    # 'top N ... in <year>' cue is gone and ``changed`` is False — the bare-topic
    # leg would be dropped. The 'Flashback <year>' prefix is only emitted for a
    # year-in-review intent, so re-detect it and re-emit the embedded topic as
    # the second leg (the prefix rewrite was 'Flashback <year> <topic>').
    m = _FLASHBACK_PREFIX_RE.match(q)
    if m:
        topic = q[m.end():].strip()
        if topic:
            return list(dict.fromkeys([q, topic]))
    scoped = range_query_topic(q)
    if scoped:
        return [scoped]
    return [q]


# Payload fields needed for ranking/display. The article `body` is intentionally
# excluded: it is large (~6KB/article) and only used for chat context, where it
# is fetched separately for the final sources (_attach_bodies).
_PAYLOAD_FIELDS = [
    "title",
    "url",
    "published_date",
    "category",
    "summary",
    "author_names",
    "industry_names",
    "dealtype_names",
    "content_type",
]


def _embed_sparse(model, text: str):
    """Sparse-embed one query, consuming fastembed's lazy generator inside the
    worker thread (a bare ``next()`` outside would run inference on the event
    loop and stall every other request)."""
    return next(iter(model.embed([text])))


async def hybrid_search(
    query: str,
    top_k: int,
    qfilter: Filter | None = None,
    with_body: bool = False,
) -> list[SourceArticle]:
    # Dense/sparse encoders are CPU/sync-bound: run them off the event loop so
    # the async handlers stay responsive under load, and serialize them so
    # concurrent requests don't thrash the inference thread pools.
    #
    # The (dense, sparse) pair for a query string is deterministic and
    # independent of the qfilter, so it is cached in Redis keyed by the
    # embedding models (a model change invalidates it). Repeated queries with
    # different facet/date filters skip encoding entirely.
    # Bound the text that reaches the encoders, for EVERY caller. /search is
    # already refused at the edge beyond SEARCH_QUERY_MAX_CHARS, so this never
    # binds for it; the bound that matters here is RETRIEVAL_QUERY_MAX_CHARS,
    # which is chat's own accepted message length (chat.MAX_CONTENT_LEN) so
    # that the LLM prompt and this retrieval always see the same question.
    # expand_query only ever grows the string, so even an in-limit input can be
    # over the limit again by the time it gets here. The clamp happens before
    # the cache key is built so the key and the embedded text always describe
    # the same string.
    query = query[: config.RETRIEVAL_QUERY_MAX_CHARS]
    vec_key = (
        f"vec:{config.EMBED_MODEL}|{config.SPARSE_MODEL}:"
        f"{_cache_key_component(query)}"
    )
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
            content_type=payload.get("content_type") or None,
            score=p.score,
        )
        for p in result.points
        # Skip points with a null/empty payload (shouldn't happen, but a
        # malformed point with no title/summary would otherwise surface as an
        # empty article that displaces real results, and a null payload would
        # raise AttributeError on p.payload.get).
        if (payload := p.payload)
    ]


async def rerank(query: str, results: list[SourceArticle]) -> list[SourceArticle]:
    """Cross-encoder rerank of RRF candidates, in place. Rewrites score with a
    sigmoid-normalized relevance score (0-1) so both ordering and the score the
    frontend shows reflect reranked relevance."""
    if len(results) <= 1:
        return results
    # Bound the query side of every (query, passage) pair: the cross-encoder
    # tokenizes both sides, so an unbounded query is the same CPU-spike path
    # the dense encoder has above. See RETRIEVAL_QUERY_MAX_CHARS in config for
    # why this is chat's limit and not /search's 512.
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
    """Lowercased alphanumeric query tokens minus stopwords/single chars,
    used for cheap lexical localization of the best-matching body region."""
    return {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS and len(w) > 1}


def _effective_step(positions: int, step: int, max_windows: int | None) -> int:
    """The stride to scan ``positions`` window starts with, honouring both a
    minimum step of 1 and an optional hard budget of ``max_windows`` windows.

    Clamping the configured stride alone does not bound the work, because a
    small stride is a legal (and deceptively cheap-looking) setting: step=1
    over a 50K body scores 48,501 windows and costs ~117ms per body, and
    ``body_rescue`` scans every body-bearing article before the candidate cap
    applies, so 20 articles cost ~2.3s for a single chat turn. Widening the
    stride bounds the work by window COUNT instead of by the raw value.

    ``max_windows=None`` (or <= 0) means no budget, which is what a direct
    caller that does not opt in gets. ``body_rescue`` always passes the
    configured ``BODY_RESCUE_MAX_WINDOWS``.

    The trade-off is recall for work: a stride coarse enough to widen can
    straddle a token-dense region and miss it. That only happens when a budget
    is configured below what the chosen stride would need, and at the defaults
    (98 windows against a budget of 200) the stride is never touched, so the
    scan is unchanged. A budget of 1 degenerates to the single window at
    start=0 rather than an empty range.
    """
    step = max(1, step)
    if max_windows and max_windows > 0 and -(-positions // step) > max_windows:
        step = max(1, -(-positions // max_windows))
    return step


def _best_body_window(
    body: str, tokens: set[str], win: int, step: int, max_windows: int | None = None
) -> str:
    """The body region with the most distinct query tokens, cheaply located by
    sliding a window over the lowercased body. Returns the window with the
    original casing (falls back to the tail region on ties).

    ``max_windows`` caps how many windows are scored per body, so the work is
    bounded by a window COUNT rather than by the raw stride; it defaults to
    None (uncapped) so the four-argument calling convention keeps its original
    behaviour. See ``_effective_step``.
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

    When the top reranked score is weak (below BODY_RESCUE_THRESHOLD) the
    title+summary cross-encoder scores are unreliable: relevant content may
    live mid-article (e.g. historical retrospectives). Re-score each candidate
    against the body region with the most lexical query overlap and keep
    max(baseline, body), so such matches can pass the chat relevance gate.

    The whole feature is gated on ENABLE_BODY_RESCUE here rather than only at
    the call sites: this is the function that pays the cost (a second
    cross-encoder pass under the process-wide inference lock), so the guard
    belongs where the cost is, not in every future caller that has to remember
    it. The call-site guards in chat.py stay as a cheap short-circuit.

    The pass runs on at most BODY_RESCUE_MAX_CANDIDATES articles: the body
    pass costs one cross-encoder prediction per candidate and dominates the
    window scan by two orders of magnitude, so the candidate count is the only
    budget that matters."""
    if not articles:
        return articles
    if not config.ENABLE_BODY_RESCUE:
        return articles
    if max((a.score for a in articles), default=0.0) >= config.BODY_RESCUE_THRESHOLD:
        return articles
    # Same bound as rerank(), and for the same reason: this is the second
    # cross-encoder call site, and the clamp in rerank() is a local that cannot
    # reach here. Chat drives this with a message of up to MAX_CONTENT_LEN plus
    # query expansion, so the query side of each pair here is genuinely
    # unbounded without it. Clamped before the tokenization below so the
    # lexical window and the rerank agree on the same string.
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
        # How many distinct query tokens the best window actually contains --
        # the same signal _best_body_window maximised, i.e. what the second
        # pass would have to work with.
        low_win = win.lower()
        overlap = sum(1 for t in tokens if t in low_win)
        candidates.append((overlap, i, (query, f"{a.title}. {a.summary or ''}. {win}".strip())))
    if not candidates:
        return articles
    # Shortlist by body-window overlap, NOT by a.score and NOT by list order.
    # A weak title+summary score is the very reason the rescue exists: taking
    # the top-N by score would drop exactly the deep-body matches it exists to
    # rescue, and taking the first N by list order would let retrieval ranking
    # decide the budget. An article whose best window contains none of the
    # query tokens is one the body pass cannot lift, so it is the correct thing
    # to drop first when the budget runs out. Ties resolve on the original
    # index so a run is reproducible.
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
    """Normalize the stored published_date for raw-string comparison.

    New records store naive RFC 3339 (e.g. '2020-01-01T12:00:00'); records
    indexed before the UTC-shift fix carry a '+00:00' suffix. Stripping the
    trailing tz offset (any of 'Z', '+HH:MM', '+HHMM', '-HH:MM', '-HHMM') lets
    the string tiebreaker order records by wall-clock uniformly, without
    reinterpreting or shifting the underlying time. Only a trailing offset is
    removed, so an embedded '+' (rare in these fields) is left intact.
    """
    if not published_date:
        return ""
    return _TZ_RE.sub("", published_date)


# Trailing ISO-8601 timezone designator (UTC 'Z' or a numeric ±HH:MM / ±HHMM
# offset). Used by _tz_stripped_pub to normalize dates for string comparison.
_TZ_RE = re.compile(r"(?:[zZ]|[+-]\d{2}:?\d{2})$")


def sort_results(
    results: list[SourceArticle],
    recency_boost: bool = False,
) -> list[SourceArticle]:
    """Recency-tempered relevance first, recency second: blended score desc,
    then published_date desc (missing dates last). When ``recency_boost`` is set
    (a recency intent like 'latest'/'recent'), freshness is weighted far more
    heavily so old evergreen articles rank below recent ones."""
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
    """Deterministic cache key fragment for a Qdrant filter."""
    if qfilter is None:
        return ""
    return json.dumps(qfilter.model_dump(), sort_keys=True, default=str)


def _cache_key_component(value: str) -> str:
    """Bounded, deterministic cache-key fragment for a request-supplied string.

    The retrieval caches (``vec:``, ``retrieve:``, ``search:``) key on the
    query text and on the facet values, which is fine for a normal request but
    makes the key as long as the request: a megabyte of ``q`` becomes a
    megabyte-scale Redis key (multi-KB across the key, plus the memory the
    server copies on every GET/SET). Rather than truncating the text — which
    would collide distinct long values onto one key and serve a different
    query's or a different facet filter's results — a long value is replaced
    by a truncated sha256 of its UTF-8 bytes, prefixed with ``h:`` so a digest
    can never be confused with a short literal value that happens to look like
    hex. Short values keep their exact previous key, so existing cache entries
    still hit.

    No caller parses a value back out of a key: every one of these keys is
    written and read only through this function's owning call site.
    """
    if len(value) <= config.CACHE_KEY_QUERY_MAX_CHARS:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]
    return f"h:{digest}"


async def _attach_bodies(articles: list[SourceArticle]) -> None:
    """Fetch the article `body` payloads for a set of articles in one Qdrant
    call. hybrid_search deliberately omits bodies to keep candidate fetches
    small; chat needs bodies for the LLM context, so they are pulled only for
    the final reranked set."""
    ids = [a.id for a in articles]
    if not ids:
        return
    resp = await state["qdrant"].retrieve(
        collection_name=config.QDRANT_COLLECTION,
        ids=ids,
        with_payload=["body"],
    )
    # Qdrant can return points keyed by either int or str ids, while
    # SourceArticle.id is an int. Normalize both sides to str so a string-id
    # point doesn't silently drop its body context.
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
) -> list[SourceArticle]:
    """Run every retrieval leg, merge RRF candidates, cross-encode rerank, and
    apply the entity-mention boost. Returns the recency-sorted articles.

    Shared by /search and /chat so the two pipelines stay consistent. A
    non-empty reranked article set is cached in Redis (same TTL as /search)
    because it is deterministic for a (query, filter) pair; chat follow-ups
    re-run the same retrieval on every turn, and this cache makes those turns
    skip embedding + rerank entirely. Empty sets are never cached (see the
    guard comment by ``cache.set``). Bodies are not cached (they are large);
    when ``need_body`` is set they are fetched from Qdrant for the returned set.
    """
    q = fix_query(q)[0]  # typo-corrected query flows to cache key, legs, boost
    # A recency intent ('latest', 'recent') weights freshness heavily in ranking
    # so new articles outrank old evergreen ones; rolling windows ('this week')
    # are already scoped by the date filter and need no extra ranking boost.
    recency_boost = is_recency_intent(q)
    cache_key = (
        f"retrieve:{_cache_key_component(q)}:{top_k}:{_filter_token(qfilter)}"
    )
    cached = await cache.get(cache_key)
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
    # Bodies are deliberately excluded from the cache entry: they are large and
    # chat re-fetches them from Qdrant on a cache hit (_attach_bodies).
    #
    # Empty result sets are never cached: a transient retrieval failure (or a
    # momentary empty candidate set) would otherwise be replayed as an
    # authoritative "no results" for the whole CACHE_TTL_SECONDS window, which
    # is exactly the bug where a date-filtered query returned nothing for
    # minutes. Skipping the write (rather than caching a short TTL) is the safer
    # default: it fails toward correctness, and re-running the pipeline costs
    # far less than serving a wrong answer.
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
    eff_from: str | None,
    eff_to: str | None,
    auto_industry: str | None,
    auto_dealtype: str | None,
    auto_content_type: str | None = None,
    need_body: bool = False,
) -> tuple[list[SourceArticle], str | None, str | None, str | None]:
    """Retrieve with the effective (explicit-or-auto) category facets applied,
    and fall back to dropping an *auto* facet when it zeroes out an otherwise
    valid query. Returns ``(results, final_industry, final_dealtype,
    final_content_type)`` where the final facets reflect any fallback, so callers
    can key caches/notes on what was actually retrieved.

    ``industry``/``dealtype``/``content_type`` are the caller-supplied (explicit)
    facets; when one is None the matching ``auto_*`` value is used instead. An auto
    facet is a *semantic* guess mapped onto the corpus's tag vocabulary (e.g.
    'edtech' -> the 'Education' industry facet). When the corpus tags most of those
    articles differently (VCCircle tags edtech articles 'TMT', not 'Education'),
    the exact-match industry filter combined with a date window returns nothing and
    silently kills the query. Only an *empty* result set triggers the retry (the
    empty attempt is not cached), and only the auto facets are dropped: an
    explicit user-supplied facet and the date window always stay, so a genuinely
    empty corpus still reports an honest "no results".

    The fallback is deliberately bounded: it fires only when the first retrieval
    returned nothing AND an auto facet is present, and the retry itself re-runs
    retrieval (so a transient miss that then succeeds simply restores the good
    path). The only cost of a double-transient miss is that the auto facet is
    relaxed into a broader result — a graceful degradation, never a crash or
    fabricated data. All auto facets are dropped together (rather than probing
    each alone): dropping any one of them is a relaxation of the same semantic
    guess, and the broader set is the safer answer for a query that otherwise
    would have returned nothing.
    """
    eff_industry = industry or auto_industry
    eff_dealtype = dealtype or auto_dealtype
    eff_content_type = content_type or auto_content_type
    qfilter = build_facet_filter(eff_industry, eff_dealtype, author, eff_from, eff_to, eff_content_type)
    results = await retrieve_and_rerank(retrieval_q, top_k, qfilter, need_body=need_body)
    # A single below-gate hit (score under the chat relevance gate) left by a
    # mis-applied auto facet is effectively a dead result set, so relax the auto
    # facet(s) for it too, not only for a fully empty set (#172).
    lone_weak_hit = len(results) == 1 and results[0].score < config.ASK_MIN_SCORE
    if (results and not lone_weak_hit) or not (auto_industry or auto_dealtype or auto_content_type):
        results = await _temporal_date_fallback(
            results, top_k, eff_from, eff_to, eff_industry, eff_dealtype, author, need_body
        )
        return results, eff_industry, eff_dealtype, eff_content_type
    # An auto facet zeroed the set: drop each auto facet that wasn't explicitly
    # supplied (an explicit facet is never dropped) and retry once.
    relaxed_industry = eff_industry if not (auto_industry and industry is None) else None
    relaxed_dealtype = eff_dealtype if not (auto_dealtype and dealtype is None) else None
    relaxed_content_type = eff_content_type if not (auto_content_type and content_type is None) else None
    relaxed = build_facet_filter(relaxed_industry, relaxed_dealtype, author, eff_from, eff_to, relaxed_content_type)
    if relaxed == qfilter:
        results = await _temporal_date_fallback(
            results, top_k, eff_from, eff_to, eff_industry, eff_dealtype, author, need_body
        )
        return results, eff_industry, eff_dealtype, eff_content_type
    relaxed_results = await retrieve_and_rerank(retrieval_q, top_k, relaxed, need_body=need_body)
    relaxed_results = await _temporal_date_fallback(
        relaxed_results, top_k, eff_from, eff_to, relaxed_industry, relaxed_dealtype, author, need_body
    )
    return relaxed_results, relaxed_industry, relaxed_dealtype, relaxed_content_type


# A temporal query whose lexical signal is too weak to pass the chat relevance
# gate (e.g. 'what happened in May 2021', which strips down to 'what happened')
# still needs to surface something. Below this many relevance-passing hits within
# a date window, the window's own recency-sorted articles are used to fill it.
_TEMPORAL_FALLBACK_MIN = 3


async def _temporal_date_fallback(
    results: list[SourceArticle],
    top_k: int,
    from_date: str | None,
    to_date: str | None,
    industry: str | None,
    dealtype: str | None,
    author: str | None,
    need_body: bool,
) -> list[SourceArticle]:
    """When a date-scoped query has too few relevance-passing hits, fill the gap
    with the window's most recent articles (retrieved purely by date, ignoring the
    weak lexical query).

    A temporal query's date window IS the intent, so recency within that window
    is a valid relevance signal even when the words carry no lexical match. Returns
    ``results`` unchanged when no date window is set, when enough hits already
    pass, or when the date window itself is empty."""
    if not (from_date or to_date):
        return results
    strong = [r for r in results if r.score >= config.ASK_MIN_SCORE]
    if len(strong) >= min(_TEMPORAL_FALLBACK_MIN, top_k):
        return results
    date_only = await retrieve_by_date_window(
        top_k, from_date, to_date,
        industry=industry, dealtype=dealtype, author=author, need_body=need_body,
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
    need_body: bool = False,
) -> list[SourceArticle]:
    """Date-scoped retrieval that ignores the lexical query entirely: returns the
    most recent articles published in [from_date, to_date], optionally narrowed by
    category facets. Used as the temporal fallback when lexical matching is too
    weak to surface anything — recency within the window becomes the relevance
    signal. Articles are scored by recency so they clear the chat relevance gate
    and sort newest-first."""
    qfilter = build_facet_filter(industry, dealtype, author, from_date, to_date)
    if qfilter is None:
        return []
    # `published_date` carries a DATETIME payload index, so `order_by` returns the
    # window's most-recent `top_k` articles directly — no full-window
    # materialization (a year-wide window could otherwise page millions of points
    # into memory). This bounds the work to O(top_k) while still selecting by true
    # recency rather than an arbitrary ID-ordered slice.
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
            # Score on a recency-agnostic base; the recency multiplier is applied
            # exactly once in sort_results so merged results share one scale with
            # lexical (cross-encoder) hits instead of double-counting recency.
            # Date-only fillers sit at a modest floor: at/below the chat
            # relevance gate but strictly below the typical lexical
            # (cross-encoder) band -- real relevant hits sigmoid-score well
            # above it -- so they surface without outranking a genuine lexical
            # match (and, via _merge_results' body preference, never drop an
            # article body). The floor is now independent of the chat gate, so
            # it clears that gate only while it is kept at or above
            # config.ASK_MIN_SCORE, which the shipped defaults (both 0.2) do.
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
    # max_length rejects an over-long query with 422 rather than truncating it:
    # a silently truncated query returns results for a query the caller never
    # asked, and the error names the limit so the UI can explain itself. The
    # bound itself is config.SEARCH_QUERY_MAX_CHARS (see config.py).
    q: str = Query(..., min_length=1, max_length=config.SEARCH_QUERY_MAX_CHARS),
    top_k: int = Query(config.TOP_K, ge=1, le=50),
    industry: str | None = Query(None),
    dealtype: str | None = Query(None),
    author: str | None = Query(None),
    content_type: str | None = Query(None),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
):
    start = time.perf_counter()
    q_fixed, _ = fix_query(q)
    retrieval_q, eff_from, eff_to, auto_dealtype, auto_industry = _effective_intent(q_fixed, from_date, to_date)
    auto_content_type = extract_content_type(q_fixed)
    # Auto-extracted category facets fill in only when the caller didn't pass an
    # explicit facet, so the UI filter (and /search callers) always win. The raw
    # explicit params are kept apart so the auto-facet fallback can tell whether
    # a facet was user-supplied (never dropped) or auto-derived (droppable).
    explicit_industry, explicit_dealtype, explicit_content_type = industry, dealtype, content_type
    dealtype = dealtype or auto_dealtype
    industry = industry or auto_industry
    content_type = content_type or auto_content_type
    # NOTE: query expansion happens exactly once inside retrieve_and_rerank ->
    # _retrieval_leg (which chat also uses), so we must NOT expand here too,
    # otherwise /search expands twice and diverges from the chat pipeline.
    eff_top_k = min(max(top_k, suggested_top_k(q) or 0), 50)
    cache_key = (
        f"search:{_cache_key_component(retrieval_q)}:{eff_top_k}:"
        f"{facet_cache_token(industry, dealtype, author, eff_from, eff_to, content_type)}"
    )
    filtered = any((industry, dealtype, author, content_type, from_date, to_date))
    cached_results = await cache.get(cache_key)
    if cached_results is not None:
        summaries = [SourceSummary.model_validate(d) for d in cached_results]
        note = weak_results_note([s.score for s in summaries], date_label(eff_from, eff_to))
        await record_search(q, len(summaries), bool(note), cached=True,
                            latency_ms=(time.perf_counter() - start) * 1000, filtered=filtered)
        return SearchResponse(query=q, results=summaries, cached=True,
                              latency_ms=(time.perf_counter() - start) * 1000, note=note)

    # Rerank on the retrieval query (date words stripped, natural phrasing kept),
    # matching chat which already passes the same query — otherwise the raw phrase
    # with month/year tokens dilutes the cross-encoder and weak scores slip through.
    # The effective industry/dealtype may relax below if an auto facet zeroed the
    # result set; cache/note on the facets actually retrieved.
    reranked, final_industry, final_dealtype, final_content_type = await retrieve_with_auto_facet_fallback(
        retrieval_q, eff_top_k,
        industry=explicit_industry, dealtype=explicit_dealtype, author=author,
        content_type=explicit_content_type,
        eff_from=eff_from, eff_to=eff_to,
        auto_industry=auto_industry, auto_dealtype=auto_dealtype,
        auto_content_type=auto_content_type,
    )
    if config.ENABLE_CLICK_BOOST:
        reranked = await apply_click_boost(q_fixed, reranked)
    if config.ENABLE_DIVERSITY:
        reranked = diversify(reranked, eff_top_k, lam=config.DIVERSITY_LAMBDA,
                             sim_thresh=config.DIVERSITY_SIM_THRESHOLD)
    results = reranked[:eff_top_k]
    note = weak_results_note([r.score for r in results], date_label(eff_from, eff_to))
    # Same empty-set guard as retrieve_and_rerank: an empty result set is never
    # cached, so a transient miss can't be replayed as "no results" for the
    # whole TTL. Non-empty sets are cached as before — except when the auto
    # facet fallback relaxed the effective filter: those results are correct
    # but their cache key (which still names the auto facet) would collide with
    # an explicit-facet request, so the /search cache is skipped for them (the
    # retrieve_and_rerank cache, keyed by the actual filter, still applies).
    fell_back = final_industry != industry or final_dealtype != dealtype or final_content_type != content_type
    if results and not fell_back:
        await cache.set(cache_key, [to_summary(r).model_dump() for r in results])
    # `filtered` (computed above from the effective facets) is used unchanged so
    # the cache-hit and cache-miss paths report the same semantics: the user's
    # query intent carried the facet even when the fallback relaxed it out.
    await record_search(q, len(results), bool(note), cached=False,
                        latency_ms=(time.perf_counter() - start) * 1000, filtered=filtered)
    return SearchResponse(query=q, results=[to_summary(r) for r in results], cached=False,
                          latency_ms=(time.perf_counter() - start) * 1000, note=note)


# Appended to a body excerpt the char cap cut short, so the model can tell the
# text stops here because of the cap rather than because the article ended.
BODY_TRUNCATION_NOTE = "\n[... body truncated ...]"


def source_context(s: SourceArticle, idx: int, body_limit: int | None = None) -> str:
    """Packs an article's metadata + summary + body into a numbered
    context block for the chat LLM prompt. The body excerpt is capped by
    CHAT_BODY_CHAR_LIMIT, or by ``body_limit`` when the caller budgets a fixed
    total across a larger source set (chat scales its source count to 'top N')."""
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
        # Apply the body cap uniformly: an explicit ``body_limit`` always wins
        # (even 0, meaning "no body"), and only when it is None do we fall back
        # to the configured default. Negative values are clamped to 0.
        limit = config.CHAT_BODY_CHAR_LIMIT if body_limit is None else max(0, int(body_limit))
        if limit and len(s.body) > limit:
            parts.append(s.body[:limit] + BODY_TRUNCATION_NOTE)
        elif limit:
            parts.append(s.body)
    return "\n".join(parts)


FACETS_CACHE_KEY = "facets:v1"
FACETS_LIMIT = 200


async def _facet_values(key: str) -> list[str]:
    """Distinct payload values for ``key``, via the public ``scroll`` API.

    Qdrant-client 1.11 does not expose a stable public facet method, so we page
    through the collection (requesting only ``key``) and collect distinct values
    instead of reaching into the client's private HTTP internals. The result is
    explicitly capped at FACETS_LIMIT and cached by the caller. Array-valued
    keyword fields (e.g. industry_names) contribute each element as a distinct
    value.

    NOTE: the cap is intentional and is NOT silently dropping data — facet
    vocabularies here are small (well under FACETS_LIMIT); if the cap is ever hit
    a warning is logged so it can be raised deliberately rather than masking a
    runaway vocabulary.
    """
    values: set[str] = set()
    next_offset = None
    while len(values) < FACETS_LIMIT:
        pts, next_offset = await state["qdrant"].scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=256,
            with_payload=[key],
            with_vectors=False,
            offset=next_offset,
        )
        for p in pts:
            v = (p.payload or {}).get(key)
            if isinstance(v, str):
                if v:
                    values.add(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, str) and item:
                        values.add(item)
        if next_offset is None or not pts:
            # `not pts` guards against a defensive edge case where the client
            # returns an empty page without clearing the offset, which would
            # otherwise loop forever.
            break
    # Explicit cap: if we stopped because the vocabulary hit FACETS_LIMIT (rather
    # than exhausting the collection), flag it — the data is truncated by design.
    if len(values) >= FACETS_LIMIT:
        logger.warning("facet %s hit FACETS_LIMIT=%d; results truncated", key, FACETS_LIMIT)
    return sorted(values)[:FACETS_LIMIT]


@app.get(
    "/facets",
    dependencies=[Depends(public_rate_limit("facets", "PUBLIC_FACETS_RATE_PER_MIN"))],
)
async def facets():
    """Distinct industry_names and dealtype_names values across the collection,
    used for filter autocomplete. Cached in Redis (small controlled vocab)."""
    cached = await cache.get(FACETS_CACHE_KEY)
    if cached is not None:
        return cached

    result = {
        "industry": await _facet_values("industry_names"),
        "dealtype": await _facet_values("dealtype_names"),
    }
    await cache.set(FACETS_CACHE_KEY, result)
    return result


class ClickEvent(BaseModel):
    query: str = ""
    position: int = 0
    id: int | None = None


@app.post(
    "/analytics/click",
    dependencies=[Depends(public_rate_limit("click", "PUBLIC_CLICK_RATE_PER_MIN"))],
)
async def analytics_click(event: ClickEvent):
    """Anonymous result-click beacon from the public search page (no data
    returned, so it stays open to keep collecting interaction analytics). The
    optional ``id`` is the clicked article's feid, used by click-driven learning."""
    await record_click(event.query, event.position, event.id)
    return {"ok": True}


def _analytics_unavailable(message: str) -> JSONResponse:
    """503 for an analytics feed whose store could not be read.

    The status line and the body must agree: a 200 carrying
    ``{"error": ...}`` is indistinguishable from a report whose counters are
    genuinely all zero, which is how a dead analytics store turned into an
    all-zero dashboard behind a healthy-looking status. The ``error`` key is
    kept so a client that only inspects the body can still detect this.
    """
    return JSONResponse(
        status_code=503,
        content={"error": message, "detail": "the analytics store could not be read"},
    )


@app.get("/analytics/summary")
async def get_analytics_summary(
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("analytics:read")),
):
    """Aggregated search/click metrics. Admin-only (analytics:read).

    Answers 503 when the analytics Redis is unreachable, so a degraded read is
    never served as a 200 all-zero report.
    """
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
    """Cross-user chat usage (sessions, messages, tokens, cost). Admin-only.

    Returns cross-user aggregates and per-session rows (opaque session id,
    message count, cost/tokens, updated_at) — no user-authored text is ever
    included. Each read is recorded in the durable admin audit trail; a failure
    to record must not break the read itself. A chat store that cannot be read
    answers 503 rather than a 200 body that looks like an empty store.
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


# =============================================================================
# Recommendation API
# =============================================================================

# Dwell time is stored as a Redis hash VALUE, not a field name or a key, so it
# does not drive key growth -- it is bounded only to keep a single request from
# writing an arbitrary-length string.
MAX_DWELL_TIME_MS = 24 * 60 * 60 * 1000


class InteractionEvent(BaseModel):
    """User article interaction event for personalization.

    ``interaction_type`` is a closed enum, not free-form text: it becomes a
    Redis hash FIELD on ``article:interactions:{id}``, so an unchecked string
    would mint a new unbounded field per call. ``article_id`` is a positive
    integer, capped at the signed 64-bit range, and is verified against the
    article index before any key is written, so a caller cannot mint keys for
    ids that do not exist.
    """

    # ge=1 rejects 0/negative; le caps at int64 because Qdrant point ids are
    # uint64/UUID and a larger value raises a client-side error rather than
    # returning empty, which would turn a reject into a 500.
    article_id: int = Field(..., ge=1, le=2**63 - 1, description="Indexed article id")
    interaction_type: InteractionType = InteractionType.CLICK
    dwell_time_ms: int | None = Field(None, ge=0, le=MAX_DWELL_TIME_MS)


class SimilarArticlesResponse(BaseModel):
    """Response for similar articles endpoint."""
    article_id: int | str
    similar_articles: list[dict]
    limit: int
    cached: bool = False


# Bumped when the cached recommendation shape changed. Entries written by the
# previous shape still carry the full article `body` (up to BODY_CHAR_LIMIT
# chars per article), and every recommend endpoint returns its cached value
# verbatim, so a versioned key is what stops those pre-deploy entries from
# being served for their remaining TTL.
RECOMMEND_CACHE_VERSION = "v2"


class RecommendationsResponse(BaseModel):
    """Response for personalized recommendations."""
    user_id: str
    recommendations: list[dict]
    limit: int
    cold_start: bool = False
    cached: bool = False


class TrendingResponse(BaseModel):
    """Response for trending articles."""
    articles: list[dict]
    limit: int
    window_days: int

# /recommend/for-you caches one entry per distinct `limit`, and `limit` is
# bounded, so a user's entire for-you cache is exactly
# FOR_YOU_MAX_LIMIT - FOR_YOU_MIN_LIMIT + 1 knowable keys. Deriving that set is
# what lets an interaction invalidate it with a single DEL; a prefix delete
# would instead SCAN the whole keyspace, and SCAN ignores MATCH when deciding
# how much work to do, so its cost tracks every key in the database rather than
# the ~20 that match. Keep the two bounds below and the `Query` in get_for_you
# in sync: widening the Query without widening this range would silently leave
# cached entries behind.
FOR_YOU_MIN_LIMIT = 1
FOR_YOU_MAX_LIMIT = 20


def _for_you_cache_key(user_id: str, limit: int) -> str:
    """The one place a /recommend/for-you cache key is spelled.

    The writer in ``get_for_you`` and the invalidator in
    ``record_user_interaction`` must agree on this string exactly. They used to
    be f-strings written out twice, and a change to one silently stopped the
    other from matching, leaving a user's feed served from a stale entry until
    its TTL ran out. Both call this instead.
    """
    return f"recommend:for-you:{user_id}:{RECOMMEND_CACHE_VERSION}:{limit}"


def _for_you_cache_keys(user_id: str) -> list[str]:
    """Every cache key /recommend/for-you can have written for ``user_id``."""
    return [
        _for_you_cache_key(user_id, limit)
        for limit in range(FOR_YOU_MIN_LIMIT, FOR_YOU_MAX_LIMIT + 1)
    ]


@app.post(
    "/recommend/interaction",
    dependencies=[
        # Both axes are required. One shared NAT address defeats the per-IP
        # bucket; one account rotating addresses defeats nothing once the
        # per-account bucket is present. Both go through the same
        # _consume_counter, so neither can drift into weaker enforcement.
        Depends(public_rate_limit("interaction", "PUBLIC_INTERACTION_RATE_PER_MIN")),
        Depends(user_rate_limit("interaction", "INTERACTION_USER_RATE_PER_MIN")),
    ],
)
async def record_user_interaction(
    event: InteractionEvent,
    request: Request,
    _auth: None = Depends(require_auth),
):
    """Record a user-article interaction for personalization.

    Authenticated users only. Logs clicks, views, and reads to build
    user preference profiles for personalized recommendations.

    Outage posture: the limiter is fail-closed, so a limiter-Redis outage
    answers 503 rather than serving an unbounded write path. This is NOT the
    /ready exception -- no load balancer probes this endpoint, so there is no
    health check to protect here. The trade is explicit: a Redis blip stops
    interaction recording for everyone until it clears, costing personalization
    signal, whereas failing open re-opens the amplification the limits close.
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
        # Not in the index, so no key was minted for it.
        raise HTTPException(status_code=404, detail="Unknown article")
    if result is InteractionResult.CAP_REACHED:
        # The article is real; this ACCOUNT has interacted with too many
        # distinct articles. Reporting 404 here would be false.
        raise HTTPException(
            status_code=429,
            detail="Interaction limit reached for this account",
            headers={"Retry-After": str(config.PUBLIC_RATE_WINDOW_SECONDS)},
        )
    if result is InteractionResult.UNAVAILABLE:
        # Redis or the index could not answer. Saying "unknown article" would
        # be a lie about a real article, and the write did not happen.
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
    """Get articles similar to the specified article.

    Uses dense vector similarity from Qdrant with optional category filtering.
    """
    cached_key = f"recommend:similar:{RECOMMEND_CACHE_VERSION}:{article_id}:{limit}:{same_category}"
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


@app.get("/recommend/for-you", response_model=RecommendationsResponse)
async def get_for_you(
    limit: int = Query(config.RECOMMEND_DEFAULT_LIMIT, ge=FOR_YOU_MIN_LIMIT, le=FOR_YOU_MAX_LIMIT),
    _auth: None = Depends(require_auth),
    request: Request = None,
):
    """Get personalized recommendations for the authenticated user.

    Uses user interaction history, category affinity, and hybrid scoring
    to surface relevant articles. A cold-start user -- one who is
    authenticated but has no interaction history -- is served latest top
    stories by ``get_personalized_recommendations`` and flagged via
    ``cold_start`` in the response.

    Every caller reaching this handler has been through ``require_auth``,
    which sets ``request.state.user_id`` to a real user id or the service
    token id, so there is no anonymous case here (#303).
    """
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
    """Get trending/popular articles based on click velocity.

    Queries Redis for recent interaction counts and returns the most
    engaged-with articles from the configured time window.
    """
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
