import asyncio
import json
import logging
import math
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Query
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

# Import config FIRST so the OMP/MKL thread caps in app.config are set before
# any inference library (torch/onnxruntime) is imported below.
from app import auth as auth_module
from app import chat as chat_module
from app.analytics import close as close_analytics
from app.analytics import record_click, record_search
from app.analytics import summary as analytics_data
from app.answer_fallback import date_label, weak_results_note
from app.auth import require_auth, require_permission
from app.click_boost import apply_click_boost
from app.config import config
from app.cost_budget import close as close_cost_budget
from app.diversity import diversify
from app.encoders import DenseEncoder
from app.health import close_redis as health_module_close_redis
from app.health import router as health_router
from app.query_expand import expand_query
from app.query_fix import fix_query, init_fixer
from app.query_intent import (
    extract_list_topic,
    extract_year_range,
    normalize_word_numbers,
    range_query_topic,
    rewrite_year_in_review,
    suggested_top_k,
)
from app.redis_cache import cache
from app.rerank_boost import apply_entity_boost
from app.reranker import Reranker

state = {}

# Serializes CPU-bound inference (dense encode, sparse embed, rerank) per
# worker so concurrent requests don't contend for the CPU and thrash torch /
# onnxruntime thread pools. Async I/O (Qdrant/Redis) is unaffected.
inference_lock = asyncio.Lock()

logger = logging.getLogger(__name__)

# Live facet vocabularies loaded once at startup from Qdrant (normalized
# lowercased value -> original-cased value). Category extraction only ever emits
# a value present in these maps, so a natural-language query like 'funding news'
# maps to the real 'Venture Capital' facet instead of guessing a label that would
# match nothing (an unknown filter value returns zero results and breaks the
# query). The actual labels (e.g. 'Venture Capital', 'M&A', 'Finance') come from
# the live index, not from hard-coded assumptions.
_DEALTYPE_FACETS: dict[str, str] = {}
_INDUSTRY_FACETS: dict[str, str] = {}

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


async def _load_facet_maps() -> None:
    """Populate the live dealtype/industry facet maps from Qdrant so category
    extraction emits only real facet values. Failures leave the maps empty, which
    makes extraction a no-op (current behavior) — startup never depends on this."""
    for target, key in ((_DEALTYPE_FACETS, "dealtype_names"), (_INDUSTRY_FACETS, "industry_names")):
        try:
            values = await _facet_values(key)
        except Exception:  # noqa: BLE001 - degraded mode, never crash startup
            logger.warning("facet map load failed for %s", key)
            continue
        target.clear()
        for v in values:
            target[v.strip().lower()] = v


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = DenseEncoder(config.EMBED_MODEL, config.EMBED_DEVICE, config.TORCH_THREADS)
    state["sparse_model"] = SparseTextEmbedding(config.SPARSE_MODEL)
    state["reranker"] = Reranker(config.RERANK_MODEL, backend=config.RERANK_BACKEND)
    state["qdrant"] = AsyncQdrantClient(url=config.QDRANT_URL, timeout=30)
    await _load_facet_maps()
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


app = FastAPI(title="VCCircle New Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

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
    resolves) still scopes results."""
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
    return retrieval_q, from_date, to_date, dealtype, industry


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
    articles (e.g. 'venture debt providers', 'unicorns created'). For
    month-scoped queries the bare topic is used directly (date filter scopes the
    month). Otherwise a single query."""
    flashback, changed = rewrite_year_in_review(q)
    if changed:
        topic = extract_list_topic(q) or q
        return list(dict.fromkeys([flashback, topic]))
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
    vec_key = f"vec:{config.EMBED_MODEL}|{config.SPARSE_MODEL}:{query}"
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
            score=p.score,
        )
        for p in result.points
        # Skip points with a null payload (shouldn't happen, but a malformed
        # point would otherwise raise AttributeError on p.payload.get).
        if (payload := p.payload or {}) is not None
    ]


async def rerank(query: str, results: list[SourceArticle]) -> list[SourceArticle]:
    """Cross-encoder rerank of RRF candidates, in place. Rewrites score with a
    sigmoid-normalized relevance score (0-1) so both ordering and the score the
    frontend shows reflect reranked relevance."""
    if len(results) <= 1:
        return results
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


def _best_body_window(body: str, tokens: set[str], win: int, step: int) -> str:
    """The body region with the most distinct query tokens, cheaply located by
    sliding a window over the lowercased body. Returns the window with the
    original casing (falls back to the tail region on ties)."""
    if not tokens or len(body) <= win:
        return body
    low = body.lower()
    best_score, best_start = -1, 0
    for start in range(0, len(body) - win + 1, step):
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
    Costs one extra cross-encoder pass per candidate and only runs on weak
    results, so normal queries are unaffected."""
    if not articles:
        return articles
    if max((a.score for a in articles), default=0.0) >= config.BODY_RESCUE_THRESHOLD:
        return articles
    tokens = _query_content_tokens(query)
    if not tokens:
        return articles
    pairs: list[tuple[str, str]] = []
    indices: list[int] = []
    for i, a in enumerate(articles):
        if not a.body:
            continue
        win = _best_body_window(a.body, tokens, config.BODY_RESCUE_WINDOW, config.BODY_RESCUE_STEP)
        pairs.append((query, f"{a.title}. {a.summary or ''}. {win}".strip()))
        indices.append(i)
    if not pairs:
        return articles
    async with inference_lock:
        logits = await asyncio.to_thread(state["reranker"].predict, pairs)
    for i, logit in zip(indices, logits):
        articles[i].score = max(articles[i].score, float(1 / (1 + math.exp(-logit))))
    articles.sort(key=lambda a: a.score, reverse=True)
    return articles


def _recency_multiplier(published_date: str | None) -> float:
    """1 - STRENGTH * (1 - exp(-age_days / DECAY)); no boost for missing dates."""
    dt = _parse_date(published_date)
    if dt is None:
        return 1.0
    age_days = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 86400.0)
    return 1.0 - config.RECENCY_STRENGTH * (1.0 - math.exp(-age_days / config.RECENCY_DECAY_DAYS))


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


def sort_results(results: list[SourceArticle]) -> list[SourceArticle]:
    """Recency-tempered relevance first, recency second: blended score desc,
    then published_date desc (missing dates last)."""
    results.sort(
        key=lambda a: (a.score * _recency_multiplier(a.published_date), _tz_stripped_pub(a.published_date)),
        reverse=True,
    )
    return results


def _filter_token(qfilter: Filter | None) -> str:
    """Deterministic cache key fragment for a Qdrant filter."""
    if qfilter is None:
        return ""
    return json.dumps(qfilter.model_dump(), sort_keys=True, default=str)


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
    bodies = {p.id: (p.payload or {}).get("body", "") for p in resp}
    for a in articles:
        a.body = bodies.get(a.id, "")


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
    cache_key = f"retrieve:{q}:{top_k}:{_filter_token(qfilter)}"
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
    reranked = sort_results(reranked)
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
    eff_from: str | None,
    eff_to: str | None,
    auto_industry: str | None,
    auto_dealtype: str | None,
    need_body: bool = False,
) -> tuple[list[SourceArticle], str | None, str | None]:
    """Retrieve with the effective (explicit-or-auto) category facets applied,
    and fall back to dropping an *auto* facet when it zeroes out an otherwise
    valid query. Returns ``(results, final_industry, final_dealtype)`` where the
    final facets reflect any fallback, so callers can key caches/notes on what
    was actually retrieved.

    ``industry``/``dealtype`` are the caller-supplied (explicit) facets; when one
    is None the matching ``auto_*`` value is used instead. An auto facet is a
    *semantic* guess mapped onto the corpus's tag vocabulary (e.g. 'edtech' ->
    the 'Education' industry facet). When the corpus tags most of those articles
    differently (VCCircle tags edtech articles 'TMT', not 'Education'), the
    exact-match industry filter combined with a date window returns nothing and
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
    qfilter = build_facet_filter(eff_industry, eff_dealtype, author, eff_from, eff_to)
    results = await retrieve_and_rerank(retrieval_q, top_k, qfilter, need_body=need_body)
    if results or not (auto_industry or auto_dealtype):
        return results, eff_industry, eff_dealtype
    # An auto facet zeroed the set: drop each auto facet that wasn't explicitly
    # supplied (an explicit facet is never dropped) and retry once.
    relaxed_industry = eff_industry if not (auto_industry and industry is None) else None
    relaxed_dealtype = eff_dealtype if not (auto_dealtype and dealtype is None) else None
    relaxed = build_facet_filter(relaxed_industry, relaxed_dealtype, author, eff_from, eff_to)
    if relaxed == qfilter:
        return results, eff_industry, eff_dealtype
    relaxed_results = await retrieve_and_rerank(retrieval_q, top_k, relaxed, need_body=need_body)
    return relaxed_results, relaxed_industry, relaxed_dealtype


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
    q_fixed, _ = fix_query(q)
    retrieval_q, eff_from, eff_to, auto_dealtype, auto_industry = _effective_intent(q_fixed, from_date, to_date)
    # Auto-extracted category facets fill in only when the caller didn't pass an
    # explicit facet, so the UI filter (and /search callers) always win. The raw
    # explicit params are kept apart so the auto-facet fallback can tell whether
    # a facet was user-supplied (never dropped) or auto-derived (droppable).
    explicit_industry, explicit_dealtype = industry, dealtype
    dealtype = dealtype or auto_dealtype
    industry = industry or auto_industry
    # NOTE: query expansion happens exactly once inside retrieve_and_rerank ->
    # _retrieval_leg (which chat also uses), so we must NOT expand here too,
    # otherwise /search expands twice and diverges from the chat pipeline.
    eff_top_k = min(max(top_k, suggested_top_k(q) or 0), 50)
    cache_key = f"search:{retrieval_q}:{eff_top_k}:{facet_cache_token(industry, dealtype, author, eff_from, eff_to)}"
    filtered = any((industry, dealtype, author, from_date, to_date))
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
    reranked, final_industry, final_dealtype = await retrieve_with_auto_facet_fallback(
        retrieval_q, eff_top_k,
        industry=explicit_industry, dealtype=explicit_dealtype, author=author,
        eff_from=eff_from, eff_to=eff_to,
        auto_industry=auto_industry, auto_dealtype=auto_dealtype,
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
    fell_back = final_industry != industry or final_dealtype != dealtype
    if results and not fell_back:
        await cache.set(cache_key, [to_summary(r).model_dump() for r in results])
    # `filtered` (computed above from the effective facets) is used unchanged so
    # the cache-hit and cache-miss paths report the same semantics: the user's
    # query intent carried the facet even when the fallback relaxed it out.
    await record_search(q, len(results), bool(note), cached=False,
                        latency_ms=(time.perf_counter() - start) * 1000, filtered=filtered)
    return SearchResponse(query=q, results=[to_summary(r) for r in results], cached=False,
                          latency_ms=(time.perf_counter() - start) * 1000, note=note)


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
        parts.append(s.body[: min(limit, len(s.body))])
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


@app.get("/facets")
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


@app.post("/analytics/click")
async def analytics_click(event: ClickEvent):
    """Anonymous result-click beacon from the public search page (no data
    returned, so it stays open to keep collecting interaction analytics). The
    optional ``id`` is the clicked article's feid, used by click-driven learning."""
    await record_click(event.query, event.position, event.id)
    return {"ok": True}


@app.get("/analytics/summary")
async def get_analytics_summary(
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("analytics:read")),
):
    """Aggregated search/click metrics. Admin-only (analytics:read)."""
    return await analytics_data()


@app.get("/analytics/chat")
async def get_analytics_chat(
    _auth: None = Depends(require_auth),
    _perm: None = Depends(require_permission("analytics:read")),
):
    """Cross-user chat usage (sessions, messages, tokens, cost). Admin-only."""
    return await chat_module._require_store().global_stats()
