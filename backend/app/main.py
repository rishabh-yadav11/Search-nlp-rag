import asyncio
import time
from contextlib import asynccontextmanager
from functools import partial

from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, Prefetch, FusionQuery, Fusion
from sentence_transformers import SentenceTransformer

from app.config import config

state = {}
cache: TTLCache = TTLCache(maxsize=config.CACHE_MAX_SIZE, ttl=config.CACHE_TTL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    state["sparse_model"] = SparseTextEmbedding(config.SPARSE_MODEL)
    state["qdrant"] = AsyncQdrantClient(url=config.QDRANT_URL, timeout=30)
    state["llm"] = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL) if config.GROQ_API_KEY else None
    yield
    await state["qdrant"].close()


app = FastAPI(title="VCCircle Semantic Search POC", lifespan=lifespan)

# POC-only: wide open for local frontend on :3000. Restrict origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class SourceArticle(BaseModel):
    id: int
    title: str
    url: str
    published_date: str | None = None
    category: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SourceArticle]
    cached: bool
    latency_ms: float


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceArticle]
    cached: bool
    latency_ms: float


async def hybrid_search(query: str, top_k: int, min_dense_score: float | None = None) -> list[SourceArticle]:
    # Sentence-transformers and fastembed are CPU/sync-bound: run them off the
    # event loop so the async handlers stay responsive under load.
    dense_vec = (await asyncio.to_thread(partial(state["model"].encode, query, normalize_embeddings=True))).tolist()
    sparse_emb = list(await asyncio.to_thread(state["sparse_model"].embed, [query]))[0]
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
            score=p.score,
        )
        for p in result.points
    ]


@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1), top_k: int = Query(config.TOP_K, ge=1, le=50)):
    start = time.perf_counter()
    cache_key = f"search:{q}:{top_k}"
    if cache_key in cache:
        results = cache[cache_key]
        return SearchResponse(query=q, results=results, cached=True, latency_ms=(time.perf_counter() - start) * 1000)

    results = await hybrid_search(q, top_k)
    cache[cache_key] = results
    return SearchResponse(query=q, results=results, cached=False, latency_ms=(time.perf_counter() - start) * 1000)


ANSWER_PROMPT = """You are answering a question using ONLY the numbered articles below. \
Cite the article number(s) for every factual claim, like [1] or [2][3]. \
If the articles don't contain enough information to answer, say so plainly instead of guessing.

Articles:
{context}

Question: {question}

Answer (with inline [n] citations):"""


@app.get("/ask", response_model=AskResponse)
async def ask(q: str = Query(..., min_length=1), top_k: int = Query(config.TOP_K, ge=1, le=20)):
    if state["llm"] is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    start = time.perf_counter()
    cache_key = f"ask:{q}:{top_k}"
    if cache_key in cache:
        answer, sources = cache[cache_key]
        return AskResponse(query=q, answer=answer, sources=sources, cached=True,
                            latency_ms=(time.perf_counter() - start) * 1000)

    sources = await hybrid_search(q, top_k, min_dense_score=config.ASK_MIN_SCORE)
    if not sources:
        answer = "No sufficiently relevant articles were found for this query."
        cache[cache_key] = (answer, sources)
        return AskResponse(query=q, answer=answer, sources=sources, cached=False,
                            latency_ms=(time.perf_counter() - start) * 1000)

    context = "\n".join(f"[{i+1}] {s.title} ({s.published_date or 'n/a'})" for i, s in enumerate(sources))
    prompt = ANSWER_PROMPT.format(context=context, question=q)

    response = await state["llm"].chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content or ""

    cache[cache_key] = (answer, sources)
    return AskResponse(query=q, answer=answer, sources=sources, cached=False,
                        latency_ms=(time.perf_counter() - start) * 1000)


@app.get("/health")
async def health():
    return {"status": "ok"}
