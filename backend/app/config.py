import os
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv()


class Config:
    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "vccircle")
    MYSQL_TABLE = os.getenv("MYSQL_TABLE", "articles")

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vccircle_articles")

    # Cache
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Embeddings
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    EMBED_DIM = 768  # matches bge-base; change if you swap models
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

    # Indexed text limits. The dense embedder gets title+facets+summary only
    # (kept short so CPU builds stay fast); the sparse/lexical embedder gets the
    # full text including body so body keywords stay searchable.
    EMBED_DENSE_CHAR_LIMIT = int(os.getenv("EMBED_DENSE_CHAR_LIMIT", "1500"))
    EMBED_CHAR_LIMIT = int(os.getenv("EMBED_CHAR_LIMIT", "6000"))
    BODY_CHAR_LIMIT = int(os.getenv("BODY_CHAR_LIMIT", "6000"))

    # In-flight encode batches during indexing. Keep this small: CPU dense
    # encoding of a batch near max-token length uses ~1-2GB, so depth * batch
    # must fit in RAM (the pipeline's value is overlapping encode with upsert,
    # not running many encodes in parallel).
    INDEXER_WORKERS = int(os.getenv("INDEXER_WORKERS", "2"))

    # Sparse (BM25) embeddings — must match the model used at index time
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # Reranker (cross-encoder) applied to RRF candidates before the top_k is kept.
    # Fewer candidates = faster CPU rerank; 12 keeps top-8 quality vs 16 while
    # trimming latency (measured 8/8 overlap on representative queries).
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))

    # LLM (Google Gemini via OpenAI-compatible endpoint). Provide the API key
    # in GEMINI_API_KEY. Set GEMINI_MODEL to the model id you want to use.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    # Per-call timeout and retry policy for the LLM (see app/llm.py).
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.0"))
    # Pricing in USD per 1M tokens, used by LLMResult.cost() for cost tracking.
    # Defaults approximate Google Gemini 3.1 Flash Lite rates.
    LLM_PRICE_INPUT_PER_1M = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.25"))
    LLM_PRICE_OUTPUT_PER_1M = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "1.50"))
    # Conversion for displaying cost in Indian Rupees (INR). Approx market rate.
    INR_PER_USD = float(os.getenv("INR_PER_USD", "95.60"))
    # Daily LLM spend cap in USD; 0 disables it. Chat fails closed (no LLM calls)
    # once today's cumulative spend reaches this value (see app/cost_budget.py).
    LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "0"))

    # Search
    TOP_K = int(os.getenv("TOP_K", "8"))
    # Minimum reranked relevance score for chat sources; weaker results are
    # dropped before the LLM sees them.
    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "120"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

    # Recency-tempered ranking: scores are multiplied by
    # 1 - RECENCY_STRENGTH * (1 - exp(-age_days / RECENCY_DECAY_DAYS))
    # so fresher news ranks higher; missing dates get no boost.
    RECENCY_STRENGTH = float(os.getenv("RECENCY_STRENGTH", "0.25"))
    RECENCY_DECAY_DAYS = float(os.getenv("RECENCY_DECAY_DAYS", "90"))

    # Retrieval-quality tuning (see app/query_expand.py, app/rerank_boost.py,
    # app/answer_fallback.py). Toggles can be disabled per-deployment.
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in ("1", "true", "yes")
    ENABLE_ENTITY_BOOST = os.getenv("ENABLE_ENTITY_BOOST", "true").lower() in ("1", "true", "yes")
    ENABLE_WEAK_FALLBACK = os.getenv("ENABLE_WEAK_FALLBACK", "true").lower() in ("1", "true", "yes")

    # Chat history (SQLite on the host; survives restarts, unlike Redis without AOF)
    # Relative CHAT_DB_PATH resolves against the backend working dir (where
    # gunicorn runs). Retention purges conversations idle for CHAT_RETENTION_DAYS.
    CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat.db")
    CHAT_RETENTION_DAYS = int(os.getenv("CHAT_RETENTION_DAYS", "180"))
    CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
    CHAT_PURGE_INTERVAL_SECONDS = int(os.getenv("CHAT_PURGE_INTERVAL_SECONDS", "86400"))

    # Analytics
    # Aggregates live in Redis DB 1 (the query cache uses DB 0 and is flushed
    # during deploys). When a view token is set, /analytics/summary requires it.
    ANALYTICS_REDIS_DB = int(os.getenv("ANALYTICS_REDIS_DB", "1"))
    ANALYTICS_VIEW_TOKEN = os.getenv("ANALYTICS_VIEW_TOKEN", "")

    # CORS: comma-separated allowed origins. Production serves the API and the
    # frontend same-origin through nginx, so this only matters for cross-origin
    # dev clients (e.g. the Next.js dev server on :3000 hitting :8000).
    CORS_ORIGINS: ClassVar[list[str]] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
        if o.strip()
    ]


config = Config()
