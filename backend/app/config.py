import os
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv()

# Cap the number of CPU threads torch/onnxruntime use per process BEFORE any
# inference library is imported. With GUNICORN_WORKERS processes sharing the
# box, leaving the default (all cores) oversubscribes the CPU and hurts
# latency under concurrent load. 2 threads per worker is the tuned default.
_TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))


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
    # CPU threads each worker's inference libs may use (torch + onnxruntime).
    # Kept small so GUNICORN_WORKERS processes don't oversubscribe the box.
    TORCH_THREADS = _TORCH_THREADS

    # Indexed text limits. The dense embedder gets title+facets+summary only
    # (kept short so CPU builds stay fast); the sparse/lexical embedder gets the
    # full text including body so body keywords stay searchable. Body-related
    # caps default to 50000 chars, which covers every article currently in the
    # corpus (longest clean body ~44K) with headroom for growth.
    EMBED_DENSE_CHAR_LIMIT = int(os.getenv("EMBED_DENSE_CHAR_LIMIT", "1500"))
    EMBED_CHAR_LIMIT = int(os.getenv("EMBED_CHAR_LIMIT", "50000"))
    BODY_CHAR_LIMIT = int(os.getenv("BODY_CHAR_LIMIT", "50000"))
    # Per-source body excerpt sent to the chat LLM (the whole stored body when
    # this matches BODY_CHAR_LIMIT; lower it to cut prompt tokens/cost).
    CHAT_BODY_CHAR_LIMIT = int(os.getenv("CHAT_BODY_CHAR_LIMIT", "50000"))
    # Chat dynamically scales the source count to the query's requested 'top N'
    # (capped here so the LLM context stays bounded) and trims each source's
    # body excerpt to fit the total budget below, so asking for more deals never
    # balloons the prompt size. 400000 matches today's 8 sources x 50K bodies.
    CHAT_MAX_SOURCES = int(os.getenv("CHAT_MAX_SOURCES", "20"))
    CHAT_TOTAL_BODY_CHARS = int(os.getenv("CHAT_TOTAL_BODY_CHARS", "400000"))

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
    # Reranker execution backend: 'onnx' (optimum/onnxruntime, ~2-3x faster on
    # CPU, exported once at startup) or 'torch' (sentence-transformers). Falls
    # back to 'torch' automatically when optimum is not installed.
    RERANK_BACKEND = os.getenv("RERANK_BACKEND", "onnx")
    # Local dir where the ONNX cross-encoder is exported on first use and
    # reloaded on later startups (relative paths resolve against the backend
    # working dir, where gunicorn runs).
    RERANK_ONNX_DIR = os.getenv("RERANK_ONNX_DIR", "data/reranker_onnx")

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
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    # TTL for cached query (dense+sparse) vectors, keyed by the embedding model
    # so a model change invalidates them automatically. Long is safe: the pair
    # for a given query string is deterministic and stable for a fixed index.
    VECTOR_CACHE_TTL_SECONDS = int(os.getenv("VECTOR_CACHE_TTL_SECONDS", "86400"))

    # Recency-tempered ranking: scores are multiplied by
    # 1 - RECENCY_STRENGTH * (1 - exp(-age_days / RECENCY_DECAY_DAYS))
    # so fresher news ranks higher; missing dates get no boost.
    RECENCY_STRENGTH = float(os.getenv("RECENCY_STRENGTH", "0.25"))
    RECENCY_DECAY_DAYS = float(os.getenv("RECENCY_DECAY_DAYS", "90"))

    # Retrieval-quality tuning (see app/query_expand.py, app/rerank_boost.py,
    # app/answer_fallback.py, app/query_fix.py). Toggles can be disabled per-deployment.
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() in ("1", "true", "yes")
    ENABLE_ENTITY_BOOST = os.getenv("ENABLE_ENTITY_BOOST", "true").lower() in ("1", "true", "yes")
    ENABLE_WEAK_FALLBACK = os.getenv("ENABLE_WEAK_FALLBACK", "true").lower() in ("1", "true", "yes")

    # Query-string typo correction (app/query_fix.py): symspellpy over a
    # corpus-derived vocabulary + curated entities, applied before embedding.
    # The vocab is generated by scripts/build_query_vocab.py; when absent the
    # fixer is a no-op. Corrected strings also normalize the cache keys, so
    # repeated typos of the same query reuse the same cached results.
    ENABLE_QUERY_FIX = os.getenv("ENABLE_QUERY_FIX", "true").lower() in ("1", "true", "yes")
    QUERY_FIX_VOCAB_PATH = os.getenv("QUERY_FIX_VOCAB_PATH", "data/query_vocab.json.gz")
    QUERY_FIX_MAX_EDIT = int(os.getenv("QUERY_FIX_MAX_EDIT", "2"))
    QUERY_FIX_MIN_COUNT = int(os.getenv("QUERY_FIX_MIN_COUNT", "5"))
    QUERY_FIX_MIN_TOKEN_LEN = int(os.getenv("QUERY_FIX_MIN_TOKEN_LEN", "3"))

    # Result diversity (app/diversity.py): greedy MMR over the reranked set to
    # avoid near-duplicate headlines filling the top-k. LAMBDA near 1 favours
    # pure relevance; lower trades relevance for headline diversity. Applied in
    # /search before the final top-k slice.
    ENABLE_DIVERSITY = os.getenv("ENABLE_DIVERSITY", "true").lower() in ("1", "true", "yes")
    DIVERSITY_LAMBDA = float(os.getenv("DIVERSITY_LAMBDA", "0.7"))
    DIVERSITY_SIM_THRESHOLD = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.4"))

    # Click-driven learning (app/click_boost.py): per-query per-article click
    # aggregates (analytics Redis) boost results users actually open. Inert until
    # a query accumulates >= CLICK_BOOST_MIN_CLICKS clicks and an article holds
    # >= CLICK_BOOST_MIN_ARTICLE_CLICKS clicks (>= CLICK_BOOST_MIN_SHARE of the
    # query's total), so it never fires on sparse/noisy traffic.
    ENABLE_CLICK_BOOST = os.getenv("ENABLE_CLICK_BOOST", "true").lower() in ("1", "true", "yes")
    CLICK_BOOST_MIN_CLICKS = int(os.getenv("CLICK_BOOST_MIN_CLICKS", "5"))
    CLICK_BOOST_MIN_ARTICLE_CLICKS = int(os.getenv("CLICK_BOOST_MIN_ARTICLE_CLICKS", "3"))
    CLICK_BOOST_MIN_SHARE = float(os.getenv("CLICK_BOOST_MIN_SHARE", "0.3"))
    CLICK_BOOST_MULT = float(os.getenv("CLICK_BOOST_MULT", "1.3"))

    # Chat-only "body rescue": when the top reranked score is below
    # BODY_RESCUE_THRESHOLD, re-score the candidates against the body region
    # with the most lexical query-token overlap and keep max(baseline, body).
    # Lets deep-body matches (e.g. historical retrospectives whose relevant
    # facts live mid-article) pass the chat relevance gate; costs one extra
    # cross-encoder pass per candidate and only runs on weak-top results.
    ENABLE_BODY_RESCUE = os.getenv("ENABLE_BODY_RESCUE", "true").lower() in ("1", "true", "yes")
    BODY_RESCUE_THRESHOLD = float(os.getenv("BODY_RESCUE_THRESHOLD", "0.3"))
    BODY_RESCUE_WINDOW = int(os.getenv("BODY_RESCUE_WINDOW", "1500"))
    BODY_RESCUE_STEP = int(os.getenv("BODY_RESCUE_STEP", "500"))

    # Chat history (SQLite on the host; survives restarts, unlike Redis without AOF)
    # Relative CHAT_DB_PATH resolves against the backend working dir (where
    # gunicorn runs). Retention purges conversations idle for CHAT_RETENTION_DAYS.
    CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat.db")
    CHAT_RETENTION_DAYS = int(os.getenv("CHAT_RETENTION_DAYS", "180"))
    CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
    CHAT_PURGE_INTERVAL_SECONDS = int(os.getenv("CHAT_PURGE_INTERVAL_SECONDS", "86400"))

    # Analytics
    # Aggregates live in Redis DB 1 (the query cache uses DB 0 and is flushed
    # during deploys). Read endpoints are gated by the auth layer (admin role).
    ANALYTICS_REDIS_DB = int(os.getenv("ANALYTICS_REDIS_DB", "1"))

    # Auth (token + RBAC). Users sign up openly; a role-based access-control
    # layer maps roles to permissions (see app/auth.py). Tokens are opaque,
    # hashed (SHA-256) in storage, expire after AUTH_TOKEN_TTL_DAYS, and can be
    # revoked individually.
    AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "data/auth.db")
    AUTH_TOKEN_TTL_DAYS = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "7"))
    # Default role granted to new accounts (public signups land in 'user').
    AUTH_DEFAULT_ROLE = os.getenv("AUTH_DEFAULT_ROLE", "user")
    # Optional machine-to-machine bypass: any request carrying this exact value
    # in X-Service-Token acts as an admin user. Leave empty to disable. Used by
    # the internal eval scripts; never expose it to browsers.
    AUTH_SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")
    # Bootstrap admin: created once at startup (role=admin) if no account with
    # this email exists. An existing account is never overwritten.
    AUTH_ADMIN_EMAIL = os.getenv("AUTH_ADMIN_EMAIL", "")
    AUTH_ADMIN_PASSWORD = os.getenv("AUTH_ADMIN_PASSWORD", "")
    # Input-validation limits for the auth endpoints.
    AUTH_PASSWORD_MIN_LEN = int(os.getenv("AUTH_PASSWORD_MIN_LEN", "8"))
    AUTH_MAX_EMAIL_LEN = int(os.getenv("AUTH_MAX_EMAIL_LEN", "254"))
    AUTH_MAX_NAME_LEN = int(os.getenv("AUTH_MAX_NAME_LEN", "60"))
    # Redis-backed per-IP rate limits on the public auth endpoints (0 disables).
    AUTH_SIGNUP_RATE_PER_MIN = int(os.getenv("AUTH_SIGNUP_RATE_PER_MIN", "5"))
    AUTH_LOGIN_RATE_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_MIN", "10"))
    AUTH_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))

    # CORS: comma-separated allowed origins. Production serves the API and the
    # frontend same-origin through nginx, so this only matters for cross-origin
    # dev clients (e.g. the Next.js dev server on :3000 hitting :8000).
    CORS_ORIGINS: ClassVar[list[str]] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
        if o.strip()
    ]


config = Config()
