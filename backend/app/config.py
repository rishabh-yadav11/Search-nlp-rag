import os
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
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "512"))
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

    # Indexed text limits. compose_index_text() orders metadata (title, authors,
    # industry, dealtype) BEFORE summary/body so the facet values survive the
    # embedder's 512-token truncation.
    EMBED_CHAR_LIMIT = int(os.getenv("EMBED_CHAR_LIMIT", "6000"))  # chars sent to the embedder
    BODY_CHAR_LIMIT = int(os.getenv("BODY_CHAR_LIMIT", "6000"))    # body chars kept in the payload

    # Indexer worker threads (dense encode + upsert pipelined across batches)
    INDEXER_WORKERS = int(os.getenv("INDEXER_WORKERS", "8"))

    # Sparse (BM25) embeddings — must match the model used at index time
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # Reranker (cross-encoder) applied to RRF candidates before the top_k is kept
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "16"))

    # LLM (Groq, OpenAI-compatible API)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Search
    TOP_K = int(os.getenv("TOP_K", "8"))
    # Minimum dense cosine similarity for /ask retrieval; filters the dense
    # prefetch so weak results don't reach the LLM.
    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "120"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

    # Recency-tempered ranking: scores are multiplied by
    # 1 - RECENCY_STRENGTH * (1 - exp(-age_days / RECENCY_DECAY_DAYS))
    # so fresher news ranks higher; missing dates get no boost.
    RECENCY_STRENGTH = float(os.getenv("RECENCY_STRENGTH", "0.25"))
    RECENCY_DECAY_DAYS = float(os.getenv("RECENCY_DECAY_DAYS", "90"))


config = Config()
