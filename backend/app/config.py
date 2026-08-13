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

    # Embeddings
    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    EMBED_DIM = 384  # matches bge-small; change if you swap models
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

    # Sparse (BM25) embeddings — must match the model used at index time
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # LLM (Groq, OpenAI-compatible API)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Search
    TOP_K = int(os.getenv("TOP_K", "8"))
    # Minimum dense cosine similarity for /ask retrieval; filters the dense
    # prefetch so weak results don't reach the LLM.
    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))


config = Config()
