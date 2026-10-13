import logging
import os
import re
import socket
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()






BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _data_path(name: str, default: str) -> str:

    raw = os.getenv(name, default).strip().strip("\"'").strip()
    if not raw:
        raise ValueError(
            f"{name} is set but empty; it must be a filesystem path "
            f"(relative paths resolve against {BACKEND_ROOT})"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return str(path)






_TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))





_DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "testserver")


def _normalize_host(entry: str) -> str:

    host = entry.strip().lower().split("://", 1)[-1]
    return host.partition(":")[0]


def _is_ipv6_literal(entry: str) -> bool:

    host = entry.strip().lower().split("://", 1)[-1]
    if host.startswith("["):
        return True
    return host.count(":") > 1


def _machine_hosts() -> tuple[str, ...]:

    hosts: list[str] = []
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            name = getter()
        except (OSError, ValueError):




            continue
        if name:
            hosts.append(name)
    try:
        hosts.extend({info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)})
    except (OSError, ValueError):
        pass
    hosts.extend(_default_route_addresses())



    return tuple(h for h in dict.fromkeys(hosts) if h and not _is_ipv6_literal(h))


def _default_route_addresses() -> tuple[str, ...]:

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 53))
            return (probe.getsockname()[0],)
    except Exception:






        logger.debug("default-route probe failed", exc_info=True)
        return ()



def _clamped_int(name: str, default: int, low: int, high: int) -> int:




    if not low <= default <= high:
        raise ValueError(f"{name} default {default} outside [{low}, {high}]")
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %d", name, raw, default)
        return default
    if value < low or value > high:
        clamped = max(low, min(high, value))
        logger.warning(
            "%s=%d is outside [%d, %d]; clamped to %d", name, value, low, high, clamped
        )
        return clamped
    return value


def _parse_allowed_hosts(raw: str | None, extra_hosts: tuple[str, ...] = ()) -> tuple[str, ...]:

    if raw is None or not raw.strip():



        usable = [h for h in extra_hosts if not _is_ipv6_literal(h)]
        derived = _DEFAULT_ALLOWED_HOSTS + tuple(_normalize_host(h) for h in usable)
        return tuple(dict.fromkeys(h for h in derived if h))

    hosts: list[str] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry == "*":
            raise ValueError(
                "ALLOWED_HOSTS=* would disable the Host header check entirely; "
                "list the hostnames the API is reachable as instead"
            )
        if "*" in entry and not (entry.startswith("*.") and "*" not in entry[2:]):
            raise ValueError(
                f"ALLOWED_HOSTS entry {entry!r} is not a valid wildcard pattern; "
                "only a leading '*.' (as in '*.example.com') is supported"
            )
        if _is_ipv6_literal(entry):




            raise ValueError(
                f"ALLOWED_HOSTS entry {entry!r} is an IPv6 literal, which this Host "
                "check cannot match; TrustedHostMiddleware compares the authority "
                "up to its first colon, so serving an IPv6-only name needs a "
                "different middleware"
            )
        host = _normalize_host(entry)
        if not host:
            raise ValueError(f"ALLOWED_HOSTS entry {entry!r} is not a usable hostname")
        hosts.append(host)
    if not hosts:
        raise ValueError("ALLOWED_HOSTS is set but contains no usable hostname")
    return tuple(dict.fromkeys(hosts))













_TRUE_SPELLINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_SPELLINGS = frozenset({"0", "false", "no", "off"})


def _env_tristate(name: str) -> bool | None:

    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE_SPELLINGS:
        return True
    if raw in _FALSE_SPELLINGS:
        return False
    return None


def _env_bool(name: str, default: bool) -> bool:

    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_SPELLINGS:
        return True
    if value in _FALSE_SPELLINGS:
        return False
    accepted = "/".join(sorted(_TRUE_SPELLINGS | _FALSE_SPELLINGS))
    if not value:


        logger.warning(
            "%s is set but blank, which reads as OFF; set it to one of %s to "
            "choose, or delete the line to use the default (%s)",
            name,
            accepted,
            str(default).lower(),
        )
        return False
    logger.warning(
        "%s=%r is not a boolean; using default %s (accepted: %s)",
        name,
        raw,
        str(default).lower(),
        accepted,
    )
    return default


def _ensure_data_dir(path: str, env_var: str) -> None:

    parent = Path(path).parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"{env_var}={path!r} cannot be used: its directory {parent} could not be "
            f"created ({exc.strerror or exc}). Point {env_var} at a writable location."
        ) from exc
    if not parent.is_dir():  # pragma: no cover - mkdir(exist_ok=True) raises first
        raise RuntimeError(
            f"{env_var}={path!r} cannot be used: {parent} is not a directory."
        )
    if not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(
            f"{env_var}={path!r} cannot be used: directory {parent} is not writable. "
            f"Starting anyway would create an EMPTY database there and silently lose "
            f"every existing row, so the app refuses to start."
        )


def ensure_data_paths_ready(cfg: "Config") -> None:

    for env_var, path in (
        ("CHAT_DB_PATH", cfg.CHAT_DB_PATH),
        ("AUTH_DB_PATH", cfg.AUTH_DB_PATH),
        ("FEED_DB_PATH", cfg.FEED_DB_PATH),
        ("QUERY_FIX_VOCAB_PATH", cfg.QUERY_FIX_VOCAB_PATH),
    ):
        _ensure_data_dir(path, env_var)


class Config:
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "vccircle")
    MYSQL_TABLE = os.getenv("MYSQL_TABLE", "articles")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")





    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vccircle_articles")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    EMBED_DIM = 768
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

    TORCH_THREADS = _TORCH_THREADS




    EMBED_DENSE_CHAR_LIMIT = int(os.getenv("EMBED_DENSE_CHAR_LIMIT", "1500"))
    EMBED_CHAR_LIMIT = int(os.getenv("EMBED_CHAR_LIMIT", "50000"))
    BODY_CHAR_LIMIT = int(os.getenv("BODY_CHAR_LIMIT", "50000"))


    CHAT_BODY_CHAR_LIMIT = int(os.getenv("CHAT_BODY_CHAR_LIMIT", "50000"))



    CHAT_MAX_SOURCES = int(os.getenv("CHAT_MAX_SOURCES", "20"))
    CHAT_TOTAL_BODY_CHARS = int(os.getenv("CHAT_TOTAL_BODY_CHARS", "400000"))







    CHAT_MAX_MULTI_ENTITIES = int(os.getenv("CHAT_MAX_MULTI_ENTITIES", "6"))



    CHAT_MULTI_ENTITY_CONCURRENCY = int(os.getenv("CHAT_MULTI_ENTITY_CONCURRENCY", "4"))





    CHAT_HISTORY_CHAR_LIMIT = int(os.getenv("CHAT_HISTORY_CHAR_LIMIT", "12000"))








    CACHE_KEY_QUERY_MAX_CHARS = _clamped_int("CACHE_KEY_QUERY_MAX_CHARS", 128, 8, 4096)













    RETRIEVAL_QUERY_MAX_CHARS = _clamped_int("RETRIEVAL_QUERY_MAX_CHARS", 8000, 64, 65536)




    INDEXER_WORKERS = int(os.getenv("INDEXER_WORKERS", "2"))


    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")



    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")










    RERANK_CANDIDATES = _clamped_int("RERANK_CANDIDATES", 12, 5, 50)






    RERANK_BACKEND = os.getenv("RERANK_BACKEND", "torch")




    RERANK_ONNX_DIR = _data_path("RERANK_ONNX_DIR", "data/reranker_onnx")








    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")



    LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))



    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.0"))

    LLM_PRICE_INPUT_PER_1M = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.25"))
    LLM_PRICE_OUTPUT_PER_1M = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "1.50"))


    INR_PER_USD = float(os.getenv("INR_PER_USD", "95.60"))


    FX_RATE_API_URL = os.getenv("FX_RATE_API_URL", "https://open.er-api.com/v6/latest/USD")


    FX_RATE_REFRESH_SECONDS = float(os.getenv("FX_RATE_REFRESH_SECONDS", "43200"))




    LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0"))



    LLM_CALL_RESERVE_USD = float(os.getenv("LLM_CALL_RESERVE_USD", "0.05"))



    COST_RESERVATION_TTL_SECONDS = int(os.getenv("COST_RESERVATION_TTL_SECONDS", "900"))


    CHAT_MAX_HISTORY_CHARS = int(os.getenv("CHAT_MAX_HISTORY_CHARS", "24000"))

    TOP_K = int(os.getenv("TOP_K", "8"))


    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))



    ASK_MIN_SCORE_FACETED = float(os.getenv("ASK_MIN_SCORE_FACETED", "0.0"))










    WEAK_RESULT_SCORE = float(os.getenv("WEAK_RESULT_SCORE", "0.3"))
    WEAK_RESULT_MIN_STRONG = int(os.getenv("WEAK_RESULT_MIN_STRONG", "3"))














    DATE_FILLER_SCORE = float(os.getenv("DATE_FILLER_SCORE", "0.2"))
    if DATE_FILLER_SCORE < ASK_MIN_SCORE:
        logger.warning(
            "DATE_FILLER_SCORE=%g is below ASK_MIN_SCORE=%g: date-only fallback "
            "fillers are filtered out by the chat relevance gate before reaching "
            "the model, so the temporal fallback contributes nothing. Raise "
            "DATE_FILLER_SCORE to at least ASK_MIN_SCORE.",
            DATE_FILLER_SCORE,
            ASK_MIN_SCORE,
        )

    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))





    CACHE_MAX_BYTES = int(os.getenv("CACHE_MAX_BYTES", "33554432"))



    VECTOR_CACHE_TTL_SECONDS = int(os.getenv("VECTOR_CACHE_TTL_SECONDS", "86400"))





    READY_CACHE_TTL_SECONDS = float(os.getenv("READY_CACHE_TTL_SECONDS", "5"))
    READY_DEP_TIMEOUT_SECONDS = float(os.getenv("READY_DEP_TIMEOUT_SECONDS", "2.0"))




    RECENCY_STRENGTH = float(os.getenv("RECENCY_STRENGTH", "0.25"))
    RECENCY_DECAY_DAYS = float(os.getenv("RECENCY_DECAY_DAYS", "90"))




    RECENCY_BOOST_STRENGTH = float(os.getenv("RECENCY_BOOST_STRENGTH", "0.85"))
    RECENCY_BOOST_DECAY_DAYS = float(os.getenv("RECENCY_BOOST_DECAY_DAYS", "30.0"))



    ENABLE_QUERY_EXPANSION = _env_bool("ENABLE_QUERY_EXPANSION", True)
    ENABLE_ENTITY_BOOST = _env_bool("ENABLE_ENTITY_BOOST", True)
    ENABLE_WEAK_FALLBACK = _env_bool("ENABLE_WEAK_FALLBACK", True)






    ENABLE_QUERY_FIX = _env_bool("ENABLE_QUERY_FIX", True)
    QUERY_FIX_VOCAB_PATH = _data_path("QUERY_FIX_VOCAB_PATH", "data/query_vocab.json.gz")
    QUERY_FIX_MAX_EDIT = int(os.getenv("QUERY_FIX_MAX_EDIT", "2"))
    QUERY_FIX_MIN_COUNT = int(os.getenv("QUERY_FIX_MIN_COUNT", "5"))
    QUERY_FIX_MIN_TOKEN_LEN = int(os.getenv("QUERY_FIX_MIN_TOKEN_LEN", "3"))





    ENABLE_DIVERSITY = _env_bool("ENABLE_DIVERSITY", True)
    DIVERSITY_LAMBDA = float(os.getenv("DIVERSITY_LAMBDA", "0.7"))
    DIVERSITY_SIM_THRESHOLD = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.4"))






    ENABLE_CLICK_BOOST = _env_bool("ENABLE_CLICK_BOOST", True)
    CLICK_BOOST_MIN_CLICKS = int(os.getenv("CLICK_BOOST_MIN_CLICKS", "5"))
    CLICK_BOOST_MIN_ARTICLE_CLICKS = int(os.getenv("CLICK_BOOST_MIN_ARTICLE_CLICKS", "3"))
    CLICK_BOOST_MIN_SHARE = float(os.getenv("CLICK_BOOST_MIN_SHARE", "0.3"))
    CLICK_BOOST_MULT = float(os.getenv("CLICK_BOOST_MULT", "1.3"))











    CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS = int(os.getenv("CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS", "3600"))





    CLICK_QUERY_MAX_LEN = int(os.getenv("CLICK_QUERY_MAX_LEN", "256"))
    CLICK_QUERY_TTL_SECONDS = int(os.getenv("CLICK_QUERY_TTL_SECONDS", str(7 * 24 * 3600)))







    ANALYTICS_QUERY_KEY = os.getenv("ANALYTICS_QUERY_KEY", "")



    COST_DAY_TTL_SECONDS = int(os.getenv("COST_DAY_TTL_SECONDS", str(7 * 24 * 3600)))



    ANALYTICS_SESSION_TTL_HOURS = _clamped_int("ANALYTICS_SESSION_TTL_HOURS", 24, 1, 24 * 365)
    TOP_QUERIES_TODAY_TTL_DAYS = _clamped_int("TOP_QUERIES_TODAY_TTL_DAYS", 7, 1, 90)



    LAST_SEEN_TOUCH_INTERVAL_SECONDS = _clamped_int("LAST_SEEN_TOUCH_INTERVAL_SECONDS", 300, 0, 3600)


    AUDIT_RECENT_LIMIT = _clamped_int("AUDIT_RECENT_LIMIT", 20, 1, 100)












    ENABLE_BODY_RESCUE = _env_bool("ENABLE_BODY_RESCUE", True)
    BODY_RESCUE_THRESHOLD = float(os.getenv("BODY_RESCUE_THRESHOLD", "0.3"))





    BODY_RESCUE_WINDOW = _clamped_int("BODY_RESCUE_WINDOW", 1500, 200, 8000)



    BODY_RESCUE_STEP = _clamped_int("BODY_RESCUE_STEP", 500, 1, 1500)











    BODY_RESCUE_MAX_WINDOWS = _clamped_int("BODY_RESCUE_MAX_WINDOWS", 200, 1, 5000)



    BODY_RESCUE_MAX_CANDIDATES = _clamped_int("BODY_RESCUE_MAX_CANDIDATES", 10, 1, 50)






    SEARCH_QUERY_MAX_CHARS = _clamped_int("SEARCH_QUERY_MAX_CHARS", 512, 32, 4000)



    CHAT_DB_PATH = _data_path("CHAT_DB_PATH", "data/chat.db")
    CHAT_RETENTION_DAYS = int(os.getenv("CHAT_RETENTION_DAYS", "180"))
    CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
    CHAT_PURGE_INTERVAL_SECONDS = int(os.getenv("CHAT_PURGE_INTERVAL_SECONDS", "86400"))








    CHAT_SESSION_MESSAGE_LIMIT = int(os.getenv("CHAT_SESSION_MESSAGE_LIMIT", "200"))



    CHAT_MESSAGE_SOURCE_LIMIT = int(os.getenv("CHAT_MESSAGE_SOURCE_LIMIT", "20"))

    ENABLE_RECOMMENDATIONS = _env_bool("ENABLE_RECOMMENDATIONS", True)
    RECOMMEND_SIMILARITY_WEIGHT = float(os.getenv("RECOMMEND_SIMILARITY_WEIGHT", "0.4"))
    RECOMMEND_CATEGORY_WEIGHT = float(os.getenv("RECOMMEND_CATEGORY_WEIGHT", "0.3"))
    RECOMMEND_RECENCY_WEIGHT = float(os.getenv("RECOMMEND_RECENCY_WEIGHT", "0.2"))
    RECOMMEND_POPULARITY_WEIGHT = float(os.getenv("RECOMMEND_POPULARITY_WEIGHT", "0.1"))
    USER_INTERACTION_TTL_DAYS = int(os.getenv("USER_INTERACTION_TTL_DAYS", "90"))





    USER_MAX_DISTINCT_INTERACTIONS = int(os.getenv("USER_MAX_DISTINCT_INTERACTIONS", "500"))
    RECOMMEND_DEFAULT_LIMIT = int(os.getenv("RECOMMEND_DEFAULT_LIMIT", "10"))




    RECOMMEND_CANDIDATES_LIMIT = int(os.getenv("RECOMMEND_CANDIDATES_LIMIT", "50"))
    USER_PROFILE_REDIS_DB = int(os.getenv("USER_PROFILE_REDIS_DB", "2"))
    TRENDING_VELOCITY_WINDOW_DAYS = int(os.getenv("TRENDING_VELOCITY_WINDOW_DAYS", "7"))



    ANALYTICS_REDIS_DB = int(os.getenv("ANALYTICS_REDIS_DB", "1"))





    AUTH_DB_PATH = _data_path("AUTH_DB_PATH", "data/auth.db")








    AUTH_RATE_LIMIT_REDIS_DB = int(os.getenv("AUTH_RATE_LIMIT_REDIS_DB", "3"))
    AUTH_TOKEN_TTL_DAYS = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "7"))






















    AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "vccircle_session")








    AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")








    AUTH_COOKIE_SECURE = _env_tristate("AUTH_COOKIE_SECURE") is not False


    AUTH_COOKIE_PATH = "/"


    AUTH_COOKIE_MAX_AGE_SECONDS = AUTH_TOKEN_TTL_DAYS * 86400
    AUTH_SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")



    AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS = int(os.getenv("AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", "86400"))



    AUTH_SERVICE_TOKEN_SCOPE: ClassVar[tuple[str, ...]] = tuple(
        p.strip()
        for p in os.getenv("AUTH_SERVICE_TOKEN_SCOPE", "chat:use").split(",")
        if p.strip()
    )


    AUTH_ADMIN_EMAIL = os.getenv("AUTH_ADMIN_EMAIL", "")
    AUTH_ADMIN_PASSWORD = os.getenv("AUTH_ADMIN_PASSWORD", "")
    AUTH_PASSWORD_MIN_LEN = int(os.getenv("AUTH_PASSWORD_MIN_LEN", "8"))
    AUTH_MAX_EMAIL_LEN = int(os.getenv("AUTH_MAX_EMAIL_LEN", "254"))
    AUTH_MAX_NAME_LEN = int(os.getenv("AUTH_MAX_NAME_LEN", "60"))

    AUTH_SIGNUP_RATE_PER_MIN = int(os.getenv("AUTH_SIGNUP_RATE_PER_MIN", "5"))
    AUTH_LOGIN_RATE_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_MIN", "10"))
    AUTH_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))

















    AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", "20"))






    AUTH_MAX_ACTIVE_TOKENS_PER_USER = int(os.getenv("AUTH_MAX_ACTIVE_TOKENS_PER_USER", "10"))







    PUBLIC_SEARCH_RATE_PER_MIN = int(os.getenv("PUBLIC_SEARCH_RATE_PER_MIN", "60"))
    PUBLIC_FACETS_RATE_PER_MIN = int(os.getenv("PUBLIC_FACETS_RATE_PER_MIN", "60"))
    PUBLIC_CLICK_RATE_PER_MIN = int(os.getenv("PUBLIC_CLICK_RATE_PER_MIN", "120"))






    PUBLIC_INTERACTION_RATE_PER_MIN = int(os.getenv("PUBLIC_INTERACTION_RATE_PER_MIN", "60"))
    INTERACTION_USER_RATE_PER_MIN = int(os.getenv("INTERACTION_USER_RATE_PER_MIN", "60"))





    PUBLIC_READY_RATE_PER_MIN = int(os.getenv("PUBLIC_READY_RATE_PER_MIN", "600"))
    PUBLIC_RATE_WINDOW_SECONDS = int(os.getenv("PUBLIC_RATE_WINDOW_SECONDS", "60"))









    AUTH_TRUST_X_FORWARDED_FOR: bool | None = _env_tristate("AUTH_TRUST_X_FORWARDED_FOR")

    AUTH_TOKEN_PURGE_INTERVAL_SECONDS = int(os.getenv("AUTH_TOKEN_PURGE_INTERVAL_SECONDS", "3600"))






    CORS_ORIGINS: ClassVar[tuple[str, ...]] = tuple(
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8001").split(",")
        if o.strip()
    )








    ALLOWED_HOSTS: ClassVar[tuple[str, ...]] = _parse_allowed_hosts(
        os.getenv("ALLOWED_HOSTS"), CORS_ORIGINS + _machine_hosts()
    )







    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()









    FEED_DB_PATH = _data_path("FEED_DB_PATH", "data/feed.db")
    FEED_MAX_SUBSCRIPTIONS = int(os.getenv("FEED_MAX_SUBSCRIPTIONS", "50"))
    FEED_DEFAULT_LIMIT = int(os.getenv("FEED_DEFAULT_LIMIT", "20"))






_GOOGLE_KEY_RE = re.compile(r"^AIza[0-9A-Za-z_-]{35}$")
_GOOGLE_API_HOST = "generativelanguage.googleapis.com"
_GOOGLE_KEY_PREFIX = "AIza"





_PLACEHOLDER_API_KEYS = frozenset(
    {
        "0",
        "changeme",
        "changethis",
        "dummy",
        "empty",
        "example",
        "insertkeyhere",
        "key",
        "na",
        "nil",
        "none",
        "null",
        "pastekeyhere",
        "placeholder",
        "putkeyhere",
        "replaceme",
        "sample",
        "secret",
        "tbd",
        "test",
        "todo",
        "unset",
        "yourapikey",
        "yourapikeyhere",
        "yourgeminiapikey",
        "yourkey",
        "yourkeyhere",
        "yoursecret",
    }
)



_REPEATED_FILLER_RE = re.compile(r"(.)\1{3,}")











_MIN_DISTINCT_KEY_CHARS = 12


def classify_gemini_api_key(value: str | None) -> str:

    if value is None or not value.strip():
        return "missing"
    stripped = value.strip()
    normalised = re.sub(r"[^0-9a-z]+", "", stripped.lower())
    if normalised in _PLACEHOLDER_API_KEYS or _REPEATED_FILLER_RE.fullmatch(normalised):
        return "placeholder"



    base_url = config.GEMINI_BASE_URL.strip().lower()
    if not base_url or _GOOGLE_API_HOST in base_url:
        if not _GOOGLE_KEY_RE.fullmatch(stripped):
            return "malformed"
        if len(set(stripped[len(_GOOGLE_KEY_PREFIX) :])) < _MIN_DISTINCT_KEY_CHARS:
            return "placeholder"
    return "ok"


config = Config()
