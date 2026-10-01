import logging
import os
import re
import socket
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Resolved from __file__, not the CWD: gunicorn, pm2, systemd and a shell each
# start the app from a different CWD, so a CWD-relative data path opens a
# brand-new empty SQLite file in the wrong place.
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _data_path(name: str, default: str) -> str:
    """Resolve a data-path knob to an absolute path (``~`` expanded)."""
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


# Set before any inference library is imported: gunicorn workers that each
# default to all cores oversubscribe the box.
_TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))

# Compared against the incoming `Host` header with the port stripped.
_DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "testserver")


def _normalize_host(entry: str) -> str:
    """Normalize a host entry; IPv6 is excluded upstream because a first-hextet
    match would accept any ``2001:*`` Host."""
    host = entry.strip().lower().split("://", 1)[-1]
    return host.partition(":")[0]


def _is_ipv6_literal(entry: str) -> bool:
    """True for an address a split-on-first-colon Host can never match."""
    host = entry.strip().lower().split("://", 1)[-1]
    if host.startswith("["):
        return True
    return host.count(":") > 1


def _machine_hosts() -> tuple[str, ...]:
    """Hostnames this box itself answers to; best effort, since a resolution
    failure here must cost allowed hosts, never the boot."""
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
    # IPv6 is dropped here, not in _normalize_host: a truncated hextet fails open.
    return tuple(h for h in dict.fromkeys(hosts) if h and not _is_ipv6_literal(h))


def _default_route_addresses() -> tuple[str, ...]:
    """The address outbound traffic would source from; on a NAT'd cloud host that
    is the private one, not the public address the site is reached at."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 53))
            return (probe.getsockname()[0],)
    except Exception:
        # Broad on purpose: this runs at import, so it must never raise. DEBUG and
        # not WARNING: a box with no default route is unremarkable.
        logger.debug("default-route probe failed", exc_info=True)
        return ()



def _clamped_int(name: str, default: int, low: int, high: int) -> int:
    """Clamp-and-warn rather than raise: this module is imported at process start,
    so a mistyped value would become a boot failure of the whole API."""
    # An explicit raise, not an `assert`: python -O strips asserts, voiding the guarantee.
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
    """Turn the ALLOWED_HOSTS knob into a de-duplicated tuple of hostnames.

    Wildcards are validated here rather than by the middleware, which builds its
    stack on the first request and would 500 every call on a bad pattern.
    """
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


# setup.sh's forced-true warning greps exactly these four spellings, so a
# spelling added here must be added to that regex in the same commit.
_TRUE_SPELLINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_SPELLINGS = frozenset({"0", "false", "no", "off"})


def _env_tristate(name: str) -> bool | None:
    """True/False force; an unrecognised value stays None so a typo cannot pick the unsafe side."""
    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE_SPELLINGS:
        return True
    if raw in _FALSE_SPELLINGS:
        return False
    return None


def _env_bool(name: str, default: bool) -> bool:
    """Set-but-blank reads OFF, not ``default``: ``KEY=`` is how an operator clears a knob."""
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
    """Create ``path``'s parent, or fail loudly: on an existing-but-unwritable
    parent ``mkdir(exist_ok=True)`` succeeds, so the app would otherwise open a
    fresh EMPTY database and serve it as if it had no history."""
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
    """Validate every WRITTEN data location; a read-only path (query vocab, ONNX dir) is deliberately not required."""
    for env_var, path in (
        ("CHAT_DB_PATH", cfg.CHAT_DB_PATH),
        ("AUTH_DB_PATH", cfg.AUTH_DB_PATH),
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
    # The client is built without the kwarg when None: qdrant-client sends
    # whatever it is given as the api-key header, so "" would authenticate.
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vccircle_articles")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    EMBED_DIM = 768  # matches bge-base; change if you swap models
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
    TORCH_THREADS = _TORCH_THREADS

    EMBED_DENSE_CHAR_LIMIT = int(os.getenv("EMBED_DENSE_CHAR_LIMIT", "1500"))
    EMBED_CHAR_LIMIT = int(os.getenv("EMBED_CHAR_LIMIT", "50000"))
    BODY_CHAR_LIMIT = int(os.getenv("BODY_CHAR_LIMIT", "50000"))
    CHAT_BODY_CHAR_LIMIT = int(os.getenv("CHAT_BODY_CHAR_LIMIT", "50000"))
    CHAT_MAX_SOURCES = int(os.getenv("CHAT_MAX_SOURCES", "20"))
    CHAT_TOTAL_BODY_CHARS = int(os.getenv("CHAT_TOTAL_BODY_CHARS", "400000"))
    # One full retrieval leg runs PER entity named, and that count is unbounded,
    # so past this cap the expansion is skipped and the turn is a single query.
    CHAT_MAX_MULTI_ENTITIES = int(os.getenv("CHAT_MAX_MULTI_ENTITIES", "6"))
    CHAT_MULTI_ENTITY_CONCURRENCY = int(os.getenv("CHAT_MULTI_ENTITY_CONCURRENCY", "4"))

    CHAT_HISTORY_CHAR_LIMIT = int(os.getenv("CHAT_HISTORY_CHAR_LIMIT", "12000"))

    # A longer query is replaced wholesale by a sha256 digest, never truncated,
    # so two long queries can never share a key.
    CACHE_KEY_QUERY_MAX_CHARS = _clamped_int("CACHE_KEY_QUERY_MAX_CHARS", 128, 8, 4096)

    # Deliberately distinct from SEARCH_QUERY_MAX_CHARS: chat puts the whole
    # message in the LLM prompt, so a lower retrieval clamp would drop the
    # caller's own words from the search. A 0 would empty every result set.
    RETRIEVAL_QUERY_MAX_CHARS = _clamped_int("RETRIEVAL_QUERY_MAX_CHARS", 8000, 64, 65536)

    INDEXER_WORKERS = int(os.getenv("INDEXER_WORKERS", "2"))

    # Must match the sparse model used at index time.
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Every candidate is one cross-encoder pair run under the process-wide
    # `inference_lock`, so a large value serialises all other inference.
    RERANK_CANDIDATES = _clamped_int("RERANK_CANDIDATES", 12, 5, 50)
    # 'torch' is the only backend; the knob is kept so existing deployments that
    # set it keep working.
    RERANK_BACKEND = os.getenv("RERANK_BACKEND", "torch")
    # Inert: kept only so a deployed .env that sets it stays valid.
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
    # 0 is a deliberate opt-out for deployments that meter spend elsewhere.
    LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0"))
    # Held before the call runs, so concurrent turns cannot each read a stale counter.
    LLM_CALL_RESERVE_USD = float(os.getenv("LLM_CALL_RESERVE_USD", "0.05"))
    # An unsettled hold is swept and CHARGED, so a crashed billed call stays charged.
    COST_RESERVATION_TTL_SECONDS = int(os.getenv("COST_RESERVATION_TTL_SECONDS", "900"))
    CHAT_MAX_HISTORY_CHARS = int(os.getenv("CHAT_MAX_HISTORY_CHARS", "24000"))

    TOP_K = int(os.getenv("TOP_K", "8"))
    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    # A resolved facet IS the relevance signal, so the cross-encoder gate drops to 0.
    ASK_MIN_SCORE_FACETED = float(os.getenv("ASK_MIN_SCORE_FACETED", "0.0"))
    # Judged over the whole result list; ASK_MIN_SCORE is the per-source inclusion filter.
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

    # Scores are scaled by 1 - RECENCY_STRENGTH * (1 - exp(-age_days / RECENCY_DECAY_DAYS)).
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

    # This dedupe, not the click-boost thresholds, is what bounds a forged click burst.
    CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS = int(os.getenv("CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS", "3600"))

    CLICK_QUERY_MAX_LEN = int(os.getenv("CLICK_QUERY_MAX_LEN", "256"))
    CLICK_QUERY_TTL_SECONDS = int(os.getenv("CLICK_QUERY_TTL_SECONDS", str(7 * 24 * 3600)))

    # Empty means the app generates a random key on first use and persists it in
    # the analytics Redis; setting it pins the digest namespace across a rebuild.
    ANALYTICS_QUERY_KEY = os.getenv("ANALYTICS_QUERY_KEY", "")

    COST_DAY_TTL_SECONDS = int(os.getenv("COST_DAY_TTL_SECONDS", str(7 * 24 * 3600)))

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
    # Caps what is serialized back per message, not what is generated (CHAT_MAX_SOURCES).
    CHAT_MESSAGE_SOURCE_LIMIT = int(os.getenv("CHAT_MESSAGE_SOURCE_LIMIT", "20"))

    ENABLE_RECOMMENDATIONS = _env_bool("ENABLE_RECOMMENDATIONS", True)
    RECOMMEND_SIMILARITY_WEIGHT = float(os.getenv("RECOMMEND_SIMILARITY_WEIGHT", "0.4"))
    RECOMMEND_CATEGORY_WEIGHT = float(os.getenv("RECOMMEND_CATEGORY_WEIGHT", "0.3"))
    RECOMMEND_RECENCY_WEIGHT = float(os.getenv("RECOMMEND_RECENCY_WEIGHT", "0.2"))
    RECOMMEND_POPULARITY_WEIGHT = float(os.getenv("RECOMMEND_POPULARITY_WEIGHT", "0.1"))
    USER_INTERACTION_TTL_DAYS = int(os.getenv("USER_INTERACTION_TTL_DAYS", "90"))
    # Each distinct article mints two Redis keys, so an uncapped self-signup'd
    # account can grow the profile keyspace without bound.
    USER_MAX_DISTINCT_INTERACTIONS = int(os.getenv("USER_MAX_DISTINCT_INTERACTIONS", "500"))
    RECOMMEND_DEFAULT_LIMIT = int(os.getenv("RECOMMEND_DEFAULT_LIMIT", "10"))
    RECOMMEND_CANDIDATES_LIMIT = int(os.getenv("RECOMMEND_CANDIDATES_LIMIT", "50"))
    USER_PROFILE_REDIS_DB = int(os.getenv("USER_PROFILE_REDIS_DB", "2"))
    TRENDING_VELOCITY_WINDOW_DAYS = int(os.getenv("TRENDING_VELOCITY_WINDOW_DAYS", "7"))

    # Must not be the query cache's DB 0, which is flushed during deploys.
    ANALYTICS_REDIS_DB = int(os.getenv("ANALYTICS_REDIS_DB", "1"))

    AUTH_DB_PATH = _data_path("AUTH_DB_PATH", "data/auth.db")
    # Pinned, not inherited from REDIS_URL, whose db segment is the flushed query cache.
    AUTH_RATE_LIMIT_REDIS_DB = int(os.getenv("AUTH_RATE_LIMIT_REDIS_DB", "3"))
    AUTH_TOKEN_TTL_DAYS = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "7"))
    # A SEED, not a standing grant. The first request presenting it creates the
    # row that is thereafter the only authority on its life, so a restart does
    # NOT revive it. Rotate by changing the value here and restarting.
    # The credential travels only in an HttpOnly cookie: a Bearer header the app
    # writes into a JSON body is reachable by any script on the page, so an XSS
    # bug would exfiltrate a durable account takeover. No Domain is ever set, so
    # the same image works on a bare-IP deployment.
    AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "vccircle_session")
    # SameSite=Lax still sends the cookie on top-level GET navigations (deep
    # links) while blocking the cross-site POST shape.
    AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")
    # Defaults TRUE. A plain-HTTP deployment (setup.sh with NGINX_TLS=off) MUST
    # set this false or login will silently not persist.
    AUTH_COOKIE_SECURE = _env_tristate("AUTH_COOKIE_SECURE") is not False
    # Not configurable: the API serves /api/..., so a narrower path would stop the cookie.
    AUTH_COOKIE_PATH = "/"
    # Derived from the token TTL so the cookie and the stored record cannot drift apart.
    AUTH_COOKIE_MAX_AGE_SECONDS = AUTH_TOKEN_TTL_DAYS * 86400
    AUTH_SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")
    # Not optional: <= 0 falls back to the default, because an eternal machine
    # admin credential is exactly the hole this closes.
    AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS = int(os.getenv("AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", "86400"))
    AUTH_SERVICE_TOKEN_SCOPE: ClassVar[tuple[str, ...]] = tuple(
        p.strip()
        for p in os.getenv("AUTH_SERVICE_TOKEN_SCOPE", "chat:use").split(",")
        if p.strip()
    )
    # Created once at startup if no account with this email exists; never overwritten.
    AUTH_ADMIN_EMAIL = os.getenv("AUTH_ADMIN_EMAIL", "")
    AUTH_ADMIN_PASSWORD = os.getenv("AUTH_ADMIN_PASSWORD", "")
    AUTH_PASSWORD_MIN_LEN = int(os.getenv("AUTH_PASSWORD_MIN_LEN", "8"))
    AUTH_MAX_EMAIL_LEN = int(os.getenv("AUTH_MAX_EMAIL_LEN", "254"))
    AUTH_MAX_NAME_LEN = int(os.getenv("AUTH_MAX_NAME_LEN", "60"))
    AUTH_SIGNUP_RATE_PER_MIN = int(os.getenv("AUTH_SIGNUP_RATE_PER_MIN", "5"))
    AUTH_LOGIN_RATE_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_MIN", "10"))
    AUTH_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))
    # Counts FAILED attempts only, after the credential check: gating every
    # attempt first turns the throttle into an account-lockout weapon an anonymous
    # caller can aim at a known address. Keyed on the submitted address alone, so
    # it is not an account-existence oracle.
    AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", "20"))
    # The purge only removes EXPIRED rows, so without this cap the table grows
    # with logins rather than users. Logging in past the cap revokes the oldest.
    AUTH_MAX_ACTIVE_TOKENS_PER_USER = int(os.getenv("AUTH_MAX_ACTIVE_TOKENS_PER_USER", "10"))
    # These endpoints are the abuse surface, so they FAIL CLOSED (503) when Redis
    # is unreachable; /ready is the one deliberate exception and fails open.
    PUBLIC_SEARCH_RATE_PER_MIN = int(os.getenv("PUBLIC_SEARCH_RATE_PER_MIN", "60"))
    PUBLIC_FACETS_RATE_PER_MIN = int(os.getenv("PUBLIC_FACETS_RATE_PER_MIN", "60"))
    PUBLIC_CLICK_RATE_PER_MIN = int(os.getenv("PUBLIC_CLICK_RATE_PER_MIN", "120"))
    # Limited on both axes: a per-IP bucket alone cannot bound one account
    # behind a shared NAT/proxy address.
    PUBLIC_INTERACTION_RATE_PER_MIN = int(os.getenv("PUBLIC_INTERACTION_RATE_PER_MIN", "60"))
    INTERACTION_USER_RATE_PER_MIN = int(os.getenv("INTERACTION_USER_RATE_PER_MIN", "60"))
    # An order of magnitude above a 1 Hz prober: a 429 makes an LB pull the node
    # from rotation, the exact outage this limiter must not cause.
    PUBLIC_READY_RATE_PER_MIN = int(os.getenv("PUBLIC_READY_RATE_PER_MIN", "600"))
    PUBLIC_RATE_WINDOW_SECONDS = int(os.getenv("PUBLIC_RATE_WINDOW_SECONDS", "60"))
    # Unset means AUTO (auth._client_ip): X-Forwarded-For is honoured only when
    # the socket peer is loopback, so a direct client cannot forge its rate-limit IP.
    AUTH_TRUST_X_FORWARDED_FOR: bool | None = _env_tristate("AUTH_TRUST_X_FORWARDED_FOR")
    AUTH_TOKEN_PURGE_INTERVAL_SECONDS = int(os.getenv("AUTH_TOKEN_PURGE_INTERVAL_SECONDS", "3600"))

    CORS_ORIGINS: ClassVar[tuple[str, ...]] = tuple(
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8001").split(",")
        if o.strip()
    )

    # Derived from CORS_ORIGINS and this box's own names/addresses, which cannot
    # cover a bare public IP behind NAT. Set it explicitly for such a deployment:
    # a wrong value answers 400 to every request.
    ALLOWED_HOSTS: ClassVar[tuple[str, ...]] = _parse_allowed_hosts(
        os.getenv("ALLOWED_HOSTS"), CORS_ORIGINS + _machine_hosts()
    )

    # Applies to the app's own loggers only, so a third-party logger's INFO output
    # stays off at every setting. Unrecognised values fall back to INFO and say so.
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()


# A real Google API key is "AIza" plus 35 URL-safe characters, but that shape is
# only required against Google's endpoint: GEMINI_BASE_URL is configurable, so a
# gateway deployment legitimately holds a differently shaped key.
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

# A MASKED key keeps its real prefix and length and fills the rest with filler,
# which neither check above can see, so the tail must also look random: a real
# 35-character tail has ~27 distinct characters, every masking style far fewer.
# Such an example is described here, never written out -- a contiguous "AIza..."
# string, comment included, is indistinguishable from a leak to a scanner.
_MIN_DISTINCT_KEY_CHARS = 12


def classify_gemini_api_key(value: str | None) -> str:
    """Classify a configured GEMINI_API_KEY for readiness reporting.

    One of "ok", "missing", "placeholder" (filler, or a correctly shaped key
    whose body is masked filler) or "malformed" (the wrong shape for the
    configured GEMINI_BASE_URL). The key itself is never returned or logged.
    """
    if value is None or not value.strip():
        return "missing"
    stripped = value.strip()
    normalised = re.sub(r"[^0-9a-z]+", "", stripped.lower())
    if normalised in _PLACEHOLDER_API_KEYS or _REPEATED_FILLER_RE.fullmatch(normalised):
        return "placeholder"
    # A blank base URL counts as Google-shaped, so clearing the variable cannot
    # switch the structural check off.
    base_url = config.GEMINI_BASE_URL.strip().lower()
    if not base_url or _GOOGLE_API_HOST in base_url:
        if not _GOOGLE_KEY_RE.fullmatch(stripped):
            return "malformed"
        if len(set(stripped[len(_GOOGLE_KEY_PREFIX) :])) < _MIN_DISTINCT_KEY_CHARS:
            return "placeholder"
    return "ok"


config = Config()
