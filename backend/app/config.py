import logging
import os
import socket
from typing import ClassVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Cap the number of CPU threads torch/onnxruntime use per process BEFORE any
# inference library is imported. With GUNICORN_WORKERS processes sharing the
# box, leaving the default (all cores) oversubscribes the CPU and hurts
# latency under concurrent load. 2 threads per worker is the tuned default.
_TORCH_THREADS = int(os.getenv("TORCH_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))

# Hostnames the API answers to, enforced by TrustedHostMiddleware (see
# app/main.py). The allowed entries are compared against the incoming `Host`
# header with the port already stripped, so every entry is normalised the same
# way here.
_DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "testserver")


def _normalize_host(entry: str) -> str:
    """Lowercase a host/origin and drop any scheme and :port suffix.

    Only IPv4 and names. Starlette compares the Host authority as
    ``headers.get("host", "").split(":")[0]`` — everything before the FIRST
    colon — so an IPv6 literal cannot be expressed in this allow-list at all: a
    bracketed ``[::1]`` arrives already split to ``[``, and an unbracketed
    ``2001:db8::5`` arrives as ``2001``. Keeping the brackets would silently
    admit an entry that can never match, and truncating to the first hextet
    would be worse: ``2001`` matches ANY ``2001:*`` Host, turning the check
    into a fail-open on a guessable header. So IPv6 is excluded by
    ``_is_ipv6_literal`` at every source instead, and serving an IPv6-only
    deployment needs a middleware that parses the authority properly.
    """
    host = entry.strip().lower().split("://", 1)[-1]
    return host.partition(":")[0]


def _is_ipv6_literal(entry: str) -> bool:
    """True for an address that a split-on-first-colon Host can never match.

    Both spellings occur in the wild: ``getaddrinfo`` and ``getsockname`` hand
    back unbracketed literals, while an operator writing ``CORS_ORIGINS`` is
    likely to bracket them.
    """
    host = entry.strip().lower().split("://", 1)[-1]
    if host.startswith("["):
        return True
    return host.count(":") > 1


def _machine_hosts() -> tuple[str, ...]:
    """Hostnames and addresses this box itself answers to.

    Production is same-origin through nginx behind a `server_name _` catch-all
    vhost that forwards whatever `Host` the client used, and the documented
    posture leaves CORS_ORIGINS at its localhost default — so neither CORS nor
    a hardcoded domain covers a site reached by IP or by the box's own name.
    These are the box's name, the addresses bound to it and its default-route
    address; without them, every public request 400s. A separately registered
    public domain still has to be added to ALLOWED_HOSTS by the operator.

    Best effort by design: this runs at import, so a name-resolution failure
    must degrade to "fewer allowed hosts", never take the whole API down.
    """
    hosts: list[str] = []
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            name = getter()
        except (OSError, ValueError):
            # Narrow on purpose: gaierror is an OSError and a non-decodable
            # hostname raises UnicodeDecodeError, which is a ValueError. A
            # best-effort identity probe that loses one hostname beats refusing
            # to boot the API.
            continue
        if name:
            hosts.append(name)
    try:
        hosts.extend({info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)})
    except (OSError, ValueError):
        pass
    hosts.extend(_default_route_addresses())
    # IPv6 literals are dropped here rather than in _normalize_host: a hextet
    # left behind by a later truncation is a silently fail-open entry, so the
    # literal must never travel as far as normalisation. See _normalize_host.
    return tuple(h for h in dict.fromkeys(hosts) if h and not _is_ipv6_literal(h))


def _default_route_addresses() -> tuple[str, ...]:
    """The address this box would source outbound traffic from.

    `getaddrinfo(gethostname())` only yields the addresses bound to the box's
    own name, which on a NAT'd cloud host is the private one — but the site is
    reached at the public address, and nginx forwards the client's `Host`
    through (`server_name _;` plus `proxy_set_header Host $host`). Without the
    public address in the allow-list, every public request answers 400.

    A connected UDP socket sends no packets: it only asks the routing table
    which source address it would pick. Any failure (no route, a sandboxed
    import, a missing address family) just means one fewer allowed host, so
    every step is guarded: this runs at import, and an exception escaping here
    would take the whole API down, which is strictly worse than a narrower
    allow-list.

    IPv4 only. An IPv6 source address cannot be put in the allow-list at all
    (see _normalize_host), so probing for one would only add an entry that
    could never match.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 53))
            return (probe.getsockname()[0],)
    except Exception:
        # Broad on purpose, and asserted by test so it cannot be narrowed
        # back: the only contract that matters here is "never raise at
        # import". No route, a sandboxed socket module and an unusable
        # address family all cost the same one allowed host.
        # DEBUG, not WARNING: a box with no default route is unremarkable, so
        # this would fire once per worker on an ordinary boot and a warning
        # traceback would be noise. The operator signal that matters is the
        # effective allow-list main.py logs at startup, which already shows a
        # public address missing from it.
        logger.debug("default-route probe failed", exc_info=True)
        return ()



def _clamped_int(name: str, default: int, low: int, high: int) -> int:
    """Read an integer env knob, clamped into ``[low, high]``.

    Clamp-and-warn, not raise. This module is imported at process start, so
    raising here would turn a mistyped deployment value into a boot failure of
    the whole API — and the knobs guarded by this helper are throughput caps
    whose *failure* mode is expensive CPU, not a wrong answer. Clamping keeps
    the service up and bounds the cost; the WARNING naming the key, the
    rejected value and the bound keeps the misconfiguration visible in the
    logs rather than silently papering over it.

    A non-integer value falls back to ``default`` for the same reason: an
    unparseable knob is an operator typo, not a client input, and the safest
    reading of it is "not configured", which is what an absent variable means.
    ``default`` itself is required to sit inside ``[low, high]``.
    """
    assert low <= default <= high, f"{name} default {default} outside [{low}, {high}]"
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

    Unset or blank falls back to the local dev/test hosts plus whatever
    extra_hosts carries (the CORS origins and this box's own identities), so a
    missing knob keeps localhost, the dev stack, the test client and the real
    deployment working without ever opening the check. An explicit value
    replaces that default wholesale: a bare "*" is rejected outright (it would
    silently disable the check, which is the exact opposite of the knob's
    purpose) and a value that contains no usable hostname is rejected too,
    because it would otherwise match nothing and 400 every request with no clue
    why.

    A wildcard is only accepted in the one shape TrustedHostMiddleware itself
    supports, a leading ``*.``. Any other placement is rejected HERE rather
    than left to the middleware, because ``add_middleware`` defers building the
    middleware stack to the first request: a malformed pattern such as
    ``a.*.com`` would otherwise boot cleanly, log a healthy-looking allow-list
    and then turn every single request into a 500 from the middleware's own
    ``assert``. Failing at config load turns that into a clear message.
    """
    if raw is None or not raw.strip():
        # IPv6 literals are dropped for the same reason as in _machine_hosts: a
        # truncated hextet would silently widen the allow-list, and a bracketed
        # one can never match.
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
            # Rejected rather than normalised: a bracketed literal would sit in
            # the list as dead weight, and a bare one would truncate to its
            # first hextet and match any Host under that prefix. See
            # _normalize_host.
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


def _env_tristate(name: str) -> bool | None:
    """Read a three-state boolean env var: True/False force a behaviour, None
    means "auto" (the variable is unset, or says so explicitly).

    An unrecognised value falls back to None rather than to a forced side, so
    a typo in an operator's .env can never silently pick the unsafe one.
    """
    raw = os.getenv(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


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

    # Total characters of prior conversation replayed into the chat prompt.
    # CHAT_MAX_HISTORY_TURNS bounds the turn COUNT but not their SIZE, and every
    # replayed turn is untrusted text the model must read as data rather than
    # instructions (#248), so the replay is also bounded by character budget,
    # newest turns first. 12000 fits several full question/answer turns.
    CHAT_HISTORY_CHAR_LIMIT = int(os.getenv("CHAT_HISTORY_CHAR_LIMIT", "12000"))

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
    # Clamped to [5, 50] — see _clamped_int for clamp-and-warn rationale. Both
    # ends are real hazards, not defensive padding:
    #   * High end: every candidate is one cross-encoder pair, and the whole
    #     batch runs under the process-wide `inference_lock` on TORCH_THREADS
    #     (2) cores, so a large value serialises every other inference in the
    #     process behind one request. 50 already exceeds what any caller can
    #     consume — /search caps top_k at 50 and CHAT_MAX_SOURCES is 20.
    #   * Low end: below 5 there is no ranking left to do. `main.py` does
    #     `max(top_k, RERANK_CANDIDATES)`, so a small value degrades quietly
    #     there, but `scripts/rerank_bench.py` passes this straight through as
    #     a Qdrant `limit`, where 0 returns nothing and a negative is invalid.
    #     A silently dead rerank is worse than a clamped one.
    RERANK_CANDIDATES = _clamped_int("RERANK_CANDIDATES", 12, 5, 50)
    # Reranker execution backend. 'torch' (sentence-transformers CrossEncoder)
    # is the only backend: the ONNX backend ('onnx', via optimum/onnxruntime)
    # is not installable — optimum-onnx requires transformers<4.58, which
    # conflicts with the pinned transformers 5.x (CVE-fix) version — so its
    # code path was removed from app/reranker.py. This knob is kept so existing
    # deployments that set RERANK_BACKEND keep working; any value other than
    # 'torch' logs a warning and uses torch.
    RERANK_BACKEND = os.getenv("RERANK_BACKEND", "torch")
    # Inert: local dir that held the exported ONNX cross-encoder cache when the
    # ONNX backend existed. Nothing reads it now; kept as a documented
    # placeholder (it is still listed in .env.example) rather than an env var
    # that silently disappears from deployed setups.
    RERANK_ONNX_DIR = os.getenv("RERANK_ONNX_DIR", "data/reranker_onnx")

    # LLM (Google Gemini via OpenAI-compatible endpoint). Provide the API key
    # in GEMINI_API_KEY. Set GEMINI_MODEL to the model id you want to use.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    # Answer model. Kept separate from GEMINI_MODEL so the eval judge can be held
    # constant while the answer model is A/B tested (the judge reads GEMINI_MODEL).
    # Falls back to GEMINI_MODEL for backwards compatibility.
    LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    # Sampling temperature for answer generation. 0.0 is deterministic and
    # minimizes the stochastic fabrication that drives hallucination on thin
    # context; raise only if more creative variation is explicitly wanted.
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
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
    # Daily LLM spend cap in USD. Chat fails closed (no LLM calls) once today's
    # cumulative spend reaches this value (see app/cost_budget.py). 0 is a
    # deliberate opt-out for deployments that meter spend elsewhere; it is not
    # the default, because an unset cap is how unbilled spend happens.
    LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0"))
    # Per-billed-call hold taken against the cap BEFORE the call runs, so
    # concurrent turns contend for the same budget instead of each reading a
    # stale counter (see reserve() in app/cost_budget.py).
    LLM_CALL_RESERVE_USD = float(os.getenv("LLM_CALL_RESERVE_USD", "0.05"))
    # Lifetime of an unsettled hold. Bounds the damage a crashed or cancelled
    # turn does to the budget: after this long the hold is swept and CHARGED to
    # the spend counter at its reserved amount, so a crashed billed call stays
    # charged for the rest of the day instead of becoming free spend.
    COST_RESERVATION_TTL_SECONDS = int(os.getenv("COST_RESERVATION_TTL_SECONDS", "900"))
    # Total-character cap on conversation history sent to the LLM. Turns the
    # cap is needed most (long histories) into the turns that cost the most.
    CHAT_MAX_HISTORY_CHARS = int(os.getenv("CHAT_MAX_HISTORY_CHARS", "24000"))

    # Search
    TOP_K = int(os.getenv("TOP_K", "8"))
    # Minimum reranked relevance score for chat sources; weaker results are
    # dropped before the LLM sees them.
    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    # When the query itself resolves a category facet (dealtype/industry), that
    # facet filter IS the relevance signal, so the cross-encoder score only ranks
    # within an already on-topic set. Drop the gate to 0 so month/year-scoped
    # category queries (e.g. 'funding news in jun 2025') surface their matches
    # instead of being rejected as "weakly related".
    ASK_MIN_SCORE_FACETED = float(os.getenv("ASK_MIN_SCORE_FACETED", "0.0"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    # Byte budget for the in-process fallback cache (the HybridCache degrades to
    # a per-worker LRU when Redis is unreachable). The entry cap alone cannot
    # bound memory because the shared cache mixes small search-result payloads
    # with large embedding vectors (768 floats as JSON, ~15KB each): a handful
    # of vectors would otherwise consume the whole entry budget and thrash out
    # the many small entries. Eviction therefore drops the largest entries first
    # until the total is back under this budget.
    CACHE_MAX_BYTES = int(os.getenv("CACHE_MAX_BYTES", "33554432"))
    # TTL for cached query (dense+sparse) vectors, keyed by the embedding model
    # so a model change invalidates them automatically. Long is safe: the pair
    # for a given query string is deterministic and stable for a fixed index.
    VECTOR_CACHE_TTL_SECONDS = int(os.getenv("VECTOR_CACHE_TTL_SECONDS", "86400"))

    # Readiness probe (/ready, /readyz). The result is cached for a few seconds
    # so a load balancer polling every second does not pay for a full Qdrant +
    # Redis probe on every hit, and each dependency is probed concurrently under
    # its own explicit timeout so the worst case is one timeout, not their sum.
    READY_CACHE_TTL_SECONDS = float(os.getenv("READY_CACHE_TTL_SECONDS", "5"))
    READY_DEP_TIMEOUT_SECONDS = float(os.getenv("READY_DEP_TIMEOUT_SECONDS", "2.0"))

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

    # Bounds on stored query strings so a hostile client can't grow Redis
    # without limit: cap the stored query length and expire the per-query click
    # sorted sets (and the /search top_queries aggregate) a few days after the
    # last write. Shared by the /analytics/click beacon and /search tracking.
    CLICK_QUERY_MAX_LEN = int(os.getenv("CLICK_QUERY_MAX_LEN", "256"))
    CLICK_QUERY_TTL_SECONDS = int(os.getenv("CLICK_QUERY_TTL_SECONDS", str(7 * 24 * 3600)))

    # Daily LLM cost counter TTL: kept well past the day it tracks so the budget
    # guardrail survives brief outages, then auto-expires instead of accumulating.
    COST_DAY_TTL_SECONDS = int(os.getenv("COST_DAY_TTL_SECONDS", str(7 * 24 * 3600)))

    # Chat-only "body rescue": when the top reranked score is below
    # BODY_RESCUE_THRESHOLD, re-score the candidates against the body region
    # with the most lexical query-token overlap and keep max(baseline, body).
    # Lets deep-body matches (e.g. historical retrospectives whose relevant
    # facts live mid-article) pass the chat relevance gate; costs one extra
    # cross-encoder pass per candidate and only runs on weak-top results.
    #
    # The rescue is left ON by default: it exists because deep-body matches
    # were being dropped by the relevance gate, and turning it off is a
    # relevance regression, not a performance fix. Its cost is bounded by the
    # three clamped knobs below instead — the expensive part is the second
    # cross-encoder pass, not the body scan (a 50K body scans in ~0.25ms at
    # these defaults), so that is what BODY_RESCUE_MAX_CANDIDATES bounds.
    ENABLE_BODY_RESCUE = os.getenv("ENABLE_BODY_RESCUE", "true").lower() in ("1", "true", "yes")
    BODY_RESCUE_THRESHOLD = float(os.getenv("BODY_RESCUE_THRESHOLD", "0.3"))
    # WINDOW is the size of the excerpt handed to the cross-encoder. Below 200
    # the excerpt is too small to carry a useful passage (and a 0 window makes
    # `_best_body_window` return an empty string, silently disabling the rescue
    # while still paying for the pass); above 8000 it inflates the model's
    # input for every candidate in the rescue batch.
    BODY_RESCUE_WINDOW = _clamped_int("BODY_RESCUE_WINDOW", 1500, 200, 8000)
    # STEP is the sliding-window stride over the body. A 0 is a hard crash —
    # `range(0, n, 0)` raises ValueError and 500s the chat turn — and a small
    # step re-scans the whole body: step=1 costs ~116ms per 50K body versus
    # ~0.25ms at the default 500, which is a 470x amplification of a knob whose
    # value is supposed to be a cost saving. The window/step ratio is also
    # capped implicitly by the 1500 ceiling (at most ~40x overlap).
    BODY_RESCUE_STEP = _clamped_int("BODY_RESCUE_STEP", 500, 1, 1500)
    # Most candidates that may enter the second cross-encoder pass. The pass is
    # the dominant cost (one pair per candidate, under `inference_lock`) and
    # chat hands body_rescue up to CHAT_MAX_SOURCES (20) articles. 10 keeps the
    # rescue available on the candidates it is designed for while halving the
    # worst case; see body_rescue() in main.py for how the shortlist is picked.
    BODY_RESCUE_MAX_CANDIDATES = _clamped_int("BODY_RESCUE_MAX_CANDIDATES", 10, 1, 50)

    # Upper bound on the /search `q` parameter, in characters. This overlaps
    # issue #241 (still open) and is deliberately the minimal version of it:
    # q reaches the embedding encoders, the cache key and every query_intent
    # regex, so an unbounded value costs CPU and Redis memory per request. It
    # is set well above CLICK_QUERY_MAX_LEN (256) because a rejected search is
    # user-visible whereas a truncated stored query string is not.
    SEARCH_QUERY_MAX_CHARS = _clamped_int("SEARCH_QUERY_MAX_CHARS", 512, 32, 4000)

    # Chat history (SQLite on the host; survives restarts, unlike Redis without AOF)
    # Relative CHAT_DB_PATH resolves against the backend working dir (where
    # gunicorn runs). Retention purges conversations idle for CHAT_RETENTION_DAYS.
    CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", "data/chat.db")
    CHAT_RETENTION_DAYS = int(os.getenv("CHAT_RETENTION_DAYS", "180"))
    CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
    CHAT_PURGE_INTERVAL_SECONDS = int(os.getenv("CHAT_PURGE_INTERVAL_SECONDS", "86400"))

    # Recommendation engine
    ENABLE_RECOMMENDATIONS = os.getenv("ENABLE_RECOMMENDATIONS", "true").lower() in ("1", "true", "yes")
    # Hybrid scoring weights
    RECOMMEND_SIMILARITY_WEIGHT = float(os.getenv("RECOMMEND_SIMILARITY_WEIGHT", "0.4"))
    RECOMMEND_CATEGORY_WEIGHT = float(os.getenv("RECOMMEND_CATEGORY_WEIGHT", "0.3"))
    RECOMMEND_RECENCY_WEIGHT = float(os.getenv("RECOMMEND_RECENCY_WEIGHT", "0.2"))
    RECOMMEND_POPULARITY_WEIGHT = float(os.getenv("RECOMMEND_POPULARITY_WEIGHT", "0.1"))
    # User profile decay rate for interaction history (exponential)
    USER_PROFILE_DECAY_LAMBDA = float(os.getenv("USER_PROFILE_DECAY_LAMBDA", "0.1"))
    USER_INTERACTION_TTL_DAYS = int(os.getenv("USER_INTERACTION_TTL_DAYS", "90"))
    RECOMMEND_DEFAULT_LIMIT = int(os.getenv("RECOMMEND_DEFAULT_LIMIT", "10"))
    RECOMMEND_CANDIDATES_LIMIT = int(os.getenv("RECOMMEND_CANDIDATES_LIMIT", "50"))
    # Redis keys for user profiles and interactions
    USER_PROFILE_REDIS_DB = int(os.getenv("USER_PROFILE_REDIS_DB", "2"))
    # Trending/viral signal TTL
    TRENDING_VELOCITY_WINDOW_DAYS = int(os.getenv("TRENDING_VELOCITY_WINDOW_DAYS", "7"))

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
    # Redis-backed per-IP rate limits on the public search surface: /search,
    # /facets, /analytics/click and /ready were unauthenticated and unrated,
    # which allowed full-corpus scraping (top_k=50) and click-analytics
    # poisoning. 0 disables an individual limit. Unlike the auth limits these
    # FAIL CLOSED (503) when Redis is unreachable: these endpoints are the
    # abuse surface, so an unrated request is not an acceptable fallback.
    # /ready is the one deliberate exception and fails open instead -- see
    # health.py and auth.public_rate_limit.
    PUBLIC_SEARCH_RATE_PER_MIN = int(os.getenv("PUBLIC_SEARCH_RATE_PER_MIN", "60"))
    PUBLIC_FACETS_RATE_PER_MIN = int(os.getenv("PUBLIC_FACETS_RATE_PER_MIN", "60"))
    PUBLIC_CLICK_RATE_PER_MIN = int(os.getenv("PUBLIC_CLICK_RATE_PER_MIN", "120"))
    # /ready is polled by load balancers and orchestrators, typically once a
    # second, and a 429 makes an LB treat the node as unhealthy and pull it
    # from rotation -- the exact outage the /ready limiter must not cause. The
    # default is therefore an order of magnitude above a 1 Hz prober (600 per
    # 60s window) rather than at it, while still bounding a runaway prober.
    PUBLIC_READY_RATE_PER_MIN = int(os.getenv("PUBLIC_READY_RATE_PER_MIN", "600"))
    PUBLIC_RATE_WINDOW_SECONDS = int(os.getenv("PUBLIC_RATE_WINDOW_SECONDS", "60"))
    # Only trust the client-supplied X-Forwarded-For header when this API is
    # reached through a reverse proxy. Otherwise the real socket peer is
    # authoritative so a client cannot spoof its IP for rate-limiting.
    #
    # Unset (the shipped default) means AUTO, resolved in auth._client_ip
    # against the actual socket peer: X-Forwarded-For is honoured only when
    # the immediate peer is loopback, i.e. a proxy on this same host (the
    # nginx config in setup.sh forwards from 127.0.0.1). That gives the
    # deployed topology per-client-IP rate limits with no .env edit, while a
    # client hitting the API directly still sees its own routable address as
    # the peer and cannot forge a header to escape its bucket.
    #
    # Set the variable to true/false to force one behaviour regardless of peer
    # (true when the proxy runs on another host, false for local/direct-only).
    AUTH_TRUST_X_FORWARDED_FOR: bool | None = _env_tristate("AUTH_TRUST_X_FORWARDED_FOR")
    # Background purge interval for expired auth_tokens rows (0 disables the loop).
    AUTH_TOKEN_PURGE_INTERVAL_SECONDS = int(os.getenv("AUTH_TOKEN_PURGE_INTERVAL_SECONDS", "3600"))

    # CORS: comma-separated allowed origins. Production serves the API and the
    # frontend same-origin through nginx, so this only matters for cross-origin
    # dev clients (e.g. the Next.js dev server on :3000 hitting :8001).
    # Immutable (tuple) so the CORS allow-list can't be mutated at runtime; the
    # ClassVar marks it as a true class constant.
    CORS_ORIGINS: ClassVar[tuple[str, ...]] = tuple(
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8001").split(",")
        if o.strip()
    )

    # Hostnames this API answers to, enforced by TrustedHostMiddleware. The
    # default is derived from CORS_ORIGINS, this box's own name and addresses
    # (bound ones plus its default-route address), and the local dev/test
    # hosts, so a site reached by IP or by the box's own name keeps working
    # with no configuration at all. Set
    # ALLOWED_HOSTS explicitly (comma separated hostnames) when the API is
    # reachable under a name none of those cover — a separately registered
    # public domain, for instance. A wrong value here answers 400 to every
    # request; the effective list is logged at startup.
    # Immutable (tuple), like CORS_ORIGINS, so the allow-list can't be mutated
    # at runtime.
    ALLOWED_HOSTS: ClassVar[tuple[str, ...]] = _parse_allowed_hosts(
        os.getenv("ALLOWED_HOSTS"), CORS_ORIGINS + _machine_hosts()
    )


config = Config()
