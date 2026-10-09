import logging
import os
import re
import socket
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# The backend package's parent directory: the stable anchor for every data
# path. Resolved from this file's own location, never from the process working
# directory -- gunicorn, pm2, systemd, a shell and a test runner can each start
# the same app from a different CWD, and a CWD-relative data path turns that
# into a brand-new empty SQLite file created in the wrong place.
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _data_path(name: str, default: str) -> str:
    """Read a data-path knob and return it as an absolute path.

    A relative value (what ``.env.example`` ships, and what an operator copies)
    is resolved against :data:`BACKEND_ROOT`, so ``data/chat.db`` always means
    ``<repo>/backend/data/chat.db`` no matter where the process was started.
    An absolute value is kept as given, so a deployment that mounts its data on
    a separate volume is unaffected. ``~`` is expanded, because an operator
    writing ``~/data/chat.db`` means their home, not a literal ``~`` directory.

    Every result is absolute, so a caller can compare, log or stat it without
    re-deriving the same guess about the CWD.
    """
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

    Starlette compares the Host authority as
    ``headers.get("host", "").split(":")[0]`` -- everything before the FIRST
    colon -- so an IPv6 literal cannot be expressed in this allow-list at all,
    and truncating to the first hextet would be worse: ``2001`` matches ANY
    ``2001:*`` Host, turning the check into a fail-open on a guessable header.
    IPv6 is excluded by `_is_ipv6_literal` at every source instead, and serving
    an IPv6-only deployment needs a middleware that parses the authority.
    """
    host = entry.strip().lower().split("://", 1)[-1]
    return host.partition(":")[0]


def _is_ipv6_literal(entry: str) -> bool:
    """True for an address that a split-on-first-colon Host can never match.

    Both spellings occur: `getaddrinfo` and `getsockname` hand back unbracketed
    literals, while an operator writing `CORS_ORIGINS` is likely to bracket them.
    """
    host = entry.strip().lower().split("://", 1)[-1]
    if host.startswith("["):
        return True
    return host.count(":") > 1


def _machine_hosts() -> tuple[str, ...]:
    """Hostnames and addresses this box itself answers to.

    Production is same-origin through nginx behind a `server_name _` catch-all
    that forwards whatever `Host` the client used, and the documented posture
    leaves CORS_ORIGINS at its localhost default -- so neither CORS nor a
    hardcoded domain covers a site reached by IP or by the box's own name.
    Without them, every public request 400s. A separately registered public
    domain still has to be added to ALLOWED_HOSTS by the operator.

    Best effort by design: this runs at import, so a name-resolution failure must
    degrade to "fewer allowed hosts", never take the whole API down.
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

    `getaddrinfo(gethostname())` only yields the addresses bound to the box's own
    name, which on a NAT'd cloud host is the private one -- but the site is
    reached at the public address, and nginx forwards the client's `Host`
    through. Without the public address in the allow-list, every public request
    answers 400.

    A connected UDP socket sends no packets: it only asks the routing table which
    source address it would pick. Any failure just means one fewer allowed host,
    so every step is guarded: this runs at import, and an exception escaping here
    would take the whole API down, which is worse than a narrower allow-list.

    IPv4 only. An IPv6 source address cannot be put in the allow-list at all
    (see _normalize_host).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 53))
            return (probe.getsockname()[0],)
    except Exception:
        # Broad on purpose, and asserted by test so it cannot be narrowed
        # back: the only contract that matters here is "never raise at
        # import". DEBUG, not WARNING: a box with no default route is
        # unremarkable, so this would fire once per worker on an ordinary boot
        # and a warning traceback would be noise. The operator signal that
        # matters is the effective allow-list main.py logs at startup.
        logger.debug("default-route probe failed", exc_info=True)
        return ()



def _clamped_int(name: str, default: int, low: int, high: int) -> int:
    """Read an integer env knob, clamped into ``[low, high]``.

    Clamp-and-warn, not raise. This module is imported at process start, so
    raising here would turn a mistyped deployment value into a boot failure of
    the whole API -- and the knobs guarded by this helper are throughput caps
    whose *failure* mode is expensive CPU, not a wrong answer. Clamping keeps
    the service up and bounds the cost; the WARNING naming the key, the
    rejected value and the bound keeps the misconfiguration visible in the
    logs rather than silently papering over it.

    A non-integer value falls back to ``default`` for the same reason: an
    unparseable knob is an operator typo, not a client input, and the safest
    reading of it is "not configured". ``default`` itself must sit inside
    ``[low, high]``; violating that raises ``ValueError``, and deliberately not
    via ``assert`` so the guarantee survives ``python -O``.
    """
    # An explicit raise, not an `assert`: CPython strips asserts under
    # `python -O`, which would turn a stated guarantee into no guarantee.
    # The assert stays only as a readable marker, never as the enforcement.
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

    Unset or blank falls back to the local dev/test hosts plus whatever
    extra_hosts carries (the CORS origins and this box's own identities), so a
    missing knob keeps localhost, the dev stack, the test client and the real
    deployment working without ever opening the check. An explicit value
    replaces that default wholesale: a bare "*" is rejected outright (it would
    silently disable the check, which is the exact opposite of the knob's
    purpose) and a value that contains no usable hostname is rejected too,
    because it would otherwise match nothing and 400 every request.

    A wildcard is only accepted in the one shape TrustedHostMiddleware itself
    supports, a leading ``*.``. Any other placement is rejected HERE rather
    than left to the middleware, because ``add_middleware`` defers building the
    middleware stack to the first request: a malformed pattern such as
    ``a.*.com`` would otherwise boot cleanly and then turn every single request
    into a 500 from the middleware's own ``assert``.
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


# The spellings that mean "on" and "off" for a boolean env knob, for every knob
# in this file. One set, two readers: _env_tristate (three-state) and _env_bool
# (two-state). A second convention here is how ENABLE_DIVERSITY=" true " came
# to read as OFF while AUTH_TRUST_X_FORWARDED_FOR understood it.
#
# The truthy set is load-bearing OUTSIDE this file: setup.sh's forced-true
# warning greps exactly these four spellings, because a forced True trusts
# X-Forwarded-For from ANY peer, so a client reaching the API port directly can
# forge it to dodge a per-IP rate limit, and that warning is the only signal
# the operator gets. Add a spelling here and setup.sh's regex must add it in
# the same commit. The two are tied together by test, not by comment.
_TRUE_SPELLINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_SPELLINGS = frozenset({"0", "false", "no", "off"})


def _env_tristate(name: str) -> bool | None:
    """Read a three-state boolean env var: True/False force a behaviour, None
    means "auto" (the variable is unset, or says so explicitly).

    An unrecognised value falls back to None rather than to a forced side, so
    a typo in an operator's .env can never silently pick the unsafe one.
    """
    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE_SPELLINGS:
        return True
    if raw in _FALSE_SPELLINGS:
        return False
    return None


def _env_bool(name: str, default: bool) -> bool:
    """Read a two-state boolean knob, normalising case and surrounding space.

    The old inline parse lowercased but did not strip and had no "on" in its
    set, so ``on``, " true " and "1 " all read as OFF: a whole retrieval feature
    sat disabled with nothing wrong visible anywhere.

    Three input classes, and the differences between them are deliberate:

    - unset -> ``default``.
    - a known spelling -> that side, after strip/lower.
    - set but BLANK -> False, with a warning. Blank keeps meaning exactly what
      it means today, which is off, because ``KEY=`` in a .env is how an
      operator clears a knob. Falling back to ``default`` here would be the
      worst possible bug in this function: .env.example ships all eight
      toggles as ``true``, so a deployment that blanked one to turn it off
      would silently get the feature switched back ON.
    - anything else -> ``default``, with a warning. Warn-and-default, agreeing
      with _clamped_int: raising ValueError would turn a mistyped deployment
      value into a boot failure of the whole API, and a knob that only picks a
      feature is a bad trade for an outage.
    """
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
        # Checked before the warning below so the two failures read
        # differently: a blank knob is off, a mistyped one is the default.
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
    """Create ``path``'s parent directory, or fail with an actionable message.

    The failure mode this refuses to swallow: the parent exists but is not
    writable. Here `os.makedirs(exist_ok=True)` SUCCEEDS and `sqlite3.connect`
    then succeeds too, creating a fresh EMPTY database that the app serves as
    if it simply had no history.

    Raises RuntimeError naming the env var, the absolute path and the reason, so
    the operator sees the fix in the first lines of the startup log instead of
    discovering it as "all my conversations are gone" days later.
    """
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
    """Validate every configured data location before the app serves traffic.

    Called from the FastAPI lifespan ahead of the store connections, so a
    misconfigured data location is a loud boot failure rather than a running
    process that answers every request as if the deployment were brand new.

    Only the paths the app WRITES are checked. A missing
    ``QUERY_FIX_VOCAB_PATH`` file is a legitimate no-op (typo correction
    degrades by design), so only its directory has to be usable.
    ``RERANK_ONNX_DIR`` is inert and is deliberately not validated: nothing
    reads it, so requiring the directory would fail a perfectly healthy deploy.
    """
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
    # The API key qdrant itself is started with (setup.sh sets
    # QDRANT__SERVICE__API_KEY on the container). None when the store runs
    # unauthenticated, and the client is then built without the kwarg rather
    # than with an empty string, because qdrant-client sends whatever it is
    # given as the api-key header.
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vccircle_articles")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    EMBED_DIM = 768  # matches bge-base; change if you swap models
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
    EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
    # Kept small so GUNICORN_WORKERS processes don't oversubscribe the box.
    TORCH_THREADS = _TORCH_THREADS

    # Indexed text limits. The dense embedder gets title+facets+summary only
    # (kept short so CPU builds stay fast); the sparse/lexical embedder gets the
    # full text including body so body keywords stay searchable.
    EMBED_DENSE_CHAR_LIMIT = int(os.getenv("EMBED_DENSE_CHAR_LIMIT", "1500"))
    EMBED_CHAR_LIMIT = int(os.getenv("EMBED_CHAR_LIMIT", "50000"))
    BODY_CHAR_LIMIT = int(os.getenv("BODY_CHAR_LIMIT", "50000"))
    # Per-source body excerpt sent to the chat LLM (the whole stored body when
    # this matches BODY_CHAR_LIMIT; lower it to cut prompt tokens/cost).
    CHAT_BODY_CHAR_LIMIT = int(os.getenv("CHAT_BODY_CHAR_LIMIT", "50000"))
    # Chat scales the source count to the query's requested 'top N' (capped here
    # so the LLM context stays bounded) and trims each source's body excerpt to
    # fit the total budget below.
    CHAT_MAX_SOURCES = int(os.getenv("CHAT_MAX_SOURCES", "20"))
    CHAT_TOTAL_BODY_CHARS = int(os.getenv("CHAT_TOTAL_BODY_CHARS", "400000"))
    # A comparison/intersection chat turn runs one full retrieval leg PER
    # entity, so the entity count is the turn's fan-out. A question can name an
    # unbounded number of proper nouns (MAX_CONTENT_LEN is 8000 chars, which an
    # attacker fills with 150+ of them), so "compare A and B and ..." could
    # otherwise fan out that many pipelines inside one turn. Above this cap the
    # multi-entity expansion is skipped and the question is answered as a single
    # query, which is what a 150-way comparison deserves anyway.
    CHAT_MAX_MULTI_ENTITIES = int(os.getenv("CHAT_MAX_MULTI_ENTITIES", "6"))
    # How many of those legs may be in flight at once. Every leg takes the
    # shared module-global inference_lock for its CPU rerank, so gathering them
    # all would only queue them on that lock.
    CHAT_MULTI_ENTITY_CONCURRENCY = int(os.getenv("CHAT_MULTI_ENTITY_CONCURRENCY", "4"))

    # Total characters of prior conversation replayed into the chat prompt.
    # CHAT_MAX_HISTORY_TURNS bounds the turn COUNT but not their SIZE, and every
    # replayed turn is untrusted text the model must read as data rather than
    # instructions, so the replay is also bounded by character budget.
    CHAT_HISTORY_CHAR_LIMIT = int(os.getenv("CHAT_HISTORY_CHAR_LIMIT", "12000"))

    # Ceiling on the raw query embedded in a Redis cache key. A query longer
    # than this is replaced by a truncated sha256 digest (see
    # main._cache_key_component) so the key stays short and bounded while
    # remaining deterministic — a long query must not silently share a key
    # with a different long query, which is why the digest replaces the text
    # rather than the text being cut. Clamped because 0 here would digest EVERY
    # key and a huge one would hand the raw text back to Redis.
    CACHE_KEY_QUERY_MAX_CHARS = _clamped_int("CACHE_KEY_QUERY_MAX_CHARS", 128, 8, 4096)

    # Ceiling on the query text the SHARED retrieval path hands to the
    # transformers (hybrid_search's dense/sparse encode, rerank's
    # cross-encoder pairs, body_rescue's second pass). Deliberately NOT equal
    # to SEARCH_QUERY_MAX_CHARS: /search refuses anything longer than that at
    # the HTTP edge, so this bound never binds for it, while chat accepts up to
    # chat.MAX_CONTENT_LEN (8000) and puts the whole message in the LLM prompt
    # — clamping its retrieval lower would silently drop the caller's own words
    # from the search while the model still read them, a relevance bug and not a
    # performance trade.
    # Clamped, not read raw: this is the one knob whose misconfiguration fails
    # as a WRONG ANSWER rather than as wasted CPU. A 0 would slice every query
    # to "" at all three clamp sites and silently empty every result set.
    RETRIEVAL_QUERY_MAX_CHARS = _clamped_int("RETRIEVAL_QUERY_MAX_CHARS", 8000, 64, 65536)

    # In-flight encode batches during indexing. Keep this small: CPU dense
    # encoding of a batch near max-token length uses ~1-2GB, so depth * batch
    # must fit in RAM.
    INDEXER_WORKERS = int(os.getenv("INDEXER_WORKERS", "2"))

    # Sparse (BM25) embeddings — must match the model used at index time
    SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

    # Reranker (cross-encoder) applied to RRF candidates before the top_k is
    # kept. Fewer candidates = faster CPU rerank.
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Clamped to [5, 50] — see _clamped_int. Both ends are real hazards:
    #   * High end: every candidate is one cross-encoder pair, and the whole
    #     batch runs under the process-wide `inference_lock` on TORCH_THREADS
    #     cores, so a large value serialises every other inference in the
    #     process behind one request. 50 already exceeds what any caller can
    #     consume — /search caps top_k at 50 and CHAT_MAX_SOURCES is 20.
    #   * Low end: below 5 there is no ranking left to do. `scripts/rerank_bench.py`
    #     passes this straight through as a Qdrant `limit`, where 0 returns
    #     nothing and a negative is invalid. A silently dead rerank is worse
    #     than a clamped one.
    RERANK_CANDIDATES = _clamped_int("RERANK_CANDIDATES", 12, 5, 50)
    # Reranker execution backend. 'torch' (sentence-transformers CrossEncoder)
    # is the only backend: the ONNX backend is not installable alongside the
    # pinned transformers version, so its code path was removed from
    # app/reranker.py. This knob is kept so existing deployments that set
    # RERANK_BACKEND keep working; any value other than 'torch' logs a warning
    # and uses torch.
    RERANK_BACKEND = os.getenv("RERANK_BACKEND", "torch")
    # Inert: local dir that held the exported ONNX cross-encoder cache when the
    # ONNX backend existed. Nothing reads it now; kept as a documented
    # placeholder (it is still listed in .env.example) rather than an env var
    # that silently disappears from deployed setups.
    RERANK_ONNX_DIR = _data_path("RERANK_ONNX_DIR", "data/reranker_onnx")

    # LLM (Google Gemini via OpenAI-compatible endpoint). Provide the API key
    # in GEMINI_API_KEY. Set GEMINI_MODEL to the model id you want to use.
    # An absent key and a placeholder key are both "chat cannot work", but they
    # are not the same fault and must be told apart: a placeholder makes 100% of
    # answers the canned fallback while the process still looks healthy, so
    # classify_gemini_api_key below is the single place that decides whether the
    # configured value can actually reach the LLM.
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
    LLM_PRICE_INPUT_PER_1M = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.25"))
    LLM_PRICE_OUTPUT_PER_1M = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "1.50"))
    # Conversion for displaying cost in Indian Rupees (INR). Fallback rate used
    # until the first live fetch succeeds and on feed failures (app/fx_rate.py).
    INR_PER_USD = float(os.getenv("INR_PER_USD", "95.60"))
    # Free, keyless USD→INR feed (open.er-api.com/v6/latest/USD). The live rate
    # replaces INR_PER_USD twice a day via the fx_rate background loop.
    FX_RATE_API_URL = os.getenv("FX_RATE_API_URL", "https://open.er-api.com/v6/latest/USD")
    # How often the fx_rate loop refreshes the USD→INR rate, in seconds.
    # 12 * 3600 = twice a day.
    FX_RATE_REFRESH_SECONDS = float(os.getenv("FX_RATE_REFRESH_SECONDS", "43200"))
    # Daily LLM spend cap in USD. Chat fails closed (no LLM calls) once today's
    # cumulative spend reaches this value (see app/cost_budget.py). 0 is a
    # deliberate opt-out for deployments that meter spend elsewhere; it is not
    # the default, because an unset cap is how unbilled spend happens.
    LLM_DAILY_BUDGET_USD = float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0"))
    # Per-billed-call hold taken against the cap BEFORE the call runs, so
    # concurrent turns contend for the same budget instead of each reading a
    # stale counter (see reserve() in app/cost_budget.py).
    LLM_CALL_RESERVE_USD = float(os.getenv("LLM_CALL_RESERVE_USD", "0.05"))
    # Lifetime of an unsettled hold. After this long the hold is swept and
    # CHARGED to the spend counter at its reserved amount, so a crashed billed
    # call stays charged for the rest of the day instead of becoming free spend.
    COST_RESERVATION_TTL_SECONDS = int(os.getenv("COST_RESERVATION_TTL_SECONDS", "900"))
    # Total-character cap on conversation history sent to the LLM. Turns the
    # cap is needed most (long histories) into the turns that cost the most.
    CHAT_MAX_HISTORY_CHARS = int(os.getenv("CHAT_MAX_HISTORY_CHARS", "24000"))

    TOP_K = int(os.getenv("TOP_K", "8"))
    # Minimum reranked relevance score for chat sources; weaker results are
    # dropped before the LLM sees them.
    ASK_MIN_SCORE = float(os.getenv("ASK_MIN_SCORE", "0.2"))
    # When the query itself resolves a category facet (dealtype/industry), that
    # facet filter IS the relevance signal, so the cross-encoder score only ranks
    # within an already on-topic set, and the gate drops to 0.
    ASK_MIN_SCORE_FACETED = float(os.getenv("ASK_MIN_SCORE_FACETED", "0.0"))
    # Answerability gates (app/answer_fallback.py). WEAK_RESULT_SCORE is the
    # score a reranked hit must exceed to count as "strong", and
    # WEAK_RESULT_MIN_STRONG is how many strong hits a result list must hold
    # before it is reported as weakly answered. Deliberately separate from
    # ASK_MIN_SCORE above: that gate is chat's inclusion filter, while this pair
    # judges the whole list the caller hands over. /search applies no inclusion
    # gate at all, so the pair is the only relevance bar its weak note sees.
    # The count is capped at the length of the list and floored at 1
    # (app/answer_fallback.py:results_are_weak), so a value of 0 or less
    # behaves as 1.
    WEAK_RESULT_SCORE = float(os.getenv("WEAK_RESULT_SCORE", "0.3"))
    WEAK_RESULT_MIN_STRONG = int(os.getenv("WEAK_RESULT_MIN_STRONG", "3"))
    # Score handed to every date-only fallback filler
    # (app/main.py:retrieve_by_date_window) when a temporal query's lexical
    # signal is too weak to fill the window. Its own knob rather than a shared
    # one with the inclusion gate above, so retuning the gate cannot silently
    # move the filler floor. The two are independent, which means this floor
    # has to be kept at or above ASK_MIN_SCORE for fillers to survive the chat
    # gate that filters sources by score: raise the gate without raising this
    # and the temporal fallback stops reaching the model.
    #
    # A warning, not a ValueError: refusing to boot over a mis-ordered pair of
    # scoring thresholds would be a worse failure than serving degraded answers
    # that are at least named in the logs. (_parse_allowed_hosts does raise for
    # ALLOWED_HOSTS=*, where the misconfiguration disables a security check
    # outright and there is no safe degraded mode to serve.)
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
    # Byte budget for the in-process fallback cache (the HybridCache degrades to
    # a per-worker LRU when Redis is unreachable). The entry cap alone cannot
    # bound memory because the shared cache mixes small search-result payloads
    # with large embedding vectors (~15KB each), so eviction drops the largest
    # entries first until the total is back under this budget.
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
    # Stronger recency weighting applied when the query itself expresses a
    # recency intent ('latest', 'recent', 'fresh'), so old evergreen articles
    # drop below newer ones instead of surfacing on relevance alone. Hard-window
    # phrases ('this week') are filtered separately and get no boost.
    RECENCY_BOOST_STRENGTH = float(os.getenv("RECENCY_BOOST_STRENGTH", "0.85"))
    RECENCY_BOOST_DECAY_DAYS = float(os.getenv("RECENCY_BOOST_DECAY_DAYS", "30.0"))

    # Retrieval-quality tuning (see app/query_expand.py, app/rerank_boost.py,
    # app/answer_fallback.py, app/query_fix.py). Toggles can be disabled per-deployment.
    ENABLE_QUERY_EXPANSION = _env_bool("ENABLE_QUERY_EXPANSION", True)
    ENABLE_ENTITY_BOOST = _env_bool("ENABLE_ENTITY_BOOST", True)
    ENABLE_WEAK_FALLBACK = _env_bool("ENABLE_WEAK_FALLBACK", True)

    # Query-string typo correction (app/query_fix.py): symspellpy over a
    # corpus-derived vocabulary + curated entities, applied before embedding.
    # The vocab is generated by scripts/build_query_vocab.py; when absent the
    # fixer is a no-op. Corrected strings also normalize the cache keys, so
    # repeated typos of the same query reuse the same cached results.
    ENABLE_QUERY_FIX = _env_bool("ENABLE_QUERY_FIX", True)
    QUERY_FIX_VOCAB_PATH = _data_path("QUERY_FIX_VOCAB_PATH", "data/query_vocab.json.gz")
    QUERY_FIX_MAX_EDIT = int(os.getenv("QUERY_FIX_MAX_EDIT", "2"))
    QUERY_FIX_MIN_COUNT = int(os.getenv("QUERY_FIX_MIN_COUNT", "5"))
    QUERY_FIX_MIN_TOKEN_LEN = int(os.getenv("QUERY_FIX_MIN_TOKEN_LEN", "3"))

    # Result diversity (app/diversity.py): greedy MMR over the reranked set to
    # avoid near-duplicate headlines filling the top-k. LAMBDA near 1 favours
    # pure relevance; lower trades relevance for headline diversity. Applied in
    # /search before the final top-k slice.
    ENABLE_DIVERSITY = _env_bool("ENABLE_DIVERSITY", True)
    DIVERSITY_LAMBDA = float(os.getenv("DIVERSITY_LAMBDA", "0.7"))
    DIVERSITY_SIM_THRESHOLD = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.4"))

    # Click-driven learning (app/click_boost.py): per-query per-article click
    # aggregates (analytics Redis) boost results users actually open. Inert until
    # a query accumulates >= CLICK_BOOST_MIN_CLICKS clicks and an article holds
    # >= CLICK_BOOST_MIN_ARTICLE_CLICKS clicks (>= CLICK_BOOST_MIN_SHARE of the
    # query's total), so it never fires on sparse/noisy traffic.
    ENABLE_CLICK_BOOST = _env_bool("ENABLE_CLICK_BOOST", True)
    CLICK_BOOST_MIN_CLICKS = int(os.getenv("CLICK_BOOST_MIN_CLICKS", "5"))
    CLICK_BOOST_MIN_ARTICLE_CLICKS = int(os.getenv("CLICK_BOOST_MIN_ARTICLE_CLICKS", "3"))
    CLICK_BOOST_MIN_SHARE = float(os.getenv("CLICK_BOOST_MIN_SHARE", "0.3"))
    CLICK_BOOST_MULT = float(os.getenv("CLICK_BOOST_MULT", "1.3"))

    # Window over which one client IP contributes at most one click to a given
    # (query, article) pair to the click-boost signal. An hour is long enough
    # that a real user re-opening the same result does not cast a second vote,
    # and short enough that genuine later interest still registers. 0 disables
    # the dedupe.
    #
    # This is the control click-boost forgery actually relies on. The
    # THRESHOLDS above are deliberately left at their shipped 5/3/0.3: raising
    # them would blunt a forged burst, but blunt legitimate signal just as hard,
    # and that is a ranking-tuning decision to make on measurement.
    CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS = int(os.getenv("CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS", "3600"))

    # Bounds on stored query strings so a hostile client can't grow Redis
    # without limit: cap the stored query length and expire the per-query click
    # sorted sets (and the /search top_queries aggregate) a few days after the
    # last write. Shared by the /analytics/click beacon and /search tracking.
    CLICK_QUERY_MAX_LEN = int(os.getenv("CLICK_QUERY_MAX_LEN", "256"))
    CLICK_QUERY_TTL_SECONDS = int(os.getenv("CLICK_QUERY_TTL_SECONDS", str(7 * 24 * 3600)))

    # Secret mixed into the per-query digest that stands in for a search query
    # in the analytics aggregates. Left empty, the app generates a random key on
    # first use and persists it in the analytics Redis, so every gunicorn worker
    # and every restart shares one key. Set it only to pin the digest namespace
    # across an analytics-Redis rebuild; changing it makes previously stored
    # digests unreachable, which resets the top-query lists but leaks nothing.
    ANALYTICS_QUERY_KEY = os.getenv("ANALYTICS_QUERY_KEY", "")

    # Daily LLM cost counter TTL: kept well past the day it tracks so the budget
    # guardrail survives brief outages, then auto-expires instead of accumulating.
    COST_DAY_TTL_SECONDS = int(os.getenv("COST_DAY_TTL_SECONDS", str(7 * 24 * 3600)))

    # Search-analytics additions (see analytics.py): per-session zset TTL and the
    # daily top-queries set TTL. Both are conservative and never nil.
    ANALYTICS_SESSION_TTL_HOURS = _clamped_int("ANALYTICS_SESSION_TTL_HOURS", 24, 1, 24 * 365)
    TOP_QUERIES_TODAY_TTL_DAYS = _clamped_int("TOP_QUERIES_TODAY_TTL_DAYS", 7, 1, 90)

    # Throttle interval for the users.last_seen touch. An in-process per-user set
    # suppresses a write for this many seconds between touches.
    LAST_SEEN_TOUCH_INTERVAL_SECONDS = _clamped_int("LAST_SEEN_TOUCH_INTERVAL_SECONDS", 300, 0, 3600)

    # How many admin_audit rows /analytics/users reports (last N, newest first).
    AUDIT_RECENT_LIMIT = _clamped_int("AUDIT_RECENT_LIMIT", 20, 1, 100)

    # Chat-only "body rescue": when the top reranked score is below
    # BODY_RESCUE_THRESHOLD, re-score the candidates against the body region
    # with the most lexical query-token overlap and keep max(baseline, body).
    # Lets deep-body matches (e.g. historical retrospectives whose relevant
    # facts live mid-article) pass the chat relevance gate; costs one extra
    # cross-encoder pass per candidate and only runs on weak-top results.
    #
    # Left ON by default: turning it off is a relevance regression, not a
    # performance fix. Its cost is bounded by the three clamped knobs below
    # instead -- the expensive part is the second cross-encoder pass, which is
    # what BODY_RESCUE_MAX_CANDIDATES bounds.
    ENABLE_BODY_RESCUE = _env_bool("ENABLE_BODY_RESCUE", True)
    BODY_RESCUE_THRESHOLD = float(os.getenv("BODY_RESCUE_THRESHOLD", "0.3"))
    # WINDOW is the size of the excerpt handed to the cross-encoder. Below 200
    # the excerpt is too small to carry a useful passage (and a 0 window makes
    # `_best_body_window` return an empty string, silently disabling the rescue
    # while still paying for the pass); above 8000 it inflates the model's
    # input for every candidate in the rescue batch.
    BODY_RESCUE_WINDOW = _clamped_int("BODY_RESCUE_WINDOW", 1500, 200, 8000)
    # STEP is the sliding-window stride over the body. A 0 is a hard crash
    # (`range(0, n, 0)` raises ValueError and 500s the chat turn) and a small
    # step re-scans the whole body.
    BODY_RESCUE_STEP = _clamped_int("BODY_RESCUE_STEP", 500, 1, 1500)
    # Hard budget on how many windows are scored per body, independent of the
    # stride. Clamping STEP alone does NOT bound the work, because a small
    # stride is legal: step=1 still scans 48,501 windows of a 50K body, and
    # body_rescue scans every body-bearing article before the candidate cap
    # applies, so 20 articles cost ~2.3s. `_best_body_window` widens the stride
    # to fit this budget, and 200 is above the 98 windows the defaults scan.
    # Recall trade-off, stated rather than implied: a budget tighter than the
    # configured stride needs widens the stride, and a stride coarse enough to
    # widen can straddle a token-dense region and miss it. Free at any budget
    # >= 98 (the default scan), so only a deliberately tight budget trades
    # recall.
    BODY_RESCUE_MAX_WINDOWS = _clamped_int("BODY_RESCUE_MAX_WINDOWS", 200, 1, 5000)
    # Most candidates that may enter the second cross-encoder pass, the dominant
    # cost (one pair per candidate, under `inference_lock`), while chat hands
    # body_rescue up to CHAT_MAX_SOURCES (20) articles.
    BODY_RESCUE_MAX_CANDIDATES = _clamped_int("BODY_RESCUE_MAX_CANDIDATES", 10, 1, 50)

    # Upper bound on the /search `q` parameter, in characters: q reaches the
    # embedding encoders, the cache key and every query_intent regex, so an
    # unbounded value costs CPU and Redis memory per request. Set well above
    # CLICK_QUERY_MAX_LEN (256) because a rejected search is user-visible whereas
    # a truncated stored query string is not.
    SEARCH_QUERY_MAX_CHARS = _clamped_int("SEARCH_QUERY_MAX_CHARS", 512, 32, 4000)

    # Chat history (SQLite on the host; survives restarts, unlike Redis without
    # AOF). Retention purges conversations idle for CHAT_RETENTION_DAYS.
    CHAT_DB_PATH = _data_path("CHAT_DB_PATH", "data/chat.db")
    CHAT_RETENTION_DAYS = int(os.getenv("CHAT_RETENTION_DAYS", "180"))
    CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
    CHAT_PURGE_INTERVAL_SECONDS = int(os.getenv("CHAT_PURGE_INTERVAL_SECONDS", "86400"))
    # Messages returned by GET /api/chat/sessions/{id}. A session lives for
    # CHAT_RETENTION_DAYS and each message deserialises its sources JSON, so an
    # unbounded read of a long thread serialises the entire history into one
    # response. The read returns the MOST RECENT CHAT_SESSION_MESSAGE_LIMIT
    # messages in chronological order and flags the truncation in the response.
    # Deliberately looser than CHAT_MAX_HISTORY_TURNS/CHAT_MAX_HISTORY_CHARS,
    # which bound the *prompt*: the user can read further back in a thread than
    # the model is given context for, and that is expected.
    CHAT_SESSION_MESSAGE_LIMIT = int(os.getenv("CHAT_SESSION_MESSAGE_LIMIT", "200"))
    # Sources returned per message on the same read. CHAT_MAX_SOURCES (20) caps
    # what is *generated*; this caps what is *serialized back* per message, so a
    # message stored with more sources than the cap cannot multiply the response.
    CHAT_MESSAGE_SOURCE_LIMIT = int(os.getenv("CHAT_MESSAGE_SOURCE_LIMIT", "20"))

    ENABLE_RECOMMENDATIONS = _env_bool("ENABLE_RECOMMENDATIONS", True)
    RECOMMEND_SIMILARITY_WEIGHT = float(os.getenv("RECOMMEND_SIMILARITY_WEIGHT", "0.4"))
    RECOMMEND_CATEGORY_WEIGHT = float(os.getenv("RECOMMEND_CATEGORY_WEIGHT", "0.3"))
    RECOMMEND_RECENCY_WEIGHT = float(os.getenv("RECOMMEND_RECENCY_WEIGHT", "0.2"))
    RECOMMEND_POPULARITY_WEIGHT = float(os.getenv("RECOMMEND_POPULARITY_WEIGHT", "0.1"))
    USER_INTERACTION_TTL_DAYS = int(os.getenv("USER_INTERACTION_TTL_DAYS", "90"))
    # Per-user cap on DISTINCT articles an account may record an interaction
    # with. Every distinct article_id mints an ``article:interactions:{id}``
    # hash plus a ``user:interaction_detail:{user_id}:{id}`` key, both held for
    # USER_INTERACTION_TTL_DAYS, so without a cap any self-signup'd account can
    # grow the profile keyspace without bound. 0 disables the cap.
    USER_MAX_DISTINCT_INTERACTIONS = int(os.getenv("USER_MAX_DISTINCT_INTERACTIONS", "500"))
    RECOMMEND_DEFAULT_LIMIT = int(os.getenv("RECOMMEND_DEFAULT_LIMIT", "10"))
    # Width of the candidate pool the personalized recommendation legs and the
    # cold-start fallback fetch, deliberately wider than the result page so
    # scoring and exclusion post-processing have something to choose from (see
    # _candidate_pool in app/recommender.py).
    RECOMMEND_CANDIDATES_LIMIT = int(os.getenv("RECOMMEND_CANDIDATES_LIMIT", "50"))
    USER_PROFILE_REDIS_DB = int(os.getenv("USER_PROFILE_REDIS_DB", "2"))
    TRENDING_VELOCITY_WINDOW_DAYS = int(os.getenv("TRENDING_VELOCITY_WINDOW_DAYS", "7"))

    # Aggregates live in Redis DB 1 (the query cache uses DB 0 and is flushed
    # during deploys). Read endpoints are gated by the auth layer (admin role).
    ANALYTICS_REDIS_DB = int(os.getenv("ANALYTICS_REDIS_DB", "1"))

    # Auth (token + RBAC). Users sign up openly; a role-based access-control
    # layer maps roles to permissions (see app/auth.py). Tokens are opaque,
    # hashed (SHA-256) in storage, expire after AUTH_TOKEN_TTL_DAYS, and can be
    # revoked individually.
    AUTH_DB_PATH = _data_path("AUTH_DB_PATH", "data/auth.db")
    # Redis DB holding the auth rate-limit counters. Pinned explicitly, like
    # ANALYTICS_REDIS_DB and USER_PROFILE_REDIS_DB, rather than inherited from
    # any db segment in REDIS_URL: the inherited value was DB 0, which this repo
    # documents as the query cache and flushes during deploys, so the limiter's
    # counters were living in the database a deploy empties and a flush
    # silently reset every bucket. These counters are a security control, so
    # they get their own database: cache 0, analytics 1, profiles 2, rate
    # limiting 3. The `db` kwarg on from_url overrides whatever the URL carries.
    AUTH_RATE_LIMIT_REDIS_DB = int(os.getenv("AUTH_RATE_LIMIT_REDIS_DB", "3"))
    AUTH_TOKEN_TTL_DAYS = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "7"))
    # Optional machine-to-machine credential: the value carried in an
    # X-Service-Token header. Leave empty to disable. Never expose it to
    # browsers.
    #
    # This is a SEED, not a standing grant. The first request presenting it
    # creates a row in auth_service_tokens with the scope and expiry below;
    # from then on that row is the only authority on the token's life.
    #
    # OPERATIONAL CONSEQUENCE, deliberate: the seeded token expires
    # AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS after it is first seeded and a restart
    # does NOT revive it, because a credential that silently came back would be
    # the permanent grant this replaced. Recovery is rotation -- change the
    # value here and restart, or mint one with POST /api/auth/service-tokens.
    # Session cookie. The credential is no longer read from an
    # `Authorization: Bearer` header: a header the app writes into a JSON body is
    # reachable by any script that runs on the page, so an XSS bug exfiltrates a
    # durable account takeover. The token now travels only in an HttpOnly
    # cookie, which script cannot read. See app/auth.py for the contract.
    #
    # No Domain attribute is ever set on the cookie: host-only is deliberate so
    # the same image works on a bare-IP deployment where a Domain would have to
    # encode an address the operator may not control.
    AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "vccircle_session")
    # Defaulted off from the old localStorage key name on purpose: keeping the
    # old name would let a stale credential silently authenticate through a
    # different transport.
    #
    # SameSite=Lax still sends the cookie on top-level GET navigations, which is
    # what keeps deep links working, while blocking the cookie on cross-site
    # POSTs (the CSRF shape). `strict` is tighter but breaks in-app navigation
    # from an external link, so it is an operator choice, not a default.
    AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")
    # Secure defaults to TRUE unless an operator explicitly sets it false.
    # Defaulting on is deliberate: a session cookie that crosses plaintext HTTP
    # is trivially captured by anything on the path, which is a worse failure
    # than a login that does not persist. A plain-HTTP deployment (setup.sh with
    # NGINX_TLS=off) MUST set AUTH_COOKIE_SECURE=false or login will silently
    # not persist. That is a deployment dependency, not something to auto-
    # detect: the app sits behind TLS termination and cannot trust
    # request.url.scheme or X-Forwarded-Proto to work it out.
    AUTH_COOKIE_SECURE = _env_tristate("AUTH_COOKIE_SECURE") is not False
    # Not configurable. It must stay "/": the API serves /api/... , and a
    # narrower path would silently stop the cookie from ever reaching it.
    AUTH_COOKIE_PATH = "/"
    # Derived from the token TTL so the cookie and the server-side record expire
    # together; not independently configurable, or the two could drift apart.
    AUTH_COOKIE_MAX_AGE_SECONDS = AUTH_TOKEN_TTL_DAYS * 86400
    AUTH_SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")
    # Lifetime of a service token. Not optional: a value <= 0 falls back to
    # the default rather than meaning "never expires", because an eternal
    # machine admin credential is exactly the hole this closes.
    AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS = int(os.getenv("AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS", "86400"))
    # Permissions a service token may exercise. Defaults to chat:use, the only
    # permission the in-repo consumer (scripts/eval_runner.py) needs, instead
    # of every permission an admin holds.
    AUTH_SERVICE_TOKEN_SCOPE: ClassVar[tuple[str, ...]] = tuple(
        p.strip()
        for p in os.getenv("AUTH_SERVICE_TOKEN_SCOPE", "chat:use").split(",")
        if p.strip()
    )
    # Bootstrap admin: created once at startup (role=admin) if no account with
    # this email exists. An existing account is never overwritten.
    AUTH_ADMIN_EMAIL = os.getenv("AUTH_ADMIN_EMAIL", "")
    AUTH_ADMIN_PASSWORD = os.getenv("AUTH_ADMIN_PASSWORD", "")
    AUTH_PASSWORD_MIN_LEN = int(os.getenv("AUTH_PASSWORD_MIN_LEN", "8"))
    AUTH_MAX_EMAIL_LEN = int(os.getenv("AUTH_MAX_EMAIL_LEN", "254"))
    AUTH_MAX_NAME_LEN = int(os.getenv("AUTH_MAX_NAME_LEN", "60"))
    # Redis-backed per-IP rate limits on the public auth endpoints (0 disables).
    AUTH_SIGNUP_RATE_PER_MIN = int(os.getenv("AUTH_SIGNUP_RATE_PER_MIN", "5"))
    AUTH_LOGIN_RATE_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_MIN", "10"))
    AUTH_RATE_WINDOW_SECONDS = int(os.getenv("AUTH_RATE_WINDOW_SECONDS", "60"))
    # Per-ACCOUNT (submitted address) limit on login, counted in addition to
    # the per-IP one above. Per-IP alone cannot see a botnet hammering ONE
    # account: every request arrives from a fresh address with a fresh bucket.
    # The counter is keyed on the normalised submitted address alone, so its
    # state and its 429 are the same whether or not the address has an account
    # here, which keeps it from being an account-existence oracle.
    #
    # It counts FAILED attempts only, and is applied after the credential
    # check. That is deliberate: counting every attempt, and gating before the
    # check, turned the throttle into an account-lockout weapon -- an anonymous
    # caller could deny a known address access indefinitely by sending the
    # limit's worth of wrong passwords from rotating source addresses, never
    # guessing anything.
    #
    # What it does NOT buy is attacker cost: the check runs after the bcrypt
    # verify, so being refused is free to the caller. AUTH_LOGIN_RATE_PER_MIN is
    # the control that bounds attacker cost. 0 disables.
    AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN = int(os.getenv("AUTH_LOGIN_RATE_PER_ACCOUNT_PER_MIN", "20"))
    # Cap on simultaneously ACTIVE (unexpired) tokens per user. Every login mints
    # one, and the periodic purge only removes EXPIRED rows, so the table grew
    # with the number of logins rather than with the number of users -- an
    # unbounded-growth / DoS vector on the auth store. Logging in past the cap
    # REVOKES (deletes) the user's oldest active tokens, so the evicted
    # credential stops working immediately. 0 disables the cap.
    AUTH_MAX_ACTIVE_TOKENS_PER_USER = int(os.getenv("AUTH_MAX_ACTIVE_TOKENS_PER_USER", "10"))
    # Redis-backed per-IP rate limits on the public search surface: /search,
    # /facets, /analytics/click and /ready were unauthenticated and unrated,
    # which allowed full-corpus scraping (top_k=50) and click-analytics
    # poisoning. 0 disables an individual limit. Unlike the auth limits these
    # FAIL CLOSED (503) when Redis is unreachable: these endpoints are the
    # abuse surface, so an unrated request is not an acceptable fallback.
    # /ready is the one deliberate exception and fails open instead.
    PUBLIC_SEARCH_RATE_PER_MIN = int(os.getenv("PUBLIC_SEARCH_RATE_PER_MIN", "60"))
    PUBLIC_FACETS_RATE_PER_MIN = int(os.getenv("PUBLIC_FACETS_RATE_PER_MIN", "60"))
    PUBLIC_CLICK_RATE_PER_MIN = int(os.getenv("PUBLIC_CLICK_RATE_PER_MIN", "120"))
    # POST /recommend/interaction is reachable by ANY self-signup'd account
    # (require_auth only) and mints Redis keys per call, so it is limited on
    # BOTH axes: per client IP here, and per account via
    # INTERACTION_USER_RATE_PER_MIN. A per-IP bucket alone cannot bound one
    # account behind a shared NAT/proxy address, which is the shape a
    # deliberate flood takes.
    PUBLIC_INTERACTION_RATE_PER_MIN = int(os.getenv("PUBLIC_INTERACTION_RATE_PER_MIN", "60"))
    INTERACTION_USER_RATE_PER_MIN = int(os.getenv("INTERACTION_USER_RATE_PER_MIN", "60"))
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
    # nginx config in setup.sh forwards from 127.0.0.1). Set the variable to
    # true/false to force one behaviour regardless of peer.
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
    # (bound ones plus its default-route address), and the local dev/test hosts,
    # so a site reached by IP or by the box's own name keeps working with no
    # configuration at all. Set ALLOWED_HOSTS explicitly (comma separated) when
    # the API is reachable under a name none of those cover. A wrong value here
    # answers 400 to every request; the effective list is logged at startup.
    ALLOWED_HOSTS: ClassVar[tuple[str, ...]] = _parse_allowed_hosts(
        os.getenv("ALLOWED_HOSTS"), CORS_ORIGINS + _machine_hosts()
    )

    # Level for the app's own loggers, read by app/logging_config.py at startup.
    # It exists because uvicorn's worker leaves the root logger at WARNING with
    # no handlers, so every logger.info in the app was dropped. Only the app's
    # loggers take this level; the root logger's level is left alone, so a
    # third-party logger's INFO output stays off at every setting. An
    # unrecognised value falls back to INFO and says so in the log.
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    # Personalized feed (app/feed.py). Users subscribe to tag/industry/dealtype
    # values, and GET /api/feed returns the most recent articles matching ANY
    # subscription. Subscriptions live in their own SQLite file (like chat and
    # auth stores), keyed by (user_id, kind, value) so re-adding an existing
    # subscription is a no-op. FEED_MAX_SUBSCRIPTIONS bounds how many a single
    # account may hold (beyond the cap POST 400s on a NEW subscription);
    # FEED_DEFAULT_LIMIT is the recency list page size (bounded 1..50 by the
    # route's Query param).
    FEED_DB_PATH = _data_path("FEED_DB_PATH", "data/feed.db")
    FEED_MAX_SUBSCRIPTIONS = int(os.getenv("FEED_MAX_SUBSCRIPTIONS", "50"))
    FEED_DEFAULT_LIMIT = int(os.getenv("FEED_DEFAULT_LIMIT", "20"))


# A real Google API key is "AIza" followed by 35 URL-safe characters. That shape
# is only required against Google's own endpoint: GEMINI_BASE_URL is
# configurable, so a deployment pointing at an OpenAI-compatible gateway
# legitimately holds a differently shaped key.
_GOOGLE_KEY_RE = re.compile(r"^AIza[0-9A-Za-z_-]{35}$")
_GOOGLE_API_HOST = "generativelanguage.googleapis.com"
_GOOGLE_KEY_PREFIX = "AIza"

# Filler spellings that turn up in templates, docs and copy-pasted examples,
# compared in normalised form: lowercased with every non-alphanumeric character
# removed, so "your_key_here", "YOUR-KEY-HERE" and "<your key here>" all reduce
# to the same token.
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

# A run of one repeated character ("xxxx", "aaaaaa") is filler, never a key.
# Checked against the normalised form, so "x-x-x-x" is caught as well.
_REPEATED_FILLER_RE = re.compile(r"(.)\1{3,}")

# ...but that check cannot see inside a correctly shaped key: a MASKED key keeps
# its real prefix and its real length and fills the rest with filler, which is
# how documentation writes an example and how an operator redacts a key they are
# not sure about. So the tail after the prefix is also required to look random:
# a real 35-character tail drawn from a 64-symbol alphabet has ~27 distinct
# characters, while every masking style (all X, all digits, all dashes, a padded
# word) lands far below.
# NB: the example above is described, never written out -- a contiguous
# 39-character "AIza..." string anywhere in this repository, comment included,
# is indistinguishable from a leaked credential to a secrets scanner.
_MIN_DISTINCT_KEY_CHARS = 12


def classify_gemini_api_key(value: str | None) -> str:
    """Classify a configured GEMINI_API_KEY for readiness reporting.

    Returns one of:
      "ok"          -- a key that can plausibly be sent to the configured
                       LLM endpoint.
      "missing"     -- unset, empty, or whitespace only.
      "placeholder" -- a known filler value ("your_key_here" and friends), or
                       a correctly shaped key whose body is masked filler.
      "malformed"   -- neither missing nor filler, but the wrong shape for the
                       configured GEMINI_BASE_URL (a typo, or a truncated key).

    A truthiness test cannot do this job: every non-empty string is truthy and
    the value shipped in .env.example is the literal "your_key_here", so
    ``bool(key)`` reported a chat-broken deployment as a healthy one. The value
    itself is never returned or logged, only its classification.
    """
    if value is None or not value.strip():
        return "missing"
    stripped = value.strip()
    normalised = re.sub(r"[^0-9a-z]+", "", stripped.lower())
    if normalised in _PLACEHOLDER_API_KEYS or _REPEATED_FILLER_RE.fullmatch(normalised):
        return "placeholder"
    # Host names are case-insensitive, and a blank base URL is an operator who
    # cleared the variable rather than one who pointed it elsewhere: both keep
    # the structural check, so re-spelling the variable cannot switch it off.
    base_url = config.GEMINI_BASE_URL.strip().lower()
    if not base_url or _GOOGLE_API_HOST in base_url:
        if not _GOOGLE_KEY_RE.fullmatch(stripped):
            return "malformed"
        if len(set(stripped[len(_GOOGLE_KEY_PREFIX) :])) < _MIN_DISTINCT_KEY_CHARS:
            return "placeholder"
    return "ok"


config = Config()
