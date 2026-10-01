import asyncio
import ipaddress
import logging
import time

import redis
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from app import close_guard
from app.auth import public_rate_limit
from app.close_guard import close_quietly
from app.config import classify_gemini_api_key, config

logger = logging.getLogger("health")


router = APIRouter()

_redis_client: aioredis.Redis | None = None

# Teardown of a client that just failed its ping gets the same 2s budget as the
# ping itself (see _redis_status): a readiness probe must answer on time even
# when the connection it is releasing is dying.
_REDIS_CLOSE_TIMEOUT = 2.0


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness only: "this process is up and serving HTTP". Nothing more.

    It touches no dependency on purpose, which is what makes it a valid liveness
    probe -- it must keep answering while Qdrant is down, so a supervisor can
    tell "restart the process" apart from "the process is fine and something it
    depends on is not". It is deliberately NOT a readiness answer and must never
    be used to gate a deploy, drive a load balancer, or decide whether to alert:
    with no dependency inspected, a deployment holding a dead Qdrant client,
    unloaded models or a placeholder API key answers 200 here.

    Use /ready (a load balancer or orchestrator polling once per node, where
    the short cache and the rate limit are welcome) or /ready/deep (host-local
    monitoring: the deploy gate in setup.sh and the cron watchdog in
    deploy/healthcheck.sh) whenever a real answer is required.
    """
    return {"status": "ok"}


# Created eagerly at import rather than lazily inside _redis_status. A lazy
# `if lock is None: lock = asyncio.Lock()` has no await between the check and the
# assignment, so every caller already shares the first lock built; eager creation
# just makes that structural, with no optional state to check and no path in
# close_redis() that resets it. Python 3.10+ no longer binds an asyncio.Lock to
# an event loop at construction, so a module-level instance is safe.
_redis_init_lock: asyncio.Lock = asyncio.Lock()


async def _close_quietly(client: aioredis.Redis) -> None:
    """Best-effort release of a client we are about to discard.

    ``aioredis.Redis`` owns a connection pool, so dropping a reference without
    closing leaves the socket to the GC, and a client that just failed its ping
    can also fail to close. The expected teardown failures -- including the
    ``TimeoutError`` ``asyncio.wait_for`` raises past ``_REDIS_CLOSE_TIMEOUT`` --
    are swallowed: the caller only cares that readiness is degraded.

    Cancellation and unexpected programming errors deliberately propagate: a
    cancelled probe (client disconnect, server shutdown) must still unwind, and
    a broken close implementation must not be mistaken for a network failure.

    The mechanics live in ``app.close_guard``, shared with the lifespan teardown
    so both release a client under the same bound.
    """
    await close_quietly(
        "readiness Redis client",
        client,
        timeout=_REDIS_CLOSE_TIMEOUT,
        suppress=(redis.exceptions.RedisError, *close_guard.EXPECTED_CLOSE_ERRORS),
    )


async def _drop_redis_client(client: aioredis.Redis) -> None:
    """Invalidate the cached client under the init lock, and only if it is still
    the client that failed.

    The ping runs outside the lock (holding it across a 2s network call would
    serialize every readiness probe), so by the time a ping fails another
    caller may already have replaced the dead client with a fresh one. A blind
    ``_redis_client = None`` would discard that replacement - and leak its
    connection - forcing yet another reconnect for no reason.

    The lock is held only for the identity check and the swap: the failing client
    is closed afterwards, outside the critical section, so no I/O (and no
    re-entry into this non-reentrant lock) happens under it.
    """
    global _redis_client
    async with _redis_init_lock:
        if _redis_client is not client:
            return
        _redis_client = None
    await _close_quietly(client)


async def close_redis() -> None:
    """Close the lazily-created readiness-check Redis client. Registered as a
    shutdown hook so the connection isn't leaked on worker exit. The init lock is
    deliberately left untouched: it is built once at import and never reset."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


async def _qdrant_ok(state: dict) -> bool:
    """Qdrant client present and the collection check succeeds (bounded <3s).

    Every client/driver failure -- the expected ``TimeoutError`` and
    ``ApiException`` as much as an unexpected transport or driver error -- is a
    readiness failure, not a crash: letting one escape turned a Qdrant problem
    into a bodiless 500, which a probe cannot distinguish from a server bug. Only
    ``Exception`` is caught, so a cancellation still propagates.
    """
    client = state.get("qdrant")
    if client is None:
        return False
    try:
        await asyncio.wait_for(client.collection_exists(config.QDRANT_COLLECTION), timeout=2.5)
        return True
    except Exception:
        logger.warning("qdrant readiness probe failed", exc_info=True)
        return False


def _models_ok(state: dict) -> bool:
    return all(state.get(key) is not None for key in ("model", "sparse_model", "reranker"))


def _llm_status() -> tuple[bool, str]:
    """(usable, reason) for the configured Gemini key.

    Delegates to config.classify_gemini_api_key so readiness and the startup log
    can never disagree about the key. A bare ``bool(config.GEMINI_API_KEY)``
    answered "true" for any non-empty string, and the value shipped in
    .env.example is the literal placeholder "your_key_here" -- so a backend whose
    every chat answer is the canned fallback reported itself healthy.
    """
    reason = classify_gemini_api_key(config.GEMINI_API_KEY)
    return reason == "ok", reason


async def _redis_status() -> tuple[bool, str]:
    """Reachability of Redis with a 2s-bounded ping. Never fails readiness:
    the HybridCache degrades silently to in-process memory, so report the
    effective cache mode instead."""
    global _redis_client
    if not config.REDIS_URL:
        return True, "memory"
    async with _redis_init_lock:
        if _redis_client is None:
            try:
                _redis_client = aioredis.from_url(
                    config.REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
                )
            except (TimeoutError, redis.exceptions.RedisError, OSError):
                return False, "down"
        client = _redis_client
    if client is None:
        return False, "degraded"
    try:
        await asyncio.wait_for(client.ping(), timeout=2.0)
    except (TimeoutError, redis.exceptions.RedisError, OSError):
        await _drop_redis_client(client)
        return False, "degraded"
    return True, "redis"


# A load balancer that polls /ready every second would otherwise re-run both
# dependency probes on every poll, so one slow dependency multiplies into
# sustained probe load. The cached entry is plain data -- (deadline, ready,
# report) with a time.monotonic() deadline -- never an event-loop-bound object,
# so it stays valid across the per-test event loops the endpoint tests run in.
_readiness_cache: tuple[float, bool, dict] | None = None
# Single-flight for the cache MISS. The entry above bounds the SERIAL probe rate,
# but the moment it expires every request arriving in the same instant misses
# together and each would fan out its own Qdrant + Redis round; the rate limiter
# does not prevent that herd either, since it bounds arrival rate, not
# concurrency.
#
# Built LAZILY and per event loop, unlike _redis_init_lock, and both parts are
# load-bearing: this lock IS contended, and a contended acquire binds an
# asyncio.Lock to its running loop for good, so reusing it from another loop
# raises "is bound to a different event loop". The check and the assignment sit
# together with no await between them, so two callers in one loop cannot each
# build a lock and defeat the single-flight, and the lock is held in a local so a
# concurrent reset cannot swap the object out from under the `async with`.
_readiness_probe_lock: asyncio.Lock | None = None
_readiness_probe_loop: asyncio.AbstractEventLoop | None = None


def reset_readiness_cache() -> None:
    """Drop the cached readiness report so the next poll re-probes, along with
    the single-flight lock so neither is carried across event loops."""
    global _readiness_cache, _readiness_probe_lock, _readiness_probe_loop
    _readiness_cache = None
    _readiness_probe_lock = None
    _readiness_probe_loop = None


async def _cached_readiness_report(state: dict) -> tuple[bool, dict]:
    """Readiness report reused for READY_CACHE_TTL_SECONDS; a hit touches
    neither Qdrant nor Redis.

    A miss is single-flight: concurrent callers that miss together run the
    probes once between them, and the waiters re-check the cache under the lock
    and reuse the entry the winner just wrote."""
    global _readiness_cache, _readiness_probe_lock, _readiness_probe_loop
    cached = _readiness_cache
    if cached is not None and cached[0] > time.monotonic():
        return cached[1], cached[2]
    # Rebuild when the loop changes: a lock bound to a dead loop would refuse this
    # acquire. Callers within one loop all see the same object, so the
    # single-flight still holds.
    loop = asyncio.get_running_loop()
    lock = _readiness_probe_lock
    if lock is None or _readiness_probe_loop is not loop:
        lock = asyncio.Lock()
        _readiness_probe_lock = lock
        _readiness_probe_loop = loop
    async with lock:
        # Re-check: another caller may have refreshed the entry while this one
        # waited for the lock, in which case there is nothing left to probe.
        cached = _readiness_cache
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1], cached[2]
        ready, report = await _readiness_report(state)
        _readiness_cache = (now + config.READY_CACHE_TTL_SECONDS, ready, report)
        return ready, report


async def _probe_bounded(name: str, coro, default):
    """Run one dependency probe under its own explicit deadline.

    Probes are launched together and bounded individually, so the worst case is
    one timeout rather than the sum of both. A timeout is a dependency failure
    and yields ``default``; any other exception is a defect in the probe
    machinery, so it propagates and the endpoint answers 500 instead of
    claiming a dependency is down.
    """
    try:
        return await asyncio.wait_for(coro, timeout=config.READY_DEP_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.warning("%s readiness probe timed out after %ss", name, config.READY_DEP_TIMEOUT_SECONDS)
        return default


async def _readiness_report(state: dict) -> tuple[bool, dict]:
    # Both dependency probes are launched together: the report costs one probe
    # budget, not the sum of both. return_exceptions keeps a raising probe from
    # orphaning the other one mid-flight; the first real error is re-raised.
    qdrant_result, redis_result = await asyncio.gather(
        _probe_bounded("qdrant", _qdrant_ok(state), False),
        _probe_bounded("redis", _redis_status(), (False, "degraded")),
        return_exceptions=True,
    )
    for result in (qdrant_result, redis_result):
        if isinstance(result, BaseException):
            raise result
    qdrant_ok, (redis_ok, cache_mode) = qdrant_result, redis_result
    models_ok = _models_ok(state)
    llm_ok, llm_reason = _llm_status()
    # The LLM key gates readiness, not just the report: without a usable key every
    # chat answer is the canned fallback, so the deployment cannot do the one
    # thing it is deployed to do. Redis stays out of this sum on purpose (see
    # _redis_status): it degrades to an in-process cache, not to a wrong answer.
    ready = qdrant_ok and models_ok and llm_ok
    report = {
        "ready": ready,
        "checks": {
            "qdrant": {"ok": qdrant_ok},
            "models": {"ok": models_ok},
            "redis": {"ok": redis_ok, "cache": cache_mode},
            # reason is a classification ("missing" / "placeholder" /
            # "malformed" / "ok"), never the key itself.
            "llm": {"ok": llm_ok, "reason": llm_reason},
        },
    }
    return ready, report


# fail_closed=False: /ready is polled by load balancers and orchestrators, and a
# Redis outage is a degraded-but-serving state here (the HybridCache falls back
# to an in-process cache). Failing this limiter closed would pull healthy nodes
# out of rotation for a dependency the service does not need to be ready. It
# still counts every poll and still answers 429; only a broken limiter store is
# tolerated, which is what stops an unrouted slow-loris.
@router.get(
    "/ready",
    dependencies=[Depends(public_rate_limit("ready", "PUBLIC_READY_RATE_PER_MIN", fail_closed=False))],
)
async def ready() -> JSONResponse:
    """Readiness: 200 only when this node can actually serve, 503 otherwise.

    This is the real answer -- Qdrant reachable, models loaded, and a usable
    Gemini key configured -- and it is the endpoint a load balancer or an
    orchestrator polls. The result is cached for READY_CACHE_TTL_SECONDS and the
    probe is rate-limited, both right for a 1 Hz prober and both wrong for a
    watchdog that must see an outage the moment it starts: use /ready/deep.

    Unlike /health, this endpoint CAN fail, and its answer is allowed to flip
    from 200 to 503 while the process itself is perfectly healthy.
    """
    from app.main import state  # lazy: avoid circular import at startup

    try:
        ok, report = await _cached_readiness_report(state)
    except Exception:
        # A dependency that is down or timing out is reported as not-ready (503)
        # inside _readiness_report. Anything escaping it is a defect in the
        # probe machinery rather than an outage, so it must not be laundered
        # into a 503 that would tell the load balancer to stop sending traffic.
        logger.exception("readiness probe raised unexpectedly")
        return JSONResponse(status_code=500, content={"ready": False, "checks": {}, "error": "readiness probe failed"})
    return JSONResponse(status_code=200 if ok else 503, content=report)


@router.get(
    "/readyz",
    # The same limiter, and deliberately the same "ready" action, as /ready: this
    # alias runs the identical readiness probe, so it shares one budget rather
    # than handing a caller a second allowance by changing one path segment.
    dependencies=[Depends(public_rate_limit("ready", "PUBLIC_READY_RATE_PER_MIN", fail_closed=False))],
)
async def readyz() -> Response:
    from app.main import state  # lazy: avoid circular import at startup

    try:
        ok, _ = await _cached_readiness_report(state)
    except Exception:
        logger.exception("readiness probe raised unexpectedly")
        return Response(status_code=500)
    return Response(status_code=200 if ok else 503)


def _is_host_local_probe(request: Request) -> bool:
    """True for a direct loopback caller on this host, False for anything else.

    /ready/deep runs uncached and unrated (see its docstring), which is safe
    only if the internet cannot reach it, so the gate is deliberately narrow:

    * the socket peer must be a loopback address, which is the shape of a
      `curl` from setup.sh or deploy/healthcheck.sh running on this box; and
    * the request must carry no X-Forwarded-For. A request that arrived through
      the local reverse proxy (nginx on this host forwards every request with
      that header) is an internet request wearing a loopback peer's address,
      and is refused. A client cannot strip the header nginx sets.

    A non-loopback peer -- a TestClient, a container on the docker bridge, a
      different host -- is not host-local and is refused. Fail closed: a false
    negative costs a watchdog that cannot probe, while a false positive exposes
    an unrated, uncached dependency probe to the internet.
    """
    peer = request.client.host if request.client else None
    if not peer:
        return False
    try:
        if not ipaddress.ip_address(peer).is_loopback:
            return False
    except ValueError:
        return False
    return not request.headers.get("x-forwarded-for", "").strip()


@router.get("/ready/deep")
async def ready_deep(request: Request) -> Response:
    """Uncached, unrated readiness for host-local monitoring.

    The same readiness contract as /ready -- same report, same 200/503 -- with
    the two things a *watchdog* must not inherit deliberately removed:

    * no cache. /ready may answer from an entry written up to
      READY_CACHE_TTL_SECONDS ago, so a poll landing just after a dependency died
      gets the pre-outage verdict. A watchdog acts on that answer, so this one
      re-probes every time; it neither reads nor writes the shared entry.
    * no rate limit. Its callers act on the status code, and a 429 is
      indistinguishable from an outage to all of them: setup.sh's `wait_http`
      uses `curl -fsS`, so a throttled probe fails the deploy outright. Such
      callers also run on a timer or a single pass, far below the budget /ready
      needs for a load balancer polling once a second.

    Neither property is safe to hand to the internet, so the route is refused
    for any caller that is not a direct loopback request on this host
    (_is_host_local_probe). This is the endpoint for deploy/healthcheck.sh and
    for setup.sh's deploy gate; it is NOT a public status endpoint.
    """
    if not _is_host_local_probe(request):
        logger.warning("refused a non-host-local /ready/deep probe")
        return Response(status_code=403, content="host-local probe endpoint")
    from app.main import state  # lazy: avoid circular import at startup

    try:
        ok, report = await _readiness_report(state)
    except Exception:
        # Same reasoning as /ready: a defect in the probe machinery must not be
        # laundered into a verdict about the dependencies.
        logger.exception("readiness probe raised unexpectedly")
        return JSONResponse(status_code=500, content={"ready": False, "checks": {}, "error": "readiness probe failed"})
    return JSONResponse(status_code=200 if ok else 503, content=report)


def warn_if_llm_key_unusable() -> None:
    """Log the LLM key's usability once at startup.

    Deliberately a log line, not a raised error. Crashing would take /health
    with it, leaving the watchdog nothing to probe and turning a diagnosable
    "GEMINI_API_KEY is still the .env.example placeholder" into an opaque boot
    loop. /ready reports the same fault as not-ready with checks.llm.reason, and
    the process stays up to answer both probes.
    """
    ok, reason = _llm_status()
    if ok:
        return
    logger.error(
        "GEMINI_API_KEY is %s: chat will answer every question with the canned fallback. "
        "Set a real key (https://aistudio.google.com/apikey) in backend/.env; /ready reports "
        "checks.llm.reason=%r until then.",
        reason,
        reason,
    )
