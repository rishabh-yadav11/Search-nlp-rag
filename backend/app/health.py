import asyncio
import contextlib
import logging
import time

import redis
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from app.auth import public_rate_limit
from app.config import config

logger = logging.getLogger("health")


router = APIRouter()

_redis_client: aioredis.Redis | None = None

# Teardown of a client that just failed its ping gets the same 2s budget as the
# ping itself (see _redis_status): a readiness probe must answer on time even
# when the connection it is releasing is dying.
_REDIS_CLOSE_TIMEOUT = 2.0


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Created eagerly at import rather than lazily inside _redis_status. The lazy
# form this replaces was not actually racy: `if lock is None: lock =
# asyncio.Lock()` has no await between the check and the assignment, so no
# other task can be scheduled in between and every caller shares the first
# lock built. The block it guards awaits nothing either, so that lock could
# never be contended -- the #183 race is unreachable. Eager creation is still
# worth it as a simplification: the lock is never None, so there is no
# optional state to check, and close_redis() no longer has a path that resets
# it. "One lock shared by every caller" becomes structural instead of a
# property of an atomic check. Python 3.10+ no longer binds an asyncio.Lock to
# an event loop at construction, so a module-level instance is safe.
_redis_init_lock: asyncio.Lock = asyncio.Lock()


async def _close_quietly(client: aioredis.Redis) -> None:
    """Best-effort release of a client we are about to discard.

    ``aioredis.Redis`` owns a connection pool, so dropping a reference without
    closing leaves the socket to the GC. A client that just failed its ping can
    also fail to close due to a Redis or operating-system network failure, and
    ``asyncio.wait_for`` raises ``TimeoutError`` when it exceeds
    ``_REDIS_CLOSE_TIMEOUT``. Those expected teardown failures are swallowed:
    the caller only cares that readiness is degraded, not about teardown.

    Cancellation and unexpected programming errors deliberately propagate: a
    cancelled probe (client disconnect, server shutdown) must still unwind,
    and a broken close implementation must not be mistaken for a network
    failure.
    """
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    with contextlib.suppress(redis.exceptions.RedisError, OSError, TimeoutError):
        # fallback for test doubles and older redis-py. Either way the call is
        # bounded: a hung close is cancelled and abandoned so /ready and /readyz
        # answer on time regardless of what the dying connection does.
        outcome = close()
        if asyncio.iscoroutine(outcome) or isinstance(outcome, asyncio.Future):
            await asyncio.wait_for(outcome, timeout=_REDIS_CLOSE_TIMEOUT)


async def _drop_redis_client(client: aioredis.Redis) -> None:
    """Invalidate the cached client under the init lock, and only if it is
    still the client that failed.

    The ping runs outside the lock (holding it across a 2s network call would
    serialize every readiness probe), so by the time a ping fails another
    caller may already have replaced the dead client with a fresh one. A blind
    ``_redis_client = None`` would discard that replacement - and leak its
    connection - forcing yet another reconnect for no reason.

    The lock is held only for the identity check and the swap: the failing
    client is closed afterwards, outside the critical section, so no I/O (and
    no re-entry into this non-reentrant lock) happens under it. A client that
    is still cached - i.e. one installed by another caller - is never closed;
    only the client actually dropped here is released."""
    global _redis_client
    async with _redis_init_lock:
        if _redis_client is not client:
            return
        _redis_client = None
    await _close_quietly(client)


async def close_redis() -> None:
    """Close the lazily-created readiness-check Redis client. Registered as a
    shutdown hook so the connection isn't leaked on worker exit. The init lock
    is deliberately left untouched: it is built once at import and never reset,
    so callers that arrive after a close await the same lock as any caller
    still in flight instead of a second, freshly built one."""
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
    readiness failure, not a crash. Letting one escape turned a Qdrant problem
    into a bodiless 500, which a probe cannot distinguish from a server bug.
    Only ``Exception`` is caught, so a cancellation still propagates.
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


def _llm_ok() -> bool:
    return bool(config.GEMINI_API_KEY)


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
# Single-flight for the cache MISS. The entry above bounds the SERIAL probe
# rate, but the moment it expires every request arriving in the same instant
# misses together, and each would otherwise fan out its own Qdrant + Redis
# probe round. The rate limiter does not prevent that herd either: it bounds
# arrival rate, not concurrency.
#
# Built LAZILY and per event loop, unlike _redis_init_lock, and both parts are
# load-bearing. _redis_init_lock guards a block that awaits nothing, so it is
# never contended and acquire() always takes the fast path, which never binds
# the lock to a loop. This one IS contended, and a contended acquire binds the
# lock to its running loop for good: reusing that lock from another loop raises
# "is bound to a different event loop". The tests here run a fresh event loop
# each, so a lock carried across them is a live hazard, and lazy creation alone
# only papers over it -- it has to be rebuilt when the loop changes. The check
# and the assignment sit together with no await between them, so two callers
# in one loop cannot each build a lock and defeat the single-flight, and the
# lock is held in a local so a concurrent reset cannot swap the object out from
# under the `async with`.
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
    # Rebuild when the loop changes: a lock bound to a dead loop would refuse
    # this acquire. Callers within one loop all see the same object, so the
    # single-flight still holds; the check and the assignment are adjacent with
    # no await between them, so they cannot both build one.
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
    llm_ok = _llm_ok()
    ready = qdrant_ok and models_ok
    report = {
        "ready": ready,
        "checks": {
            "qdrant": {"ok": qdrant_ok},
            "models": {"ok": models_ok},
            "redis": {"ok": redis_ok, "cache": cache_mode},
            "llm": {"ok": llm_ok},
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
    # The same limiter, and deliberately the same "ready" action, as /ready:
    # this alias runs the identical readiness probe, so it shares one budget
    # rather than handing a caller a second allowance by changing one path
    # segment. Same fail-open deviation, same reason.
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
