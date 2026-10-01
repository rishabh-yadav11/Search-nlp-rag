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

# Same 2s budget as the ping: the probe must answer on time even while releasing a dying client.
_REDIS_CLOSE_TIMEOUT = 2.0


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Eager: the lazy form is not racy (no await between its check and assignment) and 3.10+ locks
# are not bound to a loop at construction.
_redis_init_lock: asyncio.Lock = asyncio.Lock()


async def _close_quietly(client: aioredis.Redis) -> None:
    await close_quietly(
        "readiness Redis client",
        client,
        timeout=_REDIS_CLOSE_TIMEOUT,
        suppress=(redis.exceptions.RedisError, *close_guard.EXPECTED_CLOSE_ERRORS),
    )


async def _drop_redis_client(client: aioredis.Redis) -> None:
    global _redis_client
    async with _redis_init_lock:
        if _redis_client is not client:
            return
        _redis_client = None
    await _close_quietly(client)


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


async def _qdrant_ok(state: dict) -> bool:
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
    reason = classify_gemini_api_key(config.GEMINI_API_KEY)
    return reason == "ok", reason


async def _redis_status() -> tuple[bool, str]:
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


# Plain data with a monotonic deadline, never a loop-bound object, so it survives the per-test event loops.
_readiness_cache: tuple[float, bool, dict] | None = None
# Single-flight for the cache MISS, built lazily per event loop: a contended acquire binds an
# asyncio.Lock to its running loop for good, so a lock left over from a dead loop must be rebuilt.
_readiness_probe_lock: asyncio.Lock | None = None
_readiness_probe_loop: asyncio.AbstractEventLoop | None = None


def reset_readiness_cache() -> None:
    global _readiness_cache, _readiness_probe_lock, _readiness_probe_loop
    _readiness_cache = None
    _readiness_probe_lock = None
    _readiness_probe_loop = None


async def _cached_readiness_report(state: dict) -> tuple[bool, dict]:
    global _readiness_cache, _readiness_probe_lock, _readiness_probe_loop
    cached = _readiness_cache
    if cached is not None and cached[0] > time.monotonic():
        return cached[1], cached[2]
    loop = asyncio.get_running_loop()
    lock = _readiness_probe_lock
    if lock is None or _readiness_probe_loop is not loop:
        lock = asyncio.Lock()
        _readiness_probe_lock = lock
        _readiness_probe_loop = loop
    async with lock:
        cached = _readiness_cache
        now = time.monotonic()
        if cached is not None and cached[0] > now:
            return cached[1], cached[2]
        ready, report = await _readiness_report(state)
        _readiness_cache = (now + config.READY_CACHE_TTL_SECONDS, ready, report)
        return ready, report


async def _probe_bounded(name: str, coro, default):
    """A timeout is a dependency failure; any other exception is a defect in this probe machinery and propagates."""
    try:
        return await asyncio.wait_for(coro, timeout=config.READY_DEP_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.warning("%s readiness probe timed out after %ss", name, config.READY_DEP_TIMEOUT_SECONDS)
        return default


async def _readiness_report(state: dict) -> tuple[bool, dict]:
    # Probes run concurrently so the report costs one budget rather than two; return_exceptions
    # keeps a raising probe from orphaning its sibling, and the first real error is re-raised below.
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
    # Redis stays out of this sum on purpose: it degrades to an in-process cache, not to a wrong answer.
    ready = qdrant_ok and models_ok and llm_ok
    report = {
        "ready": ready,
        "checks": {
            "qdrant": {"ok": qdrant_ok},
            "models": {"ok": models_ok},
            "redis": {"ok": redis_ok, "cache": cache_mode},
            # A classification ("missing" / "placeholder" / "malformed" / "ok"), never the key itself.
            "llm": {"ok": llm_ok, "reason": llm_reason},
        },
    }
    return ready, report


# fail_closed=False: a Redis outage is degraded-but-serving here, so failing closed would pull healthy
# nodes out of rotation; polls are still counted and still answered 429.
@router.get(
    "/ready",
    dependencies=[Depends(public_rate_limit("ready", "PUBLIC_READY_RATE_PER_MIN", fail_closed=False))],
)
async def ready() -> JSONResponse:
    """Cached and rate-limited readiness for a load balancer; a watchdog needing an immediate
    answer must use /ready/deep instead."""
    from app.main import state  # lazy: avoid circular import at startup

    try:
        ok, report = await _cached_readiness_report(state)
    except Exception:
        # A defect in the probe machinery, not a dependency fault: 500 rather than a 503 that
        # launders a broken probe into a healthy verdict.
        logger.exception("readiness probe raised unexpectedly")
        return JSONResponse(status_code=500, content={"ready": False, "checks": {}, "error": "readiness probe failed"})
    return JSONResponse(status_code=200 if ok else 503, content=report)


@router.get(
    "/readyz",
    # Deliberately the same "ready" action as /ready: this alias runs the identical probe, so it shares one budget.
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
    """Fail closed: /ready/deep is uncached and unrated, and a request through the local reverse proxy is an
    internet request wearing a loopback peer's address, so X-Forwarded-For must be absent."""
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
    """Uncached, unrated readiness for a watchdog: a 429 is indistinguishable from an outage to a
    `curl -fsS` caller. Neither is safe to hand the internet, so this is host-local monitoring
    only, not a status endpoint."""
    if not _is_host_local_probe(request):
        logger.warning("refused a non-host-local /ready/deep probe")
        return Response(status_code=403, content="host-local probe endpoint")
    from app.main import state  # lazy: avoid circular import at startup

    try:
        ok, report = await _readiness_report(state)
    except Exception:
        logger.exception("readiness probe raised unexpectedly")
        return JSONResponse(status_code=500, content={"ready": False, "checks": {}, "error": "readiness probe failed"})
    return JSONResponse(status_code=200 if ok else 503, content=report)


def warn_if_llm_key_unusable() -> None:
    """Deliberately a log line, not a raise: crashing would take /health down with it and leave
    the watchdog nothing to probe."""
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
