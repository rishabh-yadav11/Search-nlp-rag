import asyncio

import redis
import redis.asyncio as aioredis
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from app.config import config

router = APIRouter()

_redis_client = None


@router.get("/health")
async def health():
    return {"status": "ok"}


_redis_init_lock = None


async def close_redis() -> None:
    """Close the lazily-created readiness-check Redis client. Registered as a
    shutdown hook so the connection isn't leaked on worker exit."""
    global _redis_client, _redis_init_lock
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
    _redis_init_lock = None


@router.get("/live")
async def live():
    return {"status": "ok"}


async def _qdrant_ok(state) -> bool:
    """Qdrant client present and the collection check succeeds (bounded <3s)."""
    client = state.get("qdrant")
    if client is None:
        return False
    try:
        await asyncio.wait_for(client.collection_exists(config.QDRANT_COLLECTION), timeout=2.5)
        return True
    except Exception:
        return False


def _models_ok(state) -> bool:
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
    global _redis_init_lock
    if _redis_init_lock is None:
        _redis_init_lock = asyncio.Lock()
    async with _redis_init_lock:
        if _redis_client is None:
            try:
                _redis_client = aioredis.from_url(
                    config.REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
                )
            except (redis.exceptions.RedisError, OSError, asyncio.TimeoutError):
                return False, "down"
    try:
        await asyncio.wait_for(_redis_client.ping(), timeout=2.0)
        return True, "redis"
    except (redis.exceptions.RedisError, OSError, asyncio.TimeoutError):
        _redis_client = None
        return False, "degraded"


async def _readiness_report(state) -> tuple[bool, dict]:
    qdrant_ok = await _qdrant_ok(state)
    models_ok = _models_ok(state)
    redis_ok, cache_mode = await _redis_status()
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


@router.get("/ready")
async def ready():
    from app.main import state  # lazy: avoid circular import at startup

    ready, report = await _readiness_report(state)
    return JSONResponse(status_code=200 if ready else 503, content=report)


@router.get("/readyz")
async def readyz():
    from app.main import state  # lazy: avoid circular import at startup

    ready, _ = await _readiness_report(state)
    return Response(status_code=200 if ready else 503)
