import json
import logging
import time

import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("cache")


class HybridCache:
    """Redis-backed JSON cache with an in-process TTLCache fallback.

    Redis lets multiple gunicorn workers share one cache. If Redis is
    unreachable the module degrades silently to a per-worker local cache so
    the API keeps working.
    """

    def __init__(self, redis_url: str, ttl: int, maxsize: int):
        self._url = redis_url
        self._ttl = ttl
        self._mem: dict[str, tuple[object, float]] = {}
        self._redis = None
        self._warned = False

    def _client(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
            )
        return self._redis

    def _degraded(self, exc: Exception) -> None:
        if not self._warned:
            logger.warning("Redis unavailable (%s); using in-process cache", exc)
            self._warned = True

    async def get(self, key: str):
        try:
            raw = await self._client().get(key)
        except Exception as exc:
            self._degraded(exc)
            return self._get_mem(key)
        if raw is None:
            return self._get_mem(key)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Corrupt payload in Redis: log it distinctly (don't silently swallow
            # into the in-process fallback) and degrade to the memory cache.
            self._degraded(exc)
            return self._get_mem(key)

    def _get_mem(self, key: str):
        entry = self._mem.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at <= time.monotonic():
            self._mem.pop(key, None)
            return None
        return value

    async def set(self, key: str, value, ttl: int | None = None) -> None:
        try:
            await self._client().set(key, json.dumps(value), ex=self._ttl if ttl is None else ttl)
            return
        except Exception as exc:
            self._degraded(exc)
        effective_ttl = self._ttl if ttl is None else ttl
        self._mem[key] = (value, time.monotonic() + effective_ttl)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()


cache = HybridCache(config.REDIS_URL, config.CACHE_TTL_SECONDS, config.CACHE_MAX_SIZE)