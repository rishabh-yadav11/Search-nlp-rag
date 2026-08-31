import contextlib
import json
import logging
import time
from collections import OrderedDict

import redis
import redis.asyncio as aioredis

from app.config import config

logger = logging.getLogger("cache")


_REDIS_ERRORS = (redis.exceptions.RedisError, OSError, TimeoutError)


class HybridCache:
    """Redis-backed JSON cache with an in-process TTLCache fallback.

    Redis lets multiple gunicorn workers share one cache. If Redis is
    unreachable the module degrades silently to a per-worker local cache so
    the API keeps working.
    """

    def __init__(self, redis_url: str, ttl: int, maxsize: int):
        self._url = redis_url
        self._ttl = ttl
        self._maxsize = maxsize
        self._mem: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self._redis: aioredis.Redis | None = None
        self._decode_warned = False
        self._conn_warned = False

    def _new_client(self) -> aioredis.Redis:
        return aioredis.from_url(
            self._url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )

    def _acquire(self) -> tuple[aioredis.Redis, bool]:
        """Return ``(client, is_new)`` for one cache operation.

        A brand new client is *not* published to ``self._redis`` here: it
        becomes the shared client only once its first command succeeds. If
        that first command raises, the caller discards it (see
        :meth:`_discard`) instead of leaving an open connection nobody can
        reach.
        """
        if self._redis is not None:
            return self._redis, False
        return self._new_client(), True

    @staticmethod
    async def _discard(client: aioredis.Redis) -> None:
        """Close a client this cache will not keep, either because its first
        command failed or because it lost the publish race. Close errors are
        ignored: the caller is already on the degraded path."""
        with contextlib.suppress(Exception):
            await client.aclose()

    async def _publish(self, client: aioredis.Redis) -> None:
        """Share ``client`` once one of its commands has succeeded.

        Concurrent calls can each build their own client while ``_redis`` is
        still unset; the first one to finish wins and the loser is closed
        rather than silently dropped while still holding a connection.
        """
        if self._redis is None:
            self._redis = client
        elif self._redis is not client:
            await self._discard(client)

    def _degraded(self, exc: Exception) -> None:
        if isinstance(exc, json.JSONDecodeError):
            if not self._decode_warned:
                logger.warning("Redis payload decode failed (%s); using in-process cache", exc)
                self._decode_warned = True
        else:
            if not self._conn_warned:
                logger.warning("Redis unavailable (%s); using in-process cache", exc)
                self._conn_warned = True

    async def get(self, key: str) -> object | None:
        client, is_new = self._acquire()
        try:
            raw = await client.get(key)
        except _REDIS_ERRORS as exc:
            if is_new:
                await self._discard(client)
            self._degraded(exc)
            return self._get_mem(key)
        await self._publish(client)
        if raw is None:
            return self._get_mem(key)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Corrupt payload in Redis: log it distinctly (don't silently swallow
            # into the in-process fallback) and degrade to the memory cache.
            self._degraded(exc)
            return self._get_mem(key)

    def _get_mem(self, key: str) -> object | None:
        entry = self._mem.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at <= time.monotonic():
            self._mem.pop(key, None)
            return None
        self._mem.move_to_end(key)
        return value

    async def set(self, key: str, value, ttl: int | None = None) -> None:
        payload = json.dumps(value)
        client, is_new = self._acquire()
        try:
            await client.set(key, payload, ex=self._ttl if ttl is None else ttl)
        except _REDIS_ERRORS as exc:
            if is_new:
                await self._discard(client)
            self._degraded(exc)
        else:
            await self._publish(client)
            return
        effective_ttl = self._ttl if ttl is None else ttl
        self._mem[key] = (value, time.monotonic() + effective_ttl)
        self._mem.move_to_end(key)
        while len(self._mem) > self._maxsize:
            self._mem.popitem(last=False)

    async def delete_prefix(self, prefix: str) -> None:
        """Delete every cached key starting with ``prefix`` (Redis + memory).

        Used to invalidate per-user recommendation caches (whose keys embed a
        varying limit component, e.g. ``recommend:for-you:{user}:{limit}``)
        when a new interaction lands. Scans Redis and also purges any matching
        in-process entries so the fallback cache does not return stale data.
        """
        for key in list(self._mem.keys()):
            if key.startswith(prefix):
                self._mem.pop(key, None)
        client, is_new = self._acquire()
        try:
            async for key in client.scan_iter(match=f"{prefix}*", count=100):
                await client.delete(key)
        except _REDIS_ERRORS as exc:
            if is_new:
                await self._discard(client)
            self._degraded(exc)
        else:
            await self._publish(client)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


cache = HybridCache(config.REDIS_URL, config.CACHE_TTL_SECONDS, config.CACHE_MAX_SIZE)