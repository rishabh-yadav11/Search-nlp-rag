import contextlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Iterable

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

    def __init__(self, redis_url: str, ttl: int, maxsize: int, max_bytes: int | None = None):
        self._url = redis_url
        self._ttl = ttl
        self._maxsize = maxsize
        # Byte ceiling for the in-process fallback. Entries are wildly
        # different sizes (small search-result payloads vs ~15KB embedding
        # vectors), so an entry-count cap alone lets a handful of vectors
        # thrash out every small entry. ``None``/<=0 means unbounded.
        self._max_bytes = max_bytes
        self._mem: OrderedDict[str, tuple[object, float, int]] = OrderedDict()
        self._mem_bytes = 0
        self._redis: aioredis.Redis | None = None
        self._decode_warned = False
        self._conn_warned = False

    def _drop_mem(self, key: str) -> None:
        """Remove one in-process entry, keeping the byte total in sync."""
        entry = self._mem.pop(key, None)
        if entry is not None:
            self._mem_bytes -= entry[2]

    def _evict_mem(self) -> None:
        """Enforce both capacity caps on the in-process fallback.

        Entry count first (oldest first), then the byte budget, evicting the
        largest entries first so big vectors are sacrificed before the many
        small search results they would otherwise evict.
        """
        while len(self._mem) > self._maxsize:
            self._drop_mem(next(iter(self._mem)))
        if not self._max_bytes or self._max_bytes <= 0:
            return
        while self._mem and self._mem_bytes > self._max_bytes:
            # max() by (cost, -position) picks the largest entry and, among
            # equal costs, the oldest one.
            key = max(
                enumerate(self._mem.items()), key=lambda pair: (pair[1][1][2], -pair[0])
            )[1][0]
            self._drop_mem(key)

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
        value, expires_at, _cost = entry
        if expires_at <= time.monotonic():
            self._drop_mem(key)
            return None
        # The expiry is fixed when the entry is created: a read must never
        # extend it, or a hot key would live forever during a Redis outage and
        # the fallback would serve unbounded stale data. Redis itself does not
        # slide its TTL, so this keeps the fallback faithful to it.
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
        cost = len(payload.encode())
        # Re-caching an existing key replaces the old entry, so its bytes must
        # come back before the new cost is added or the total inflates on every
        # refresh until the budget evicts good entries early.
        self._drop_mem(key)
        self._mem[key] = (value, time.monotonic() + effective_ttl, cost)
        self._mem.move_to_end(key)
        self._mem_bytes += cost
        self._evict_mem()

    async def delete_keys(self, keys: Iterable[str]) -> None:
        """Delete an explicitly known set of keys (Redis + memory).

        This is the request-path invalidation primitive. Cost is O(len(keys))
        -- a single ``DEL`` for the whole set -- because the caller derives the
        key set from what it actually wrote instead of asking Redis to find it.
        Use it whenever the key space of a cache entry is small and knowable,
        which is the case for every per-user cache in this service.

        In-process entries are dropped first so the fallback cache cannot serve
        a stale value even if the Redis round trip then fails.
        """
        targets = set(keys)
        if not targets:
            return
        for key in list(self._mem.keys()):
            if key in targets:
                self._drop_mem(key)
        client, is_new = self._acquire()
        try:
            await client.delete(*targets)
        except _REDIS_ERRORS as exc:
            if is_new:
                await self._discard(client)
            self._degraded(exc)
        else:
            await self._publish(client)

    async def delete_prefix(self, prefix: str) -> None:
        """Delete every cached key starting with ``prefix`` (Redis + memory).

        O(keyspace): implemented with ``SCAN``, which walks *every* key in the
        database server-side and filters by ``MATCH`` only after the fact. The
        cost therefore scales with the total number of keys in the cache, not
        with the number that match, and a single call can walk the whole
        keyspace.

        NEVER call this from a request handler. It has no production call site
        left for exactly that reason: the per-user recommendation cache that
        used to be purged this way has a knowable key set and is invalidated
        with :meth:`delete_keys` instead. Reserve this for maintenance and
        teardown paths, where a one-off full walk is acceptable.
        """
        for key in list(self._mem.keys()):
            if key.startswith(prefix):
                self._drop_mem(key)
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


cache = HybridCache(
    config.REDIS_URL, config.CACHE_TTL_SECONDS, config.CACHE_MAX_SIZE, config.CACHE_MAX_BYTES
)