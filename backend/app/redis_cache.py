import contextlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Iterable

import redis
import redis.asyncio as aioredis

from app.config import config
from app.degraded import DegradedLatch

logger = logging.getLogger("cache")


_REDIS_ERRORS = (redis.exceptions.RedisError, OSError, TimeoutError)


class HybridCache:
    """Redis with an in-process fallback; corrupt payloads are logged, since silence would hide a broken Redis."""

    def __init__(self, redis_url: str, ttl: int, maxsize: int, max_bytes: int | None = None):
        self._url = redis_url
        self._ttl = ttl
        self._maxsize = maxsize
        self._max_bytes = max_bytes
        self._mem: OrderedDict[str, tuple[object, float, int]] = OrderedDict()
        self._mem_bytes = 0
        self._redis: aioredis.Redis | None = None
        self._decode_latch = DegradedLatch(logger, "cache payload decode")
        self._conn_latch = DegradedLatch(logger, "cache Redis")

    def _drop_mem(self, key: str) -> None:
        entry = self._mem.pop(key, None)
        if entry is not None:
            self._mem_bytes -= entry[2]

    def _evict_mem(self) -> None:
        while len(self._mem) > self._maxsize:
            self._drop_mem(next(iter(self._mem)))
        if not self._max_bytes or self._max_bytes <= 0:
            return
        while self._mem and self._mem_bytes > self._max_bytes:
            key = max(
                enumerate(self._mem.items()), key=lambda pair: (pair[1][1][2], -pair[0])
            )[1][0]
            self._drop_mem(key)

    def _new_client(self) -> aioredis.Redis:
        return aioredis.from_url(
            self._url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
        )

    def _acquire(self) -> tuple[aioredis.Redis, bool]:
        if self._redis is not None:
            return self._redis, False
        return self._new_client(), True

    @staticmethod
    async def _discard(client: aioredis.Redis) -> None:
        """Closed rather than kept: a client whose first command failed would fail every later call."""
        with contextlib.suppress(Exception):
            await client.aclose()

    async def _publish(self, client: aioredis.Redis) -> None:
        if self._redis is None:
            self._redis = client
        elif self._redis is not client:
            await self._discard(client)

    def _degraded(self, exc: Exception) -> None:
        if isinstance(exc, json.JSONDecodeError):
            self._decode_latch.warn_degraded(
                "Redis payload decode failed (%s); using in-process cache", exc
            )
        else:
            self._conn_latch.warn_degraded("Redis unavailable (%s); using in-process cache", exc)

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
        self._conn_latch.log_recovered()
        self._decode_latch.log_recovered()
        if raw is None:
            return self._get_mem(key)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._degraded(exc)
            return self._get_mem(key)
        return value

    async def get_many(self, keys: list[str]) -> list[object | None]:
        if not keys:
            return []
        client, is_new = self._acquire()
        try:
            raws = await client.mget(keys)
        except _REDIS_ERRORS as exc:
            if is_new:
                await self._discard(client)
            self._degraded(exc)
            raws = [None] * len(keys)
        else:
            await self._publish(client)
        values: list[object | None] = []
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                values.append(self._get_mem(key))
                continue
            try:
                values.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                self._degraded(exc)
                values.append(self._get_mem(key))
        return values

    def _get_mem(self, key: str) -> object | None:
        entry = self._mem.get(key)
        if entry is None:
            return None
        value, expires_at, _cost = entry
        if expires_at <= time.monotonic():
            self._drop_mem(key)
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
            self._conn_latch.log_recovered()
            return
        effective_ttl = self._ttl if ttl is None else ttl
        cost = len(payload.encode())
        self._drop_mem(key)
        self._mem[key] = (value, time.monotonic() + effective_ttl, cost)
        self._mem.move_to_end(key)
        self._mem_bytes += cost
        self._evict_mem()

    async def delete_keys(self, keys: Iterable[str]) -> None:
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
            self._conn_latch.log_recovered()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


cache = HybridCache(
    config.REDIS_URL, config.CACHE_TTL_SECONDS, config.CACHE_MAX_SIZE, config.CACHE_MAX_BYTES
)