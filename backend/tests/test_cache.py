import asyncio
import json

from app.redis_cache import HybridCache


class _FakeRedis:
    """Redis stand-in that always raises -> forces the in-process fallback."""

    def __init__(self, error=None):
        self.error = error if error is not None else ConnectionError("redis unreachable")

    async def get(self, key):
        raise self.error

    async def set(self, key, value, ex=None):
        raise self.error


class _RecordingRedis:
    """Redis stand-in that records calls for the happy (non-degraded) path."""

    def __init__(self, store=None):
        self.store = store if store is not None else {}
        self.sets = []
        self.closed = False

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.store[key] = value

    async def aclose(self):
        self.closed = True


def _run(coro):
    return asyncio.run(coro)


def test_set_get_round_trip():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()
    value = {"results": [{"id": 1, "title": "A"}]}

    async def scenario():
        await cache.set("k", value)
        assert await cache.get("k") == value
        assert await cache.get("missing") is None

    _run(scenario())


def test_ttl_expiry():
    cache = HybridCache("redis://fake:6379/0", ttl=0.1, maxsize=10)
    cache._redis = _FakeRedis()

    async def scenario():
        await cache.set("k", "v")
        assert await cache.get("k") == "v"
        await asyncio.sleep(0.25)
        assert await cache.get("k") is None

    _run(scenario())


def test_degraded_mode_falls_back_and_warns_once(monkeypatch):
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()
    assert cache._warned is False

    warnings = []
    monkeypatch.setattr("app.redis_cache.logger.warning", lambda *a, **k: warnings.append(a))

    async def scenario():
        await cache.set("k", "v")
        assert await cache.get("k") == "v"
        assert await cache.get("other") is None
        await cache.set("k2", {"nested": [1, 2]})
        assert await cache.get("k2") == {"nested": [1, 2]}

    _run(scenario())

    assert cache._warned is True
    assert len(warnings) == 1, "degraded warning should be logged exactly once"


def test_degraded_warn_flag_stays_set(monkeypatch):
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()

    async def scenario():
        await cache.set("a", 1)
        await cache.set("b", 2)

    _run(scenario())
    assert cache._warned is True
    _run(scenario())
    assert cache._warned is True


def test_get_redis_hit_decodes_json():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _RecordingRedis({"k": '{"a": 1, "nested": [true, null]}'})
    assert _run(cache.get("k")) == {"a": 1, "nested": [True, None]}


def test_get_redis_miss_falls_through_to_mem():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._mem["k"] = {"from": "mem"}
    cache._redis = _RecordingRedis({})
    assert _run(cache.get("k")) == {"from": "mem"}
    assert _run(cache.get("missing")) is None


def test_set_success_writes_json_with_ttl():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    redis = _RecordingRedis()
    cache._redis = redis
    _run(cache.set("k", {"b": 2}))
    _run(cache.set("k2", {"b": 3}, ttl=5))
    assert json.loads(redis.sets[0][1]) == {"b": 2}
    assert redis.sets[0][2] == 60  # default ttl from the cache
    assert redis.sets[1][2] == 5  # per-call override
    assert redis.store["k"] == '{"b": 2}'


def test_client_lazy_init_and_reuse(monkeypatch):
    redis = _RecordingRedis()
    created = []

    def fake_from_url(url, **kwargs):
        created.append((url, kwargs))
        return redis

    monkeypatch.setattr("app.redis_cache.aioredis.from_url", fake_from_url)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    assert cache._client() is redis
    assert cache._client() is redis  # reused, not recreated
    assert len(created) == 1
    assert created[0][0] == "redis://fake:6379/0"
    assert created[0][1]["decode_responses"] is True


def test_close_with_active_client():
    redis = _RecordingRedis()
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = redis
    _run(cache.close())
    assert redis.closed is True


def test_close_without_client_is_noop():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    _run(cache.close())  # must not raise
