import asyncio

from app.redis_cache import HybridCache


class _FakeRedis:
    """Redis stand-in that always raises -> forces the in-process fallback."""

    def __init__(self, error=None):
        self.error = error if error is not None else ConnectionError("redis unreachable")

    async def get(self, key):
        raise self.error

    async def set(self, key, value, ex=None):
        raise self.error


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
