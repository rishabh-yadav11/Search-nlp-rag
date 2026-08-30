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
        self.close_attempts = 0

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.store[key] = value

    async def aclose(self):
        self.close_attempts += 1
        self.closed = True


class _FailingRedis(_RecordingRedis):
    """Redis stand-in whose every command fails on first use."""

    def __init__(self, error=None):
        super().__init__()
        self.error = error if error is not None else ConnectionError("redis unreachable")

    async def get(self, key):
        raise self.error

    async def set(self, key, value, ex=None):
        raise self.error


class _FlakyRedis(_RecordingRedis):
    """Redis stand-in that works until ``fail`` is flipped on."""

    def __init__(self, error=None):
        super().__init__()
        self.fail = False
        self.error = error if error is not None else ConnectionError("redis unreachable")

    async def get(self, key):
        if self.fail:
            raise self.error
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        if self.fail:
            raise self.error
        self.sets.append((key, value, ex))
        self.store[key] = value


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
    assert cache._conn_warned is False

    warnings = []
    monkeypatch.setattr("app.redis_cache.logger.warning", lambda *a, **k: warnings.append(a))

    async def scenario():
        await cache.set("k", "v")
        assert await cache.get("k") == "v"
        assert await cache.get("other") is None
        await cache.set("k2", {"nested": [1, 2]})
        assert await cache.get("k2") == {"nested": [1, 2]}

    _run(scenario())

    assert cache._conn_warned is True
    assert len(warnings) == 1, "degraded warning should be logged exactly once"


def test_degraded_warn_flag_stays_set(monkeypatch):
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()

    async def scenario():
        await cache.set("a", 1)
        await cache.set("b", 2)

    _run(scenario())
    assert cache._conn_warned is True
    _run(scenario())
    assert cache._conn_warned is True


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


def _patch_from_url(monkeypatch, factory):
    """Route client construction to ``factory`` and record every client built."""
    created = []

    def fake_from_url(url, **kwargs):
        client = factory()
        created.append(client)
        return client

    monkeypatch.setattr("app.redis_cache.aioredis.from_url", fake_from_url)
    return created


def test_client_lazy_init_and_reuse(monkeypatch):
    client = _RecordingRedis()
    built = []

    def fake_from_url(url, **kwargs):
        built.append((url, kwargs))
        return client

    monkeypatch.setattr("app.redis_cache.aioredis.from_url", fake_from_url)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    assert built == []  # nothing is built until the cache is actually used

    async def scenario():
        await cache.set("k", "v")
        await cache.get("k")
        await cache.get("other")

    _run(scenario())

    assert len(built) == 1  # built once, then reused
    assert built[0][0] == "redis://fake:6379/0"
    assert built[0][1]["decode_responses"] is True
    assert cache._redis is client  # published after its first successful command


def test_new_client_closed_when_first_get_fails(monkeypatch):
    created = _patch_from_url(monkeypatch, _FailingRedis)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)

    assert _run(cache.get("k")) is None

    assert len(created) == 1
    assert created[0].closed is True, "client that failed its first use must be closed"
    assert cache._redis is None, "a failed client must not become the shared client"


def test_new_client_closed_when_first_set_fails(monkeypatch):
    created = _patch_from_url(monkeypatch, _FailingRedis)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)

    _run(cache.set("k", "v"))

    assert len(created) == 1
    assert created[0].closed is True, "client that failed its first use must be closed"
    assert cache._redis is None, "a failed client must not become the shared client"
    assert cache._mem["k"][0] == "v"  # still degraded to the in-process cache


def test_close_failure_on_discarded_client_is_swallowed(monkeypatch):
    class _UnclosableRedis(_FailingRedis):
        async def aclose(self):
            self.close_attempts += 1
            raise RuntimeError("close failed")

    created = _patch_from_url(monkeypatch, _UnclosableRedis)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)

    assert _run(cache.get("k")) is None  # must not propagate the close error
    assert created[0].close_attempts == 1
    assert cache._redis is None


def test_losing_client_closed_when_another_task_published_first():
    winner = _RecordingRedis()
    loser = _RecordingRedis()
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = winner

    _run(cache._publish(loser))

    assert cache._redis is winner
    assert loser.closed is True, "a client that lost the publish race must be closed"
    assert winner.closed is False


def test_published_client_kept_when_a_later_command_fails():
    redis = _FlakyRedis()
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = redis  # already published by an earlier success

    async def scenario():
        await cache.set("k", "v")
        redis.fail = True
        await cache.set("k2", "v2")

    _run(scenario())

    assert cache._redis is redis  # connection pool survives a transient failure
    assert redis.closed is False


def test_close_with_active_client():
    redis = _RecordingRedis()
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = redis
    _run(cache.close())
    assert redis.closed is True


def test_close_without_client_is_noop():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    _run(cache.close())  # must not raise
