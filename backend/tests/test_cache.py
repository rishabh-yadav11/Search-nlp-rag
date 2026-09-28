import asyncio
import json
import time

from _support import run_sync as _run

from app import redis_cache
from app.redis_cache import HybridCache


class _FakeRedis:
    """Redis stand-in that always raises -> forces the in-process fallback."""

    def __init__(self, error=None):
        self.error = error if error is not None else ConnectionError("redis unreachable")

    async def get(self, key):
        raise self.error

    async def set(self, key, value, ex=None):
        raise self.error

    async def mget(self, keys, *_rest):

        # Production calls this BOTH ways: redis_cache.py:148 `mget(keys)` and

        # :201 `mget(*keys)`. Accept either shape rather than pinning one.

        if isinstance(keys, str):

            keys = [keys, *_rest]

        else:

            keys = [*keys, *_rest]
        # A Redis that is down fails every command, so the batched read must
        # take the same degraded branch ``get`` does.
        raise self.error

    async def delete(self, *keys):
        # A Redis that is down fails every command, not just reads and writes;
        # without this the delete paths would raise AttributeError instead of
        # taking the degraded branch they are meant to exercise.
        raise self.error


class _RecordingRedis:
    """Redis stand-in that records calls for the happy (non-degraded) path."""

    def __init__(self, store=None):
        self.store = store if store is not None else {}
        self.mgets = []
        self.sets = []
        self.closed = False
        self.close_attempts = 0

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.store[key] = value

    async def mget(self, keys, *_rest):

        # Production calls this BOTH ways: redis_cache.py:148 `mget(keys)` and

        # :201 `mget(*keys)`. Accept either shape rather than pinning one.

        if isinstance(keys, str):

            keys = [keys, *_rest]

        else:

            keys = [*keys, *_rest]
        self.mgets.append(list(keys))
        return [self.store.get(key) for key in keys]

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

    async def mget(self, keys, *_rest):

        # Production calls this BOTH ways: redis_cache.py:148 `mget(keys)` and

        # :201 `mget(*keys)`. Accept either shape rather than pinning one.

        if isinstance(keys, str):

            keys = [keys, *_rest]

        else:

            keys = [*keys, *_rest]
        if self.fail:
            raise self.error
        self.mgets.append(list(keys))
        return [self.store.get(key) for key in keys]


def test_set_get_round_trip():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()
    value = {"results": [{"id": 1, "title": "A"}]}

    async def scenario():
        await cache.set("k", value)
        assert await cache.get("k") == value
        assert await cache.get("missing") is None

    _run(scenario())


def test_degraded_fallback_honors_per_write_ttl_override():
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()

    async def scenario():
        await cache.set("k", "v", ttl=0.1)
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
    # _mem entries are (value, expiry, byte cost) tuples -- the only shape the
    # cache itself writes (see HybridCache.set) and the shape _get_mem
    # unpacks. The expiry is fixed at creation: _get_mem no longer slides it.
    # This hand-written entry is deliberately not counted in _mem_bytes.
    cache._mem["k"] = ({"from": "mem"}, time.monotonic() + 60, 60)
    cache._redis = _RecordingRedis({})
    assert _run(cache.get("k")) == {"from": "mem"}
    assert _run(cache.get("missing")) is None


def test_get_many_reads_every_key_in_one_round_trip():
    """The batched read is positional and costs exactly one command."""
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    redis = _RecordingRedis({"a": '{"n": 1}', "b": '{"n": 2}'})
    cache._redis = redis
    assert _run(cache.get_many(["a", "b", "missing"])) == [{"n": 1}, {"n": 2}, None]
    assert redis.mgets == [["a", "b", "missing"]]
    assert not redis.sets, "a read must not write"


def test_get_many_falls_back_per_key_to_the_in_process_cache():
    """A Redis miss falls through to memory exactly as ``get`` does."""
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    # _mem entries are (value, expiry, byte cost) tuples; see the same note in
    # test_get_redis_miss_falls_through_to_mem.
    cache._mem["a"] = ({"from": "mem"}, time.monotonic() + 60, 60)
    cache._redis = _RecordingRedis({})
    assert _run(cache.get_many(["a", "b"])) == [{"from": "mem"}, None]


def test_get_many_degrades_to_memory_when_redis_is_down(monkeypatch):
    """A down Redis must not fail the batched read or lose the fallback.

    ``get`` already degrades to the in-process cache; a batched read that did
    not would turn a Redis outage into a /search 500, and one that degraded
    without consulting memory would silently drop warm entries.
    """
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()
    warnings = []
    monkeypatch.setattr("app.redis_cache.logger.warning", lambda *a, **k: warnings.append(a))

    async def scenario():
        await cache.set("k", {"n": 1})
        assert await cache.get_many(["k", "absent"]) == [{"n": 1}, None]

    _run(scenario())

    assert cache._conn_warned is True
    assert len(warnings) == 1, "the batched read must warn once, like get()"


def test_get_many_degrades_per_key_on_a_corrupt_payload(monkeypatch):
    """One unparseable value must not discard the rest of the batch."""
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _RecordingRedis({"good": '{"n": 1}', "bad": "not json"})
    cache._mem["bad"] = ({"from": "mem"}, time.monotonic() + 60, 60)
    warnings = []
    monkeypatch.setattr("app.redis_cache.logger.warning", lambda *a, **k: warnings.append(a))

    assert _run(cache.get_many(["good", "bad"])) == [{"n": 1}, {"from": "mem"}]
    assert any("decode" in str(w) for w in warnings), (
        "a corrupt payload must be logged as a decode failure, not a connect failure"
    )


def test_get_many_does_not_slide_the_in_process_ttl(monkeypatch):
    """Batched reads honour the no-slide rule that single reads do."""
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(redis_cache, "time", clock)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()
    value = {"results": [{"id": 1}]}

    async def scenario():
        await cache.set("hot", value)
        clock.advance(1.0)
        assert await cache.get_many(["hot"]) == [value]
        clock.advance(59.0)  # 60s after the write, at the TTL boundary
        assert await cache.get_many(["hot"]) == [None], "a batched read must not slide the TTL"

    _run(scenario())


def test_get_many_with_no_keys_issues_no_command():
    """An empty batch never opens a connection."""
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    assert _run(cache.get_many([])) == []
    assert cache._redis is None


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


class _FakeClock:
    """Stand-in for the ``time`` module holding a controllable monotonic clock."""

    def __init__(self, now: float):
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_mem_ttl_does_not_slide_when_the_entry_is_read(monkeypatch):
    """A read must not extend the in-process TTL.

    The expiry is fixed when the entry is created, so a hot key still
    disappears once its original window has passed. Sliding it would keep a
    frequently-read key alive forever during a Redis outage and serve stale
    data without bound.
    """
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(redis_cache, "time", clock)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=10)
    cache._redis = _FakeRedis()
    value = {"results": [{"id": 1}]}

    async def scenario():
        await cache.set("hot", value)
        clock.advance(1.0)
        assert await cache.get("hot") == value  # well inside the window
        clock.advance(1.0)
        assert await cache.get("hot") == value  # this read must not re-arm it
        clock.advance(59.0)  # 61s after the write, past the 60s TTL
        assert await cache.get("hot") is None, "a read must not slide the TTL"

    _run(scenario())


def test_byte_budget_evicts_large_vectors_before_small_results():
    """A few large entries must not push the many small ones out of the cache.

    Both live under one entry-count cap, so counting entries alone lets a
    handful of embedding vectors consume the whole budget. The byte budget
    must shed the largest entries first, keeping small search results
    retrievable, and must never be exceeded.
    """
    big_cost = len(json.dumps([round(0.001 * (i % 977), 8) for i in range(768)]).encode())
    small_values = {f"search:{i}": {"id": i, "title": "x" * 20} for i in range(50)}
    small_total = sum(len(json.dumps(v).encode()) for v in small_values.values())
    # The budget comfortably fits every small entry many times over, plus a
    # handful of vectors -- but nowhere near all 20 vectors together.
    budget = small_total * 4 + big_cost * 3
    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=1000, max_bytes=budget)
    cache._redis = _FakeRedis()
    vectors = {f"vec:{i}": [round(0.001 * (j % 977), 8) for j in range(768)] for i in range(20)}

    async def scenario():
        for key, value in small_values.items():
            await cache.set(key, value)
        # Re-caching an existing key must not double-count its bytes.
        refreshed = {"id": 0, "title": "y" * 40}
        await cache.set("search:0", refreshed)
        small_values["search:0"] = refreshed
        before_vectors = cache._mem_bytes
        for key, value in vectors.items():
            await cache.set(key, value)

        assert cache._mem_bytes == sum(entry[2] for entry in cache._mem.values())
        assert cache._mem_bytes >= before_vectors
        assert cache._mem_bytes <= budget, "the byte budget must hold after inserts"

        for key, value in small_values.items():
            assert await cache.get(key) == value, f"small entry {key} was evicted by vectors"

        kept_vectors = [key for key in vectors if await cache.get(key) is not None]
        assert 0 < len(kept_vectors) < len(vectors), "vectors must be shed, and not all of them"

    _run(scenario())


def test_delete_keys_purge_keeps_byte_total_in_sync():
    """Purging a known key set must give back its bytes, or the budget drifts
    low over time and the cache silently evicts far earlier than configured."""
    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=100, max_bytes=1 << 20)
    cache._redis = _FakeRedis()
    purged = {f"recommend:u{i}:10": {"v": "x" * 50} for i in range(5)}
    kept = {f"search:{i}": {"v": "y" * 10} for i in range(5)}

    async def scenario():
        for key, value in {**purged, **kept}.items():
            await cache.set(key, value)
        before = cache._mem_bytes

        await cache.delete_keys(purged)

        released = sum(len(json.dumps(v).encode()) for v in purged.values())
        assert cache._mem_bytes == before - released
        assert cache._mem_bytes == sum(entry[2] for entry in cache._mem.values())
        for key, value in purged.items():
            assert await cache.get(key) is None, f"{key} should have been purged"
        for key, value in kept.items():
            assert await cache.get(key) == value

    _run(scenario())


def test_delete_keys_with_no_keys_issues_no_redis_command():
    """An empty invalidation set must be a true no-op: no command, no
    connection. Asserted structurally by counting the commands a spy Redis
    sees, so a regression that acquires a client cannot pass."""
    calls = []

    class _CountingRedis:
        def __getattr__(self, name):
            async def _record(*args, **kwargs):
                calls.append(name)
            return _record

    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=10)
    cache._redis = _CountingRedis()

    async def scenario():
        await cache.delete_keys([])
        await cache.delete_keys(iter([]))

    _run(scenario())
    assert calls == [], f"an empty key set must not talk to Redis, saw {calls}"


def test_expired_entry_read_gives_its_bytes_back(monkeypatch):
    """Reading an entry past its TTL must subtract it from the byte total.

    The count-eviction and purge paths were pinned, but the expiry path was
    not: a read that returns None while leaving the cost behind inflates
    _mem_bytes on every expiry, and the byte budget then evicts live entries
    far earlier than CACHE_MAX_BYTES says.
    """
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(redis_cache, "time", clock)
    cache = HybridCache("redis://fake:6379/0", ttl=60, maxsize=100, max_bytes=1 << 20)
    cache._redis = _FakeRedis()
    values = {f"search:{i}": {"v": "x" * 40} for i in range(3)}

    async def scenario():
        for key, value in values.items():
            await cache.set(key, value)
        clock.advance(61.0)  # every entry is now past its 60s TTL

        for key in values:
            assert await cache.get(key) is None, f"{key} should have expired"

        assert cache._mem_bytes == 0
        assert not cache._mem

    _run(scenario())


def test_count_eviction_gives_evicted_bytes_back():
    """Dropping an entry to satisfy the entry-count cap must subtract its cost.

    The byte budget was checked after inserts, which is why a count eviction
    that discards the entry but keeps its bytes went unnoticed: the total
    drifts upward with every insert until the budget starts evicting live
    entries for no reason.
    """
    cache = HybridCache("redis://fake:6379/0", ttl=600, maxsize=2, max_bytes=1 << 20)
    cache._redis = _FakeRedis()

    async def scenario():
        for i in range(6):
            await cache.set(f"search:{i}", {"v": "x" * (40 + i)})
            assert len(cache._mem) <= 2, "the entry-count cap must hold"
        assert cache._mem_bytes == sum(entry[2] for entry in cache._mem.values())
        assert cache._mem_bytes <= sum(len(json.dumps({"v": "x" * (40 + i)}).encode()) for i in range(2, 6))

    _run(scenario())
