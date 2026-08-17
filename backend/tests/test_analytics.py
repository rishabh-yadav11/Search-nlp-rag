"""Analytics recording/summary uses Redis aggregates and is best-effort."""

import asyncio

from app import analytics


class _FakeRedis:
    """In-memory Redis stand-in recording every pipeline command."""

    def __init__(self):
        self.store: dict = {}
        self.last_pipe = []

    def pipeline(self):
        return self

    def incr(self, key, amount=1):
        self.last_pipe.append(("incr", key, amount))
        return self

    def zincrby(self, key, amount, member):
        self.last_pipe.append(("zincrby", key, amount, member))
        return self

    def incrbyfloat(self, key, amount):
        self.last_pipe.append(("incrbyfloat", key, amount))
        return self

    async def execute(self):
        for cmd in self.last_pipe:
            if cmd[0] in ("incr", "zincrby"):
                self.store[cmd[1]] = self.store.get(cmd[1], 0) + cmd[2]
            elif cmd[0] == "incrbyfloat":
                self.store[cmd[1]] = self.store.get(cmd[1], 0.0) + cmd[2]
        self.last_pipe = []
        return []

    async def mget(self, keys):
        return [self.store.get(k) for k in keys]

    async def get(self, key):
        return self.store.get(key)

    async def zrevrange(self, key, start, stop, withscores=False):
        return []


def _run(coro):
    return asyncio.run(coro)


def test_record_search_increments_counters(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_search("fintech funding", 5, weak=False, cached=False, latency_ms=210, filtered=True))

    assert fake.store["analytics:search:total"] == 1
    assert fake.store["analytics:search:filtered"] == 1
    assert fake.store["analytics:search:latency:sum"] == 210
    assert fake.store["analytics:top_queries"] == 1
    assert "analytics:search:zero_results" not in fake.store


def test_record_search_zero_results(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_search("no match anything", 0, weak=False, cached=False, latency_ms=50, filtered=False))

    assert fake.store["analytics:search:zero_results"] == 1
    assert "analytics:search:weak" not in fake.store


def test_record_search_weak_and_cached(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_search("weak query", 3, weak=True, cached=True, latency_ms=30, filtered=False))

    assert fake.store["analytics:search:weak"] == 1
    assert fake.store["analytics:search:cached"] == 1


def test_record_click(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_click("fintech funding", 2))
    _run(analytics.record_click("healthtech funding", 4))

    assert fake.store["analytics:click:total"] == 2
    assert fake.store["analytics:click:pos:2"] == 1
    assert fake.store["analytics:click:pos:4"] == 1


def test_summary_reads_aggregates(monkeypatch):
    fake = _FakeRedis()
    fake.store.update(
        {
            "analytics:search:total": 10,
            f"analytics:search:day:{analytics._today()}": 4,
            "analytics:search:zero_results": 2,
            "analytics:search:weak": 1,
            "analytics:search:filtered": 3,
            "analytics:search:latency:sum": 2000,
            "analytics:search:latency:count": 10,
            "analytics:search:cached": 6,
            "analytics:click:total": 7,
        }
    )
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    s = _run(analytics.summary())

    assert s["searches_total"] == 10
    assert s["searches_today"] == 4
    assert s["zero_result_rate"] == 20.0
    assert s["weak_result_rate"] == 10.0
    assert s["filtered_rate"] == 30.0
    assert s["cache_hit_rate"] == 60.0
    assert s["avg_latency_ms"] == 200.0
    assert s["clicks_total"] == 7


def test_recording_never_raises_when_redis_down(monkeypatch):
    class _BrokenRedis:
        def pipeline(self):
            return self

        def incr(self, key, amount=1):
            return self

        def zincrby(self, key, amount, member):
            return self

        async def execute(self):
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())

    _run(analytics.record_search("anything", 1, weak=False, cached=False, latency_ms=10, filtered=False))
    _run(analytics.record_click("anything", 1))
    assert analytics._warned is True


def test_summary_returns_error_dict_when_redis_down(monkeypatch):
    class _BrokenRedis:
        async def mget(self, keys):
            raise ConnectionError("redis unreachable")

        async def get(self, key):
            raise ConnectionError("redis unreachable")

        async def zrevrange(self, key, start, stop, withscores=False):
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())

    s = _run(analytics.summary())
    assert "error" in s
