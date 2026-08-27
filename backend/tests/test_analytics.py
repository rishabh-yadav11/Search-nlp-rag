"""Analytics recording/summary uses Redis aggregates and is best-effort."""

import asyncio

import pytest

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

    def expire(self, key, seconds):
        self.last_pipe.append(("expire", key, seconds))
        return self

    async def execute(self):
        for cmd in self.last_pipe:
            if cmd[0] in ("incr", "zincrby"):
                self.store[cmd[1]] = self.store.get(cmd[1], 0) + cmd[2]
            elif cmd[0] == "incrbyfloat":
                self.store[cmd[1]] = self.store.get(cmd[1], 0.0) + cmd[2]
            elif cmd[0] == "expire":
                pass  # TTL not tracked by the fake
        self.last_pipe = []
        return []

    async def mget(self, keys):
        return [self.store.get(k) for k in keys]

    async def get(self, key):
        return self.store.get(key)

    async def zrevrange(self, key, start, stop, withscores=False):
        return []


class _SignalsRedis:
    """Redis stand-in returning a fixed zrevrange payload for click_signals."""

    def __init__(self, raw):
        self.raw = raw
        self.queries = []

    async def zrevrange(self, key, start, stop, withscores=False):
        self.queries.append(key)
        return self.raw


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

        def expire(self, key, seconds):
            return self

        async def execute(self):
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())
    # Reset the process-global warning flag so this test independently verifies
    # that a Redis failure triggers the warn-once path (order-independent).
    monkeypatch.setattr(analytics, "_warned", False)

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


# --- _client / _degraded / close ---


def test_client_lazy_init_replaces_redis_db(monkeypatch):
    """_client builds REDIS_URL pointing at ANALYTICS_REDIS_DB and reuses the
    connection across calls."""
    created = []

    class FakeRedis:
        pass

    def fake_from_url(url, **kwargs):
        created.append((url, kwargs))
        return FakeRedis()

    monkeypatch.setattr(analytics, "_redis", None)
    monkeypatch.setattr(analytics.aioredis, "from_url", fake_from_url)
    monkeypatch.setattr(analytics.config, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(analytics.config, "ANALYTICS_REDIS_DB", 1)
    client = analytics._client()
    assert analytics._client() is client  # cached, not recreated
    assert len(created) == 1
    assert created[0][0] == "redis://localhost:6379/0"
    assert created[0][1]["db"] == 1
    assert created[0][1]["decode_responses"] is True


def test_degraded_warns_once(monkeypatch):
    monkeypatch.setattr(analytics, "_warned", False)
    warnings = []
    monkeypatch.setattr(analytics.logger, "warning", lambda *a, **k: warnings.append(a))
    analytics._degraded(ConnectionError("down"))
    analytics._degraded(ConnectionError("down"))
    assert len(warnings) == 1
    assert analytics._warned is True


def test_close_resets_redis(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    redis = FakeRedis()
    monkeypatch.setattr(analytics, "_redis", redis)
    _run(analytics.close())
    assert redis.closed is True
    assert analytics._redis is None


def test_close_noop_when_no_redis(monkeypatch):
    monkeypatch.setattr(analytics, "_redis", None)
    _run(analytics.close())  # must not raise


# --- record_click with article_id ---


def test_record_click_with_article_id_tallies_query_click(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    _run(analytics.record_click("fintech funding", 3, article_id=42))
    _run(analytics.record_click("fintech funding", 1, article_id=42))  # repeat click
    assert fake.store["analytics:click:total"] == 2
    assert fake.store["analytics:query_click:fintech funding"] == 2


def test_record_click_without_article_id_skips_query_key(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    _run(analytics.record_click("q", 1))
    assert not any(k.startswith("analytics:query_click:") for k in fake.store)


# --- click_signals ---


def test_click_signals_no_raw_returns_none(monkeypatch):
    fake = _SignalsRedis([])
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) is None
    assert fake.queries == ["analytics:query_click:q"]


def test_click_signals_below_min_clicks_returns_none(monkeypatch):
    monkeypatch.setattr(analytics.config, "CLICK_BOOST_MIN_CLICKS", 5)
    fake = _SignalsRedis([("12", 2.0), ("7", 2.0)])  # total 4 < 5
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) is None


def test_click_signals_success_builds_dict(monkeypatch):
    monkeypatch.setattr(analytics.config, "CLICK_BOOST_MIN_CLICKS", 3)
    fake = _SignalsRedis([("42", 3.0), ("7", 1.0), ("99", 0.0)])  # zero-count filtered
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) == {"total": 4, "by_id": {42: 3, 7: 1}}


def test_click_signals_redis_down_returns_none(monkeypatch):
    class _BrokenRedis:
        async def zrevrange(self, key, start, stop, withscores=False):
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(analytics, "_warned", False)
    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())
    assert _run(analytics.click_signals("q")) is None  # degraded -> None
    assert analytics._warned is True


# --- _i / _f malformed-value branches ---


@pytest.mark.parametrize("value", ["abc", [1], {"a": 1}])
def test_i_malformed_value_returns_zero(value):
    assert analytics._i(value) == 0


@pytest.mark.parametrize("value", ["xyz", [1], {"a": 1}])
def test_f_malformed_value_returns_zero(value):
    assert analytics._f(value) == 0.0


def test_i_and_f_parse_values():
    assert analytics._i("7") == 7
    assert analytics._i(7.0) == 7
    assert analytics._i(None) == 0
    assert analytics._i("") == 0
    assert analytics._f("2.5") == 2.5
    assert analytics._f(None) == 0.0
