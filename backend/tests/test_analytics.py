"""Analytics recording/summary uses Redis aggregates and is best-effort."""

import json
import logging

import pytest
from _support import run_sync as _run
from conftest import auth_cookie

from app import analytics
from app.degraded import REANNOUNCE_SECONDS, DegradedLatch


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

 
def _outage_lines(caplog, logger_name="analytics"):
    """Latch lines only, selected by text: the digest-key warning is a separate flag."""
    return [
        r
        for r in caplog.records
        if r.name == logger_name and "analytics Redis" in r.getMessage()
    ]


class _FakeRedis:
    def __init__(self):
        self.store: dict = {}
        self.last_pipe = []
        self.zsets: list = []

    def zincrby_members(self, key):
        return [m for k, m in self.zsets if k == key]

    def pipeline(self):
        return self

    def incr(self, key, amount=1):
        self.last_pipe.append(("incr", key, amount))
        return self

    def zincrby(self, key, amount, member):
        self.last_pipe.append(("zincrby", key, amount, member))
        self.zsets.append((key, member))
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
                pass
        self.last_pipe = []
        return []

    async def mget(self, keys, *_rest):
        # mget is only ever called with a list; a bare str is tolerated for leniency.
        if isinstance(keys, str):
            keys = [keys, *_rest]
        else:
            keys = [*keys, *_rest]
        return [self.store.get(k) for k in keys]

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, nx=False):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def zrevrange(self, key, start, stop, withscores=False):
        return []


@pytest.fixture(autouse=True)
def _reset_digest_key():
    """Process-wide globals outlive a test; reset so no fake leaks state into the next."""
    analytics._QUERY_DIGEST_KEY = None
    analytics._digest_warned = False
    analytics._legacy_scrubbed = False
    yield
    analytics._QUERY_DIGEST_KEY = None
    analytics._digest_warned = False
    analytics._legacy_scrubbed = False


class _SignalsRedis:
    def __init__(self, raw):
        self.raw = raw
        self.queries = []
        self.keys: dict = {}

    async def get(self, key):
        return self.keys.get(key)

    async def set(self, key, value, nx=False):
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True

    async def zrevrange(self, key, start, stop, withscores=False):
        self.queries.append(key)
        return self.raw


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


class _SummaryRedis:
    """Real sorted sets, so zrevrange's INCLUSIVE end index is exercised for real."""

    def __init__(self):
        self.store: dict = {}
        self.zsets: dict = {}
        self.calls: list = []

    async def mget(self, keys, *_rest):
        if isinstance(keys, str):
            keys = [keys, *_rest]
        else:
            keys = [*keys, *_rest]
        return [str(self.store[k]) if k in self.store else None for k in keys]

    async def get(self, key):
        return str(self.store[key]) if key in self.store else None

    async def set(self, key, value, nx=False):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def zrevrange(self, key, start, stop, withscores=False):
        self.calls.append((key, start, stop))
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        window = items[start:] if stop == -1 else items[start : stop + 1]
        if withscores:
            return list(window)
        return [m for m, _ in window]

    async def zrange(self, key, start, stop):
        return list(self.zsets.get(key, {}))

    async def zrem(self, key, *members):
        self.zsets.setdefault(key, {})
        removed = 0
        for m in members:
            if self.zsets[key].pop(m, None) is not None:
                removed += 1
        return removed


def test_summary_click_positions_follow_click_position_max(monkeypatch):
    fake = _SummaryRedis()
    for i in range(1, 26):
        fake.store[f"analytics:click:pos:{i}"] = i
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics, "CLICK_POSITION_MAX", 20)

    s = _run(analytics.summary())

    assert list(s["click_positions"]) == [str(i) for i in range(1, 21)]
    assert s["click_positions"]["20"] == 20
    assert "21" not in s["click_positions"]


def test_summary_click_positions_shrink_with_click_position_max(monkeypatch):
    fake = _SummaryRedis()
    for i in range(1, 26):
        fake.store[f"analytics:click:pos:{i}"] = i
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics, "CLICK_POSITION_MAX", 4)

    s = _run(analytics.summary())

    assert list(s["click_positions"]) == ["1", "2", "3", "4"]


def test_summary_default_click_positions_cover_one_to_ten(monkeypatch):
    fake = _SummaryRedis()
    for i in range(1, 31):
        fake.store[f"analytics:click:pos:{i}"] = i
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert analytics.CLICK_POSITION_MAX == 10

    s = _run(analytics.summary())

    assert s["click_positions"] == {str(i): i for i in range(1, 11)}


def test_summary_top_lists_window_sizes_are_exact(monkeypatch):
    """Exactly N members, not N+1: zrevrange's end index is INCLUSIVE."""
    fake = _SummaryRedis()
    # Digest-shaped members: a dropped non-digest member would measure the filter, not the window.
    fake.zsets["analytics:top_queries"] = {f"q1:{i:032x}": 100 - i for i in range(1, 31)}
    fake.zsets["analytics:click_top_queries"] = {f"q1:{i:032x}": 100 - i for i in range(1, 21)}
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    s = _run(analytics.summary())

    assert len(s["top_queries"]) == analytics.TOP_QUERIES_N == 20
    assert [q for q, _ in s["top_queries"]] == [f"q1:{i:032x}" for i in range(1, 21)]
    assert len(s["click_top_queries"]) == analytics.TOP_CLICKED_QUERIES_N == 10
    assert [q for q, _ in s["click_top_queries"]] == [f"q1:{i:032x}" for i in range(1, 11)]


def test_recording_never_raises_when_redis_down(monkeypatch, caplog, clock):
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
    monkeypatch.setattr(
        analytics,
        "_latch",
        DegradedLatch(analytics.logger, "analytics Redis", now=clock),
    )

    _run(analytics.record_search("anything", 1, weak=False, cached=False, latency_ms=10, filtered=False))
    _run(analytics.record_click("anything", 1))
    assert [r.levelname for r in _outage_lines(caplog)] == ["WARNING"]


def test_summary_raises_when_redis_is_down(monkeypatch):
    """A failed read must raise, not return a payload indistinguishable from an all-zero report."""

    class _BrokenRedis:
        async def mget(self, keys, *_rest):
            if isinstance(keys, str):
                keys = [keys, *_rest]
            else:
                keys = [*keys, *_rest]
            raise ConnectionError("redis unreachable")

        async def get(self, key):
            raise ConnectionError("redis unreachable")

        async def zrevrange(self, key, start, stop, withscores=False):
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())

    with pytest.raises(analytics.AnalyticsUnavailableError):
        _run(analytics.summary())


@pytest.fixture
def analytics_client(tmp_path):
    """Admin client over the real app on throwaway SQLite stores; yields (client, cookie, chat_store)."""
    from fastapi.testclient import TestClient

    from app import auth as auth_module
    from app import chat as chat_module
    from app import main
    from app.auth import AuthStore
    from app.chat import ChatStore

    auth_store = AuthStore(str(tmp_path / "auth.db"))
    _run(auth_store.connect())
    chat_store = ChatStore(str(tmp_path / "chat.db"))
    _run(chat_store.connect())
    auth_module.store = auth_store
    chat_module.store = chat_store
    try:
        user = _run(auth_store.create_user("admin@example.com", "secret1", "admin", "admin"))
        token = _run(auth_store.issue_token(user.id, 7))
        client = TestClient(main.app, raise_server_exceptions=False)
        yield client, auth_cookie(token), chat_store
    finally:
        auth_module.store = None
        chat_module.store = None
        _run(auth_store.close())
        _run(chat_store.close())


class _UnreachableRedis:
    async def mget(self, keys, *_rest):
        if isinstance(keys, str):
            keys = [keys, *_rest]
        else:
            keys = [*keys, *_rest]
        raise ConnectionError("redis unreachable")

    async def get(self, key):
        raise ConnectionError("redis unreachable")

    async def zrevrange(self, key, start, stop, withscores=False):
        raise ConnectionError("redis unreachable")


def test_analytics_summary_endpoint_is_503_when_redis_is_down(analytics_client, monkeypatch):
    """503, not a 200 all-zero report the dashboard would render as real during an outage."""
    client, cookie, _ = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _UnreachableRedis())

    res = client.get("/analytics/summary", cookies=cookie)

    assert res.status_code == 503
    body = res.json()
    assert body["error"]
    assert "searches_total" not in body


def test_analytics_chat_endpoint_is_503_when_chat_store_is_down(analytics_client, monkeypatch):
    client, cookie, chat_store = analytics_client

    async def boom(*args, **kwargs):
        raise RuntimeError("chat db gone")

    monkeypatch.setattr(chat_store, "_fetchone", boom)
    monkeypatch.setattr(chat_store, "_fetchall", boom)

    res = client.get("/analytics/chat", cookies=cookie)

    assert res.status_code == 503
    body = res.json()
    assert body["error"]
    assert "sessions" not in body


def test_analytics_endpoints_still_serve_200_when_stores_are_up(analytics_client, monkeypatch):
    """The 200 path is unchanged: a real report, not swallowed by the 503 mapping."""
    client, cookie, _ = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _FakeRedis())

    summary_res = client.get("/analytics/summary", cookies=cookie)
    chat_res = client.get("/analytics/chat", cookies=cookie)

    assert summary_res.status_code == 200
    assert "searches_total" in summary_res.json()
    assert chat_res.status_code == 200
    assert chat_res.json()["sessions"] == 0


def test_summary_read_is_recorded_in_the_admin_audit_trail(analytics_client, monkeypatch):
    client, cookie, chat_store = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _FakeRedis())

    before = len(_run(chat_store.admin_audit_log(limit=1000)))
    res = client.get("/analytics/summary", cookies=cookie)
    assert res.status_code == 200

    log = _run(chat_store.admin_audit_log(limit=1000))
    assert len(log) == before + 1
    assert log[0]["action"] == "analytics.summary.read"


def test_summary_serves_text_free_rows_over_http(analytics_client, monkeypatch):
    client, cookie, _ = analytics_client
    secret = "who bought northwind capital"
    fake = _SummaryRedis()
    fake.store["analytics:search:total"] = 12
    fake.zsets["analytics:top_queries"] = {analytics.query_digest(secret, "k"): 12}
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    body = client.get("/analytics/summary", cookies=cookie).json()

    assert secret not in json.dumps(body)
    assert body["top_queries"] == [[analytics.query_digest(secret, "k"), 12]]


def test_summary_still_serves_when_the_audit_write_fails(analytics_client, monkeypatch):
    """A broken audit table is best-effort and must not take the dashboard read down."""
    client, cookie, chat_store = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _FakeRedis())

    async def boom(*args, **kwargs):
        raise RuntimeError("audit db gone")

    monkeypatch.setattr(chat_store, "record_admin_audit", boom)

    res = client.get("/analytics/summary", cookies=cookie)

    assert res.status_code == 200
    assert "searches_total" in res.json()


def test_client_lazy_init_replaces_redis_db(monkeypatch):
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
    assert analytics._client() is client
    assert len(created) == 1
    assert created[0][0] == "redis://localhost:6379/0"
    assert created[0][1]["db"] == 1
    assert created[0][1]["decode_responses"] is True


def test_degraded_warns_once(monkeypatch, caplog, clock):
    """One line however many failures arrive: the frozen clock keeps the re-announce window shut."""
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

    monkeypatch.setattr(
        analytics,
        "_latch",
        DegradedLatch(analytics.logger, "analytics Redis", now=clock),
    )
    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())

    for _ in range(5):
        _run(analytics.record_search("q", 1, weak=False, cached=False, latency_ms=1, filtered=False))

    assert [r.levelname for r in _outage_lines(caplog)] == ["WARNING"]


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
    _run(analytics.close())


def test_record_click_with_article_id_tallies_query_click(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    _run(analytics.record_click("fintech funding", 3, article_id=42))
    _run(analytics.record_click("fintech funding", 1, article_id=42))
    assert fake.store["analytics:click:total"] == 2
    qkey = next(k for k in fake.store if k.startswith("analytics:query_click:"))
    assert qkey == "analytics:query_click:" + analytics.query_digest(
        "fintech funding", analytics._QUERY_DIGEST_KEY
    )
    assert fake.store[qkey] == 2


def test_record_click_without_article_id_skips_query_key(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    _run(analytics.record_click("q", 1))
    assert not any(k.startswith("analytics:query_click:") for k in fake.store)


def test_click_signals_no_raw_returns_none(monkeypatch):
    fake = _SignalsRedis([])
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) is None
    assert fake.queries == [
        "analytics:query_click:" + analytics.query_digest("q", analytics._QUERY_DIGEST_KEY)
    ]


def test_click_signals_below_min_clicks_returns_none(monkeypatch):
    monkeypatch.setattr(analytics.config, "CLICK_BOOST_MIN_CLICKS", 5)
    fake = _SignalsRedis([("12", 2.0), ("7", 2.0)])
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) is None


def test_click_signals_success_builds_dict(monkeypatch):
    monkeypatch.setattr(analytics.config, "CLICK_BOOST_MIN_CLICKS", 3)
    fake = _SignalsRedis([("42", 3.0), ("7", 1.0), ("99", 0.0)])
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) == {"total": 4, "by_id": {42: 3, 7: 1}}


def test_click_signals_redis_down_returns_none(monkeypatch, caplog, clock):
    """No signal, and the outage surfaces at the digest-key read, before any raw-text lookup."""

    class _BrokenRedis:
        async def get(self, key):
            raise ConnectionError("redis unreachable")

        async def set(self, key, value, nx=False):
            raise ConnectionError("redis unreachable")

        async def zrevrange(self, key, start, stop, withscores=False):
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(analytics, "_digest_warned", False)
    monkeypatch.setattr(
        analytics,
        "_latch",
        DegradedLatch(analytics.logger, "analytics Redis", now=clock),
    )
    monkeypatch.setattr(analytics, "_client", lambda: _BrokenRedis())
    with caplog.at_level(logging.WARNING, logger="analytics"):
        assert _run(analytics.click_signals("q")) is None
    assert analytics._digest_warned is True
    # A digest hiccup is not a store outage and must not consume the latch.
    assert _outage_lines(caplog) == []
    assert "analytics digest key unavailable" in caplog.records[0].getMessage()


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


def test_recorded_search_never_stores_the_query_text(monkeypatch):
    """Asserts on what was WRITTEN: a redacted response would still leave the corpus in Redis."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    secret = "acme corp series b"

    _run(analytics.record_search(secret, 5, weak=False, cached=False, latency_ms=10, filtered=False))

    assert secret not in repr(fake.store)
    assert fake.zincrby_members("analytics:top_queries") == [
        analytics.query_digest(secret, analytics._QUERY_DIGEST_KEY)
    ]


def test_summary_never_reports_the_query_text(monkeypatch):
    secret = "who acquired northwind capital"
    fake = _SummaryRedis()
    digest = analytics.query_digest(secret, "test-key")
    fake.zsets["analytics:top_queries"] = {digest: 12}
    fake.zsets["analytics:click_top_queries"] = {digest: 4}
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    s = _run(analytics.summary())

    assert secret not in repr(s)
    assert s["top_queries"] == [[digest, 12]]
    assert s["click_top_queries"] == [[digest, 4]]


def test_summary_drops_legacy_verbatim_members(monkeypatch):
    fake = _SummaryRedis()
    legacy = "project falcon acquisition terms"
    fake.zsets["analytics:top_queries"] = {legacy: 30, analytics.query_digest("ok", "k"): 5}
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    s = _run(analytics.summary())

    assert legacy not in repr(s)
    assert [q for q, _ in s["top_queries"]] == [analytics.query_digest("ok", "k")]


def test_summary_still_reports_a_full_list_when_legacy_rows_dominate(monkeypatch):
    """Legacy rows ranked above the digests must not shrink the window; hence the over-fetch."""
    fake = _SummaryRedis()
    rows = {f"legacy verbatim query {i}": 1000 - i for i in range(40)}
    digests = [f"q1:{i:032x}" for i in range(1, analytics.TOP_QUERIES_N + 6)]
    rows.update({d: 100 - i for i, d in enumerate(digests)})
    fake.zsets["analytics:top_queries"] = rows
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    s = _run(analytics.summary())

    assert "legacy verbatim" not in repr(s)
    assert len(s["top_queries"]) == analytics.TOP_QUERIES_N
    assert all(analytics._is_digest(q) for q, _ in s["top_queries"])
    assert [q for q, _ in s["top_queries"]] == digests[: analytics.TOP_QUERIES_N]


def test_summary_deletes_legacy_verbatim_members_from_the_store(monkeypatch):
    """The write path re-arms EXPIRE on the whole key each event, so a pre-upgrade member's TTL never lapses."""
    fake = _SummaryRedis()
    secret = "project falcon acquisition terms"
    fake.zsets["analytics:top_queries"] = {
        secret: 30,
        "another pre-upgrade query": 20,
        analytics.query_digest("live", "k"): 5,
    }
    fake.zsets["analytics:click_top_queries"] = {secret: 9}
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics, "_legacy_scrubbed", False)

    _run(analytics.summary())

    assert secret not in repr(fake.zsets)
    assert "another pre-upgrade query" not in repr(fake.zsets)
    assert list(fake.zsets["analytics:top_queries"]) == [analytics.query_digest("live", "k")]
    assert fake.zsets["analytics:click_top_queries"] == {}


def test_legacy_scrub_runs_once_not_on_every_read(monkeypatch):
    fake = _SummaryRedis()
    fake.zsets["analytics:top_queries"] = {analytics.query_digest("live", "k"): 5}
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics, "_legacy_scrubbed", False)

    scans = []
    original = fake.zrange

    async def counting_zrange(*args, **kwargs):
        scans.append(args[0])
        return await original(*args, **kwargs)

    monkeypatch.setattr(fake, "zrange", counting_zrange)

    _run(analytics.summary())
    first = len(scans)
    _run(analytics.summary())

    assert first > 0
    assert len(scans) == first


def test_summary_still_reads_when_the_scrub_fails(monkeypatch):
    """An unscannable store must not 503 the dashboard; the read-side filter still withholds the text."""

    class _NoZrangeRedis(_SummaryRedis):
        async def zrange(self, key, start, stop):
            raise ConnectionError("redis unreachable")

    fake = _NoZrangeRedis()
    fake.zsets["analytics:top_queries"] = {analytics.query_digest("live", "k"): 5}
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics, "_legacy_scrubbed", False)

    s = _run(analytics.summary())

    assert s["top_queries"] == [[analytics.query_digest("live", "k"), 5]]


def test_click_beacon_never_stores_the_query_text(monkeypatch):
    """The unauthenticated beacon is the widest door in: its query must not reach the store in any form."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    secret = "private acquisition rumour"

    _run(analytics.record_click(secret, 2, article_id=7))

    assert secret not in repr(fake.store)


def test_click_signals_still_finds_its_own_signal_after_hashing(monkeypatch):
    """The fake keeps real sorted sets: a stub ignoring keys cannot show write and read agreeing."""

    class _RoundTripRedis(_SummaryRedis):
        def pipeline(self):
            return self

        def incr(self, key, amount=1):
            return self

        def zincrby(self, key, amount, member):
            self.zsets.setdefault(key, {})
            self.zsets[key][member] = self.zsets[key].get(member, 0) + amount
            return self

        def expire(self, key, seconds):
            return self

        async def execute(self):
            return []

    fake = _RoundTripRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics.config, "CLICK_BOOST_MIN_CLICKS", 2)
    _run(analytics.record_click("fintech funding", 1, article_id=42))
    _run(analytics.record_click("fintech funding", 2, article_id=42))

    assert _run(analytics.click_signals("fintech funding")) == {"total": 2, "by_id": {42: 2}}


def test_digest_is_stable_across_calls_so_counts_aggregate():
    a = analytics.query_digest("fintech funding", "k")
    assert a == analytics.query_digest("fintech funding", "k")
    assert a != analytics.query_digest("healthtech funding", "k")


def test_digest_is_keyed_not_a_bare_hash():
    """A published key would let a dashboard reader compute the digest of any guessed query."""
    assert analytics.query_digest("fintech funding", "key-one") != analytics.query_digest(
        "fintech funding", "key-two"
    )


def test_digest_key_is_generated_once_and_shared(monkeypatch):
    """Persisted so every worker resolves the same key instead of splitting one query's counts."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics.config, "ANALYTICS_QUERY_KEY", "")

    first = _run(analytics._digest_key(fake))
    assert first
    assert fake.store[analytics.QUERY_DIGEST_KEY_REDIS_KEY] == first
    analytics._QUERY_DIGEST_KEY = None
    assert _run(analytics._digest_key(fake)) == first


def test_configured_query_key_overrides_the_stored_one(monkeypatch):
    fake = _FakeRedis()
    fake.store[analytics.QUERY_DIGEST_KEY_REDIS_KEY] = "stored-key"
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics.config, "ANALYTICS_QUERY_KEY", "configured-key")

    assert _run(analytics._digest_key(fake)) == "configured-key"


def test_counters_survive_a_digest_key_failure(monkeypatch):
    """A digest-key failure costs the top-query member only; the text-free counters must survive."""

    class _NoKeyRedis(_FakeRedis):
        async def set(self, key, value, nx=False):
            raise ConnectionError("redis unreachable")

    fake = _NoKeyRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_search("fintech funding", 5, weak=False, cached=False, latency_ms=210, filtered=True))

    assert fake.store["analytics:search:total"] == 1
    assert fake.store["analytics:search:latency:sum"] == 210
    assert "analytics:top_queries" not in fake.store


def test_click_signals_without_a_key_returns_none_rather_than_looking_up_raw_text(monkeypatch):
    class _NoKeyRedis:
        def pipeline(self):
            return self

        async def get(self, key):
            raise ConnectionError("redis unreachable")

        async def set(self, key, value, nx=False):
            raise ConnectionError("redis unreachable")

        async def zrevrange(self, key, start, stop, withscores=False):
            raise AssertionError("must not query using a raw-query key")

    monkeypatch.setattr(analytics, "_client", lambda: _NoKeyRedis())

    assert _run(analytics.click_signals("fintech funding")) is None


def test_empty_query_still_records_counters(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_search("", 0, weak=False, cached=False, latency_ms=5, filtered=False))
    _run(analytics.record_click("", 1, article_id=3))

    assert fake.store["analytics:search:total"] == 1
    assert fake.store["analytics:search:zero_results"] == 1
    assert fake.store["analytics:click:total"] == 1


def test_unicode_query_is_never_stored_verbatim(monkeypatch):
    """The digest hashes encoded bytes, so it cannot split a codepoint the way a raw str slice can."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    secret = "acquisition of Ünicode çompany 株式会社"

    _run(analytics.record_search(secret, 1, weak=False, cached=False, latency_ms=1, filtered=False))

    assert secret not in repr(fake.store)


def test_digest_width_is_bounded_regardless_of_query_length(monkeypatch):
    monkeypatch.setattr(analytics.config, "CLICK_QUERY_MAX_LEN", 256)
    digest = analytics.query_digest("x" * 10_000, "k")
    assert len(digest) == len(analytics.QUERY_DIGEST_PREFIX) + analytics.QUERY_DIGEST_HEX_LEN


class _SwitchableRedis:
    def __init__(self):
        self.broken = False

    def pipeline(self):
        return self

    def incr(self, key, amount=1):
        return self

    def zincrby(self, key, amount, member):
        return self

    def expire(self, key, seconds):
        return self

    async def execute(self):
        if self.broken:
            raise ConnectionError("redis unreachable")
        return []


def _record_search():
    return analytics.record_search("q", 1, weak=False, cached=False, latency_ms=1, filtered=False)


@pytest.fixture
def latch(monkeypatch, clock):
    """A fresh latch on the injected clock, so each test owns the transition state and window."""
    fresh = DegradedLatch(analytics.logger, "analytics Redis", now=clock)
    monkeypatch.setattr(analytics, "_latch", fresh)
    return fresh


@pytest.fixture
def clock():
    return _FakeClock()


def test_degraded_latch_logs_warn_warn_warn_across_flap(
    monkeypatch, caplog, latch, clock
):
    """failure -> success -> failure is 3 events (W, W, W); the clock jumps past the re-announce window."""
    fake = _SwitchableRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    fake.broken = True
    _run(_record_search())
    fake.broken = False
    _run(_record_search())
    clock.advance(REANNOUNCE_SECONDS + 1)
    fake.broken = True
    _run(_record_search())

    outage = _outage_lines(caplog)
    assert [r.levelname for r in outage] == ["WARNING"] * 3
    assert "analytics Redis unavailable" in outage[0].getMessage()
    assert outage[1].getMessage() == "analytics Redis recovered"
    assert "analytics Redis unavailable" in outage[2].getMessage()


def test_degraded_latch_logs_once_for_many_consecutive_failures(monkeypatch, caplog, latch):
    fake = _SwitchableRedis()
    fake.broken = True
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    for _ in range(5):
        _run(_record_search())

    assert [r.levelname for r in _outage_lines(caplog)] == ["WARNING"]


def test_degraded_latch_logs_nothing_when_no_failure_preceded(monkeypatch, caplog, latch):
    monkeypatch.setattr(analytics, "_client", lambda: _SwitchableRedis())
    _run(_record_search())

    assert _outage_lines(caplog) == []
