"""Analytics recording/summary uses Redis aggregates and is best-effort."""

import json

import pytest
from _support import run_sync as _run

from app import analytics


class _FakeRedis:
    """In-memory Redis stand-in recording every pipeline command."""

    def __init__(self):
        self.store: dict = {}
        self.last_pipe = []
        # Every sorted-set member ever written, as (key, member). The store
        # itself collapses a zset to a running total, so this is the only way a
        # test can assert on WHAT was stored -- which is the whole question
        # #348 turns on.
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
                pass  # TTL not tracked by the fake
        self.last_pipe = []
        return []

    async def mget(self, keys):
        return [self.store.get(k) for k in keys]

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, nx=False):
        # ``nx`` is how the digest key is seeded without two workers minting
        # two different keys for one deployment.
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def zrevrange(self, key, start, stop, withscores=False):
        return []


@pytest.fixture(autouse=True)
def _reset_digest_key():
    """Clear the process-wide digest key around every test.

    ``_QUERY_DIGEST_KEY`` is deliberately cached for the life of the process
    (it is the secret shared by all workers), so without this a key seeded
    through one test's fake Redis would still be in force for the next test.
    """
    analytics._QUERY_DIGEST_KEY = None
    yield
    analytics._QUERY_DIGEST_KEY = None


class _SignalsRedis:
    """Redis stand-in returning a fixed zrevrange payload for click_signals."""

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


# --- summary() read-path ranges are derived, not hardcoded ---


class _SummaryRedis:
    """Redis stand-in with real sorted sets, so summary()'s top-N windows and
    click-position buckets are exercised for real (zrevrange end index is
    INCLUSIVE, exactly as Redis treats it)."""

    def __init__(self):
        self.store: dict = {}
        self.zsets: dict = {}
        self.calls: list = []

    async def mget(self, keys):
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


def test_summary_click_positions_follow_click_position_max(monkeypatch):
    """Raising CLICK_POSITION_MAX must widen the read path too: every bucket the
    write path can record has to be reported back, not silently truncated."""
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
    """The read path follows the bound in both directions (no stale buckets)."""
    fake = _SummaryRedis()
    for i in range(1, 26):
        fake.store[f"analytics:click:pos:{i}"] = i
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics, "CLICK_POSITION_MAX", 4)

    s = _run(analytics.summary())

    assert list(s["click_positions"]) == ["1", "2", "3", "4"]


def test_summary_default_click_positions_cover_one_to_ten(monkeypatch):
    """At the shipped default the documented shape is unchanged: ascending 1..10."""
    fake = _SummaryRedis()
    for i in range(1, 31):
        fake.store[f"analytics:click:pos:{i}"] = i
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert analytics.CLICK_POSITION_MAX == 10

    s = _run(analytics.summary())

    assert s["click_positions"] == {str(i): i for i in range(1, 11)}


def test_summary_top_lists_window_sizes_are_exact(monkeypatch):
    """Named top-N limits must yield exactly N members, not N or N+1, which is
    what a mis-transcribed inclusive zrevrange end index would cause."""
    fake = _SummaryRedis()
    # Members are digests now, so seed the store with members the read path
    # will actually keep -- a non-digest member is dropped, which would make
    # this test measure the filter instead of the window size.
    fake.zsets["analytics:top_queries"] = {f"q1:{i:032x}": 100 - i for i in range(1, 31)}
    fake.zsets["analytics:click_top_queries"] = {f"q1:{i:032x}": 100 - i for i in range(1, 21)}
    monkeypatch.setattr(analytics, "_client", lambda: fake)


    s = _run(analytics.summary())

    assert len(s["top_queries"]) == analytics.TOP_QUERIES_N == 20
    assert [q for q, _ in s["top_queries"]] == [f"q1:{i:032x}" for i in range(1, 21)]
    assert len(s["click_top_queries"]) == analytics.TOP_CLICKED_QUERIES_N == 10
    assert [q for q, _ in s["click_top_queries"]] == [f"q1:{i:032x}" for i in range(1, 11)]


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


def test_summary_raises_when_redis_is_down(monkeypatch):
    """A failed analytics read must be a failure, not a payload the caller
    cannot tell from a report whose counters are legitimately all zero
    (#281). The HTTP layer turns this into 503."""

    class _BrokenRedis:
        async def mget(self, keys):
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
    """An admin-authenticated TestClient over the real app, with both SQLite
    stores on throwaway files. Yields (client, admin_headers, chat_store)."""
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
        yield client, {"Authorization": f"Bearer {token}"}, chat_store
    finally:
        auth_module.store = None
        chat_module.store = None
        _run(auth_store.close())
        _run(chat_store.close())


class _UnreachableRedis:
    async def mget(self, keys):
        raise ConnectionError("redis unreachable")

    async def get(self, key):
        raise ConnectionError("redis unreachable")

    async def zrevrange(self, key, start, stop, withscores=False):
        raise ConnectionError("redis unreachable")


def test_analytics_summary_endpoint_is_503_when_redis_is_down(analytics_client, monkeypatch):
    """/analytics/summary must not answer 200 while the analytics store is
    down: the status has to agree with the body, or the dashboard renders an
    all-zero report during an outage (#281)."""
    client, headers, _ = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _UnreachableRedis())

    res = client.get("/analytics/summary", headers=headers)

    assert res.status_code == 503
    body = res.json()
    assert body["error"]
    # A degraded body must not be mistakable for a report.
    assert "searches_total" not in body


def test_analytics_chat_endpoint_is_503_when_chat_store_is_down(analytics_client, monkeypatch):
    """/analytics/chat must answer 503 when the chat store cannot be read, for
    the same reason as /analytics/summary (#281)."""
    client, headers, chat_store = analytics_client

    async def boom(*args, **kwargs):
        raise RuntimeError("chat db gone")

    monkeypatch.setattr(chat_store, "_fetchone", boom)
    monkeypatch.setattr(chat_store, "_fetchall", boom)

    res = client.get("/analytics/chat", headers=headers)

    assert res.status_code == 503
    body = res.json()
    assert body["error"]
    assert "sessions" not in body


def test_analytics_endpoints_still_serve_200_when_stores_are_up(analytics_client, monkeypatch):
    """The healthy path is unchanged: a 200 whose body is a real report. Pins
    that the 503 mapping did not swallow working reads."""
    client, headers, _ = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _FakeRedis())

    summary_res = client.get("/analytics/summary", headers=headers)
    chat_res = client.get("/analytics/chat", headers=headers)

    assert summary_res.status_code == 200
    assert "searches_total" in summary_res.json()
    assert chat_res.status_code == 200
    # A real (empty) report: zero sessions is genuine data, not a degraded body.
    assert chat_res.json()["sessions"] == 0


def test_summary_read_is_recorded_in_the_admin_audit_trail(analytics_client, monkeypatch):
    """The cross-user read leaves a trail, as #273 established for
    /analytics/chat. Without one, an admin reading these aggregates is
    indistinguishable from nobody having looked."""
    client, headers, chat_store = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _FakeRedis())

    before = len(_run(chat_store.admin_audit_log(limit=1000)))
    res = client.get("/analytics/summary", headers=headers)
    assert res.status_code == 200

    log = _run(chat_store.admin_audit_log(limit=1000))
    assert len(log) == before + 1
    assert log[0]["action"] == "analytics.summary.read"


def test_summary_serves_text_free_rows_over_http(analytics_client, monkeypatch):
    """End-to-end: what the admin dashboard actually receives contains no
    search text, only digests."""
    client, headers, _ = analytics_client
    secret = "who bought northwind capital"
    fake = _SummaryRedis()
    fake.store["analytics:search:total"] = 12
    fake.zsets["analytics:top_queries"] = {analytics.query_digest(secret, "k"): 12}
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    body = client.get("/analytics/summary", headers=headers).json()

    assert secret not in json.dumps(body)
    assert body["top_queries"] == [[analytics.query_digest(secret, "k"), 12]]


def test_summary_still_serves_when_the_audit_write_fails(analytics_client, monkeypatch):
    """The audit is best-effort: if it cannot be written the (text-free) read
    must still succeed, or a broken audit table would take the dashboard down
    with it."""
    client, headers, chat_store = analytics_client
    monkeypatch.setattr(analytics, "_client", lambda: _FakeRedis())

    async def boom(*args, **kwargs):
        raise RuntimeError("audit db gone")

    monkeypatch.setattr(chat_store, "record_admin_audit", boom)

    res = client.get("/analytics/summary", headers=headers)

    assert res.status_code == 200
    assert "searches_total" in res.json()


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
    # Keyed by the query's digest, not the query: click_signals can still reach
    # the set, without the text ever becoming a Redis key.
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


# --- click_signals ---


def test_click_signals_no_raw_returns_none(monkeypatch):
    fake = _SignalsRedis([])
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    assert _run(analytics.click_signals("q")) is None
    assert fake.queries == [
        "analytics:query_click:" + analytics.query_digest("q", analytics._QUERY_DIGEST_KEY)
    ]


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


# --- #348: cross-user search text must not be stored or reported ---


def test_recorded_search_never_stores_the_query_text(monkeypatch):
    """The store must not hold user-authored search text at all.

    Load-bearing for #348: redacting the response alone would leave the corpus
    in Redis for the next reader to return, so this asserts on what was
    WRITTEN, not on how the response looks.
    """
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    secret = "acme corp series b"

    _run(analytics.record_search(secret, 5, weak=False, cached=False, latency_ms=10, filtered=False))

    assert secret not in repr(fake.store)
    # The only member written to top_queries is the digest, not the query.
    assert fake.zincrby_members("analytics:top_queries") == [
        analytics.query_digest(secret, analytics._QUERY_DIGEST_KEY)
    ]


def test_summary_never_reports_the_query_text(monkeypatch):
    """A digest is what the dashboard receives, and the text is not in it."""
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
    """Rows written before this change are verbatim queries and sit in the store
    until their TTL lapses. An upgraded deployment must not keep serving them
    for the length of that TTL, so the read path drops any non-digest member."""
    fake = _SummaryRedis()
    legacy = "project falcon acquisition terms"
    fake.zsets["analytics:top_queries"] = {legacy: 30, analytics.query_digest("ok", "k"): 5}
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    s = _run(analytics.summary())

    assert legacy not in repr(s)
    assert [q for q, _ in s["top_queries"]] == [analytics.query_digest("ok", "k")]


def test_click_beacon_never_stores_the_query_text(monkeypatch):
    """The unauthenticated beacon is the widest door in: its query reaches the
    top-click set AND becomes a Redis key. Neither may contain the text."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    secret = "private acquisition rumour"

    _run(analytics.record_click(secret, 2, article_id=7))

    assert secret not in repr(fake.store)


def test_click_signals_still_finds_its_own_signal_after_hashing(monkeypatch):
    """Hashing the key must not break the click-boost layer: the signal recorded
    for a query must still be found by looking that same query up. This is the
    regression the hashing could have caused, and why the digest is a stable
    function of the query rather than a random id.

    The fake needs real sorted sets: the point is that the write and the read
    agree on the key, which a stub that ignores keys cannot demonstrate.
    """

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
    """Same query, same digest, always -- otherwise a query's searches scatter
    across per-worker digests and the counts mean nothing."""
    a = analytics.query_digest("fintech funding", "k")
    assert a == analytics.query_digest("fintech funding", "k")
    assert a != analytics.query_digest("healthtech funding", "k")


def test_digest_is_keyed_not_a_bare_hash():
    """The key has to matter. Under a published key the digest of any guessed
    query is computable by whoever can read the dashboard, which would leave
    the fix cosmetic."""
    assert analytics.query_digest("fintech funding", "key-one") != analytics.query_digest(
        "fintech funding", "key-two"
    )


def test_digest_key_is_generated_once_and_shared(monkeypatch):
    """With no ANALYTICS_QUERY_KEY the key is minted and persisted, so every
    worker resolves the same one instead of each inventing a private namespace
    (which would split one query's counts four ways)."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics.config, "ANALYTICS_QUERY_KEY", "")

    first = _run(analytics._digest_key(fake))
    assert first
    assert fake.store[analytics.QUERY_DIGEST_KEY_REDIS_KEY] == first
    analytics._QUERY_DIGEST_KEY = None
    assert _run(analytics._digest_key(fake)) == first


def test_configured_query_key_overrides_the_stored_one(monkeypatch):
    """ANALYTICS_QUERY_KEY pins the namespace so digests survive a rebuild."""
    fake = _FakeRedis()
    fake.store[analytics.QUERY_DIGEST_KEY_REDIS_KEY] = "stored-key"
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(analytics.config, "ANALYTICS_QUERY_KEY", "configured-key")

    assert _run(analytics._digest_key(fake)) == "configured-key"


def test_counters_survive_a_digest_key_failure(monkeypatch):
    """A Redis hiccup resolving the key must cost the top-query member only.
    The volume/latency/cache counters carry no user text, so losing them would
    be a reporting regression worse than the leak being closed."""

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
    """With the key unresolvable the only options are no signal, or a lookup
    keyed by the raw query. It must be no signal -- the raw fallback would
    reintroduce exactly the text this removed."""

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
    """An empty query has no text to leak and must not crash the writer."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)

    _run(analytics.record_search("", 0, weak=False, cached=False, latency_ms=5, filtered=False))
    _run(analytics.record_click("", 1, article_id=3))

    assert fake.store["analytics:search:total"] == 1
    assert fake.store["analytics:search:zero_results"] == 1
    assert fake.store["analytics:click:total"] == 1


def test_unicode_query_is_never_stored_verbatim(monkeypatch):
    """Non-ASCII search text must not raise and must not appear in the store.
    The digest is computed from encoded bytes, so it also cannot split a
    codepoint the way a raw str slice can."""
    fake = _FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: fake)
    secret = "acquisition of Ünicode çompany 株式会社"

    _run(analytics.record_search(secret, 1, weak=False, cached=False, latency_ms=1, filtered=False))

    assert secret not in repr(fake.store)


def test_digest_width_is_bounded_regardless_of_query_length(monkeypatch):
    """A fixed-width digest means the stored member cannot grow with beacon
    input, which is what the old length cap existed to prevent."""
    monkeypatch.setattr(analytics.config, "CLICK_QUERY_MAX_LEN", 256)
    digest = analytics.query_digest("x" * 10_000, "k")
    assert len(digest) == len(analytics.QUERY_DIGEST_PREFIX) + analytics.QUERY_DIGEST_HEX_LEN
