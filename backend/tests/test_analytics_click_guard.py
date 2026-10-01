"""Write-amplification guards on the unauthenticated /analytics/click beacon."""
import asyncio
import importlib.util
import math
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import analytics, auth, click_boost, main
from app import config as _config_module
from app.config import config
from app.main import SourceArticle

_client = TestClient(main.app, raise_server_exceptions=False)

# Tallies are derived from this policy rather than observed, so nothing here describes a policy the app is not running.
_CLICK_POLICY_KNOBS = (
    "ENABLE_CLICK_BOOST",
    "CLICK_BOOST_MIN_CLICKS",
    "CLICK_BOOST_MIN_ARTICLE_CLICKS",
    "CLICK_BOOST_MIN_SHARE",
    "CLICK_BOOST_MULT",
    "CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS",
    "CLICK_QUERY_MAX_LEN",
    "CLICK_QUERY_TTL_SECONDS",
)


def _shipped_config(monkeypatch, **env):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(os, "environ", dict(env))
    src = Path(_config_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("click_guard_shipped_config", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


class FakeRedis:
    def __init__(self):
        self.counters: dict[str, int] = {}
        self.sets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, int] = {}
        self.nx_keys: set[str] = set()
        self.queried: list[str] = []

    def pipeline(self):
        return _Pipeline(self)

    def incr(self, key, amount=1):
        self.counters[key] = self.counters.get(key, 0) + amount
        return self

    def zincrby(self, key, amount, member):
        bucket = self.sets.setdefault(key, {})
        bucket[member] = bucket.get(member, 0) + amount
        return self

    def expire(self, key, seconds):
        self.ttls[key] = seconds
        return self

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.nx_keys:
            return None
        if nx:
            self.nx_keys.add(key)
        if ex is not None:
            self.ttls[key] = ex
        self.counters[key] = value
        return True

    async def delete(self, key):
        self.nx_keys.discard(key)
        self.counters.pop(key, None)
        return 1

    async def zrevrange(self, key, start, end, withscores=False):
        self.queried.append(key)
        items = sorted(self.sets.get(key, {}).items(), key=lambda kv: -kv[1])[start : end + 1]
        return [(m, float(s)) for m, s in items] if withscores else [m for m, _ in items]


class _Pipeline:
    def __init__(self, redis):
        self.redis = redis

    def __getattr__(self, name):
        return getattr(self.redis, name)

    async def execute(self):
        return []


class QdrantStub:
    def __init__(self, ids=(), boom=False):
        self.ids = set(ids)
        self.boom = boom
        self.asked: list = []

    async def retrieve(self, collection_name, ids, with_payload=False, with_vectors=False):
        self.asked.append(list(ids))
        if self.boom:
            raise RuntimeError("qdrant down")
        return [type("P", (), {"id": i})() for i in ids if i in self.ids]


class BrokenDedupeRedis(FakeRedis):
    async def set(self, key, value, nx=False, ex=None):
        raise RuntimeError("redis write failed")


class FlakyPipelineRedis(FakeRedis):
    def __init__(self, fail_pipeline: bool = True):
        super().__init__()
        self.fail_pipeline = fail_pipeline

    def pipeline(self):
        return _FlakyPipeline(self)


class _FlakyPipeline(_Pipeline):
    async def execute(self):
        if self.redis.fail_pipeline:
            raise RuntimeError("redis pipeline failed")
        return []


def _run(coro):
    return asyncio.run(coro)

_TEST_DIGEST_KEY = "click-guard-test-digest-key"


@pytest.fixture(autouse=True)
def pinned_query_digest_key():
    analytics._QUERY_DIGEST_KEY = _TEST_DIGEST_KEY
    yield
    analytics._QUERY_DIGEST_KEY = None


def _digest(query):
    return analytics.query_digest(query, _TEST_DIGEST_KEY)


def _qkey(query):
    return f"analytics:query_click:{_digest(query)}"


@pytest.fixture(autouse=True)
def shipped_click_policy(monkeypatch):
    shipped = _shipped_config(monkeypatch)
    for knob in _CLICK_POLICY_KNOBS:
        monkeypatch.setattr(config, knob, getattr(shipped, knob))
    return shipped


def _votes_short_of_boost(target_votes):
    """Filler picked so the per-article floor and the share gate each fail on the SAME tally; neither implies the other."""
    share = config.CLICK_BOOST_MIN_SHARE
    if not 0 < share < 1:
        raise AssertionError(
            f"CLICK_BOOST_MIN_SHARE={share!r} admits no minority share; the "
            "click-guard thresholds are not a usable policy for this test"
        )
    total = max(
        config.CLICK_BOOST_MIN_CLICKS,
        target_votes + 1,
        math.ceil((target_votes + 1) / share),
    )
    return total, total - target_votes

@pytest.fixture
def store(monkeypatch):
    fake = FakeRedis()

    class _RateRedis:
        async def set(self, key, value, nx=False, ex=None):
            return True

        async def incr(self, key):
            fake.counters[key] = fake.counters.get(key, 0) + 1
            return fake.counters[key]

    monkeypatch.setattr(analytics, "_client", lambda: fake)
    monkeypatch.setattr(auth, "_rate_client", _RateRedis())
    return fake


@pytest.fixture
def index(monkeypatch):
    stub = QdrantStub(ids=(7, 42, 99, 123))
    monkeypatch.setitem(main.state, "qdrant", stub)
    return stub


def _as(ip):
    async def wrapper(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": (ip, 40000)}
        await main.app(scope, receive, send)

    return TestClient(wrapper, raise_server_exceptions=False)


def _beacon(query, position=1, article_id=None, ip="1.2.3.4"):
    return _as(ip).post(
        "/analytics/click",
        json={"query": query, "position": position, "id": article_id},
    )


def _results():
    return [
        SourceArticle(id=7, title="b", url="u7", score=0.90),
        SourceArticle(id=42, title="a", url="u42", score=0.85),
    ]


def _boosted(store, query="ola ipo"):
    return _run(click_boost.apply_click_boost(query, _results()))


def test_forged_beacon_burst_does_not_move_ranking(store, index):
    for _ in range(5):
        assert _beacon("ola ipo", 1, 42).status_code == 200
    total, filler = _votes_short_of_boost(1)
    for i in range(filler):
        _beacon("ola ipo", 2, 99, ip=f"10.4.4.{i}")

    assert store.sets[_qkey("ola ipo")] == {"42": 1.0, "99": float(filler)}, (
        "5 beacons, one vote"
    )
    assert sum(store.sets[_qkey("ola ipo")].values()) == total
    assert _boosted(store) == _results()
    assert index.asked[:5] == [[42]] * 5, "the id is looked up in the index on every beacon"


def test_forged_beacons_from_distinct_clients_still_cannot_boost_one_article(store, index):
    forged = max(1, config.CLICK_BOOST_MIN_ARTICLE_CLICKS)
    total, filler = _votes_short_of_boost(forged)

    for i in range(forged):
        assert _beacon("ola ipo", 1, 42, ip=f"10.0.0.{i}").status_code == 200
    for i in range(filler):
        _beacon("ola ipo", 2, 99, ip=f"10.9.9.{i}")

    assert total == forged + filler >= config.CLICK_BOOST_MIN_CLICKS
    tally = store.sets[_qkey("ola ipo")]
    assert tally["42"] == forged, "every forged beacon is a distinct client's vote"
    assert round(sum(tally.values()) * config.CLICK_BOOST_MIN_SHARE) > forged, (
        "the forged votes are a minority of the query's clicks"
    )
    assert _boosted(store) == _results()


def test_one_client_cannot_manufacture_click_share(store, index):
    flood = max(config.CLICK_BOOST_MIN_CLICKS, config.PUBLIC_CLICK_RATE_PER_MIN // 2)
    for _ in range(flood):
        _beacon("ola ipo", 1, 42, ip="10.1.1.1")
        _beacon("ola ipo", 1, 99, ip="10.1.1.1")
    total, filler = _votes_short_of_boost(2)
    for i in range(filler):
        _beacon("ola ipo", 3, 123, ip=f"10.1.2.{i}")

    tally = store.sets[_qkey("ola ipo")]
    assert tally["42"] == 1.0 and tally["99"] == 1.0, "one client, one vote per article"
    assert sum(tally.values()) == total >= config.CLICK_BOOST_MIN_CLICKS
    assert _boosted(store) == _results()


def test_distinct_clients_still_boost_a_genuinely_clicked_article(store, index):
    voters = max(
        config.CLICK_BOOST_MIN_CLICKS,
        config.CLICK_BOOST_MIN_ARTICLE_CLICKS,
        1,
    )
    for i in range(voters):
        assert _beacon("ola ipo", 1, 42, ip=f"10.0.0.{i}").status_code == 200

    out = _boosted(store)
    assert out[0].id == 42, "the clicked result should now rank first"
    assert out[0].score == pytest.approx(0.85 * config.CLICK_BOOST_MULT)


def test_shipped_click_boost_thresholds_are_the_shipped_ones(shipped_click_policy):
    shipped = shipped_click_policy
    assert (
        shipped.CLICK_BOOST_MIN_CLICKS,
        shipped.CLICK_BOOST_MIN_ARTICLE_CLICKS,
        shipped.CLICK_BOOST_MIN_SHARE,
    ) == (5, 3, 0.3), (
        "the click-boost thresholds changed. That is a ranking-tuning decision, "
        "not a consequence of the #242 dedupe fix: it makes the boost harder to "
        "trigger for genuine signal too, so it needs its own justification."
    )
    assert shipped.CLICK_BOOST_MULT == 1.3
    assert shipped.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS == 3600, (
        "the dedupe window is the control #242 actually relies on"
    )


def test_a_clients_repeat_click_adds_no_second_vote_but_others_still_count(store, index):
    for _ in range(5):
        _beacon("ola ipo", 1, 42, ip="10.0.0.1")
    _beacon("ola ipo", 1, 42, ip="10.0.0.2")

    assert store.sets[_qkey("ola ipo")] == {"42": 2.0}


def test_raw_click_analytics_are_untouched_by_the_dedupe(store, index):
    fired = 8
    for position in range(1, fired + 1):
        assert _beacon("ola ipo", position, 42, ip="10.5.5.5").status_code == 200

    assert store.counters["analytics:click:total"] == fired
    assert {k: v for k, v in store.counters.items() if k.startswith("analytics:click:pos:")} == {
        f"analytics:click:pos:{i}": 1 for i in range(1, fired + 1)
    }
    assert store.sets["analytics:click_top_queries"] == {_digest("ola ipo"): float(fired)}
    assert store.sets[_qkey("ola ipo")] == {"42": 1.0}


def test_position_outside_the_display_range_is_clamped_not_a_new_bucket(store, index):
    for position in (0, -5, 4, 11, 9999):
        assert _beacon("ola ipo", position, 42, ip="10.5.5.6").status_code == 200

    assert {k: v for k, v in store.counters.items() if k.startswith("analytics:click:pos:")} == {
        "analytics:click:pos:1": 2,
        "analytics:click:pos:4": 1,
        "analytics:click:pos:10": 2,
    }
    assert store.counters["analytics:click:total"] == 5


def test_click_without_an_id_is_still_counted(store, index):
    assert _beacon("ola ipo", 3).status_code == 200

    assert store.counters["analytics:click:total"] == 1
    assert store.counters["analytics:click:pos:3"] == 1
    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]


def test_a_non_numeric_article_id_never_becomes_a_ranking_vote(store, index):
    _run(analytics.record_click("ola ipo", 1, article_id="not-an-int", client_ip="10.9.9.9"))
    _run(analytics.record_click("ola ipo", 1, article_id=None, client_ip="10.9.9.10"))

    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]
    assert store.counters["analytics:click:total"] == 2


def test_beacon_for_an_unknown_article_id_records_no_ranking_vote(store, index):
    assert _beacon("ola ipo", 1, 999).status_code == 200

    assert store.counters["analytics:click:total"] == 1
    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]


def test_beacon_id_check_fails_closed_when_qdrant_is_unreachable(store, monkeypatch):
    monkeypatch.setitem(main.state, "qdrant", QdrantStub(boom=True))

    assert _beacon("ola ipo", 1, 42).status_code == 200

    assert store.counters["analytics:click:total"] == 1
    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]


def test_dedupe_claim_failure_drops_the_ranking_vote(store, index, monkeypatch):
    broken = BrokenDedupeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: broken)

    assert _beacon("ola ipo", 1, 42).status_code == 200

    assert broken.counters["analytics:click:total"] == 1
    assert not [k for k in broken.sets if k.startswith("analytics:query_click:")]


def test_query_spellings_collapse_to_a_single_boost_key(store, index):
    for spelling in ("ola ipo", "OLA IPO", "ola   ipo", "  Ola\tIPO\n", "Ola  Ipo"):
        _beacon(spelling, 1, 42, ip="10.2.2.2")

    assert [k for k in store.sets if k.startswith("analytics:query_click:")] == [
        _qkey("ola ipo")
    ]
    assert store.sets[_qkey("ola ipo")] == {"42": 1.0}


def test_normalisation_is_identical_on_the_read_path(store, index):
    _beacon("Ola   IPO", 1, 42, ip="10.2.2.3")
    (written,) = [k for k in store.sets if k.startswith("analytics:query_click:")]

    _run(analytics.click_signals("ola ipo"))

    assert store.queried == [written]


@pytest.mark.parametrize(
    "label,query",
    [
        ("short", "Ola   IPO"),
        ("long with whitespace run", "a" * 100 + " " * 200 + "b" * 100),
        ("long trailing run", "a " * 300),
        ("long unbroken", "x" * 5_000),
    ],
)
def test_a_vote_is_never_written_to_a_key_the_ranking_path_never_reads(store, index, label, query):
    _beacon(query, 1, 42, ip="10.8.8.8")
    (written,) = [k for k in store.sets if k.startswith("analytics:query_click:")]

    _run(analytics.click_signals(query))

    assert store.queried == [written], f"read path looked up a different key for {label}"


def test_over_long_query_is_stored_bounded_and_expires(store, index):
    huge = "ola ipo " + ("x" * 5_000_000)
    _beacon(huge, 1, 42, ip="10.3.3.3")

    (key,) = [k for k in store.sets if k.startswith("analytics:query_click:")]
    assert len(key) == len(_qkey("x"))
    assert key == _qkey("ola ipo " + "x" * config.CLICK_QUERY_MAX_LEN)
    assert store.ttls[key] == config.CLICK_QUERY_TTL_SECONDS


def test_dedupe_claim_keys_carry_a_ttl_and_leak_no_ip_or_query(store, index):
    _beacon("ola ipo", 1, 42, ip="203.0.113.9")

    (claim,) = [k for k in store.nx_keys]
    assert claim.startswith("analytics:click:seen:")
    assert "203.0.113.9" not in claim
    assert "ola" not in claim
    assert store.ttls[claim] == config.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS


def test_beacon_remains_public_and_per_ip_rate_limited(store, index, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_CLICK_RATE_PER_MIN", 2)

    first = _as("10.7.7.7")
    second = _as("10.7.7.8")
    for _ in range(2):
        assert first.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 200

    over = first.post("/analytics/click", json={"query": "q", "position": 1})
    assert over.status_code == 429
    assert second.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 200


def test_a_failed_write_gives_the_click_vote_back(store, index, monkeypatch):
    flaky = FlakyPipelineRedis()
    monkeypatch.setattr(analytics, "_client", lambda: flaky)

    assert _beacon("ola ipo", 1, 42, ip="10.4.4.4").status_code == 200
    assert not flaky.nx_keys, "the claim must not stay spent when the tally failed"

    healthy = FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: healthy)
    assert _beacon("ola ipo", 1, 42, ip="10.4.4.4").status_code == 200
    assert healthy.sets[_qkey("ola ipo")] == {"42": 1.0}


def test_a_lost_claim_is_never_released_for_its_owner(store, index, monkeypatch):
    flaky = FlakyPipelineRedis(fail_pipeline=False)
    monkeypatch.setattr(analytics, "_client", lambda: flaky)

    assert _beacon("ola ipo", 1, 42, ip="10.6.6.6").status_code == 200
    (claim,) = list(flaky.nx_keys)
    assert flaky.sets[_qkey("ola ipo")] == {"42": 1.0}

    flaky.fail_pipeline = True
    assert _beacon("ola ipo", 1, 42, ip="10.6.6.6").status_code == 200

    assert claim in flaky.nx_keys, "the winner's claim must survive a loser's failed write"
    assert flaky.sets[_qkey("ola ipo")] == {"42": 1.0}, "and still count once"
