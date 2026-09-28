"""Write-amplification guards on the anonymous /analytics/click beacon (#242).

The beacon is intentionally unauthenticated (it is fired by anonymous search
traffic), so these tests pin the controls that keep an unauthenticated caller
from moving the ranking other users see: the per-client dedupe of the ranking
vote, the index check on the client-supplied article id, and the canonical form
of the stored query. They exercise the real route, the real recording and the
real boosting code, standing in only for Redis and Qdrant.

Hermeticity: ``app/config.py`` calls ``load_dotenv()`` at import, so the live
``config`` object carries whatever a developer's ``backend/.env`` says -- and an
env file that pins ``CLICK_BOOST_MIN_*`` silently replaces the shipped
thresholds. These tests pin a security property, so they must not read whatever
this machine happens to have: the autouse fixture below loads the SHIPPED policy
and installs it on the live config. Vote counts are then derived FROM that
policy, so each test states its claim for any valid thresholds rather than
relying on one particular ratio between them holding.
"""
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

# The click-boost policy these tests run under. The values themselves are
# pinned explicitly in test_shipped_click_boost_thresholds_are_the_shipped_ones,
# so retuning the boost is a deliberate, visible act rather than something an
# operator's .env can quietly redefine.
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
    """Parse ``app/config.py`` in a private module under a controlled environment.

    Two things have to be neutralised for the result to be the SHIPPED policy
    rather than an echo of this machine: ``load_dotenv()`` runs inside
    config.py and would repopulate ``os.environ`` from a developer's
    ``backend/.env`` (and it resolves the file from the CALLING file's
    location, not the CWD, so chdir alone would not keep it out), and the
    ambient environment is replaced outright so an unrelated shell export
    cannot leak in. monkeypatch undoes both. The load is private, so no other
    module's ``config`` object is touched and the rest of the suite still sees
    the process-wide config it had before.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(os, "environ", dict(env))
    src = Path(_config_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("click_guard_shipped_config", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


class FakeRedis:
    """Redis stand-in with real SET NX EX semantics, so the click-signal dedupe
    behaves as it does in production (first writer wins, the rest are told the
    key already exists)."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.sets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, int] = {}
        self.nx_keys: set[str] = set()
        self.queried: list[str] = []

    # -- pipeline (record_click) --
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

    # -- direct (dedupe claim) --
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
    """Qdrant stand-in holding a fixed set of point ids."""

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
    """Redis whose SET NX fails, standing in for a partial outage."""

    async def set(self, key, value, nx=False, ex=None):
        raise RuntimeError("redis write failed")


class FlakyPipelineRedis(FakeRedis):
    """Redis that accepts the dedupe claim but can be made to fail the write
    that was supposed to record it, standing in for a failure between the two.
    The claim and any release of it hit the same state, as they would in Redis.
    """

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

# The secret every query digest is keyed with in this file. Pinned rather than
# minted so the expected Redis key is a fixed, checkable value, and so none of
# the fakes above has to implement the ``analytics:query_digest_key`` read that
# ``_digest_key`` would otherwise make. The stored key is an opaque digest, not
# the query text (#348) -- the click-boost signal is addressed the same way,
# which is what these guards still exercise.
_TEST_DIGEST_KEY = "click-guard-test-digest-key"


@pytest.fixture(autouse=True)
def pinned_query_digest_key():
    """Install a fixed query-digest key for every test in this file.

    ``_QUERY_DIGEST_KEY`` is cached process-wide by design (it is the secret
    shared by every gunicorn worker), so it must be seeded and cleared per test
    or one test's key leaks into the next. Pinning it also keeps these
    expectations independent of whatever ``ANALYTICS_QUERY_KEY`` this machine's
    .env names, and -- because the key is resolved without a Redis round trip
    -- leaves ``nx_keys`` holding only dedupe CLAIMS, which is what the claim
    assertions below read.
    """
    analytics._QUERY_DIGEST_KEY = _TEST_DIGEST_KEY
    yield
    analytics._QUERY_DIGEST_KEY = None


def _digest(query):
    """The opaque id a query is stored and looked up under."""
    return analytics.query_digest(query, _TEST_DIGEST_KEY)


def _qkey(query):
    """The per-query click sorted-set key for ``query``."""
    return f"analytics:query_click:{_digest(query)}"





@pytest.fixture(autouse=True)
def shipped_click_policy(monkeypatch):
    """Install the SHIPPED click-boost policy on the live config for every test.

    Without this, these assertions read whatever the developer running them has
    in ``backend/.env`` -- and #242 proved the cost of that: a .env pinning
    ``CLICK_BOOST_MIN_*`` redefines the bar these tests claim to pin, so
    ``_CLICK_POLICY_KNOBS`` are read from a clean parse of config.py and set
    explicitly. The knobs are read at call time by the code under test, so
    setting them here is enough to steer the real route, recording and
    boosting. monkeypatch restores the ambient values afterwards, leaving the
    rest of the suite exactly as it found them.
    """
    shipped = _shipped_config(monkeypatch)
    for knob in _CLICK_POLICY_KNOBS:
        monkeypatch.setattr(config, knob, getattr(shipped, knob))
    return shipped


def _votes_short_of_boost(target_votes):
    """Votes for a filler article that leave ``target_votes`` a minority share.

    ``apply_click_boost`` boosts an article holding ``c`` clicks only when
    ``c >= CLICK_BOOST_MIN_ARTICLE_CLICKS`` AND ``c >= max(1, round(total *
    CLICK_BOOST_MIN_SHARE))``. The share gate is the one an attacker's own
    clicks cannot satisfy on their own: the more total traffic the query has,
    the more filler is needed. Solving ``round(total * share) > target_votes``
    for ``total`` gives the padding below, so the target's share sits strictly
    under the threshold BY CONSTRUCTION -- it does not depend on
    ``min_article / min_clicks < min_share`` happening to hold for whatever
    thresholds are in force, which is what broke under a retuned or
    .env-overridden policy.

    Returns ``(total_clicks, filler_votes)``; the caller is responsible for
    firing the filler from distinct clients so each one really is a vote.
    """
    share = config.CLICK_BOOST_MIN_SHARE
    if not 0 < share < 1:
        # A share at or above 1 can never be crossed by any article, and a
        # share of 0 makes every article a majority of the query. Neither is a
        # threshold this test can demonstrate a minority share against, so the
        # tests that use this helper say so rather than assert something the
        # policy has made vacuous.
        raise AssertionError(
            f"CLICK_BOOST_MIN_SHARE={share!r} admits no minority share; the "
            "click-guard thresholds are not a usable policy for this test"
        )
    # The smallest total whose share gate lands strictly above target_votes.
    # +1 keeps the result clear of banker's rounding on an exact .5.
    total = max(
        config.CLICK_BOOST_MIN_CLICKS,
        target_votes + 1,
        math.ceil((target_votes + 1) / share),
    )
    return total, total - target_votes

@pytest.fixture
def store(monkeypatch):
    """Install a fresh analytics Redis and a working rate-limiter store."""
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
    """Install a Qdrant stub holding ids 7, 42, 99 and 123, as the search path
    does. 123 is a clickable result that ``_results()`` never returns, so tests
    can give the query real traffic that contributes to the click total without
    appearing in the ranking -- the traffic a boost must not be fooled by.
    """
    stub = QdrantStub(ids=(7, 42, 99, 123))
    monkeypatch.setitem(main.state, "qdrant", stub)
    return stub


def _as(ip):
    """A TestClient whose requests present as coming from ``ip``, so the
    per-client dedupe and the per-IP rate limit see distinct callers."""

    async def wrapper(scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "client": (ip, 40000)}
        await main.app(scope, receive, send)

    return TestClient(wrapper, raise_server_exceptions=False)


def _beacon(query, position=1, article_id=None, ip="1.2.3.4"):
    """Fire one beacon as a distinct client."""
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


# --- the attack from the issue: a burst of forged beacons ---


def test_forged_beacon_burst_does_not_move_ranking(store, index):
    """The issue's repro, post-fix: 5 forged beacons from one client for a real
    article id leave both the scores and the order exactly as they were.

    The dedupe is the load-bearing part and holds for any policy: five beacons,
    one vote. The ranking assertion is made independent of the thresholds by
    giving the query real traffic from uninvolved clients, so the lone forged
    vote is both under the per-article bar and a minority share. It is not left
    to the accident that one vote happens to be too few at today's thresholds.
    """
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
    """A handful of clients voting for the same article is not enough, even once
    the query is busy enough for the signal to be live at all: their votes hold
    too small a share of the query's total to be believed.

    The attacker spends exactly the per-article allowance they are entitled to
    (``CLICK_BOOST_MIN_ARTICLE_CLICKS`` distinct clients, one vote each -- the
    most a single forged campaign can buy under the dedupe), and every other
    click on the query comes from a different, uninvolved client. The filler
    count is DERIVED so the target's share sits below ``CLICK_BOOST_MIN_SHARE``
    by construction: the claim is about a minority never being believed, and
    holds for any valid thresholds. It is not the case that
    ``min_article / min_clicks < min_share`` happens to be true, which is what
    the old count arithmetic silently assumed -- under a .env or a retuned
    policy that ratio inverts, and a handful of clients really would boost the
    article, so the test would fail for a reason that has nothing to do with the
    dedupe it is here to pin.
    """
    forged = max(1, config.CLICK_BOOST_MIN_ARTICLE_CLICKS)
    total, filler = _votes_short_of_boost(forged)

    for i in range(forged):
        assert _beacon("ola ipo", 1, 42, ip=f"10.0.0.{i}").status_code == 200
    for i in range(filler):
        _beacon("ola ipo", 2, 99, ip=f"10.9.9.{i}")

    # The signal really is live -- the target is not spared by a quiet query.
    assert total == forged + filler >= config.CLICK_BOOST_MIN_CLICKS
    tally = store.sets[_qkey("ola ipo")]
    assert tally["42"] == forged, "every forged beacon is a distinct client's vote"
    assert round(sum(tally.values()) * config.CLICK_BOOST_MIN_SHARE) > forged, (
        "the forged votes are a minority of the query's clicks"
    )
    assert _boosted(store) == _results()


def test_one_client_cannot_manufacture_click_share(store, index):
    """One client cannot vote its way to a majority share: however hard it
    floods the query, and however busy the query is from other clients, it is
    left with a single vote on each article it names.

    The flood is bounded by the shipped per-IP rate limit rather than by
    ``CLICK_BOOST_MIN_CLICKS``, so the test does not change shape if the
    thresholds are retuned, and the query is filled with other clients' clicks
    so the single client's vote is a minority rather than being spared by a
    quiet query.
    """
    flood = max(config.CLICK_BOOST_MIN_CLICKS, config.PUBLIC_CLICK_RATE_PER_MIN // 2)
    for _ in range(flood):
        _beacon("ola ipo", 1, 42, ip="10.1.1.1")
        _beacon("ola ipo", 1, 99, ip="10.1.1.1")
    total, filler = _votes_short_of_boost(2)  # one vote on 42 and one on 99
    for i in range(filler):
        _beacon("ola ipo", 3, 123, ip=f"10.1.2.{i}")

    tally = store.sets[_qkey("ola ipo")]
    assert tally["42"] == 1.0 and tally["99"] == 1.0, "one client, one vote per article"
    assert sum(tally.values()) == total >= config.CLICK_BOOST_MIN_CLICKS
    assert _boosted(store) == _results()


# --- the legitimate path still works ---


def test_distinct_clients_still_boost_a_genuinely_clicked_article(store, index):
    """The feature is not switched off: once enough distinct users click one
    result, that result is boosted and re-sorted.

    The vote count is derived from the policy (every distinct client that the
    boost needs at least), and the click is unanimous, so this holds for any
    thresholds rather than for the particular counts shipped today.
    """
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
    """The shipped click-boost thresholds are pinned, deliberately.

    #242 fixes the forged-burst attack with the per-client vote dedupe, which
    defeats the attack at ANY threshold. It deliberately does NOT raise these
    values: raising them also makes the boost harder to trigger for genuine
    signal, which is a ranking-tuning decision that needs measurement, not a
    side effect of a security fix. This test is where that decision gets made
    visible if it is ever made.

    The values are read from a clean parse of config.py with ``load_dotenv``
    neutralised (see ``_shipped_config``), so what is asserted is what the code
    ships -- not what a developer's ``backend/.env`` happens to override it
    with. That override is exactly how #242's fix could be silently disabled:
    an env file pinning ``CLICK_BOOST_MIN_*`` wins over the code default, and
    every threshold-sensitive test here would then be asserting the operator's
    policy while appearing to assert the repository's.
    """
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
    """Re-opening the same result carries no new ranking information, so a repeat
    is dropped -- but the dedupe is per client, so it cannot be used to silence
    another user's genuine click."""
    for _ in range(5):
        _beacon("ola ipo", 1, 42, ip="10.0.0.1")
    _beacon("ola ipo", 1, 42, ip="10.0.0.2")

    assert store.sets[_qkey("ola ipo")] == {"42": 2.0}


def test_raw_click_analytics_are_untouched_by_the_dedupe(store, index):
    """The dedupe gates the RANKING signal only. Every beacon a client sends is
    still counted in the raw click analytics exactly as before, so the numbers
    the product reports do not change."""
    fired = 8  # every position inside the display range, so no clamping happens
    for position in range(1, fired + 1):
        assert _beacon("ola ipo", position, 42, ip="10.5.5.5").status_code == 200

    assert store.counters["analytics:click:total"] == fired
    # Exact buckets, not a sum: a total alone is blind to WHERE the clicks landed.
    assert {k: v for k, v in store.counters.items() if k.startswith("analytics:click:pos:")} == {
        f"analytics:click:pos:{i}": 1 for i in range(1, fired + 1)
    }
    assert store.sets["analytics:click_top_queries"] == {_digest("ola ipo"): float(fired)}
    # ...while the ranking vote is one.
    assert store.sets[_qkey("ola ipo")] == {"42": 1.0}


def test_position_outside_the_display_range_is_clamped_not_a_new_bucket(store, index):
    """A beacon cannot mint an ``analytics:click:pos:{n}`` key outside the range
    the summary reads: out-of-range positions collapse onto the first/last
    tracked slot rather than each creating a new bucket."""
    for position in (0, -5, 4, 11, 9999):
        assert _beacon("ola ipo", position, 42, ip="10.5.5.6").status_code == 200

    assert {k: v for k, v in store.counters.items() if k.startswith("analytics:click:pos:")} == {
        "analytics:click:pos:1": 2,  # 0 and -5
        "analytics:click:pos:4": 1,
        "analytics:click:pos:10": 2,  # 11 and 9999
    }
    assert store.counters["analytics:click:total"] == 5


def test_click_without_an_id_is_still_counted(store, index):
    """A beacon with no id (the frontend may omit it) still records the click."""
    assert _beacon("ola ipo", 3).status_code == 200

    assert store.counters["analytics:click:total"] == 1
    assert store.counters["analytics:click:pos:3"] == 1
    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]


def test_a_non_numeric_article_id_never_becomes_a_ranking_vote(store, index):
    """``record_click`` is the module's public entry point, so its id handling
    is pinned directly as well as through the route. A value that is not an
    integer must not be written into the per-query sorted set -- a garbage
    member there is a member the ranking layer can never interpret. The raw
    click is still counted, exactly as for a beacon with no id at all."""
    _run(analytics.record_click("ola ipo", 1, article_id="not-an-int", client_ip="10.9.9.9"))
    _run(analytics.record_click("ola ipo", 1, article_id=None, client_ip="10.9.9.10"))

    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]
    # Both beacons were still counted -- the guard drops the vote, not the click.
    assert store.counters["analytics:click:total"] == 2


# --- the client-supplied id must exist in the index ---


def test_beacon_for_an_unknown_article_id_records_no_ranking_vote(store, index):
    """id 999 is not in the collection: the click is counted for analytics, but
    no boost record is minted for an article that can never be returned."""
    assert _beacon("ola ipo", 1, 999).status_code == 200

    assert store.counters["analytics:click:total"] == 1
    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]


def test_beacon_id_check_fails_closed_when_qdrant_is_unreachable(store, monkeypatch):
    """An unconfirmable id must not become a ranking vote."""
    monkeypatch.setitem(main.state, "qdrant", QdrantStub(boom=True))

    assert _beacon("ola ipo", 1, 42).status_code == 200

    assert store.counters["analytics:click:total"] == 1
    assert not [k for k in store.sets if k.startswith("analytics:query_click:")]


def test_dedupe_claim_failure_drops_the_ranking_vote(store, index, monkeypatch):
    """If the dedupe store cannot answer, the click is counted but not allowed
    to steer ranking."""
    broken = BrokenDedupeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: broken)

    assert _beacon("ola ipo", 1, 42).status_code == 200

    assert broken.counters["analytics:click:total"] == 1
    assert not [k for k in broken.sets if k.startswith("analytics:query_click:")]


# --- one logical query must not mint many boost keys ---


def test_query_spellings_collapse_to_a_single_boost_key(store, index):
    """Case and whitespace variations are one query, so they share one key
    instead of each minting a boost record nothing reads back."""
    for spelling in ("ola ipo", "OLA IPO", "ola   ipo", "  Ola\tIPO\n", "Ola  Ipo"):
        _beacon(spelling, 1, 42, ip="10.2.2.2")

    assert [k for k in store.sets if k.startswith("analytics:query_click:")] == [
        _qkey("ola ipo")
    ]
    # All five came from ONE client, so the canonicalised claim must collapse
    # them to a single vote -- a per-spelling claim would let one client buy
    # five votes just by varying its whitespace and case.
    assert store.sets[_qkey("ola ipo")] == {"42": 1.0}


def test_normalisation_is_identical_on_the_read_path(store, index):
    """A click recorded as 'Ola   IPO' lands on the very key the ranking path
    reads for 'ola ipo', so the signal is not stranded under a key nothing ever
    looks up."""
    _beacon("Ola   IPO", 1, 42, ip="10.2.2.3")
    (written,) = [k for k in store.sets if k.startswith("analytics:query_click:")]

    _run(analytics.click_signals("ola ipo"))

    assert store.queried == [written]


@pytest.mark.parametrize(
    "label,query",
    [
        # Under the cap: the two orders of collapse-then-bound and
        # bound-then-collapse happen to agree, so this case proves nothing.
        ("short", "Ola   IPO"),
        # Over the cap AND containing a whitespace run: bounding the raw string
        # first throws the tail away, collapsing first keeps it. These two
        # orders disagree here, which is what stranded the vote.
        ("long with whitespace run", "a" * 100 + " " * 200 + "b" * 100),
        ("long trailing run", "a " * 300),
        ("long unbroken", "x" * 5_000),
    ],
)
def test_a_vote_is_never_written_to_a_key_the_ranking_path_never_reads(store, index, label, query):
    """The write and the read path must derive the SAME boost key, whatever the
    query's length and whitespace. A vote stored under any other key is a vote
    that can never move a ranking."""
    _beacon(query, 1, 42, ip="10.8.8.8")
    (written,) = [k for k in store.sets if k.startswith("analytics:query_click:")]

    _run(analytics.click_signals(query))

    assert store.queried == [written], f"read path looked up a different key for {label}"


def test_over_long_query_is_stored_bounded_and_expires(store, index):
    """A megabyte of query text cannot become a megabyte Redis key, and the key
    is given a TTL so it cannot outlive the traffic that made it."""
    huge = "ola ipo " + ("x" * 5_000_000)
    _beacon(huge, 1, 42, ip="10.3.3.3")

    (key,) = [k for k in store.sets if k.startswith("analytics:query_click:")]
    # The key is a fixed-width digest, so it no longer grows with the beacon at
    # all. The length cap still bounds what gets hashed, so a query past the cap
    # keys identically to its bounded prefix -- the vote is not stranded.
    assert len(key) == len(_qkey("x"))
    assert key == _qkey("ola ipo " + "x" * config.CLICK_QUERY_MAX_LEN)
    assert store.ttls[key] == config.CLICK_QUERY_TTL_SECONDS


def test_dedupe_claim_keys_carry_a_ttl_and_leak_no_ip_or_query(store, index):
    """The dedupe set is bounded by TTL and the claim key is a digest, so the
    client's address and the query text are not recoverable from Redis."""
    _beacon("ola ipo", 1, 42, ip="203.0.113.9")

    (claim,) = [k for k in store.nx_keys]
    assert claim.startswith("analytics:click:seen:")
    assert "203.0.113.9" not in claim
    assert "ola" not in claim
    assert store.ttls[claim] == config.CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS


# --- the beacon is still a public, per-IP rate-limited endpoint ---


def test_beacon_remains_public_and_per_ip_rate_limited(store, index, monkeypatch):
    """No auth header, no cookie, no session: the beacon stays open so anonymous
    traffic is still measured. And the per-IP limit still bites -- one client
    cannot flood it, while a second client is unaffected."""
    monkeypatch.setattr(config, "PUBLIC_CLICK_RATE_PER_MIN", 2)

    first = _as("10.7.7.7")
    second = _as("10.7.7.8")
    for _ in range(2):
        # No Authorization header was sent and none was demanded.
        assert first.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 200

    over = first.post("/analytics/click", json={"query": "q", "position": 1})
    assert over.status_code == 429
    assert second.post("/analytics/click", json={"query": "q", "position": 1}).status_code == 200


def test_a_failed_write_gives_the_click_vote_back(store, index, monkeypatch):
    """If the write that a claim unlocks never lands, the claim is released, so
    a transient Redis failure does not silence that user's click for the rest of
    the dedupe window."""
    flaky = FlakyPipelineRedis()
    monkeypatch.setattr(analytics, "_client", lambda: flaky)

    assert _beacon("ola ipo", 1, 42, ip="10.4.4.4").status_code == 200
    assert not flaky.nx_keys, "the claim must not stay spent when the tally failed"

    # The vote is still castable: the same client clicking again records it.
    healthy = FakeRedis()
    monkeypatch.setattr(analytics, "_client", lambda: healthy)
    assert _beacon("ola ipo", 1, 42, ip="10.4.4.4").status_code == 200
    assert healthy.sets[_qkey("ola ipo")] == {"42": 1.0}


def test_a_lost_claim_is_never_released_for_its_owner(store, index, monkeypatch):
    """A repeat click that LOSES the claim race owns no claim, so a write failure
    on its behalf must not release the claim the winner holds -- otherwise a
    client could free its own vote by making the write fail."""
    flaky = FlakyPipelineRedis(fail_pipeline=False)
    monkeypatch.setattr(analytics, "_client", lambda: flaky)

    assert _beacon("ola ipo", 1, 42, ip="10.6.6.6").status_code == 200
    (claim,) = list(flaky.nx_keys)
    assert flaky.sets[_qkey("ola ipo")] == {"42": 1.0}

    flaky.fail_pipeline = True
    assert _beacon("ola ipo", 1, 42, ip="10.6.6.6").status_code == 200

    assert claim in flaky.nx_keys, "the winner's claim must survive a loser's failed write"
    assert flaky.sets[_qkey("ola ipo")] == {"42": 1.0}, "and still count once"
