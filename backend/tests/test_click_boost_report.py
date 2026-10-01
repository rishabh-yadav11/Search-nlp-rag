"""Tests for the click-boost measurement report (scripts/click_boost_report.py)."""
import asyncio
from types import SimpleNamespace

import pytest

from app import analytics, click_boost
from app.config import config
from scripts import click_boost_report as report


class TallyStore:
    """Stand-in covering only the Redis surface the report and the click path touch."""

    def __init__(self):
        self.sets: dict[str, dict[str, float]] = {}
        self.counters: dict[str, int] = {}
        self.claims: set[str] = set()

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
        return self

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.claims:
            return None
        self.claims.add(key)
        return True

    async def delete(self, key):
        self.claims.discard(key)
        return 1

    async def zrange(self, key, start, end, withscores=False):
        items = sorted(self.sets.get(key, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        # Redis reads a -1 end index as "through the last member"; a naive slice drops the last vote.
        if end == -1:
            end = len(items) - 1
        items = items[start : end + 1]
        return [(m, float(s)) for m, s in items] if withscores else [m for m, _ in items]

    async def scan_iter(self, match="*", count=None):
        prefix = match.rstrip("*")
        for key in sorted(self.sets):
            if key.startswith(prefix):
                yield key


class _Pipeline:
    def __init__(self, store):
        self.store = store
        self.reads: list[tuple[str, int, int]] = []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def zrange(self, key, start, end, withscores=False):
        # The fake records the caller's range instead of answering: one ignoring it would make the window untestable.
        self.reads.append((key, start, end))
        return self

    async def execute(self):
        replies = [
            await self.store.zrange(key, start, end, withscores=True)
            for key, start, end in self.reads
        ]
        self.reads = []
        return replies


@pytest.fixture
def store(monkeypatch):
    fresh = TallyStore()
    monkeypatch.setattr(analytics, "_client", lambda: fresh)
    monkeypatch.setattr(analytics, "_QUERY_DIGEST_KEY", "report-test-digest-key")
    yield fresh
    analytics._QUERY_DIGEST_KEY = None


def _vote(store, query, article_id, client_ip):
    return asyncio.run(analytics.record_click(query, 1, article_id=article_id, client_ip=client_ip))


def _scan(store):
    return asyncio.run(report.collect(store))


# Spelled out so no test reads its expectation from a machine's .env.
SHIPPED = report.Policy("shipped", 5, 3, 0.3)
PROPOSED = report.Policy("proposed", 20, 8, 0.5)


@pytest.mark.parametrize(
    ("total", "min_share", "expected"),
    [
        (1, 0.3, 1),
        (5, 0.3, 2),
        (7, 0.3, 2),
        (13, 0.3, 4),
        (33, 0.3, 10),
        (10, 0.5, 5),
        (7, 1.0, 7),
    ],
)
def test_the_share_gate_is_the_documented_rounding(total, min_share, expected):
    assert click_boost.share_gate(total, min_share) == expected


def test_the_report_compares_the_shipped_policy_with_242s_proposal():
    assert report.proposed_policy().key == (20, 8, 0.5)


@pytest.mark.parametrize("key", [(5, 3, 0.3), (999, 999, 0.99), (20, 8, 0.5)])
def test_the_policy_the_report_measures_is_the_one_in_force(key, monkeypatch):
    clicks, article, share = key
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_CLICKS", clicks)
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_ARTICLE_CLICKS", article)
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_SHARE", share)

    assert report.effective_policy().key == key


@pytest.fixture
def shipped_policy_in_force(monkeypatch):
    for knob, value in (
        ("CLICK_BOOST_MIN_CLICKS", 5),
        ("CLICK_BOOST_MIN_ARTICLE_CLICKS", 3),
        ("CLICK_BOOST_MIN_SHARE", 0.3),
    ):
        monkeypatch.setattr(config, knob, value)


def test_the_report_reads_exactly_the_keys_the_click_path_writes(store):
    """A drifted key pattern finds nothing, which the report would read as an inert boost."""
    for i in range(4):
        _vote(store, "ola ipo", 42, f"10.0.0.{i}")
    for i in range(2):
        _vote(store, "ola ipo", 99, f"10.0.1.{i}")

    scan = _scan(store)

    (row,) = scan.rows
    assert scan.keys == 1
    assert row.key == f"{analytics.CLICK_SIGNAL_KEY_PREFIX}{analytics.query_digest('ola ipo', 'report-test-digest-key')}"
    assert row.counts == {"42": 4, "99": 2}
    assert row.total == 6


def test_every_member_is_counted_not_just_a_top_window(store):
    for i in range(300):
        _vote(store, "election result", 1000 + i, f"10.1.{i}.1")
    _vote(store, "election result", 42, "10.2.0.1")
    _vote(store, "election result", 42, "10.2.0.2")

    (row,) = _scan(store).rows

    assert row.total == 302, "every clicked article contributes to the query's total"
    assert row.counts["42"] == 2
    assert len(row.counts) == 301
    assert not SHIPPED.boosted_ids(row)


def test_keys_holding_no_votes_are_not_reported_as_no_data(store):
    store.sets[f"{analytics.CLICK_SIGNAL_KEY_PREFIX}q1:" + "0" * 32] = {"42": 0.0}
    for i in range(3):
        _vote(store, "ola ipo", 42, f"10.3.0.{i}")

    scan = _scan(store)
    text = report.render(report.analyse(scan, (SHIPPED, PROPOSED)))

    assert scan.keys == 2 and len(scan.rows) == 1
    assert "queries with a stored tally: 1 of 2 keys scanned" in text
    assert "keys holding no votes: 1" in text
    assert "not in a shape this report can read" in text
    assert "There is nothing to tune" not in text


def test_a_scan_that_finds_only_empty_tallies_is_called_unmeasured(store):
    store.sets[f"{analytics.CLICK_SIGNAL_KEY_PREFIX}q1:" + "0" * 32] = {"42": 0.0, "43": 0.0}

    scan = _scan(store)
    text = report.render(report.analyse(scan, (SHIPPED, PROPOSED)))

    assert scan.keys == 1 and scan.rows == ()
    assert "the scan matched 1 per-query click keys" in text
    assert "unmeasured deployment, not as a quiet one" in text
    assert "INERT" not in text


@pytest.mark.parametrize(
    ("total", "counts"),
    [
        (6, {"42": 4, "99": 2}),
        (18, {"11": 12, "12": 6}),
        (7, {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1}),
        (5, {"9": 2, "8": 2, "7": 1}),
        (20, {"1": 11, "2": 9}),
        (13, {"1": 3, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2}),
        (200, {"1": 61, **{str(i): 1 for i in range(2, 140)}}),
    ],
)
def test_the_report_scores_a_tally_exactly_as_the_boost_would(total, counts, monkeypatch):
    sig = {"total": total, "by_id": {int(aid): n for aid, n in counts.items()}}
    ids = sorted(int(aid) for aid in counts)
    results = [SimpleNamespace(id=i, score=1.0) for i in ids]

    async def fake_signals(_query):
        return sig

    monkeypatch.setattr(click_boost, "click_signals", fake_signals)
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_ARTICLE_CLICKS", SHIPPED.min_article)
    monkeypatch.setattr(config, "CLICK_BOOST_MIN_SHARE", SHIPPED.min_share)
    monkeypatch.setattr(config, "CLICK_BOOST_MULT", 1.3)

    out = asyncio.run(click_boost.apply_click_boost("q", results))
    multiplied = {r.id for r in out if r.score != 1.0}
    predicted = SHIPPED.boosted_ids(report.QueryRow("k", counts))

    assert multiplied == {int(aid) for aid in predicted}, (
        f"the report says {sorted(predicted)} would be boosted for {total} clicks "
        f"({counts}), the ranking path boosted {sorted(multiplied)}"
    )


def test_a_tally_below_min_clicks_is_scored_inert_by_both_paths(store, monkeypatch):
    for i in range(4):
        _vote(store, "ola ipo", 42, f"10.4.0.{i}")

    (row,) = _scan(store).rows
    assert row.total == 4 < config.CLICK_BOOST_MIN_CLICKS

    sig = asyncio.run(analytics.click_signals("ola ipo"))
    assert sig is None
    assert not SHIPPED.boosted_ids(row)
    verdict = report.analyse(_scan(store), (SHIPPED, PROPOSED)).effective
    assert verdict.live_queries == 0 and verdict.boosted_queries == 0


def test_no_click_data_at_all_is_reported_as_inert_not_as_a_tuning_result():
    text = report.render(report.analyse(report.Scan(keys=0, rows=()), (SHIPPED, PROPOSED)))

    assert "INERT" in text
    assert "There is nothing to tune" in text
    assert "any threshold, however high, would behave the same" in text
    assert "queries boosted" not in text


def test_a_busy_deployment_still_boosting_nothing_names_the_gap_in_votes():
    row = report.QueryRow("analytics:query_click:q1:abc", {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1})
    text = report.render(report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)))

    assert "boosts\n  NOTHING in this data" in text
    assert "2 more votes on it would clear the gate" in text
    assert "inert here because of the click volume" in text


@pytest.mark.parametrize(
    "counts",
    [
        {"3": 2},
        {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1},
        {"1": 2, "2": 1, "3": 1},
        {"1": 3, "2": 2},
        {"1": 4, "2": 2, "3": 1},
        {"1": 1, **{str(i): 1 for i in range(2, 62)}},
        {"1": 12, "2": 6},
    ],
)
@pytest.mark.parametrize("policy", [SHIPPED, PROPOSED])
def test_the_quoted_shortfall_is_exactly_what_clears_the_gate(counts, policy):
    row = report.QueryRow("k", counts)
    need = policy.shortfall(row)

    if need == report._SHORTFALL_UNREACHABLE:
        assert not any(
            policy.clears(report.QueryRow("k", {**counts, row.top_id: row.top_clicks + n}))
            for n in (1, 2, 5, 20, 100)
        )
        return

    def grown(n):
        return report.QueryRow("k", {**counts, row.top_id: row.top_clicks + n})

    if need == 0:
        assert policy.clears(row)
        return

    assert policy.clears(grown(need)), f"{need} extra votes should clear, and do not"
    assert not policy.clears(grown(need - 1)), f"{need - 1} extra votes should not clear, but do"


def test_a_query_under_the_liveness_bar_is_not_told_one_vote_would_do_it():
    row = report.QueryRow("k", {"3": 2})

    assert not SHIPPED.clears(row)
    assert SHIPPED.shortfall(row) == 3, "the query needs to reach MIN_CLICKS first"
    assert not SHIPPED.clears(report.QueryRow("k", {"3": 3})), "3 total is still under the bar"
    assert SHIPPED.clears(report.QueryRow("k", {"3": 5}))


def test_the_report_never_prints_a_credential(monkeypatch, capsys):
    secret_url = "redis://admin:hunter2@cache.internal:6380/1?token=s3cr3t"
    monkeypatch.setattr(config, "REDIS_URL", secret_url)
    row = report.QueryRow("k", {"42": 4, "99": 2, "7": 1})

    header = report.render(report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)))
    assert "cache.internal:6380" in header
    for secret in ("hunter2", "s3cr3t", "admin", secret_url):
        assert secret not in header, f"the report header leaked {secret!r}"

    async def boom():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(report, "_run", boom)
    assert report.main([]) == 2
    err = capsys.readouterr().err
    assert "cache.internal:6380" in err
    for secret in ("hunter2", "s3cr3t", "admin", secret_url):
        assert secret not in err, f"the failure message leaked {secret!r}"


def test_the_two_policies_are_scored_independently_on_one_distribution():
    """Both policies are scored on the SAME tallies, so the gap between them is measured, not argued.

    The tallies are SYNTHETIC: this exercises the instrument and describes no real deployment."""
    tallies = (
        {"42": 4, "99": 2, "7": 1},
        {"11": 12, "12": 6},
        {"5": 3, "6": 2, "7": 1},
        {"3": 2},
        {"1": 1, **{str(i): 1 for i in range(2, 62)}},
    )
    scan = report.Scan(
        keys=len(tallies),
        rows=tuple(report.QueryRow(f"k{i}", counts) for i, counts in enumerate(tallies)),
    )
    result = report.analyse(scan, (SHIPPED, PROPOSED))
    text = report.render(result)

    shipped, proposed = result.verdicts
    assert shipped.boosted_queries == 3, "three of the five tallies have a real majority"
    assert shipped.live_queries == 4, "the 2-click tally is never a live signal"
    assert proposed.live_queries == 1
    assert proposed.boosted_queries == 0

    assert "The proposed 20/8/0.5 clears" in text
    assert "Zero is not a hardening result" in text
    assert "The per-client dedupe is what defeats a forged burst" in text


def test_a_proposal_that_loses_nothing_is_not_reported_as_a_regression():
    row = report.QueryRow("k", {"42": 30, "43": 10})
    text = report.render(report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)))

    assert report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)).verdicts[1].boosted_queries == 1
    assert "The proposed" not in text


def test_the_defaults_this_report_calls_shipped_are_the_defaults_the_code_ships(parse_config):
    """``backend/.env`` outranks the code default, so "shipped" must be pinned in the code."""
    shipped = parse_config()
    assert (
        shipped.CLICK_BOOST_MIN_CLICKS,
        shipped.CLICK_BOOST_MIN_ARTICLE_CLICKS,
        shipped.CLICK_BOOST_MIN_SHARE,
    ) == report.PINNED_SHIPPED


def test_the_report_names_the_policy_in_force_and_flags_an_override(
    monkeypatch, shipped_policy_in_force
):
    row = report.QueryRow("k", {"42": 4, "99": 2, "7": 1})
    scan = report.Scan(keys=1, rows=(row,))

    baseline = report.render(report.analyse(scan, (report.effective_policy(), PROPOSED)))
    assert "policy: effective 5/3/0.3" in baseline
    assert "NOTE: not the shipped" not in baseline

    for knob, value in (
        ("CLICK_BOOST_MIN_CLICKS", 20),
        ("CLICK_BOOST_MIN_ARTICLE_CLICKS", 8),
        ("CLICK_BOOST_MIN_SHARE", 0.5),
    ):
        monkeypatch.setattr(config, knob, value)
    overridden = report.render(report.analyse(scan, (report.effective_policy(), PROPOSED)))

    assert "policy: effective 20/8/0.5" in overridden
    assert "NOTE: not the shipped 5/3/0.3" in overridden
    assert "would\n        never reach this deployment" in overridden


def test_main_reports_a_failed_measurement_instead_of_printing_a_verdict(monkeypatch, capsys):
    async def boom():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(report, "_run", boom)

    assert report.main([]) == 2
    err = capsys.readouterr().err
    assert "could not measure the click-boost gate" in err
    assert "No conclusion is drawn from this run" in err
