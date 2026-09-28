"""Tests for the click-boost measurement report (scripts/click_boost_report.py).

#391 exists because the click-boost thresholds were raised 4x inside the #242
security fix without anyone measuring how often a genuinely-clicked article
reaches them. These tests pin what the measurement itself is worth: that it
reads the tallies the click path really writes, and that it scores them with
the same gate the ranking path really runs. A report that drifted from either
would produce a confident number about a rule nobody applies.

The two properties that make that worth asserting, in order of how badly they
would mislead:

* the verdict must never say "the boost is inert" when there is data to score.
  A scan pointed at the wrong database, or a key prefix that moved, looks
  exactly like a deployment that has never taken a click.
* the gate must be the production gate. ``Policy.boosted_ids`` is compared
  against the real ``apply_click_boost`` below, not against a restatement of it.

Nothing here measures a real deployment: the numbers in the synthetic
distribution are labelled as synthetic in the test that uses them, and the
report's own answer for a deployment with no data is a separate case.
"""
import asyncio
from types import SimpleNamespace

import pytest

from app import analytics, click_boost
from app.config import config
from scripts import click_boost_report as report


class TallyStore:
    """Analytics Redis stand-in covering only what the report and the click
    write path touch: sorted sets, counters, the dedupe claim, and SCAN."""

    def __init__(self):
        self.sets: dict[str, dict[str, float]] = {}
        self.counters: dict[str, int] = {}
        self.claims: set[str] = set()

    # -- write path (record_click) --
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

    # -- read path (the report) --
    async def zrange(self, key, start, end, withscores=False):
        items = sorted(self.sets.get(key, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        # Redis reads a -1 end index as "through the last member"; a naive
        # slice would quietly return everything-but-the-last and every total
        # built from it would be one vote short.
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
        # The range the caller asked for is kept, not restated: a fake that
        # answered with the whole set whatever it was given would make the
        # report's read window untestable.
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


# The two policies the issue compares, spelled out here so a test can score a
# tally under each without going through config (which this machine's .env can
# move). ``test_the_report_compares_the_shipped_policy_with_242s_proposal``
# pins that the report's own constructors are these two.
SHIPPED = report.Policy("shipped", 5, 3, 0.3)
PROPOSED = report.Policy("proposed", 20, 8, 0.5)


@pytest.mark.parametrize(
    ("total", "min_share", "expected"),
    [
        (1, 0.3, 1),     # 0.3 votes: floored to one click, never to zero
        (5, 0.3, 2),     # exactly 1.5 -- rounds to 2, where truncation gives 1
        (7, 0.3, 2),     # 2.1
        (13, 0.3, 4),    # 3.9 -- rounds up, where flooring the product gives 3
        (33, 0.3, 10),   # 9.9
        (10, 0.5, 5),    # exactly 5.0
        (7, 1.0, 7),     # a share of the whole query is the whole query
    ],
)
def test_the_share_gate_is_the_documented_rounding(total, min_share, expected):
    """Pin the share gate's arithmetic itself, not just that the report and the
    boost agree on it.

    Both call the same function, so an equivalence test between them would
    pass just as happily if the function were changed -- and the gate decides
    which results get re-ranked for every user, so a silent change from
    rounding to truncation is a ranking change nobody asked for. These are the
    values the documented rule produces: the query's clicks times the share,
    rounded to a whole number of votes, never below one.
    """
    assert click_boost.share_gate(total, min_share) == expected


def test_the_report_compares_the_shipped_policy_with_242s_proposal(parse_config):
    """The report is only worth running if its two columns are the two the
    issue asks about: the thresholds the repository ships, and the 20/8/0.5
    #242 proposed before they were split back out. Both are pinned here, the
    shipped one against a clean parse of config.py so no ``backend/.env`` on
    this machine can answer for the repository."""
    shipped = parse_config()
    assert report.effective_policy().key == (
        shipped.CLICK_BOOST_MIN_CLICKS,
        shipped.CLICK_BOOST_MIN_ARTICLE_CLICKS,
        shipped.CLICK_BOOST_MIN_SHARE,
    )
    assert report.proposed_policy().key == (20, 8, 0.5), (
        "#242's proposal is the alternative this measurement exists to test; a "
        "different number here would be measuring a policy nobody proposed"
    )


# --- the report reads what the click path writes ---


def test_the_report_reads_exactly_the_keys_the_click_path_writes(store):
    """A report pointed at the wrong key pattern finds nothing and reports the
    boost as inert, so the scan has to be reading the keys the app really
    writes -- digest namespace and all."""
    for i in range(4):
        _vote(store, "ola ipo", 42, f"10.0.0.{i}")
    for i in range(2):
        _vote(store, "ola ipo", 99, f"10.0.1.{i}")

    scan = _scan(store)

    (row,) = scan.rows
    assert scan.keys == 1
    assert row.key == f"{analytics.CLICK_SIGNAL_KEY_PREFIX}{analytics.query_digest('ola ipo', 'report-test-digest-key')}"
    # One vote per distinct client, exactly as the ranking path reads it back.
    assert row.counts == {"42": 4, "99": 2}
    assert row.total == 6


def test_every_member_is_counted_not_just_a_top_window(store):
    """The share gate is a share of the query's true total, so a total that
    undercounts overstates every article's share -- and would report a boost
    that the ranking path, which paginates the whole set, never applies."""
    for i in range(300):
        _vote(store, "election result", 1000 + i, f"10.1.{i}.1")
    _vote(store, "election result", 42, "10.2.0.1")
    _vote(store, "election result", 42, "10.2.0.2")

    (row,) = _scan(store).rows

    assert row.total == 302, "every clicked article contributes to the query's total"
    assert row.counts["42"] == 2
    assert len(row.counts) == 301
    # A majority the article does not hold: 2 of 302 is far below the gate.
    assert not SHIPPED.boosted_ids(row)


def test_keys_holding_no_votes_are_not_reported_as_no_data(store):
    """An empty tally is not a shape the write path can produce, so it has to
    be surfaced rather than dropped. Silently discarding it would let a drifted
    key prefix or a wrong database read as "this deployment has never taken a
    click" -- the one answer this report must never get wrong."""
    # A member scored zero: only reachable by writing to Redis directly, which
    # is exactly why the report must not treat it as a vote and must not
    # divide by a total of zero when it works out the share.
    store.sets[f"{analytics.CLICK_SIGNAL_KEY_PREFIX}q1:" + "0" * 32] = {"42": 0.0}
    for i in range(3):
        _vote(store, "ola ipo", 42, f"10.3.0.{i}")

    scan = _scan(store)
    text = report.render(report.analyse(scan, (SHIPPED, PROPOSED)))

    assert scan.keys == 2 and len(scan.rows) == 1
    assert "queries with a stored tally: 1 of 2 keys scanned" in text
    assert "keys holding no votes: 1" in text
    assert "not in a shape this report can read" in text
    # Real traffic was found, so the "never taken a click" verdict must not appear.
    assert "There is nothing to tune" not in text


def test_a_scan_that_finds_only_empty_tallies_is_called_unmeasured(store):
    """The other half of the same hazard: keys present, no votes anywhere. That
    is not evidence of a quiet deployment, it is evidence that this report
    cannot read the store it was pointed at, and it has to say so rather than
    conclude the feature is inert."""
    store.sets[f"{analytics.CLICK_SIGNAL_KEY_PREFIX}q1:" + "0" * 32] = {"42": 0.0, "43": 0.0}

    scan = _scan(store)
    text = report.render(report.analyse(scan, (SHIPPED, PROPOSED)))

    assert scan.keys == 1 and scan.rows == ()
    assert "the scan matched 1 per-query click keys" in text
    assert "unmeasured deployment, not as a quiet one" in text
    assert "INERT" not in text


# --- the gate is the production gate ---


@pytest.mark.parametrize(
    ("total", "counts"),
    [
        (6, {"42": 4, "99": 2}),          # 4 of 6 is a clear majority
        (18, {"11": 12, "12": 6}),        # both articles clear the share gate
        (7, {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1}),  # no majority
        (5, {"9": 2, "8": 2, "7": 1}),    # exact MIN_CLICKS, no majority
        (20, {"1": 11, "2": 9}),          # a bare majority at the top
        (13, {"1": 3, "2": 2, "3": 2, "4": 2, "5": 2, "6": 2}),  # gate is 3.9: rounds to 4
        (200, {"1": 61, **{str(i): 1 for i in range(2, 140)}}),  # 61 of 200: a minority
    ],
)
def test_the_report_scores_a_tally_exactly_as_the_boost_would(total, counts, monkeypatch):
    """The report's gate and the ranking path's gate must be the same gate.

    Driven through the real ``apply_click_boost`` rather than a restatement of
    it: if the two ever diverge, this is where it shows, and the report's
    answer to "how often would this have fired" becomes a number about a rule
    the product does not run.
    """
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
    """The liveness half of the gate lives in ``click_signals``, not in
    ``apply_click_boost``: a query under MIN_CLICKS returns no signal at all,
    so there is nothing for the boost to act on and nothing for the report to
    score. Counting such a tally as boostable would be the report inventing a
    boost the ranking path cannot make."""
    for i in range(4):
        _vote(store, "ola ipo", 42, f"10.4.0.{i}")

    (row,) = _scan(store).rows
    assert row.total == 4 < config.CLICK_BOOST_MIN_CLICKS

    # The real read path refuses the signal...
    sig = asyncio.run(analytics.click_signals("ola ipo"))
    assert sig is None
    # ...and the report agrees, even though article 42 holds every vote.
    assert not SHIPPED.boosted_ids(row)
    verdict = report.analyse(_scan(store), (SHIPPED, PROPOSED)).effective
    assert verdict.live_queries == 0 and verdict.boosted_queries == 0


# --- the answer the operator reads ---


def test_no_click_data_at_all_is_reported_as_inert_not_as_a_tuning_result():
    """A deployment that has never taken a click has nothing to tune. Saying
    "the thresholds are too strict" here would send an operator to raise a bar
    that no measurement supports."""
    text = report.render(report.analyse(report.Scan(keys=0, rows=()), (SHIPPED, PROPOSED)))

    assert "INERT" in text
    assert "There is nothing to tune" in text
    assert "any threshold, however high, would behave the same" in text
    # The report must not present a table of zeroes as a measurement.
    assert "queries boosted" not in text


def test_a_busy_deployment_still_boosting_nothing_names_the_gap_in_votes():
    """When there IS traffic and the shipped gate clears nothing, the useful
    answer is how far the closest genuine article is from clearing it -- the
    number a tuning decision needs, and the one that distinguishes 'the bar is
    too high' from 'the bar is unreachable'."""
    row = report.QueryRow("analytics:query_click:q1:abc", {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1})
    text = report.render(report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)))

    # 1 of 5 leaves 20% of the query; the gate needs round(total * 0.3).
    assert "boosts\n  NOTHING in this data" in text
    assert "2 more votes on it would clear the gate" in text
    assert "inert here because of the click volume" in text


def test_the_two_policies_are_scored_independently_on_one_distribution():
    """The point of the report: the shipped policy and #242's proposal are
    scored on the SAME tallies, so the difference between them is a
    measurement rather than an argument.

    The distribution below is SYNTHETIC and exists only to exercise the
    instrument. It is not a measurement of any deployment, and nothing here
    should be read as evidence about real click volume.
    """
    tallies = (
        {"42": 4, "99": 2, "7": 1},                                    # 7 total, 57%
        {"11": 12, "12": 6},                                           # 18 total, 67%
        {"5": 3, "6": 2, "7": 1},                                      # 6 total, 50%
        {"3": 2},                                                      # 2 total: never live
        {"1": 1, **{str(i): 1 for i in range(2, 62)}},                 # 61 total, 1.6%
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
    # 20/8/0.5 needs 8 clicks AND half the query: only the 18-click tally is
    # busy enough, and at 12/18 it clears -- nothing else can.
    assert proposed.live_queries == 1
    assert proposed.boosted_queries == 0

    assert "The proposed 20/8/0.5 clears" in text
    assert "Zero is not a hardening result" in text
    assert "The per-client dedupe is what defeats a forged burst" in text


def test_a_proposal_that_loses_nothing_is_not_reported_as_a_regression():
    """The finding text is conditional on the proposal actually costing
    something, so a proposal that matches the shipped policy is described as
    equal rather than as a loss."""
    row = report.QueryRow("k", {"42": 30, "43": 10})
    text = report.render(report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)))

    assert report.analyse(report.Scan(keys=1, rows=(row,)), (SHIPPED, PROPOSED)).verdicts[1].boosted_queries == 1
    assert "The proposed" not in text


# --- what the report believes the shipped policy is ---


def test_the_defaults_this_report_calls_shipped_are_the_defaults_the_code_ships(parse_config):
    """``backend/.env`` wins over the code default, so the report's idea of
    "shipped" has to come from the code rather than from whatever this machine
    happens to have configured -- otherwise the override warning below would go
    quiet exactly when an operator needed it. Parsed with ``load_dotenv``
    neutralised, so no env file can answer for the repository."""
    shipped = parse_config()
    assert (
        shipped.CLICK_BOOST_MIN_CLICKS,
        shipped.CLICK_BOOST_MIN_ARTICLE_CLICKS,
        shipped.CLICK_BOOST_MIN_SHARE,
    ) == report.PINNED_SHIPPED


def test_the_report_names_the_policy_in_force_and_flags_an_override(monkeypatch):
    """The report must describe what is RUNNING, and say so loudly when that is
    not the code default. A retune in config.py does not reach a deployment
    whose .env pins these knobs, and an operator reading a report that silently
    described the code default would be reading a policy nobody applies."""
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
    """An unreachable Redis is an unknown answer, not a deployment with no
    clicks. Reporting it as the latter would be the most damaging thing this
    tool can do, so it exits non-zero and says what it could not do."""
    async def boom():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(report, "_run", boom)

    assert report.main([]) == 2
    err = capsys.readouterr().err
    assert "could not measure the click-boost gate" in err
    assert "No conclusion is drawn from this run" in err
