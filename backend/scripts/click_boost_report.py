"""Measure how often a genuinely-clicked article would clear the click-boost gate.

The thresholds (``CLICK_BOOST_MIN_CLICKS`` / ``CLICK_BOOST_MIN_ARTICLE_CLICKS`` /
``CLICK_BOOST_MIN_SHARE``) decide when a result is re-ranked on the strength of
what users actually clicked. Raising them is defeated for security purposes by the
per-client vote dedupe at ANY threshold, and makes the boost just as much harder
to trigger for real signal: if real click volume never reaches the bar, ranking
quietly stops learning. That is a tuning decision needing the per-query click
distribution, not argument.

This reads the analytics Redis the boost itself reads and scores every stored
per-query tally against the policy this deployment actually runs AND against the
proposed 20/8/0.5, on the same data, so the two can be compared. Read-only, so
it is safe to point at production.

A tally is deduped votes (one per client per article per
``CLICK_SIGNAL_DEDUPE_WINDOW_SECONDS``) accumulated since the query was last
quiet for ``CLICK_QUERY_TTL_SECONDS``. A query that stops receiving clicks loses
its tally entirely, so this is a rolling window over recently active queries --
not a fixed calendar period -- and counts never decay within it.

The gate is imported from ``app.click_boost``, the code the ranking path runs,
so this report cannot drift from what the product would do. Exit codes: 0 the
measurement ran (including "no click data is stored yet"); 2 it could not be taken.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Mirrors the bootstrap every script here performs, so ``app`` imports from any cwd.
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from _common import redact_url

from app import analytics
from app.click_boost import boosting_ids, share_gate
from app.config import config

# SCAN cursor hint, not a limit: every matching key is read, because a truncated
# sample would report a share of a subset as if it were the whole.
SCAN_COUNT = 500

# Bounded so a very large keyspace does not build one enormous reply in memory.
READ_BATCH = 200

# The thresholds the repository ships. Compared against what this deployment
# runs, so a report cannot describe one policy while the app runs another --
# which is what a ``backend/.env`` pinning CLICK_BOOST_MIN_* does.
PINNED_SHIPPED = (5, 3, 0.3)


@dataclass(frozen=True)
class Policy:
    """One candidate set of click-boost thresholds."""

    label: str
    min_clicks: int
    min_article: int
    min_share: float

    @property
    def key(self) -> tuple[int, int, float]:
        return (self.min_clicks, self.min_article, self.min_share)

    def boosted_ids(self, row: QueryRow) -> set[str]:
        """Ids of the articles this policy would boost for this query, if any."""
        # click_signals returns None below min_clicks, so a tally the ranking
        # path never reads must not be scored as boostable here.
        if row.total < self.min_clicks:
            return set()
        return boosting_ids(row.counts, row.total, self.min_article, self.min_share)

    def clears(self, row: QueryRow) -> bool:
        """Whether this policy would boost at least one article for this query."""
        return bool(self.boosted_ids(row))

    def shortfall(self, row: QueryRow) -> int:
        """Extra votes on the top article that would first clear this policy.

        ``0`` when it already clears. A query's total only grows as more votes
        land, so the share gate's requirement grows with it: this is the smallest
        number of further votes on the top article satisfying EVERY half of the
        gate at the total those votes produce. The liveness half is easy to
        forget -- votes that satisfy the other two while the query stays under
        ``min_clicks`` buy nothing, and quoting that smaller number would send an
        operator to count votes that change nothing.

        ``_SHORTFALL_UNREACHABLE`` when no number can.
        """
        if self.clears(row):
            return 0
        for extra in range(1, _SHORTFALL_LIMIT):
            clicks = row.top_clicks + extra
            if (
                row.total + extra >= self.min_clicks
                and clicks >= self.min_article
                and clicks >= share_gate(row.total + extra, self.min_share)
            ):
                return extra
        return _SHORTFALL_UNREACHABLE


# How far past its current count the report searches before calling the gap
# unbridgeable, so a policy no tally could satisfy cannot spin forever. 100k is
# far past any real click volume.
_SHORTFALL_LIMIT = 100_000
_SHORTFALL_UNREACHABLE = -1


@dataclass(frozen=True)
class QueryRow:
    """One query's stored click tally, as the ranking path would read it."""

    key: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def top_id(self) -> str:
        # Ties broken on the id so the report is stable between runs.
        return min(self.counts, key=lambda aid: (-self.counts[aid], aid))

    @property
    def top_clicks(self) -> int:
        return self.counts[self.top_id]

    @property
    def top_share(self) -> float:
        return self.top_clicks / self.total


@dataclass(frozen=True)
class Scan:
    """What the SCAN found: every matching key, and the tallies that hold votes."""

    keys: int
    rows: tuple[QueryRow, ...]


@dataclass(frozen=True)
class Verdict:
    """What one policy would have done with the measured data."""

    policy: Policy
    live_queries: int
    boosted_queries: int
    boosted_articles: int


@dataclass(frozen=True)
class Report:
    scan: Scan
    verdicts: tuple[Verdict, ...]

    @property
    def rows(self) -> tuple[QueryRow, ...]:
        return self.scan.rows

    @property
    def effective(self) -> Verdict:
        return self.verdicts[0]


def effective_policy() -> Policy:
    """The policy this deployment runs right now, env and .env included."""
    return Policy(
        label="effective",
        min_clicks=config.CLICK_BOOST_MIN_CLICKS,
        min_article=config.CLICK_BOOST_MIN_ARTICLE_CLICKS,
        min_share=config.CLICK_BOOST_MIN_SHARE,
    )


def proposed_policy() -> Policy:
    """The proposed 20/8/0.5, kept so the claim is testable on real data.

    Recorded as the historical alternative, not a live default: it was never
    shipped and nothing in the app reads it.
    """
    return Policy(label="proposed (#391)", min_clicks=20, min_article=8, min_share=0.5)


async def collect(client, prefix: str | None = None) -> Scan:
    """Every stored per-query click tally, and how many keys the scan saw.

    A tally's total is the sum of ALL its members, not a top-N window, because
    the share gate is a share of the true total: an undercounted total would
    overstate every article's share and report a boost that never fires.

    The key count rides alongside the rows so "there is no click data" is only
    said when there is genuinely none -- otherwise a scan pointed at the wrong
    database, or one whose prefix had drifted, would be indistinguishable from a
    deployment that has never taken a click.
    """
    pattern = f"{prefix or analytics.CLICK_SIGNAL_KEY_PREFIX}*"
    keys: list[str] = []
    rows: list[QueryRow] = []
    seen = 0
    async for key in client.scan_iter(match=pattern, count=SCAN_COUNT):
        keys.append(key)
        seen += 1
        if len(keys) >= READ_BATCH:
            rows.extend(await _read_batch(client, keys))
            keys = []
    if keys:
        rows.extend(await _read_batch(client, keys))
    return Scan(keys=seen, rows=tuple(rows))


async def _read_batch(client, keys: Sequence[str]) -> list[QueryRow]:
    pipe = client.pipeline()
    for key in keys:
        pipe.zrange(key, 0, -1, withscores=True)
    replies = await pipe.execute()
    return [row for row in (_row(key, pairs) for key, pairs in zip(keys, replies, strict=True))
            if row is not None]


def _row(key: str, pairs: Iterable | None) -> QueryRow | None:
    counts: dict[str, int] = {}
    for member, score in pairs or ():
        clicks = int(score)
        # A member scored zero is not a vote -- zincrby only ever increments, and
        # click_signals drops it too -- so a tally of nothing but zeros carries no
        # signal and is left out rather than divided by.
        if clicks:
            counts[str(member)] = clicks
    return QueryRow(key=key, counts=counts) if counts else None


def analyse(scan: Scan, policies: Sequence[Policy]) -> Report:
    """Score the same tallies under each policy."""
    verdicts = []
    for policy in policies:
        live = boosted_queries = boosted_articles = 0
        for row in scan.rows:
            if row.total < policy.min_clicks:
                continue
            live += 1
            boosted = policy.boosted_ids(row)
            if boosted:
                boosted_queries += 1
                boosted_articles += len(boosted)
        verdicts.append(Verdict(policy, live, boosted_queries, boosted_articles))
    return Report(scan=scan, verdicts=tuple(verdicts))


def render(report: Report, top: int = 5) -> str:
    """The report as an operator reads it in a terminal."""
    rows = report.rows
    effective = report.effective.policy
    out: list[str] = [
        "click-boost measurement (#391)",
        "=" * 60,
        (
            f"data:   {redact_url(config.REDIS_URL)} db={config.ANALYTICS_REDIS_DB}"
            f"  (key prefix {analytics.CLICK_SIGNAL_KEY_PREFIX})"
        ),
        "window: every tally still in Redis. A query's tally is the deduped votes it has",
        (
            f"        taken since it was last quiet for {config.CLICK_QUERY_TTL_SECONDS // 86400}d;"
            " counts are cumulative and never decay, so this is not a per-day rate."
        ),
        (
            f"policy: effective {effective.min_clicks}/{effective.min_article}/{effective.min_share}"
            f"  (ENABLE_CLICK_BOOST={config.ENABLE_CLICK_BOOST})"
        ),
    ]
    if effective.key != PINNED_SHIPPED:
        shipped = "/".join(str(v) for v in PINNED_SHIPPED)
        out += [
            f"        NOTE: not the shipped {shipped}. An env var or backend/.env is",
            "        overriding the code default, so a retune in config.py alone would",
            "        never reach this deployment.",
        ]

    if not rows:
        out += ["", f"FINDING: the scan matched {report.scan.keys} per-query click keys."]
        if report.scan.keys:
            out += [
                "  None of them holds a single vote, which is not a shape the write path can",
                "  produce. Treat this as an unmeasured deployment, not as a quiet one: check",
                "  that the report is pointed at the analytics Redis this app writes to before",
                "  reading anything into the thresholds.",
            ]
        else:
            out += [
                "  The click boost is INERT -- not because the thresholds are strict, but",
                "  because no client has ever voted on a query here. There is nothing to tune:",
                "  any threshold, however high, would behave the same. If this deployment is",
                "  supposed to be taking clicks, the anonymous /analytics/click beacon is not",
                "  reaching Redis, and that is worth checking before any threshold is discussed.",
                "  Re-run this once real traffic exists.",
            ]
        return "\n".join(out)

    out += ["", f"queries with a stored tally: {len(rows)} of {report.scan.keys} keys scanned"]
    if report.scan.keys != len(rows):
        out += [
            (
                f"  keys holding no votes: {report.scan.keys - len(rows)} (left out). The write"
                " path cannot produce one, so the store is not in a shape this report can read."
            )
        ]

    totals = sorted(r.total for r in rows)
    shares = sorted(r.top_share for r in rows)
    out += [
        "",
        (
            f"  clicks per query:      min {totals[0]}, median {statistics.median(totals):g}, "
            f"p90 {_percentile(totals, 0.9):g}, max {totals[-1]}"
        ),
        (
            f"  top article's share:   min {shares[0]:.2f}, median {statistics.median(shares):.2f}, "
            f"p90 {_percentile(shares, 0.9):.2f}, max {shares[-1]:.2f}"
        ),
        "  distribution of per-query totals:",
        *_histogram(totals, effective),
        "",
        "what each policy would have done with this data:",
        (
            f"  {'policy':<18}{'live queries':>14}{'queries boosted':>18}"
            f"{'articles boosted':>18}{'share':>8}"
        ),
    ]
    for verdict in report.verdicts:
        out.append(
            f"  {verdict.policy.label:<18}{verdict.live_queries:>14}"
            f"{verdict.boosted_queries:>18}{verdict.boosted_articles:>18}"
            f"{_pct(verdict.boosted_queries, len(rows)):>8}"
        )
    out += ["", "FINDING:", *_findings(report)]

    if top:
        out += ["", f"busiest {min(top, len(rows))} tallies (key is the stored query digest):"]
        for row in sorted(rows, key=lambda r: (-r.total, r.key))[:top]:
            out.append(
                f"  {row.key}  total={row.total} top article={row.top_id} "
                f"({row.top_clicks} clicks, {row.top_share:.2f} of the query)"
            )
    return "\n".join(out)


def _findings(report: Report) -> list[str]:
    """The sentences an operator is meant to act on -- or, more often, not to."""
    rows = report.rows
    effective = report.verdicts[0]
    proposed = report.verdicts[1] if len(report.verdicts) > 1 else None
    lines: list[str] = []

    if effective.boosted_queries == 0:
        closest = max(rows, key=lambda r: (r.top_share, r.total))
        policy = effective.policy
        need = policy.shortfall(closest)
        gap = (
            "no number of further votes on it could clear the gate"
            if need == _SHORTFALL_UNREACHABLE
            else f"{need} more votes on it would clear the gate"
        )
        lines += [
            f"  The shipped {policy.min_clicks}/{policy.min_article}/{policy.min_share} boosts",
            f"  NOTHING in this data. The closest call is {closest.key}: its top article holds",
            f"  {closest.top_clicks} of {closest.total} clicks ({closest.top_share:.2f} of the",
            f"  query) and {gap}. The boost is inert here because of the click volume, not",
            "  because of the thresholds: raising the bar cannot harden an already-inert",
            "  feature, it can only stop a live one from ever starting.",
        ]

    if proposed is not None and proposed.boosted_queries < effective.boosted_queries:
        lost = effective.boosted_queries - proposed.boosted_queries
        policy = proposed.policy
        lines += [
            f"  The proposed {policy.min_clicks}/{policy.min_article}/{policy.min_share} clears",
            f"  {proposed.boosted_queries} of the {effective.boosted_queries} queries the shipped",
            f"  thresholds clear ({lost} fewer).",
        ]
        if proposed.boosted_queries == 0:
            lines += [
                "  Zero is not a hardening result. It means the feature would be switched off,",
                "  and the only symptom would be ranking that quietly stops learning.",
            ]
        lines += [
            (
                "  The per-client dedupe is what defeats a forged burst, and it defeats it at any"
                "  threshold."
            )
        ]

    lines += [
        "",
        "  A threshold is a ranking-tuning decision, not a security control. Keep the shipped",
        "  values unless this report says a specific alternative fits real traffic.",
    ]
    return lines


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value at or above the q-th fraction."""
    if not sorted_values:
        return 0.0
    return sorted_values[max(1, math.ceil(q * len(sorted_values))) - 1]


def _pct(part: int, whole: int) -> str:
    return "n/a" if not whole else f"{100 * part / whole:.1f}%"


def _histogram(totals: Sequence[int], policy: Policy) -> list[str]:
    """How many queries sit in each click-count band, and where the liveness bar falls."""
    lines = []
    for low, high in ((1, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)):
        label = f"{low}-{high}" if high < 10**9 else f"{low}+"
        count = sum(1 for t in totals if low <= t <= high)
        marker = "   <- below the live-signal bar, boost cannot act" if high < policy.min_clicks else ""
        lines.append(f"    {label:>8} clicks:{count:>6}{marker}")
    return lines


async def _run() -> Report:
    # The app's own client factory, not a second connection setup: the DB index
    # and URL decide where the tallies live, and a report building its own client
    # would quietly measure an empty database if either moved.
    client = analytics._client()
    try:
        return analyse(await collect(client), (effective_policy(), proposed_policy()))
    finally:
        await analytics.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure how often a clicked article would clear the click-boost gate.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="list the N busiest per-query tallies (0 to omit)",
    )
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must not be negative")

    try:
        report = asyncio.run(_run())
    except Exception as exc:  # any failure means the answer is unknown, not "no clicks"
        print(
            f"could not measure the click-boost gate: {type(exc).__name__}: {exc}\n"
            f"  expected a reachable analytics Redis at {redact_url(config.REDIS_URL)} "
            f"db={config.ANALYTICS_REDIS_DB}. No conclusion is drawn from this run.",
            file=sys.stderr,
        )
        return 2

    print(render(report, top=args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
