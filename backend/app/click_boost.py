"""Click-driven result boosting (self-learning ranking signal).

Uses per-query per-article click aggregates recorded by app/analytics to nudge
results users demonstrably open. Gated so it never acts on sparse traffic: a
query must accumulate >= CLICK_BOOST_MIN_CLICKS total clicks, and an article
must hold >= CLICK_BOOST_MIN_ARTICLE_CLICKS clicks that are >= CLICK_BOOST_MIN_SHARE
of the query's total before its score is multiplied. At today's near-zero click
volume this is inert by design and becomes active only as real traffic arrives.
"""
from collections.abc import Mapping

from app.analytics import click_signals
from app.config import config


def share_gate(total: int, min_share: float) -> int:
    """The click count an article must reach to count as the query's favourite.

    Defined once, here, because ``scripts/click_boost_report.py`` scores the stored per-query
    click tallies against this same rule to answer "how often would a real clicked article
    actually clear the gate". A report that computed the threshold its own way would be
    measuring a rule the ranking path does not run, and its number would be worthless.

    ``round`` is Python's banker's rounding, unchanged.
    """
    return max(1, round(total * min_share))


def boosting_ids(
    by_id: Mapping[object, int],
    total: int,
    min_article: int,
    min_share: float,
) -> set:
    """Ids in a per-query click tally that the boost would act on.

    The same two-part test ``apply_click_boost`` applies per result: enough clicks of the
    article's own, and enough of the query's total to be believed. The share half is the one a
    single client cannot satisfy alone, which is why the two are kept as two terms.

    The key type is left open because only the counts are read: the app keys this by article id
    and the ops report that scores the same tallies keys them by the stored member. Ids absent
    from the tally are not returned -- with a positive ``min_article`` they can never clear the
    count half anyway.
    """
    gate = share_gate(total, min_share)
    return {aid for aid, clicks in by_id.items() if clicks >= min_article and clicks >= gate}


async def apply_click_boost(query: str, results: list) -> list:
    """Return ``results`` with scores boosted for confidently-clicked articles,
    re-sorted descending. Inputs are mutated (score) and re-sorted in place."""
    if not config.ENABLE_CLICK_BOOST or not results:
        return results
    sig = await click_signals(query)
    if not sig:
        return results
    total = sig["total"]
    by_id = sig["by_id"]
    gate = share_gate(total, config.CLICK_BOOST_MIN_SHARE)
    changed = False
    for r in results:
        c = by_id.get(getattr(r, "id", None), 0)
        if c >= config.CLICK_BOOST_MIN_ARTICLE_CLICKS and c >= gate:
            r.score *= config.CLICK_BOOST_MULT
            changed = True
    if changed:
        results.sort(key=lambda a: a.score, reverse=True)
    return results
