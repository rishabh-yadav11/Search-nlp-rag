"""Click-driven result boosting (self-learning ranking signal).

Uses per-query per-article click aggregates recorded by app/analytics to nudge
results users demonstrably open. Gated so it never acts on sparse traffic: a
query must accumulate >= CLICK_BOOST_MIN_CLICKS total clicks, and an article
must hold >= CLICK_BOOST_MIN_ARTICLE_CLICKS clicks that are >= CLICK_BOOST_MIN_SHARE
of the query's total before its score is multiplied. At today's near-zero click
volume this is inert by design and becomes active only as real traffic arrives.
"""
from app.analytics import click_signals
from app.config import config


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
    min_share = max(1, int(total * config.CLICK_BOOST_MIN_SHARE))
    changed = False
    for r in results:
        c = by_id.get(getattr(r, "id", None), 0)
        if c >= config.CLICK_BOOST_MIN_ARTICLE_CLICKS and c >= min_share:
            r.score *= config.CLICK_BOOST_MULT
            changed = True
    if changed:
        results.sort(key=lambda a: a.score, reverse=True)
    return results
