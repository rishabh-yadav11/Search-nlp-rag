
from collections.abc import Mapping

from app.analytics import click_signals
from app.config import config


def share_gate(total: int, min_share: float) -> int:

    return max(1, round(total * min_share))


def boosting_ids(
    by_id: Mapping[object, int],
    total: int,
    min_article: int,
    min_share: float,
) -> set:

    gate = share_gate(total, min_share)
    return {aid for aid, clicks in by_id.items() if clicks >= min_article and clicks >= gate}


async def apply_click_boost(query: str, results: list) -> list:

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
