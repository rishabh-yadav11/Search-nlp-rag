"""Maximal Marginal Relevance (MMR) diversity for search results.

News coverage repeats itself: one deal is covered by several near-identical
articles. Greedy MMR selects a diverse top-n while keeping relevance dominant:

    score(candidate) = LAMBDA * relevance - (1 - LAMBDA) * max sim(candidate, chosen)

where sim is Jaccard similarity of title word-tokens. Lambda near 1 favours
pure relevance; lower values trade a little relevance for headline diversity.
"""
import logging
import math
import re
from typing import Protocol

from app.lexical import jaccard

logger = logging.getLogger("diversity")

_WORD_RE = re.compile(r"[a-z0-9]+")


class _Result(Protocol):
    title: str
    score: float | None


def _tokens(title: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall((title or "").lower()))


def diversify(
    results: list[_Result], n: int, lam: float = 0.7, sim_thresh: float = 0.4
) -> list[_Result]:
    """Return a greedily MMR-diverse reordering of ``results`` (length ``n``).

    ``results`` are objects with a ``title`` and a numeric ``score`` (assumed
    roughly comparable, 0..1). When there are <= n results the leading ``n``
    are returned in original order. ``sim_thresh`` is a floor on pairwise
    similarity: pairs below it contribute 0 to the diversity penalty.

    A NaN ``score`` is treated as ``-inf`` (dropped): it never outranks a real
    score, and selection continues to the remaining valid candidates instead of
    silently truncating the list. If every remaining candidate is NaN or ``-inf``
    the selection stops early, warns and returns a short list rather than
    raising: /search is a read path, so one bad score must not fail the request.
    """
    if len(results) <= n:
        return list(results[:n])
    tok = [_tokens(r.title) for r in results]

    def max_sim(i: int, chosen_idx: list[int]) -> float:
        best = 0.0
        for j in chosen_idx:
            s = jaccard(tok[i], tok[j])
            if s >= sim_thresh and s > best:
                best = s
        return best
    # ``order`` is never mutated -- each round scans it once and skips indices
    # already in ``chosen_set`` -- which is what keeps the per-round O(m)
    # ``list.remove`` out. The dominant cost remains the O(len(chosen_idx))
    # ``max_sim`` on every candidate visit.
    # Termination: every round appends to ``chosen_idx`` (capped at n) or breaks,
    # so the loop cannot spin, and no "chosen_set smaller than order" guard is
    # needed: the early return above guarantees n < len(order).
    order = list(range(len(results)))
    chosen_idx: list[int] = []
    chosen_set: set[int] = set()
    while len(chosen_idx) < n:
        best_k = -1
        best_val = float("-inf")
        for k in order:
            if k in chosen_set:
                continue
            sim = max_sim(k, chosen_idx)
            score = results[k].score
            if score is None:
                score = 0.0
            elif math.isnan(score):
                score = float("-inf")
            mmr = lam * score - (1 - lam) * sim
            if mmr > best_val:
                best_val = mmr
                best_k = k
        if best_k < 0:
            # Nothing beat -inf: every remaining score is NaN or -inf. Stop
            # instead of re-picking the sentinel, and warn -- a short list reaches
            # /search as missing results, so the cause must be visible in the logs.
            logger.warning(
                "diversify: no candidate beat -inf after %d of %d picks; "
                "remaining scores are NaN, returning %d results",
                len(chosen_idx),
                n,
                len(chosen_idx),
            )
            break
        chosen_idx.append(best_k)
        chosen_set.add(best_k)
    return [results[i] for i in chosen_idx]
