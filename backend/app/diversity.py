
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
