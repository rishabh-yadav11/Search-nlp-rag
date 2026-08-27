"""Maximal Marginal Relevance (MMR) diversity for search results.

News coverage repeats itself: one deal is covered by several near-identical
articles. Greedy MMR selects a diverse top-n while keeping relevance dominant:

    score(candidate) = LAMBDA * relevance - (1 - LAMBDA) * max sim(candidate, chosen)

where sim is Jaccard similarity of title word-tokens. Lambda near 1 favours
pure relevance; lower values trade a little relevance for headline diversity.
"""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(title: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall((title or "").lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def diversify(results, n: int, lam: float = 0.7, sim_thresh: float = 0.4):
    """Return a greedily MMR-diverse reordering of ``results`` (length ``n``).

    ``results`` are objects with a ``title`` and a numeric ``score`` (assumed
    roughly comparable, 0..1). When there are <= n results the leading ``n``
    are returned in original order. ``sim_thresh`` is a floor on pairwise
    similarity: pairs below it contribute 0 to the diversity penalty.
    """
    if len(results) <= n:
        return list(results[:n])
    tok = [_tokens(r.title) for r in results]

    def max_sim(i: int, chosen_idx: list[int]) -> float:
        best = 0.0
        for j in chosen_idx:
            s = _jaccard(tok[i], tok[j])
            if s > sim_thresh and s > best:
                best = s
        return best

    order = list(range(len(results)))
    chosen_idx: list[int] = []
    while len(chosen_idx) < n and order:
        best_k = -1
        best_val = float("-inf")
        for k in order:
            sim = max_sim(k, chosen_idx)
            score = results[k].score
            if score is None:
                score = 0.0
            mmr = lam * score - (1 - lam) * sim
            if mmr > best_val:
                best_val = mmr
                best_k = k
        chosen_idx.append(best_k)
        order.remove(best_k)
    return [results[i] for i in chosen_idx]
