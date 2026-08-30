"""Diversity tests: MMR `diversify` short-circuit, the greedy MMR loop
(sim_thresh floor, lam weighting, multi-chosen max_sim), and the
`_tokens`/`_jaccard` helpers."""

import itertools
import random
from types import SimpleNamespace

from app.diversity import _jaccard, _tokens, diversify


def _res(title, score):
    return SimpleNamespace(title=title, score=score)


def _legacy_diversify(results, n, lam=0.7, sim_thresh=0.4):
    """Reference implementation of the pre-#193 loop.

    It popped the winning index out of ``order`` with ``list.remove`` (O(n) per
    round). Selection must stay byte-for-byte identical to this.
    """
    if len(results) <= n:
        return list(results[:n])
    tok = [_tokens(r.title) for r in results]

    def max_sim(i, chosen_idx):
        best = 0.0
        for j in chosen_idx:
            s = _jaccard(tok[i], tok[j])
            if s >= sim_thresh and s > best:
                best = s
        return best

    order = list(range(len(results)))
    chosen_idx = []
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


def _ids(selected, results):
    """Index of each selected result in ``results``, matched by identity.

    ``list.index`` would match an equal-but-distinct result first and mask a
    real ordering difference when two results share a title and a score.
    """
    by_id = {id(r): i for i, r in enumerate(results)}
    return [by_id[id(r)] for r in selected]


# --- _tokens ---


def test_tokens_splits_lowercases_and_strips_punctuation():
    assert _tokens("Acme Buys Widget Corp!") == frozenset(
        {"acme", "buys", "widget", "corp"}
    )


def test_tokens_empty_or_none_title_returns_empty():
    assert _tokens("") == frozenset()
    assert _tokens(None) == frozenset()


# --- _jaccard ---


def test_jaccard_empty_set_returns_zero():
    assert _jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert _jaccard(frozenset({"a"}), frozenset()) == 0.0
    assert _jaccard(frozenset(), frozenset()) == 0.0


def test_jaccard_overlap_ratio():
    assert _jaccard(frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})) == 2 / 4


# --- diversify short-circuit (line 34) ---


def test_diversify_short_circuit_when_len_at_or_below_n():
    results = [_res("one", 0.9), _res("two", 0.8)]
    assert diversify(results, 2) == results
    assert diversify(results, 3) == results


def test_diversify_short_circuit_single_result_kept():
    results = [_res("a", 0.9)]
    assert diversify(results, 5) == results


# --- diversify MMR loop (lines 38-59) ---


def test_diversify_mmr_greedy_picks_diverse_over_similar_second():
    results = [
        _res("acme buys widget corp", 0.9),
        _res("acme buys widget corp again", 0.8),
        _res("completely different news story", 0.7),
    ]
    # Pair jaccard 0.8 exceeds sim_thresh -> the similar-but-high-scoring r1 is
    # penalised and the diverse r2 is chosen second.
    assert diversify(results, 2) == [results[0], results[2]]


def test_diversify_sim_thresh_floor_disables_penalty():
    results = [
        _res("acme buys widget corp", 0.9),
        _res("acme buys widget corp again", 0.8),
        _res("completely different news story", 0.7),
    ]
    # Pair similarity (0.8) is below the 0.9 floor -> contributes 0 to the
    # penalty, so relevance ordering wins: r1 over r2.
    assert diversify(results, 2, sim_thresh=0.9) == [results[0], results[1]]


def test_diversify_lam_one_is_pure_relevance():
    results = [
        _res("acme buys widget corp", 0.9),
        _res("acme buys widget corp again", 0.8),
        _res("completely different news story", 0.7),
    ]
    assert diversify(results, 2, lam=1.0) == [results[0], results[1]]


def test_diversify_lam_zero_weights_only_diversity():
    results = [
        _res("acme buys widget corp", 0.9),
        _res("acme buys widget corp again", 0.8),
        _res("completely different news story", 0.7),
    ]
    # sim all zero on the first pass -> strict > keeps the first result; then
    # only the sim term (score weight 0) separates the rest.
    assert diversify(results, 2, lam=0.0) == [results[0], results[2]]


def test_diversify_mmr_loop_max_sim_over_multiple_chosen():
    results = [
        _res("a b c", 0.9),
        _res("a b c d", 0.8),
        _res("x y z", 0.7),
        _res("a b c e", 0.6),
    ]
    # Third pick evaluates sim against both already-chosen indices; the best of
    # the remaining similar articles (r1) is recovered after the diverse r2.
    assert diversify(results, 3) == [results[0], results[2], results[1]]


# --- #193: O(n) selection must keep the O(n^2) selection order identical ---


def test_diversify_matches_legacy_order_on_random_inputs():
    rng = random.Random(193)
    vocab = ["acme", "buys", "widget", "corp", "fund", "raise", "ipo", "seed"]
    for trial in range(25):
        results = [
            _res(
                " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 4))),
                None if rng.random() < 0.1 else round(rng.uniform(0.0, 1.0), 3),
            )
            for _ in range(rng.randint(2, 30))
        ]
        n = rng.randint(1, len(results))
        kwargs = {
            "lam": rng.choice([0.0, 0.3, 0.7, 1.0]),
            "sim_thresh": rng.choice([0.0, 0.2, 0.4, 0.6, 0.9, 1.0]),
        }
        expected = _ids(_legacy_diversify(results, n, **kwargs), results)
        assert _ids(diversify(results, n, **kwargs), results) == expected, (
            trial,
            n,
            kwargs,
        )


def test_diversify_matches_legacy_order_on_every_permuted_case():
    # Exhaustive over small inputs: distinct scores exercise the tie-breaks.
    vocab = ["a", "b", "c"]
    titles = [" ".join(p) for p in itertools.product(vocab, repeat=2)]
    for size in (3, 4):
        for combo in itertools.combinations(titles, size):
            results = [_res(t, 0.9 - 0.1 * i) for i, t in enumerate(combo)]
            for n in range(1, size + 1):
                for lam in (0.0, 0.5, 0.7, 1.0):
                    for sim_thresh in (0.0, 0.4, 1.0):
                        assert _ids(
                            diversify(results, n, lam=lam, sim_thresh=sim_thresh),
                            results,
                        ) == _ids(
                            _legacy_diversify(
                                results, n, lam=lam, sim_thresh=sim_thresh
                            ),
                            results,
                        )


def test_diversify_does_not_mutate_caller_lists():
    results = [
        _res("acme buys widget corp", 0.9),
        _res("acme buys widget corp again", 0.8),
        _res("completely different news story", 0.7),
    ]
    before = list(results)
    diversify(results, 2)
    assert results == before
    assert [id(r) for r in results] == [id(r) for r in before]


def test_diversify_stops_after_all_candidates_when_n_exceeds_len():
    results = [_res("a b", 0.9), _res("a b c", 0.8)]
    assert _ids(diversify(results, 5), results) == [0, 1]
    assert _ids(_legacy_diversify(results, 5), results) == [0, 1]
