"""Diversity tests: MMR `diversify` short-circuit, the greedy MMR loop
(sim_thresh floor, lam weighting, multi-chosen max_sim), and the
`_tokens`/`_jaccard` helpers."""

from types import SimpleNamespace

from app.diversity import _jaccard, _tokens, diversify


def _res(title, score):
    return SimpleNamespace(title=title, score=score)


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