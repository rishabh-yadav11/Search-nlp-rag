"""Property-based tests for app.lexical.jaccard and app.diversity.diversify.

Invariants (from the current source):

* jaccard: result in [0, 1], symmetric, empty side -> 0.0.
* diversify: output is a reorder+subset of the input, no duplicates,
  len <= min(n, len(input)), and stable given the same inputs.
"""

from __future__ import annotations

import math
import string
from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

from app.diversity import diversify
from app.lexical import jaccard

_WORD = st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=8)


@given(
    a=st.frozensets(_WORD, max_size=8),
    b=st.frozensets(_WORD, max_size=8),
)
def test_jaccard_range_and_symmetry(a, b) -> None:
    value = jaccard(a, b)
    assert 0.0 <= value <= 1.0
    assert value == jaccard(b, a)


@given(st.frozensets(_WORD, max_size=8))
def test_jaccard_empty_side(a) -> None:
    assert jaccard(a, frozenset()) == 0.0
    assert jaccard(frozenset(), a) == 0.0


@given(st.frozensets(_WORD, min_size=1, max_size=8))
def test_jaccard_self_is_one(a) -> None:
    assert jaccard(a, a) == 1.0


def _make_records():
    return st.lists(
        st.builds(
            SimpleNamespace,
            title=st.text(alphabet=string.ascii_lowercase + " ", min_size=0, max_size=30),
            score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=10,
    )


@given(
    records=_make_records(),
    n=st.integers(min_value=0, max_value=12),
    lam=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    thresh=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_diversify_subset_no_dups_bounded(records, n, lam, thresh) -> None:
    out = diversify(records, n, lam=lam, sim_thresh=thresh)
    ids_of = lambda rs: [id(r) for r in rs]
    input_ids = ids_of(records)
    out_ids = ids_of(out)
    # Every output object came from the input (identity) — a reorder + subset.
    assert set(out_ids) <= set(input_ids)
    # No duplicates.
    assert len(out_ids) == len(set(out_ids))
    # Length bound.
    assert len(out) <= min(n, len(records))


@given(
    records=_make_records(),
    n=st.integers(min_value=0, max_value=12),
    lam=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    thresh=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_diversify_stable(records, n, lam, thresh) -> None:
    first = diversify(records, n, lam=lam, sim_thresh=thresh)
    second = diversify(records, n, lam=lam, sim_thresh=thresh)
    assert [id(r) for r in first] == [id(r) for r in second]


@given(records=_make_records(), lam=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_diversify_short_input_unchanged_order(records, lam) -> None:
    n = len(records)  # >= len(input): must stay in original order/objects
    out = diversify(records, n, lam=lam)
    assert [id(r) for r in out] == [id(r) for r in records[:n]]


@given(titles=st.lists(_WORD, min_size=0, max_size=8), n=st.integers(min_value=0, max_value=8))
def test_diversify_zero_or_empty(titles, n) -> None:
    records = [SimpleNamespace(title=t, score=1.0) for t in titles]
    out = diversify(records, 0)
    assert out == []
    assert diversify([], n) == []


def test_diversify_drops_nan_scores() -> None:
    good = SimpleNamespace(title="a b", score=0.9)
    recs = [
        good,
        SimpleNamespace(title="a b", score=float("nan")),
        SimpleNamespace(title="a b", score=float("-inf")),
    ]
    # With n smaller than the list the greedy path runs and a NaN/-inf
    # candidate can never outrank the real score, so the short list stops
    # early instead of returning a NaN item.
    out = diversify(recs, 1)
    assert out == [good]
    assert all(math.isfinite(r.score) for r in out)
