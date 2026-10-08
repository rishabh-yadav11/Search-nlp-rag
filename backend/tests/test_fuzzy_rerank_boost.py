"""Property-based tests for app.rerank_boost.apply_entity_boost / extract_entities.

Invariants (from the current source):

* apply_entity_boost returns new result objects with a score >= the input,
  and never mutates the input objects or lists it was given.
* extract_entities produces lowercased names whose tokens are a subset of the
  query's own (normalized) tokens — extraction never invents words.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.rerank_boost import apply_entity_boost, extract_entities

_RESULT = st.builds(
    SimpleNamespace,
    title=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", max_size=40),
    summary=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", max_size=40),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


def _norm(q: str) -> str:
    """The query the way the module normalises it (lowercased, apostrophes
    stripped, whitespace collapsed) — used to check entity provenance."""
    return re.sub(r"\s+", " ", re.sub(r"['\u2019]", "", q or "").lower()).strip()


@given(
    q=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '", max_size=80)
)
def test_extract_entities_come_from_query(q: str) -> None:
    nq = _norm(q)
    for entity in extract_entities(q):
        # An entity is always built from text that actually appears in the
        # (normalized) query — extraction never invents a word.
        assert entity, f"empty entity for query {q!r}"
        assert nq
        assert entity in nq, f"{entity!r} not derived from {q!r} (norm {nq!r})"


@given(results=st.lists(_RESULT, min_size=0, max_size=6), qi=st.text(max_size=80))
def test_apply_entity_boost_never_mutates_input(results, qi: str) -> None:
    before = [(r.title, r.summary, r.score) for r in results]
    apply_entity_boost(qi, results)
    after = [(r.title, r.summary, r.score) for r in results]
    assert before == after


@given(
    results=st.lists(_RESULT, min_size=0, max_size=6),
    qi=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", max_size=80),
)
def test_apply_entity_boost_scores_non_decreasing(results, qi: str) -> None:
    # Tag each input so a clone can be paired with its source even after the
    # re-sort (multiple copies can share a title/summary).
    for i, r in enumerate(results):
        r.tag = f"r{i}"
    out = apply_entity_boost(qi, results)
    assert len(out) == len(results)
    by_tag = {r.tag: r.score for r in results}
    for clone in out:
        assert clone.score >= by_tag[clone.tag]


@given(
    results=st.lists(_RESULT, min_size=1, max_size=6),
    qa=st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz", max_size=60),
)
def test_apply_entity_boost_returns_copies_when_entities(results, qa: str) -> None:
    out = apply_entity_boost(qa, results)
    assert len(out) == len(results)
    if extract_entities(qa):
        # With entities extracted the objects are copies, never the originals.
        original_ids = {id(r) for r in results}
        for clone in out:
            assert id(clone) not in original_ids


def test_apply_entity_boost_known_title_boost() -> None:
    r = SimpleNamespace(title="Ola Electric posts strong results", summary="", score=0.5)
    out = apply_entity_boost("Ola Electric results", [r])
    assert len(out) == 1
    assert r.score == 0.5  # untouched
    assert out[0].score > 0.5  # title boost applied


def test_apply_entity_boost_known_summary_boost() -> None:
    r = SimpleNamespace(title="Quarterly results review", summary="Ola Electric posted strong numbers", score=0.4)
    out = apply_entity_boost("ola electric", [r])
    assert len(out) == 1
    assert r.score == 0.4
    assert out[0].score == pytest.approx(0.4 * 1.1)
