"""Property-based tests for app.input_hygiene normalisation/bounds.

Invariants asserted (from the current source):

* normalize_text: idempotent, NFKC-canonical, no C0 control characters.
* split_facet_values: a list of non-empty normalized strings, no duplicates,
  bounded in count and length; ``None``/empty input -> [].
* build_cache_key: deterministic, never empty, injective across distinct
  tuples (same parts in different orders -> different keys).
"""

from __future__ import annotations

import unicodedata

import pytest
from fastapi import HTTPException
from hypothesis import given
from hypothesis import strategies as st

from app.input_hygiene import (
    MAX_FACET_VALUE_LEN,
    MAX_FACET_VALUES,
    build_cache_key,
    normalize_text,
    split_facet_values,
)

# Anything <= U+001F plus DEL must never survive normalisation.
_CONTROLS = [chr(c) for c in range(0x20)] + ["\x7f"]


@given(st.text(max_size=200))
def test_normalize_text_idempotent(value: str) -> None:
    once = normalize_text(value)
    assert normalize_text(once) == once


@given(st.text(max_size=200))
def test_normalize_text_has_no_control_characters(value: str) -> None:
    out = normalize_text(value)
    for ch in _CONTROLS:
        assert ch not in out


@given(st.text(max_size=200))
def test_normalize_text_is_nfkc_canonical(value: str) -> None:
    out = normalize_text(value)
    assert unicodedata.normalize("NFKC", out) == out


def test_normalize_text_whitespace_shape() -> None:
    # Whitespace-bearing controls become spaces, never deleted/fused.
    assert normalize_text("Ola\tIPO") == "Ola IPO"
    assert normalize_text("  a   b  ") == "a b"
    # Case is deliberately preserved.
    assert normalize_text("PhonePe Funding") == "PhonePe Funding"


@given(raw=st.one_of(st.none(), st.text(max_size=1200)))
def test_split_facet_values_invariants(raw: str | None) -> None:
    try:
        values = split_facet_values("industry", raw)
    except HTTPException:
        # Rejected input: too long, or too many values. No invariants to hold.
        return
    assert isinstance(values, list)
    assert len(values) <= MAX_FACET_VALUES
    for v in values:
        assert isinstance(v, str)
        assert v != ""
        assert normalize_text(v) == v
        assert len(v) <= MAX_FACET_VALUE_LEN


def test_split_facet_values_none_and_blank() -> None:
    assert split_facet_values("industry", None) == []
    assert split_facet_values("industry", "") == []
    assert split_facet_values("industry", " , , ") == []


def test_split_facet_values_normalises_but_preserves_order_and_duplicates() -> None:
    # The current source normalises and bounds but does NOT deduplicate; the
    # duplicate 'a' entries pass through in raw order (downstream MatchAny is
    # a set, so this costs nothing but is observable at this surface).
    assert split_facet_values("industry", "Finance, finance, FINANCE,,") == [
        "Finance",
        "finance",
        "FINANCE",
    ]
    assert split_facet_values("tag", "a,a,a,") == ["a", "a", "a"]
    assert split_facet_values("tag", "a,a,a,") != ["a"]


def test_split_facet_values_raises_on_too_many() -> None:
    raw = ",".join(f"v{i}" for i in range(MAX_FACET_VALUES + 1))
    with pytest.raises(HTTPException) as exc:
        split_facet_values("industry", raw)
    assert exc.value.status_code == 400


def test_split_facet_values_raises_on_overlong_raw() -> None:
    raw = "x" * (MAX_FACET_VALUES * (MAX_FACET_VALUE_LEN + 1) + 1)
    with pytest.raises(HTTPException) as exc:
        split_facet_values("author", raw)
    assert exc.value.status_code == 400


def test_split_facet_values_raises_on_overlong_value() -> None:
    with pytest.raises(HTTPException) as exc:
        split_facet_values("dealtype", f"a,{'y' * (MAX_FACET_VALUE_LEN + 1)}")
    assert exc.value.status_code == 400


@given(
    parts=st.lists(st.text(max_size=40), min_size=1, max_size=5),
    ns=st.text(max_size=12),
)
def test_build_cache_key_deterministic(parts: list[str], ns: str) -> None:
    a = build_cache_key(*parts, namespace=ns)
    b = build_cache_key(*parts, namespace=ns)
    assert a == b
    assert a != ""
    if ns:
        assert a.startswith(f"{ns}:")


@given(
    parts=st.lists(st.text(max_size=40), min_size=1, max_size=5),
    ns=st.text(max_size=12),
)
def test_build_cache_key_never_empty(parts: list[str], ns: str) -> None:
    assert build_cache_key(*parts, namespace=ns) != ""


@given(
    left=st.lists(st.text(max_size=40), min_size=1, max_size=5),
    extra=st.lists(st.text(max_size=40), min_size=1, max_size=3),
    ns=st.text(max_size=12),
)
def test_build_cache_key_injective_across_distinct_tuples(
    left: list[str], extra: list[str], ns: str
) -> None:
    right = left + extra
    assert build_cache_key(*right, namespace=ns) != build_cache_key(*left, namespace=ns)


@given(
    a=st.lists(st.text(max_size=40), min_size=1, max_size=5),
    b=st.lists(st.text(max_size=40), min_size=1, max_size=5),
    ns=st.text(max_size=12),
)
def test_build_cache_key_order_sensitive(a: list[str], b: list[str], ns: str) -> None:
    # Same parts in a different order must never collide (a merge is a real
    # cache-poisoning risk; the encoding is order-sensitive).
    if a != b:
        assert build_cache_key(*a, namespace=ns) != build_cache_key(*b, namespace=ns)


def test_build_cache_key_delimiter_ambiguity() -> None:
    # 'a|b' as ONE part vs ('a', 'b') as two parts — a naive join would collide.
    assert build_cache_key("a|b") != build_cache_key("a", "b")


def test_build_cache_key_digests_long_key() -> None:
    big = "x" * 1000
    key = build_cache_key(big)
    assert key.startswith("sha256:")
    assert len(key) < 100
    # Two different long payloads must still differ.
    assert build_cache_key("y" * 1000) != key
