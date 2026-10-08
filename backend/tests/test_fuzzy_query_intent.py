"""Property-based tests for app.query_intent date/recency helpers.

The module's single clock seam is ``_now()``; ``frozen_clock`` pins it so every
regular expression resolves against one deterministic calendar (2026-09-15).

Invariants (from the current source):

* extract_recency_range / extract_year_range / extract_month_range: every
  result is None or a (from, to) pair of ISO YYYY-MM-DD strings with
  from <= to and both parseable.
* _full_year: expands a 2-digit year to within 50 years of its pivot, and
  passes 4-digit years through unchanged.
* normalize_word_numbers: identity when there is no top/best hint; word form
  counts after a hint become digits.
* is_recency_intent: True implies a soft-recency expression was present.
* strip_recency_window / strip_recency_intent are idempotent.
"""

from __future__ import annotations

import re
from datetime import datetime

from hypothesis import given
from hypothesis import strategies as st

import app.query_intent as qi

_DATE_WORDS = [
    "today",
    "this week",
    "past week",
    "last week",
    "this month",
    "past month",
    "last month",
    "past 3 days",
    "2 weeks ago",
    "5 days ago",
    "january 2025",
    "february 2024",
    "march 2025",
    "may 2021",
    "september 2024",
    "jan-march 2025",
    "january to february 2024",
    "Q1 2025",
    "quarter 3 of 2025",
    "FY25",
    "FY 2024-25",
    "fiscal year 2025",
    "2020",
    "2024-2025",
    "2025 to 2024",
    "last year",
    "this year",
    "2024-25",
    "top 10 deals in 2025",
    "biggest funding rounds in 2023",
]
_FILLER = [
    "funding", "ipo", "news", "deals", "startup", "mumbai", "india", "acquisition",
    "earnings", "report", "grew", "raised", "valuation", "round", "quarter",
]

def _query_with() -> st.SearchStrategy[str]:
    return st.lists(st.sampled_from(_DATE_WORDS + _FILLER), min_size=1, max_size=4).map(
        lambda ws: " ".join(ws)
    )


def _assert_iso_pair(result) -> None:
    if result is None:
        return
    frm, to = result
    assert isinstance(frm, str) and isinstance(to, str)
    ft, tt = datetime.fromisoformat(frm), datetime.fromisoformat(to)
    assert ft <= tt, f"inverted window {frm} > {to}"


@given(_query_with())
def test_recency_range_is_iso_pair_or_none(q: str) -> None:
    _assert_iso_pair(qi.extract_recency_range(q))


@given(_query_with())
def test_year_range_is_iso_pair_or_none(q: str) -> None:
    _assert_iso_pair(qi.extract_year_range(q))


@given(_query_with())
def test_month_range_is_iso_pair_or_none(q: str) -> None:
    _assert_iso_pair(qi.extract_month_range(q))


@given(y=st.integers(min_value=0, max_value=99), near=st.integers(min_value=1900, max_value=2100))
def test_full_year_stays_within_50_of_pivot(y: int, near: int) -> None:
    assert abs(qi._full_year(y, near) - near) <= 50


@given(y=st.integers(min_value=100, max_value=9999), near=st.integers(min_value=1900, max_value=2100))
def test_full_year_passthrough_four_digits(y: int, near: int) -> None:
    assert qi._full_year(y, near) == y


_GENERIC = st.lists(st.sampled_from(_FILLER), min_size=1, max_size=5).map(lambda ws: " ".join(ws))


@given(_GENERIC)
def test_normalize_word_numbers_identity_without_hint(q: str) -> None:
    assert qi.normalize_word_numbers(q) == q


_NUM_CASES = [
    ("top ten ipo deals", "top 10 ipo deals"),
    ("best twenty five rounds", "best 25 rounds"),
    ("top 10 deals", "top 10 deals"),
    ("top three five", "top 8"),
    ("leading hundred startups", "leading 100 startups"),
]


def test_normalize_word_numbers_examples() -> None:
    for raw, expected in _NUM_CASES:
        assert qi.normalize_word_numbers(raw) == expected


@given(_query_with())
def test_strip_recency_window_idempotent(q: str) -> None:
    once = qi.strip_recency_window(q)
    assert qi.strip_recency_window(once) == once


@given(_query_with())
def test_strip_recency_intent_idempotent(q: str) -> None:
    once = qi.strip_recency_intent(q)
    assert qi.strip_recency_intent(once) == once


_SOFT_SIGNALS = re.compile(
    r"\b(latest|recent|newest|freshest|fresh|lately|breaking|of\s+late)\b", re.IGNORECASE
)


@given(_query_with())
def test_recency_intent_implies_soft_signal(q: str) -> None:
    if qi.is_recency_intent(q):
        assert _SOFT_SIGNALS.search(q), f"is_recency_intent True without a signal: {q!r}"


@given(st.lists(st.sampled_from(_FILLER), min_size=1, max_size=5).map(lambda ws: " ".join(ws)))
def test_recency_intent_false_without_signal(q: str) -> None:
    # Filler words contain none of the soft signals, so intent must be False.
    assert qi.is_recency_intent(q) is False


def test_recency_intent_latest_true() -> None:
    assert qi.is_recency_intent("latest funding news") is True
    assert qi.is_recency_intent("recent rounds") is True
    # Hard-window phrases are NOT soft recency intent.
    assert qi.is_recency_intent("this week") is False


def test_recency_range_this_week_anchored(frozen_clock) -> None:
    frm, to = qi.extract_recency_range("this week")
    assert frm == "2026-09-14"  # Monday of the frozen week
    assert to == "2026-09-15"


def test_recency_range_past_3_days(frozen_clock) -> None:
    frm, to = qi.extract_recency_range("past 3 days")
    assert frm == "2026-09-12"
    assert to == "2026-09-15"


def test_year_range_explicit_year(frozen_clock) -> None:
    assert qi.extract_year_range("top deals in 2024") == ("2024-01-01", "2024-12-31")


def test_year_range_span(frozen_clock) -> None:
    assert qi.extract_year_range("2023 to 2025") == ("2023-01-01", "2025-12-31")


def test_month_range_explicit_month(frozen_clock) -> None:
    assert qi.extract_month_range("january 2025") == ("2025-01-01", "2025-01-31")


def test_month_range_default_year(frozen_clock) -> None:
    assert qi.extract_month_range("june") == ("2026-06-01", "2026-06-30")


def test_month_range_span(frozen_clock) -> None:
    assert qi.extract_month_range("jan-march 2025") == ("2025-01-01", "2025-03-31")
