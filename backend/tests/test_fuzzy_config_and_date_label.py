"""Property-based tests for app.config clamp helpers and app.answer_fallback.

Invariants (from the current source):

* ``_clamped_int``-style helpers always return an int within [low, high]
  (unset -> default, non-integer -> default, out-of-range -> clamped).
* ``date_label`` never raises for any from<=to ISO inputs and returns str/None.
"""

from __future__ import annotations

import os
from datetime import date

from hypothesis import given
from hypothesis import strategies as st

from app.answer_fallback import date_label
from app.config import _clamped_int


# Unique env names: the helper reads ``os.getenv(name)`` at call time, and the
# names are namespaced so a property run can never collide with a real knob.
def _env_name() -> st.SearchStrategy[str]:
    return st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=8, max_size=24).map(
        lambda s: f"TEST_CLAMP_{s}"
    )


def _set_env(name: str, value: str) -> None:
    import os

    os.environ[name] = value


@given(
    name=_env_name(),
    low=st.integers(min_value=-50, max_value=50),
    high=st.integers(min_value=51, max_value=200),
)
def test_clamped_int_unset_returns_default(name: str, low: int, high: int) -> None:
    default = (low + high) // 2
    assert low <= default <= high
    env_key = f"TEST_CLAMP_UNSET_{name}"
    os.environ.pop(env_key, None)
    assert _clamped_int(env_key, default, low, high) == default


@given(
    name=_env_name(),
    low=st.integers(min_value=-50, max_value=50),
    high=st.integers(min_value=51, max_value=200),
    raw=st.text(max_size=120).filter(lambda s: "\x00" not in s),
)
def test_clamped_int_non_integer_returns_default(name, low, high, raw) -> None:
    default = (low + high) // 2
    # A value that int() cannot parse falls back to default.
    if not raw.lstrip("-").isdigit():
        _set_env(f"TEST_CLAMP_NONINT_{name}", raw)
        assert _clamped_int(f"TEST_CLAMP_NONINT_{name}", default, low, high) == default


@given(
    name=_env_name(),
    low=st.integers(min_value=-50, max_value=50),
    high=st.integers(min_value=51, max_value=200),
    value=st.integers(min_value=-10_000, max_value=10_000),
)
def test_clamped_int_clamps_to_range(name, low, high, value) -> None:
    default = (low + high) // 2
    env_key = f"TEST_CLAMP_RANGE_{name}"
    os.environ[env_key] = str(value)
    result = _clamped_int(env_key, default, low, high)
    assert low <= result <= high
    # In-range values pass through unchanged.
    if low <= value <= high:
        assert result == value
    else:
        assert result in (low, high) or result == default


# ---------------------------------------------------------------------------
# answer_fallback.date_label: never raises, returns str or None.
# ---------------------------------------------------------------------------


@given(
    a=st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31)),
    b=st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31)),
)
def test_date_label_never_raises_on_iso_pairs(a: date, b: date) -> None:
    frm, to = sorted([a, b])
    label = date_label(frm.isoformat(), to.isoformat())
    assert label is None or isinstance(label, str)


@given(
    a=st.text(max_size=60),
    b=st.text(max_size=60),
)
def test_date_label_never_raises_on_arbitrary_text(a: str, b: str) -> None:
    # Garbage that does not parse is a None answer, never an exception.
    label = date_label(a, b)
    assert label is None or isinstance(label, str)


def test_date_label_month_window() -> None:
    assert date_label("2025-01-01", "2025-01-31") == "January 2025"
    assert date_label("2025-02-01", "2025-02-28") == "February 2025"
    assert date_label("2024-02-01", "2024-02-29") == "February 2024"  # leap year


def test_date_label_year_window() -> None:
    assert date_label("2025-01-01", "2025-12-31") == "2025"


def test_date_label_partial_window_is_none() -> None:
    # A window that is not exactly a full month or a full year.
    assert date_label("2025-01-01", "2025-01-15") is None
    assert date_label("2025-06-01", "2025-12-31") is None
    assert date_label(None, None) is None
    assert date_label("2025-01-01", None) is None
