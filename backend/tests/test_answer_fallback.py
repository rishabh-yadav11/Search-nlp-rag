
from app.answer_fallback import (
    TOP_WEAK_THRESHOLD,
    date_label,
    fallback_answer,
    results_are_weak,
    weak_results_note,
)


def test_results_are_weak_all_low_scores():
    assert results_are_weak([0.1, 0.2, 0.15]) is True


def test_results_are_weak_all_high_scores():
    assert results_are_weak([0.9, 0.8, 0.7]) is False


def test_results_are_weak_mixed_just_below_limit():
    assert results_are_weak([0.9, 0.1, 0.1]) is True
    assert results_are_weak([0.9, 0.8, 0.1, 0.2]) is True


def test_results_are_weak_empty_list():
    assert results_are_weak([]) is True


def test_results_are_weak_few_strong_matches_not_weak():
    """A topic with only 1-2 strong matches must not be suppressed: the corpus
    may simply have few articles on it (regression: niche queries were refused
    even when retrieval found a solid match)."""
    assert results_are_weak([0.5, 0.9]) is False
    assert results_are_weak([0.761, 0.457]) is False
    assert results_are_weak([0.5]) is False
    assert results_are_weak([0.5, 0.2]) is True  # one strong + one weak is still weak


def test_results_are_weak_custom_limit():
    assert results_are_weak([0.9, 0.8], limit=1) is False
    assert results_are_weak([0.1, 0.8], limit=2) is True


def test_results_are_weak_edge_near_threshold():
    assert results_are_weak([TOP_WEAK_THRESHOLD] * 3) is True
    assert results_are_weak([TOP_WEAK_THRESHOLD + 0.001] * 3) is False
    assert results_are_weak([0.31, 0.31, 0.29]) is True


def test_fallback_answer_contains_query_and_is_nonempty():
    q = "who acquired Housing.com"
    answer = fallback_answer(q, 0)
    assert answer
    assert q in answer


def test_fallback_answer_no_fabricated_numbers():
    answer = fallback_answer("who acquired Housing.com", 0)
    assert not any(ch.isdigit() for ch in answer)


def test_fallback_answer_mentions_weak_count():
    answer = fallback_answer("startup layoffs India 2025", 3)
    assert "startup layoffs India 2025" in answer
    assert "3" in answer


def test_weak_results_note_strong_results_none():
    assert weak_results_note([0.9, 0.8, 0.7]) is None


def test_weak_results_note_weak_results_string():
    assert isinstance(weak_results_note([0.1, 0.2, 0.15]), str)
    assert isinstance(weak_results_note([]), str)


def test_date_label_month_and_year():
    assert date_label("2025-01-01", "2025-01-31") == "January 2025"
    assert date_label("2025-02-01", "2025-02-28") == "February 2025"
    assert date_label("2025-01-01", "2025-12-31") == "2025"
    assert date_label("2024-03-01", "2024-03-31") == "March 2024"


def test_date_label_impossible_month_returns_none_instead_of_raising():
    """An out-of-range month must fall back to None like any other window that
    isn't a plain month/year. Regression: the except clause named
    calendar.IllegalYearError, which does not exist, so evaluating the except
    tuple itself raised AttributeError (issue #169)."""
    assert date_label("2025-13-01", "2025-12-31") is None
    assert date_label("2025-00-01", "2025-12-31") is None
    assert date_label("2025-99-01", "2025-99-31") is None


def test_date_label_degenerate_year_does_not_raise():
    """A year calendar cannot represent (e.g. 0000) must not escape as an
    exception; it returns None or a label, depending on the interpreter."""
    assert date_label("0000-01-01", "0000-01-31") in (None, "January 0")


def test_date_label_unknown_window():
    assert date_label(None, None) is None
    assert date_label("2025-01-05", "2025-06-30") is None
    assert date_label("2025-01-01", "2025-06-30") is None


def test_weak_results_note_with_label_is_softer():
    assert weak_results_note([0.1, 0.2], "January 2025") == (
        "Showing the closest January 2025 matches — only a few articles cover this exact topic."
    )


def test_weak_results_note_strong_with_label_none():
    assert weak_results_note([0.9, 0.8, 0.7], "January 2025") is None


def test_fallback_answer_with_label_best_effort():
    answer = fallback_answer("top pharma deals of month january 2025", 3, "January 2025")
    assert "January 2025" in answer
    assert "3" in answer
    assert "closest" in answer
