
from app.answer_fallback import (
    TOP_WEAK_THRESHOLD,
    date_label,
    fallback_answer,
    search_note,
    weak_results,
)


def test_weak_results_all_low_scores():
    assert weak_results("startup layoffs India 2025", [0.1, 0.2, 0.15]) is True


def test_weak_results_all_high_scores():
    assert weak_results("who acquired Housing.com", [0.9, 0.8, 0.7]) is False


def test_weak_results_mixed_just_below_limit():
    assert weak_results("top edtech companies raising money", [0.9, 0.1, 0.1]) is True
    assert weak_results("query", [0.9, 0.8, 0.1, 0.2]) is True


def test_weak_results_empty_list():
    assert weak_results("anything", []) is True


def test_weak_results_fewer_than_limit_results():
    assert weak_results("query", [0.5, 0.9]) is True
    assert weak_results("query", [0.5]) is True


def test_weak_results_custom_limit():
    assert weak_results("query", [0.9, 0.8], limit=1) is False
    assert weak_results("query", [0.1, 0.8], limit=2) is True


def test_weak_results_edge_near_threshold():
    assert weak_results("query", [TOP_WEAK_THRESHOLD] * 3) is True
    assert weak_results("query", [TOP_WEAK_THRESHOLD + 0.001] * 3) is False
    assert weak_results("query", [0.31, 0.31, 0.29]) is True


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


def test_search_note_strong_results_none():
    assert search_note([0.9, 0.8, 0.7]) is None


def test_search_note_weak_results_string():
    assert isinstance(search_note([0.1, 0.2, 0.15]), str)
    assert isinstance(search_note([]), str)


def test_date_label_month_and_year():
    assert date_label("2025-01-01", "2025-01-31") == "January 2025"
    assert date_label("2025-02-01", "2025-02-28") == "February 2025"
    assert date_label("2025-01-01", "2025-12-31") == "2025"
    assert date_label("2024-03-01", "2024-03-31") == "March 2024"


def test_date_label_unknown_window():
    assert date_label(None, None) is None
    assert date_label("2025-01-05", "2025-06-30") is None
    assert date_label("2025-01-01", "2025-06-30") is None


def test_search_note_with_label_is_softer():
    assert search_note([0.1, 0.2], "January 2025") == (
        "Showing the closest January 2025 matches — only a few articles cover this exact topic."
    )


def test_search_note_strong_with_label_none():
    assert search_note([0.9, 0.8, 0.7], "January 2025") is None


def test_fallback_answer_with_label_best_effort():
    answer = fallback_answer("top pharma deals of month january 2025", 3, "January 2025")
    assert "January 2025" in answer
    assert "3" in answer
    assert "closest" in answer
