

from app.query_intent import (
    _current_year,
    _referenced_year,
    extract_list_topic,
    extract_month_range,
    extract_year_range,
    month_query_topic,
    normalize_word_numbers,
    range_query_topic,
    rewrite_year_in_review,
    suggested_top_k,
)


def test_extract_year_range_explicit_year():
    assert extract_year_range("top deals of 2025") == ("2025-01-01", "2025-12-31")
    assert extract_year_range("2025 funding") == ("2025-01-01", "2025-12-31")
    assert extract_year_range("top 20 startups in 2025") == ("2025-01-01", "2025-12-31")


def test_extract_year_range_span():
    assert extract_year_range("deals from 2023-2025") == ("2023-01-01", "2025-12-31")
    assert extract_year_range("deals 2023 to 2025") == ("2023-01-01", "2025-12-31")
    assert extract_year_range("deals 2023 through 2025") == ("2023-01-01", "2025-12-31")


def test_extract_year_range_span_reversed():
    """A span written end-first is the same window, not an inverted one:
    '2025 to 2024' must match '2024 to 2025' rather than from_date > to_date."""
    assert extract_year_range("deals 2025 to 2024") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 2025-2024") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 2025 through 2024") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 1998-1995") == ("1995-01-01", "1998-12-31")


def test_extract_year_range_span_reversed_matches_ascending():
    assert extract_year_range("deals 2025 to 2024") == extract_year_range("deals 2024 to 2025")


def test_extract_year_range_last_and_this_year():
    # query_intent derives "this/last year" from _CURRENT_YEAR, so assert relative
    # to it instead of hardcoding 2025/2026.
    cur = _current_year()
    last = cur - 1
    assert extract_year_range("top articles last year") == (f"{last}-01-01", f"{last}-12-31")
    assert extract_year_range("the last year's highlights") == (f"{last}-01-01", f"{last}-12-31")
    assert extract_year_range("previous year deals") == (f"{last}-01-01", f"{last}-12-31")
    assert extract_year_range("this year's funding") == (f"{cur}-01-01", f"{cur}-12-31")
    assert extract_year_range("current year trends") == (f"{cur}-01-01", f"{cur}-12-31")


def test_extract_year_range_no_year_returns_none():
    assert extract_year_range("latest startup deals") is None
    assert extract_year_range("top 10 companies") is None
    assert extract_year_range("") is None


def test_extract_year_range_event_year_not_a_date_filter():
    """A year naming a historical event is a topic reference, not a
    publication-date filter: 'the 2008 crisis' must find retrospectives written
    later instead of being restricted to 2008 articles."""
    assert extract_year_range("lessons from the 2008 crisis") is None
    assert extract_year_range("the 2008 financial crisis") is None
    assert extract_year_range("financial crisis of 2008") is None
    assert extract_year_range("what caused the 2016 demonetisation") is None
    assert extract_year_range("how did companies fare in the 2020 pandemic") is None


def test_extract_year_range_event_year_ignored_when_other_year_mentioned():
    assert extract_year_range("funding 2024 during the 2008 crisis") == ("2024-01-01", "2024-12-31")


def test_extract_year_range_plain_year_still_filters():
    assert extract_year_range("2024 funding") == ("2024-01-01", "2024-12-31")
    assert extract_year_range("top deals of 2025") == ("2025-01-01", "2025-12-31")


def test_suggested_top_k_numeric():
    assert suggested_top_k("top 5 deals") == 5
    assert suggested_top_k("show me top 20 startups") == 20
    assert suggested_top_k("best 10 companies") == 10
    # Multi-digit counts must not be misparsed as a 2-digit prefix.
    assert suggested_top_k("top 100 companies") == 100
    assert suggested_top_k("top 150 deals") == 150


def test_suggested_top_k_word_numbers():
    assert suggested_top_k("top ten deals") == 10
    assert suggested_top_k("show me top twenty startups") == 20
    assert suggested_top_k("best ten companies") == 10
    assert suggested_top_k("biggest five funding rounds") == 5
    assert suggested_top_k("top twenty five deals") == 25
    assert suggested_top_k("largest ten deals in 2025") == 10
    assert suggested_top_k("top fortyfive deals") == 45
    assert suggested_top_k("top twentyone startups") == 21


def test_extract_list_topic_word_numbers():
    assert extract_list_topic("top ten fintech deals") == "fintech deals"
    assert extract_list_topic("best ten companies") == "companies"
    assert extract_list_topic("top twenty five deals in 2024") == "deals"


def test_normalize_word_numbers():
    assert normalize_word_numbers("top ten ipo") == "top 10 ipo"
    assert normalize_word_numbers("top ten deals") == "top 10 deals"
    assert normalize_word_numbers("best ten companies") == "best 10 companies"
    assert normalize_word_numbers("top twenty five deals") == "top 25 deals"
    assert normalize_word_numbers("top 10 ipo") == "top 10 ipo"
    assert normalize_word_numbers("top tenfold growth") == "top tenfold growth"
    assert normalize_word_numbers("latest deals") == "latest deals"


def test_suggested_top_k_default_and_none():
    assert suggested_top_k("top deals in fintech") == 10
    assert suggested_top_k("best fintech companies") == 10
    assert suggested_top_k("leading funds") == 10
    assert suggested_top_k("latest news") is None
    assert suggested_top_k("") is None


def test_rewrite_top_deals_in_year():
    new_q, changed = rewrite_year_in_review("top 20 deals in 2025")
    assert changed is True
    assert new_q == "Flashback 2025 deals"


def test_rewrite_top_articles_last_year():
    new_q, changed = rewrite_year_in_review("top articles last year")
    assert changed is True
    assert new_q == f"Flashback {_current_year() - 1} articles"


def test_rewrite_unchanged_without_top_hint():
    q, changed = rewrite_year_in_review("funding deals in 2025")
    assert changed is False
    assert q == "funding deals in 2025"


def test_rewrite_unchanged_without_year():
    q, changed = rewrite_year_in_review("top deals")
    assert changed is False
    assert q == "top deals"


def test_extract_list_topic_basic():
    assert extract_list_topic("top 3 unicorns created in 2025") == "unicorns created"
    assert extract_list_topic("top 10 fintech deals in 2025") == "fintech deals"
    assert extract_list_topic("biggest PE funds raised last year") == "PE funds raised"


def test_extract_list_topic_no_intent():
    assert extract_list_topic("funding deals in 2025") is None
    assert extract_list_topic("top deals") == "deals"


def test_extract_list_topic_year_span_removed():
    assert extract_list_topic("best M&A deals in 2023-2025") == "M&A deals"


def test_rewrite_niche_topic_flashback_kept_but_topic_extractable():
    """The Flashback rewrite still fires, but the bare topic is recoverable for
    the second (direct) retrieval leg used to surface niche articles."""
    new_q, changed = rewrite_year_in_review("top venture debt providers in 2024")
    assert changed is True
    assert new_q == "Flashback 2024 venture debt providers"
    assert extract_list_topic("top venture debt providers in 2024") == "venture debt providers"


def test_extract_list_topic_ignores_explicit_flashback_prefix():
    assert extract_list_topic("Flashback 2025 biggest deals") == "deals"


def test_extract_month_range_full_and_abbrev():
    assert extract_month_range("january 2025") == ("2025-01-01", "2025-01-31")
    assert extract_month_range("deals in feb 2024") == ("2024-02-01", "2024-02-29")
    assert extract_month_range("no month here 2025") is None


def test_extract_month_range_defaults_to_current_year():
    cur = _current_year()
    rng = extract_month_range("top deals in march")
    assert rng is not None
    assert rng[0].startswith(f"{cur}-03-")
    assert rng[1].startswith(f"{cur}-03-")


def test_extract_year_range_month_takes_precedence():
    assert extract_year_range("top pharma deals of month january 2025") == ("2025-01-01", "2025-01-31")
    assert extract_year_range("deals in feb 2024") == ("2024-02-01", "2024-02-29")


def test_rewrite_skipped_for_month_query():
    q, changed = rewrite_year_in_review("top pharma deals of month january 2025")
    assert changed is False
    assert q == "top pharma deals of month january 2025"


def test_month_query_topic_top_n_query():
    assert month_query_topic("top pharma deals of month january 2025") == "pharma deals"
    assert month_query_topic("deals in feb 2024") == "deals"


def test_month_query_topic_plain_date_query():
    assert month_query_topic("january 2025") is None
    assert month_query_topic("ChrysCapital Intas Pharma deals in january 2025") == (
        "ChrysCapital Intas Pharma deals"
    )


def test_month_query_topic_no_month_returns_none():
    assert month_query_topic("top pharma deals 2025") is None
    assert month_query_topic("venture funding") is None


def test_extract_list_topic_strips_month_words():
    assert extract_list_topic("top pharma deals of month january 2025") == "pharma deals"
    assert extract_list_topic("top deals in feb 2024") == "deals"


def test_extract_year_range_short_span():
    assert extract_year_range("top 15 deals in 2024-25") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 1999-00") == ("1999-01-01", "2000-12-31")


def test_extract_year_range_short_span_rollover():
    assert extract_year_range("deals 2024-23") == ("2023-01-01", "2024-12-31")


def test_extract_year_range_month_span():
    cur = _current_year()
    assert extract_year_range("deals in jan-march") == (f"{cur}-01-01", f"{cur}-03-31")
    assert extract_year_range("x in jan-march 2025") == ("2025-01-01", "2025-03-31")
    assert extract_year_range("top 15 deals in jan to march 2025") == ("2025-01-01", "2025-03-31")
    assert extract_year_range("top 15 deals in january 2025 to march 2025") == ("2025-01-01", "2025-03-31")
    assert extract_year_range("deals between january and march 2025") == ("2025-01-01", "2025-03-31")


def test_extract_year_range_month_span_crosses_year():
    cur = _current_year()
    assert extract_year_range("deals from may to march") == (f"{cur}-05-01", f"{cur + 1}-03-31")


def test_extract_month_range_month_span():
    cur = _current_year()
    assert extract_month_range("deals in jan-march") == (f"{cur}-01-01", f"{cur}-03-31")
    assert extract_month_range("january to march 2025") == ("2025-01-01", "2025-03-31")


def test_extract_year_range_quarter():
    cur = _current_year()
    assert extract_year_range("deals in Q1 2025") == ("2025-01-01", "2025-03-31")
    assert extract_year_range("deals in Q2-2024") == ("2024-04-01", "2024-06-30")
    assert extract_year_range("deals in Q3 2024") == ("2024-07-01", "2024-09-30")
    assert extract_year_range("deals in q4") == (f"{cur}-10-01", f"{cur}-12-31")
    assert extract_year_range("deals in first quarter of 2025") == ("2025-01-01", "2025-03-31")


def test_extract_year_range_fiscal_year():
    assert extract_year_range("top 15 deals in FY25") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("top 15 deals in FY 2024-25") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("top 15 deals in fy24-25") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("deals in fiscal year 2025") == ("2024-04-01", "2025-03-31")


def test_extract_year_range_fiscal_span_rollover():
    assert extract_year_range("deals in fy 2025-24") == ("2024-04-01", "2025-03-31")


def test_referenced_year_explicit_flashback_prefix():
    assert _referenced_year("flashback 2025 ipos") == 2025
    assert _referenced_year("what happened in flashback 2020") == 2020


def test_rewrite_not_fired_for_range_queries():
    """Range/fiscal/quarter queries must not collapse into a single annual
    Flashback roundup — they want the span's own data."""
    for q in (
        "top 15 deals in 2024-25",
        "top 15 deals in jan to march 2025",
        "top 15 deals in Q1 2025",
        "top 15 deals in FY25",
    ):
        new_q, changed = rewrite_year_in_review(q)
        assert changed is False
        assert new_q == q


def test_extract_list_topic_strips_new_range_words():
    assert extract_list_topic("top 15 deals in 2024-25") == "deals"
    assert extract_list_topic("top 15 deals in Q1 2025") == "deals"
    assert extract_list_topic("top 15 deals in FY25") == "deals"
    assert extract_list_topic("top 15 deals in jan to march 2025") == "deals"


def test_range_query_topic():
    assert range_query_topic("top 15 deals in 2024-25") == "deals"
    assert range_query_topic("deals in jan-march") == "deals"
    assert range_query_topic("top 15 deals in Q1 2025") == "deals"
    assert range_query_topic("top 15 deals in FY25") == "deals"
    assert range_query_topic("funding deals 2024-25") == "funding deals"
    assert range_query_topic("top deals in 2025") is None
    assert range_query_topic("venture funding") is None


def test_chart_request_filler_stripped_from_topic():
    """Chart/table request words ('make a table of') describe the output format,
    not the topic: they must not leak into the retrieval query or embedding
    match is diluted and retrieval can fail ('make a table deals')."""
    for q, expected in (
        ("make a table of top 15 deals in 2024-25", "deals"),
        ("show me a bar chart of top 15 deals in 2024-25", "deals"),
        ("create a pie chart for top deals in Q1 2025", "deals"),
        ("top 15 deals in 2024-25 as a table", "deals"),
        ("give me a graph of deals in jan-march", "deals"),
        ("draw a line chart of top 10 funding rounds in FY25", "funding rounds"),
        ("make a table of top pharma deals of month january 2025", "pharma deals"),
    ):
        assert range_query_topic(q) == expected
    assert extract_list_topic("make a table of top 15 deals in 2024-25") == "deals"
    assert extract_list_topic("draw a line chart of top 10 funding rounds in FY25") == "funding rounds"


def test_chart_filler_not_stripped_from_real_topic_words():
    """'table' as a real topic word (not part of a chart request) must survive."""
    assert extract_list_topic("top table manufacturing deals in 2025") == "table manufacturing deals"
    assert range_query_topic("top table games funding") is None
