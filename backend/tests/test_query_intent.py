

from app.query_intent import (
    extract_list_topic,
    extract_month_range,
    extract_year_range,
    month_query_topic,
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


def test_extract_year_range_last_and_this_year():
    assert extract_year_range("top articles last year") == ("2025-01-01", "2025-12-31")
    assert extract_year_range("the last year's highlights") == ("2025-01-01", "2025-12-31")
    assert extract_year_range("previous year deals") == ("2025-01-01", "2025-12-31")
    assert extract_year_range("this year's funding") == ("2026-01-01", "2026-12-31")
    assert extract_year_range("current year trends") == ("2026-01-01", "2026-12-31")


def test_extract_year_range_no_year_returns_none():
    assert extract_year_range("latest startup deals") is None
    assert extract_year_range("top 10 companies") is None
    assert extract_year_range("") is None


def test_suggested_top_k_numeric():
    assert suggested_top_k("top 5 deals") == 5
    assert suggested_top_k("show me top 20 startups") == 20
    assert suggested_top_k("best 10 companies") == 10


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
    assert new_q == "Flashback 2025 articles"


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
    rng = extract_month_range("top deals in march")
    assert rng is not None
    assert rng[0].startswith("2026-03-")
    assert rng[1].startswith("2026-03-")


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
