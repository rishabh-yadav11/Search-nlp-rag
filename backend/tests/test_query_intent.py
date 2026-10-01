

import datetime as dt
import re
import time

import pytest

from app import query_intent
from app.query_intent import (
    _current_year,
    _referenced_year,
    _today,
    extract_list_topic,
    extract_month_range,
    extract_year_range,
    is_aggregation_intent,
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
    """A reversed span must resolve to the same ascending window, never from_date > to_date."""
    assert extract_year_range("deals 2025 to 2024") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 2025-2024") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 2025 through 2024") == ("2024-01-01", "2025-12-31")
    assert extract_year_range("deals 1998-1995") == ("1995-01-01", "1998-12-31")


def test_extract_year_range_span_reversed_matches_ascending():
    assert extract_year_range("deals 2025 to 2024") == extract_year_range("deals 2024 to 2025")


def test_extract_year_range_last_and_this_year():
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
    """A year naming a historical event is a topic word; filtering on it hides retrospectives written later."""
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


def test_suggested_top_k_superlative_default():
    assert suggested_top_k("biggest funding rounds") == 10
    assert suggested_top_k("highest valued startups") == 10
    assert suggested_top_k("most active investors") == 10


def test_is_aggregation_intent():
    assert is_aggregation_intent("top 5 deals")
    assert is_aggregation_intent("biggest funding rounds in 2025")
    assert is_aggregation_intent("most active investors")
    assert is_aggregation_intent("highest valued startups this year")
    assert not is_aggregation_intent("latest news")
    assert not is_aggregation_intent("who invested in Ola Electric")
    assert not is_aggregation_intent("most of the time the market is up")


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


def test_range_query_topic_covers_month_scoped_queries():
    """Pin on range_query_topic, the only path the app calls, so its test-only duplicate cannot drift unnoticed."""
    assert range_query_topic("top pharma deals of month january 2025") == "pharma deals"
    assert range_query_topic("deals in feb 2024") == "deals"
    assert range_query_topic("january 2025") is None
    assert range_query_topic("ChrysCapital Intas Pharma deals in january 2025") == (
        "ChrysCapital Intas Pharma deals"
    )
    assert range_query_topic("top pharma deals 2025") is None
    assert range_query_topic("venture funding") is None


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


def test_extract_month_range_span_crosses_year_with_end_year():
    """A year written after the END month anchors the end, so a crossing span must start the year BEFORE it."""
    assert extract_month_range("deals in dec to jan 2024") == ("2023-12-01", "2024-01-31")
    assert extract_month_range("deals from december to january 2024") == ("2023-12-01", "2024-01-31")
    assert extract_month_range("deals in dec-jan 2024") == ("2023-12-01", "2024-01-31")


def test_extract_month_range_span_crosses_year_with_start_year():
    """A year written after the START month anchors the start, so a crossing span ends the year after."""
    assert extract_month_range("deals in dec 2023 to jan") == ("2023-12-01", "2024-01-31")
    assert extract_month_range("deals in december 2023 through january") == ("2023-12-01", "2024-01-31")


def test_extract_month_range_span_single_year_not_crossing_boundary():
    assert extract_month_range("deals in jan to mar 2024") == ("2024-01-01", "2024-03-31")
    assert extract_month_range("deals in january to march 2024") == ("2024-01-01", "2024-03-31")
    assert extract_month_range("deals in jan 2024 to mar") == ("2024-01-01", "2024-03-31")


def test_extract_month_range_span_two_years_honored_exactly():
    assert extract_month_range("deals in dec 2023 to jan 2024") == ("2023-12-01", "2024-01-31")
    assert extract_month_range("deals in jan 2024 to mar 2024") == ("2024-01-01", "2024-03-31")


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


def test_extract_year_range_fiscal_span_multi_year():
    """'fy 2020-2025' spans FY2020 through FY2024, not only the last fiscal year."""
    assert extract_year_range("deals fy 2020-2025") == ("2020-04-01", "2025-03-31")
    assert extract_year_range("deals fy 2019-2021") == ("2019-04-01", "2021-03-31")
    assert extract_year_range("deals fy 2025-2020") == ("2020-04-01", "2025-03-31")


def test_extract_year_range_fiscal_span_same_year_is_single_fy():
    """A span naming one year twice is one fiscal year; reading it as a span inverts the window to match zero rows."""
    for q in ("deals fy 25-25", "deals fy 2025-25", "deals fy 2025 to 2025"):
        from_date, to_date = extract_year_range(q)
        assert from_date < to_date, f"inverted window for {q!r}: {from_date}..{to_date}"
        assert (from_date, to_date) == ("2024-04-01", "2025-03-31")
        assert extract_year_range(q) == extract_year_range("deals fy 2025")


def test_extract_year_range_fiscal_single_and_consecutive_unchanged():
    assert extract_year_range("deals fy 2025") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("top 15 deals in FY25") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("top 15 deals in FY 2024-25") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("top 15 deals in fy24-25") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("deals in fiscal year 2025") == ("2024-04-01", "2025-03-31")
    assert extract_year_range("deals fy 2025-24") == ("2024-04-01", "2025-03-31")


# _fiscal_range and _strip_time_tokens share one compiled regex, so each case pins the range AND the stripped text.
_FY_CASES = (
    ("fy 2020-2021", ("2020-04-01", "2021-03-31"), " "),
    ("fy 2020 to 2021", ("2020-04-01", "2021-03-31"), " "),
    ("fy 2020 through 2021", ("2020-04-01", "2021-03-31"), " "),
    ("fy 2020", ("2019-04-01", "2020-03-31"), " "),
    ("fiscal 2025", ("2024-04-01", "2025-03-31"), " "),
    ("fiscal year 2025", ("2024-04-01", "2025-03-31"), " "),
    ("deals in fiscal year 2025", ("2024-04-01", "2025-03-31"), "deals    "),
    ("latest startup deals", None, "latest startup deals"),
    ("top 10 companies", None, "top 10 companies"),
    # Tier order, not leftmost: a span outranks a bare fiscal year, and 'fy' outranks 'fiscal'.
    ("fiscal 2025 and fy 2020-2021", ("2020-04-01", "2021-03-31"), "  and  "),
    ("fiscal 2025 plus fy 2020", ("2019-04-01", "2020-03-31"), "  plus  "),
    ("fy 2020-2021 then fy 2022", ("2020-04-01", "2021-03-31"), "  then  "),
    # The fused 'fiscal2020' must still RESOLVE a range; a stripping-only '\bfiscal\s+' form would silently drop it.
    ("fiscal2020", ("2019-04-01", "2020-03-31"), " "),
    ("deals in fiscal2020", ("2019-04-01", "2020-03-31"), "deals    "),
)


@pytest.mark.parametrize(("query", "expected_range", "expected_stripped"), _FY_CASES)
def test_fiscal_range_and_stripping_share_one_grammar(query, expected_range, expected_stripped):
    got_range = query_intent._fiscal_range(query)
    assert got_range == expected_range, f"{query!r}: range {got_range!r} != {expected_range!r}"
    got_stripped = query_intent._strip_time_tokens(query)
    assert got_stripped == expected_stripped, f"{query!r}: stripped {got_stripped!r} != {expected_stripped!r}"


def test_referenced_year_explicit_flashback_prefix():
    assert _referenced_year("flashback 2025 ipos") == 2025
    assert _referenced_year("what happened in flashback 2020") == 2020


def test_rewrite_not_fired_for_range_queries():
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
    """Chart/table request words name the output format; leaking one yields 'make a table deals'."""
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
    assert extract_list_topic("top table manufacturing deals in 2025") == "table manufacturing deals"
    assert range_query_topic("top table games funding") is None
    assert query_intent._strip_chart_filler("share table tennis news") == "share table tennis news"
    assert not query_intent._is_chart_request("share table tennis news")


def _freeze_utc(monkeypatch, when: dt.datetime) -> None:
    """Pin the module clock; ``when`` MUST be tz-aware (no stdlib type says so) or it is read in the host's timezone."""
    if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
        raise ValueError(f"_freeze_utc needs an aware datetime, got {when!r}")
    frozen = when.astimezone(dt.UTC)
    monkeypatch.setattr(query_intent, "_now", lambda: frozen)


def test_today_is_resolved_in_indian_timezone(monkeypatch):
    """20:00Z on 2023-12-31 is already 2024-01-01 in Asia/Kolkata, so a naive UTC clock resolves a whole year behind."""
    _freeze_utc(monkeypatch, dt.datetime(2023, 12, 31, 20, 0, tzinfo=dt.UTC))
    assert _today() == dt.date(2024, 1, 1)
    assert _current_year() == 2024


def test_current_year_uses_indian_date_across_new_year(monkeypatch):
    """19:00Z on 2024-12-31 is already 2025 in Asia/Kolkata, so relative-year defaults must resolve there."""
    _freeze_utc(monkeypatch, dt.datetime(2024, 12, 31, 19, 0, tzinfo=dt.UTC))
    assert _current_year() == 2025
    assert extract_year_range("top articles last year") == ("2024-01-01", "2024-12-31")
    assert extract_year_range("this year's funding") == ("2025-01-01", "2025-12-31")
    assert extract_month_range("top deals in march") == ("2025-03-01", "2025-03-31")


def test_today_matches_utc_date_outside_the_offset_window(monkeypatch):
    """Outside 18:30Z-24:00Z the two calendars agree, so the Asia/Kolkata conversion must not shift the date."""
    _freeze_utc(monkeypatch, dt.datetime(2024, 6, 15, 12, 0, tzinfo=dt.UTC))
    assert _today() == dt.date(2024, 6, 15)
    assert _current_year() == 2024


def test_freeze_rejects_naive_datetime(monkeypatch):
    naive = dt.datetime(2023, 12, 31, 20, 0)  # noqa: DTZ001
    with pytest.raises(ValueError, match="aware datetime"):
        _freeze_utc(monkeypatch, naive)


# Frozen PRE-FIX copies of the live patterns; resyncing them would make the equivalence asserts vacuous.
_ORIG_BOTH_AND_RE = re.compile(r"\bboth\b.+?\band\b", re.IGNORECASE)
_ORIG_ALL_OF_RE = re.compile(r"\ball (?:of )?.+?\b(?:and|with)\b", re.IGNORECASE)
_ORIG_ACQUIRED_BY_RE = re.compile(
    r"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b[^.?!]*?\bby\b[^.?!]*?\b(who|whom)\b",
    re.IGNORECASE,
)
_ORIG_BUYER_AUX_RE = re.compile(
    r"\b(?:who|whom|what|which\s+\w+)\b[^.?!]*?\b(?:did|does|do|has|have|had|is|are|was|were)\b"
    r"[^.?!]*?\b(acquir\w+|bought|buys?|buying|purchas\w+|take\s*over|taken\s*over|took\s*over|takeover)\b",
    re.IGNORECASE,
)
_ORIG_BUYER_TRAILING_RE = re.compile(
    r"\b(acquir\w+|bought|take\s*over|took\s*over|takeover)\b.*\b(what|whom|who)\b",
    re.IGNORECASE,
)

# The first two feed _multi_entity_scaffold's substitution (compare removed text); the rest only as booleans.
_SUB_PAIRS = (
    (query_intent._BOTH_AND_RE, _ORIG_BOTH_AND_RE),
    (query_intent._ALL_OF_RE, _ORIG_ALL_OF_RE),
)
_SEARCH_PAIRS = (
    (query_intent._ACQUIRED_BY_RE, _ORIG_ACQUIRED_BY_RE),
    (query_intent._BUYER_AUX_RE, _ORIG_BUYER_AUX_RE),
    (query_intent._BUYER_TRAILING_RE, _ORIG_BUYER_TRAILING_RE),
)
_ORACLES_BY_NAME = {
    "_ACQUIRED_BY_RE": _ORIG_ACQUIRED_BY_RE,
    "_BUYER_AUX_RE": _ORIG_BUYER_AUX_RE,
    "_BUYER_TRAILING_RE": _ORIG_BUYER_TRAILING_RE,
}

# A zero-width gap after '\ball ' matches "all and", flipping detect_multi_entity to 'intersection'.
_ZERO_GAP_FIXTURES = (
    "all and",
    "all with",
    "all  and",
    "all of and",
    "all of with",
    "all with revenue and both Acme and Beta",
    "both and",
    "both with",
)
_INTERSECTION_CORPUS = (
    "companies backed by both SoftBank and Tiger Global",
    "funds that have backed both Acme and Beta Capital",
    "deals with all of A, B and C",
    "investments common to all of these funds and their LPs",
    "all of Acme, Beta and Gamma",
    "all of the seed cohort and their angels",
    "what do all of these investors have in common and who led them",
    "both companies are large",
    "all of the above",
    "startups that SoftBank and Tiger Global have both backed",
    "small allocations across the portfolio and the follow-on",
)
_INTERSECTION_CORPUS = _INTERSECTION_CORPUS + _ZERO_GAP_FIXTURES

_ACQUISITION_CORPUS = (
    # Target direction: the named company is the one that was acquired.
    "who acquired Freshworks in 2024",
    "which company bought Skio",
    "who was acquired by SoftBank",
    "the payments startup was bought by a consortium led by SoftBank, who will take the stake",
    "Softbank's Vision Fund acquired a majority stake in the payments startup",
    # Buyer direction: the named company is the one doing the acquiring.
    "who did Freshworks acquire",
    "what did Acme buy",
    "which company was acquired by Beta",
    "whom did the consortium take over last quarter",
    "what company was bought by the consortium",
    "the startup was bought by Acme, and the analysts asked who had advised on the deal",
    # Negative: an acquire verb with no interrogative anywhere.
    "the company acquired a stake in the market",
    "Softbank acquired Accelgo for three billion dollars",
)


@pytest.mark.parametrize(
    "new,orig", _SUB_PAIRS, ids=["both_and", "all_of"]
)
def test_intersection_substitutions_match_pre_fix_patterns(new, orig):
    for text in _INTERSECTION_CORPUS:
        assert new.sub(" ", text) == orig.sub(" ", text), text


@pytest.mark.parametrize(
    "new,orig", _SEARCH_PAIRS, ids=["acquired_by", "buyer_aux", "buyer_trailing"]
)
def test_acquisition_detection_matches_pre_fix_patterns(new, orig):
    for text in _ACQUISITION_CORPUS:
        assert bool(new.search(text)) == bool(orig.search(text)), text


def test_intersection_corpus_exercises_both_polarities():
    """Equivalence alone would pass if every pattern matched nothing, so pin the oracle's hits and misses."""
    both_hit = "companies backed by both SoftBank and Tiger Global"
    assert _ORIG_BOTH_AND_RE.search(both_hit)
    assert _ORIG_BOTH_AND_RE.sub(" ", both_hit) != both_hit
    all_hit = "deals with all of A, B and C"
    assert _ORIG_ALL_OF_RE.search(all_hit)
    assert _ORIG_ALL_OF_RE.sub(" ", all_hit) != all_hit
    for text in (
        "both companies are large",
        "all of the above",
        "startups that SoftBank and Tiger Global have both backed",
        "small allocations across the portfolio and the follow-on",
    ):
        assert _ORIG_BOTH_AND_RE.sub(" ", text) == text, text
        assert _ORIG_ALL_OF_RE.sub(" ", text) == text, text


def test_acquisition_corpus_exercises_both_polarities():
    for _, orig in _SEARCH_PAIRS:
        assert any(orig.search(t) for t in _ACQUISITION_CORPUS), orig.pattern
    for text in (
        "the company acquired a stake in the market",
        "Softbank acquired Accelgo for three billion dollars",
    ):
        for _, orig in _SEARCH_PAIRS:
            assert not orig.search(text), text


# The connective spans run to tens of characters, so a too-low bound or a literal-brace escape bug shows up here.
_LONG_BOTH_AND = (
    "Among the Series B rounds announced across India and Southeast Asia this quarter, "
    "the two funds that showed up on the most term sheets were the ones backed by both "
    "SoftBank's Vision Fund and Tiger Global, and the overlap with their earlier fintech "
    "bets is the part investors keep asking about in the follow-up calls"
)
_LONG_ALL_OF = (
    "all of the seed-stage accelerator programmes, the micro-SAT fund and the deep-tech "
    "fellowship, and the shared pipeline is where the overlap between the three turns out "
    "to be largest once the follow-on rounds from last year are taken out of the picture"
)
_LONG_ACQUIRED_BY = (
    "According to two people familiar with the talks, the decade-old logistics startup, "
    "founded in 2014 and rebranded twice since, was bought by a consortium of six mid-market "
    "investors led by a domestic private equity firm, who are expected to keep the existing "
    "management team in place through the transition"
)
_LONG_BUYER_AUX = (
    "What did the Bengaluru-based payments company, which had spent two years trying to build "
    "a credit book before pivoting back to merchant acquiring, acquire from the seller in the "
    "carve-out that was announced on Tuesday evening"
)
_LONG_BUYER_TRAILING = (
    "SoftBank acquired Greystone Digital Infrastructure's data centre portfolio, and the "
    "buyer confirmed whom it will keep on the existing contracts, what capacity it will add "
    "in the next two quarters, and which sites come under the same management team"
)


def test_long_realistic_queries_still_match_exactly_as_before():
    assert len(_LONG_BOTH_AND) > 200
    assert query_intent._BOTH_AND_RE.sub(" ", _LONG_BOTH_AND) == _ORIG_BOTH_AND_RE.sub(" ", _LONG_BOTH_AND)
    assert _LONG_BOTH_AND != query_intent._BOTH_AND_RE.sub(" ", _LONG_BOTH_AND)

    assert len(_LONG_ALL_OF) > 200
    assert query_intent._ALL_OF_RE.sub(" ", _LONG_ALL_OF) == _ORIG_ALL_OF_RE.sub(" ", _LONG_ALL_OF)
    assert _LONG_ALL_OF != query_intent._ALL_OF_RE.sub(" ", _LONG_ALL_OF)

    for text, orig in (
        (_LONG_ACQUIRED_BY, _ORIG_ACQUIRED_BY_RE),
        (_LONG_BUYER_AUX, _ORIG_BUYER_AUX_RE),
        (_LONG_BUYER_TRAILING, _ORIG_BUYER_TRAILING_RE),
    ):
        assert len(text) > 200, text
        assert orig.search(text), text
        assert bool(query_intent._ACQUIRED_BY_RE.search(text)) == bool(_ORIG_ACQUIRED_BY_RE.search(text))
        assert bool(query_intent._BUYER_AUX_RE.search(text)) == bool(_ORIG_BUYER_AUX_RE.search(text))
        assert bool(query_intent._BUYER_TRAILING_RE.search(text)) == bool(_ORIG_BUYER_TRAILING_RE.search(text))


def test_connective_gaps_are_bounded_and_groups_preserved():
    """Lower bound is per-pattern: _ALL_OF_RE needs {1,N} (its pre-fix gap was one-or-more); the rest {0,N}."""
    n = query_intent._MAX_CONNECTIVE_SPAN
    for new, orig in _SUB_PAIRS + _SEARCH_PAIRS:
        assert f"{{0,{n}}}" in new.pattern or f"{{1,{n}}}" in new.pattern, new.pattern
        assert ".*" not in new.pattern, new.pattern
        assert "*?" not in new.pattern, new.pattern
        assert new.groups == orig.groups, new.pattern
    assert "{1," in query_intent._ALL_OF_RE.pattern
    assert "{0," in query_intent._BOTH_AND_RE.pattern


def test_oracles_reproduce_pre_fix_reachability():
    """Pin the oracles' own behaviour: one drifting the same way as the live pattern voids the equivalence."""
    # The ONLY discriminating inputs: "all of and" and the longer fixtures DO match pre-fix (gap >= 1 char).
    for text in ("all and", "all with"):
        assert not _ORIG_ALL_OF_RE.search(text), text
        assert not query_intent._ALL_OF_RE.search(text), text

    for text in ("deals with all of A, B and C", "all of Acme, Beta and Gamma"):
        assert _ORIG_ALL_OF_RE.sub(" ", text) != text, text
        assert query_intent._ALL_OF_RE.sub(" ", text) == _ORIG_ALL_OF_RE.sub(" ", text), text
    for text in ("backed by both SoftBank and Tiger Global", "both Acme and Beta"):
        assert _ORIG_BOTH_AND_RE.sub(" ", text) != text, text
        assert query_intent._BOTH_AND_RE.sub(" ", text) == _ORIG_BOTH_AND_RE.sub(" ", text), text

    _POSITIVES = {
        "_ACQUIRED_BY_RE": (
            "the payments startup was bought by a consortium led by SoftBank, "
            "who will take the stake"
        ),
        "_BUYER_AUX_RE": "what did Acme buy",
        "_BUYER_TRAILING_RE": (
            "SoftBank acquired Greystone Digital's portfolio, and whom will they keep"
        ),
    }
    for name, text in _POSITIVES.items():
        oracle = _ORACLES_BY_NAME[name]
        assert oracle.search(text), (name, text)
        assert getattr(query_intent, name).search(text), (name, text)
    for name in _POSITIVES:
        assert not _ORACLES_BY_NAME[name].search("the company acquired a stake in the market"), name


# Closing term left out so all k start positions scan to EOF -- the shape unbounded gaps made quadratic.
_PATHOLOGICAL = (
    ("both_and", query_intent._BOTH_AND_RE, lambda n: "both " * n),
    ("all_of", query_intent._ALL_OF_RE, lambda n: "all bbbb " * n),
    ("acquired_by", query_intent._ACQUIRED_BY_RE, lambda n: "acquired " * n),
    ("buyer_aux", query_intent._BUYER_AUX_RE, lambda n: "who " * n),
    ("buyer_trailing", query_intent._BUYER_TRAILING_RE, lambda n: "takeover " * n),
)


def _time_search(pat: re.Pattern, text: str, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        pat.search(text)
    return (time.perf_counter() - start) / repeats


@pytest.mark.parametrize("name,pat,make", _PATHOLOGICAL, ids=[p[0] for p in _PATHOLOGICAL])
def test_pathological_query_scales_linearly(name, pat, make):
    """Timed per pattern over min() of trials: summing hides a quadratic matcher; one trial admits CI noise."""
    small, large = 256, 2048
    _time_search(pat, make(8), 1)  # keep cold-start cost out of the ratio
    t_small = min(_time_search(pat, make(small), 3) for _ in range(3))
    t_large = min(_time_search(pat, make(large), 1) for _ in range(3))
    ratio = t_large / t_small
    assert ratio < 20.0, (
        f"{name}: 8x input cost {ratio:.1f}x ({t_small * 1e3:.3f}ms -> {t_large * 1e3:.3f}ms); "
        f"an unbounded gap gives ~64x"
    )
    assert t_large < 20.0, f"{name}: pathological query took {t_large:.3f}s"
