"""Shared record/text helpers used by the fetch/build/update index scripts."""

from datetime import UTC, datetime

from app.index_text import (
    clean,
    compose_dense_text,
    compose_sparse_text,
    normalize_date,
    record_from_row,
    split_names,
)


def _row(**overrides) -> dict:
    row = {
        "feid": 42,
        "title": " <h1>Ola Electric IPO</h1> ",
        "summary": "  Funding  news.  ",
        "body": "<p>Full body</p>",
        "slug": "ola-electric-ipo",
        "ext_url": "",
        "publish": "2025-06-01 00:00:00",
        "content_type": "article",
        "dealtype_names": "Series A, M&A",
        "author_names": '["Alice", "Alice", "Bob"]',
        "industry_names": "Fintech/Healthtech",
    }
    row.update(overrides)
    return row


def test_clean_strips_html_and_collapses_whitespace():
    assert clean(" <h1>Hello</h1>  World\n  ") == "Hello World"
    assert clean(None) == ""


def test_split_names_dedupes_and_handles_json_lists():
    assert split_names('["Alice", "Alice", "Bob"]') == ["Alice", "Alice", "Bob"]
    assert split_names("Fintech/Healthtech") == ["Fintech", "Healthtech"]
    assert split_names("TMT, Technology") == ["TMT", "Technology"]
    assert split_names("") == []
    assert split_names(None) == []


def test_split_names_whitespace_only_and_malformed_json():
    assert split_names("   ") == []
    assert split_names("[not json") == ["[not json"]


def test_normalize_date_edge_values():
    assert normalize_date(None) is None
    assert normalize_date("   ") is None
    assert normalize_date("not-a-date") == "not-a-date"


def test_normalize_date_datetime_instance():
    assert normalize_date(datetime(2025, 1, 15, 12, 30, tzinfo=UTC)) == "2025-01-15T12:30:00+00:00"


def test_record_from_row_builds_canonical_payload():
    rec = record_from_row(_row())
    assert rec["id"] == 42
    assert rec["title"] == "Ola Electric IPO"
    assert rec["body"] == "Full body"
    assert rec["url"] == "https://www.vccircle.com/ola-electric-ipo"
    assert rec["published_date"].startswith("2025-06-01T00:00:00")
    assert rec["category"] == "Series A, M&A"
    assert rec["author_names"] == ["Alice", "Alice", "Bob"]
    assert rec["industry_names"] == ["Fintech", "Healthtech"]


def test_record_from_row_external_url_wins():
    rec = record_from_row(_row(ext_url="https://example.com/x"))
    assert rec["url"] == "https://example.com/x"


def test_compose_dense_text_excludes_body():
    rec = record_from_row(_row())
    dense = compose_dense_text(rec)
    assert "Ola Electric IPO" in dense
    assert "Full body" not in dense


def test_compose_sparse_text_includes_body():
    rec = record_from_row(_row())
    sparse = compose_sparse_text(rec)
    assert "Ola Electric IPO" in sparse
    assert "Full body" in sparse
