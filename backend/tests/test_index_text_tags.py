"""``tag_names``: the MySQL tag column, from row to embedded text.

Tags are the one facet that also enters the *embedded* text, so an article
tagged in MySQL but indexed without its tags is invisible to a tag filter and
no later repair fixes it.
"""

from app.index_text import compose_dense_text, compose_sparse_text, record_from_row


def _row(**overrides) -> dict:
    row = {
        "feid": 42,
        "title": "Ola Electric IPO",
        "summary": "Funding news.",
        "body": "Full body",
        "slug": "ola-electric-ipo",
        "ext_url": "",
        "publish": "2025-06-01 00:00:00",
        "content_type": "article",
        "dealtype_names": "Series A, M&A",
        "author_names": "Alice",
        "industry_names": "Fintech",
        "tag_names": "A,B,A",
    }
    row.update(overrides)
    return row


def test_record_from_row_splits_and_dedupes_the_tag_string():
    """The column is comma-separated in MySQL, like the other *_names columns."""
    assert record_from_row(_row())["tag_names"] == ["A", "B"]


def test_record_from_row_tags_are_a_list_even_when_absent():
    """A KEYWORD-indexed field that is None on some points and a list on others
    is a mixed-type field Qdrant indexes inconsistently, and a read side that
    iterates the value would crash on None."""
    assert record_from_row(_row(tag_names=None))["tag_names"] == []
    assert record_from_row(_row(tag_names=""))["tag_names"] == []


def test_record_from_row_tolerates_a_row_without_the_tag_column():
    """backfill_body.py and backfill_missing.py reuse this builder with a
    narrower SELECT; a missing key must not fail an unrelated payload repair."""
    row = _row()
    del row["tag_names"]

    assert record_from_row(row)["tag_names"] == []


def test_tags_reach_both_embedded_texts():
    """The lead is shared, so a tag is a semantic term AND a lexical one."""
    rec = record_from_row(_row(tag_names="Waaree Energies, Niveshaay"))

    assert "Waaree Energies" in compose_dense_text(rec)
    assert "Niveshaay" in compose_sparse_text(rec)


def test_tags_follow_dealtype_in_the_lead():
    """A tag is appended last, so it can never displace the title or the facet
    values the embedder's truncation is meant to protect."""
    rec = record_from_row(_row(tag_names="VCC Startups"))

    lead = compose_dense_text(rec)
    assert lead.index("Series A, M&A") < lead.index("VCC Startups")
