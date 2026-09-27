"""The shared ops-script helpers: one payload builder, one pool factory.

These tests pin the exact payload produced by ``_common.make_point``. That is
the regression this file exists for: the payload used to be built by three
separate copies (one per write path), so a key could be silently dropped from
one path and not another without any test noticing. Asserting the full dict
rather than a subset means a dropped or added key now fails loudly, and the
expected dict in the test is the single source of truth for the key set.
"""
import asyncio

import numpy as np
import pytest
from _common import log, make_point, make_pool


class _FakeSparse:
    def __init__(self, indices, values):
        self.indices = np.asarray(indices)
        self.values = np.asarray(values)


def _record(**overrides) -> dict:
    rec = {
        "id": 42,
        "title": "Ola Electric IPO",
        "summary": "Funding news.",
        "body": "x" * 50,
        "url": "https://www.vccircle.com/ola-electric-ipo",
        "published_date": "2025-06-01T00:00:00",
        "category": "Series A",
        "content_type": "article",
        "author_names": ["Alice", "Bob"],
        "industry_names": ["Fintech"],
        "dealtype_names": ["Series A", "M&A"],
    }
    rec.update(overrides)
    return rec


DENSE = np.array([0.1, 0.2, 0.3], dtype=np.float32)
SPARSE = _FakeSparse([1, 5], [0.5, 0.25])


def test_make_point_payload_is_exactly_the_expected_dict():
    """The whole payload, asserted as a dict — not a subset check.

    A copy-paste divergence between two write paths for the same article is
    silent index corruption, and a subset assertion cannot catch it.
    """
    point = make_point(_record(), DENSE, SPARSE)

    assert point.payload == {
        "title": "Ola Electric IPO",
        "url": "https://www.vccircle.com/ola-electric-ipo",
        "published_date": "2025-06-01T00:00:00",
        "category": "Series A",
        "content_type": "article",
        "summary": "Funding news.",
        "body": "x" * 50,
        "author_names": ["Alice", "Bob"],
        "industry_names": ["Fintech"],
        "dealtype_names": ["Series A", "M&A"],
    }
    assert point.id == 42


def test_make_point_vector_shape_is_preserved():
    """Both hybrid vectors are stored, sparse as an explicit SparseVector."""
    point = make_point(_record(), DENSE, SPARSE)

    assert point.vector["dense"] == pytest.approx([0.1, 0.2, 0.3])
    assert point.vector["sparse"].indices == [1, 5]
    assert point.vector["sparse"].values == pytest.approx([0.5, 0.25])


def test_make_point_truncates_body_to_the_configured_limit():
    from app.config import config

    point = make_point(_record(body="y" * (config.BODY_CHAR_LIMIT + 500)), DENSE, SPARSE)

    assert point.payload["body"] == "y" * config.BODY_CHAR_LIMIT


def test_make_point_normalises_missing_optional_fields():
    """Absent/None optional fields get the same defaults the old copies used."""
    rec = _record(summary=None, body=None, published_date=None, category=None)
    rec.update({"author_names": None, "industry_names": None, "dealtype_names": None})

    payload = make_point(rec, DENSE, SPARSE).payload

    assert payload["summary"] == ""
    assert payload["body"] == ""
    assert payload["published_date"] is None
    assert payload["category"] is None
    assert payload["author_names"] == []
    assert payload["industry_names"] == []
    assert payload["dealtype_names"] == []


def test_content_type_is_in_the_payload():
    """The record carries content_type and so must the stored payload.

    This assertion used to be the reverse — it pinned the key as absent, which
    was precisely what kept the content_type feature dead end to end: the read
    path (``main._PAYLOAD_FIELDS`` -> ``SourceArticle.content_type``) and the
    facet vocabulary both read a field that no write path ever stored. Now that
    ``make_point`` stores it, this test is the guard that a future refactor
    cannot silently drop the key again: removing it from the payload fails here
    and in the exact-dict test above.
    """
    assert "content_type" in _record()  # record_from_row does supply it

    payload = make_point(_record(), DENSE, SPARSE).payload

    assert payload["content_type"] == "article"


def test_content_type_is_normalised_to_a_string():
    """Every point stores a str for a KEYWORD-indexed field, never None.

    A payload that is ``str`` on some points and ``None`` on others is a
    mixed-type field, which Qdrant's payload index handles inconsistently. The
    read side maps the empty string back to None
    (``payload.get("content_type") or None``), so nothing is lost.
    """
    for absent in (None, ""):
        payload = make_point(_record(content_type=absent), DENSE, SPARSE).payload

        assert payload["content_type"] == ""
        assert isinstance(payload["content_type"], str)

    # Whitespace is deliberately not re-stripped: both producers of this record
    # (record_from_row for MySQL, and the jsonl it dumps for the build path)
    # already strip the column, so re-stripping would add a second convention
    # rather than a guarantee. A real value passes through verbatim so the read
    # side can match it exactly against the live facet vocabulary.
    assert make_point(_record(content_type="Interview"), DENSE, SPARSE).payload["content_type"] == (
        "Interview"
    )


def test_payload_keys_cover_what_the_read_path_requests():
    """Every field the search read path asks Qdrant for is one we actually store.

    hybrid_search passes _PAYLOAD_FIELDS as `with_payload`, so a field listed
    there but never written is exactly the dead end this issue is about:
    `content_type` was requested on every read and produced by no write path.
    This asserts the two sides agree, in both directions, so the next field
    added to one without the other fails here.
    """
    from app import main

    stored = set(make_point(_record(), DENSE, SPARSE).payload)
    requested = set(main._PAYLOAD_FIELDS)

    assert "content_type" in requested & stored
    # `body` is deliberately excluded from _PAYLOAD_FIELDS (fetched separately
    # via with_body=True), so the read set must be a subset of the stored keys.
    assert requested <= stored


def test_make_pool_passes_configured_credentials_and_no_unsupported_kwargs(monkeypatch):
    """One pool factory, config-sourced, and free of kwargs aiomysql rejects.

    read_timeout/write_timeout were passed by two of the old call sites but are
    not accepted by the installed aiomysql, so those scripts raised TypeError
    before opening a connection. Asserting the exact kwargs keeps that from
    creeping back in.
    """
    import aiomysql

    captured = {}

    async def fake_create_pool(**kwargs):
        captured.update(kwargs)
        return "pool"

    monkeypatch.setattr(aiomysql, "create_pool", fake_create_pool)

    assert asyncio.run(make_pool()) == "pool"
    from app.config import config

    assert captured == {
        "host": config.MYSQL_HOST,
        "port": config.MYSQL_PORT,
        "user": config.MYSQL_USER,
        "password": config.MYSQL_PASSWORD,
        "db": config.MYSQL_DATABASE,
        "autocommit": True,
        "minsize": 1,
        "maxsize": 3,
        "connect_timeout": None,
    }


def test_make_pool_maxsize_and_connect_timeout_are_overridable(monkeypatch):
    import aiomysql

    captured = {}

    async def fake_create_pool(**kwargs):
        captured.update(kwargs)
        return "pool"

    monkeypatch.setattr(aiomysql, "create_pool", fake_create_pool)

    asyncio.run(make_pool(maxsize=5, connect_timeout=10))

    assert captured["maxsize"] == 5
    assert captured["connect_timeout"] == 10


def test_log_emits_a_single_utc_stamped_line(capsys):
    log("upserted 10 points")

    out = capsys.readouterr().out
    assert out.count("\n") == 1
    stamp, _, msg = out.strip().partition("] ")
    assert msg == "upserted 10 points"
    assert stamp.startswith("[") and stamp.endswith(" UTC")
