"""The shared ops-script helpers: one payload builder, one pool factory.

These tests pin the exact payload key set produced by ``_common.make_point``.
That is the regression this file exists for: the payload used to be built by
three separate copies (one per write path), so a key present in the record could
be silently absent from the stored payload without any test noticing — which is
precisely how ``content_type`` ended up produced by ``record_from_row`` and
never persisted. Asserting the full dict (not a subset) means a dropped or
extra key now fails loudly.
"""
import asyncio

import numpy as np
import pytest
from _common import PAYLOAD_KEYS, log, make_point, make_pool


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
        "summary": "Funding news.",
        "body": "x" * 50,
        "author_names": ["Alice", "Bob"],
        "industry_names": ["Fintech"],
        "dealtype_names": ["Series A", "M&A"],
    }
    assert point.id == 42
    assert set(point.payload) == set(PAYLOAD_KEYS)


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


def test_content_type_is_only_written_when_a_caller_asks_for_it():
    """The opt-in seam for the still-unpersisted content_type field.

    The record carries content_type, but no write path has ever stored it, so
    make_point must not add it to the payload on its own — doing so silently
    would change every stored point as a side effect of a refactor. Callers opt
    in by name, which makes the omission a deliberate, visible choice rather
    than an accident of which copy of the builder a script happened to hold.
    """
    assert "content_type" in _record()  # the record does carry it

    without = make_point(_record(), DENSE, SPARSE).payload
    assert "content_type" not in without

    with_it = make_point(_record(), DENSE, SPARSE, content_type="article").payload
    assert with_it["content_type"] == "article"
    # Opting in must not disturb the other keys.
    assert {k: v for k, v in with_it.items() if k != "content_type"} == without


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
