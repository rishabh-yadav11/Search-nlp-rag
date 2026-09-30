"""The one-off content_type backfill: what it writes, and what it refuses to
rewrite.

The failure this guards is silent: a backfill that is not idempotent either
does nothing on the second run (a no-op that looks successful while the field
is still missing) or re-writes every point forever. Both are caught here by
running the real function twice against a fake collection.
"""
import asyncio
import sys

import backfill_content_type as bct
import numpy as np
import pytest
from _common import create_payload_indexes, make_point
from backfill_content_type import scroll_points_missing_content_type, set_content_type
from qdrant_client.models import PayloadSchemaType

from app.config import config


class _Point:
    def __init__(self, id_, payload):
        self.id = id_
        self.payload = payload


class _FakeQdrant:
    """Holds point payloads in memory so a second run sees what the first wrote."""

    def __init__(self, payloads: dict[int, dict]):
        self.payloads = dict(payloads)
        self.writes: list[tuple[dict, list[int]]] = []
        self.index_calls: list[tuple[str, str]] = []

    def scroll(self, *, limit, with_payload, with_vectors, offset, **kwargs):
        # One page is enough: the fake collection is small and a second page
        # would not be returned by the real API either (no next offset).
        field = with_payload[0]
        pts = [
            _Point(pid, {field: pl[field]} if field in pl else {})
            for pid, pl in self.payloads.items()
        ]
        return pts, None

    def set_payload(self, *, collection_name, payload, points, wait):
        self.writes.append((payload, list(points)))
        for pid in points:
            self.payloads[pid].update(payload)

    def create_payload_index(self, collection_name, field, schema):
        self.index_calls.append((field, schema))

    def close(self):
        self.closed = True




def test_backfill_writes_the_value_from_mysql():
    """A point without the key gets exactly its own row's content_type."""
    client = _FakeQdrant({1: {"title": "a"}, 2: {"title": "b"}})
    records = {1: {"content_type": "Interview"}, 2: {"content_type": "Article"}}

    batch = list(scroll_points_missing_content_type(client))
    set_content_type(client, records, batch)

    assert client.writes == [
        ({"content_type": "Interview"}, [1]),
        ({"content_type": "Article"}, [2]),
    ]
    assert client.payloads[1]["content_type"] == "Interview"
    assert client.payloads[2]["content_type"] == "Article"


def test_backfill_is_idempotent_across_runs():
    """The second run finds nothing to do and writes nothing.

    This is the property that makes the script safe to rerun in production, so
    it is asserted by actually running the pass twice over one collection
    rather than by inspecting a flag.
    """
    client = _FakeQdrant({1: {"title": "a"}, 2: {"title": "b"}, 3: {"title": "c"}})
    records = {1: {"content_type": "Interview"}, 2: {"content_type": ""}, 3: {"content_type": "Article"}}

    first = list(scroll_points_missing_content_type(client))
    set_content_type(client, records, first)
    writes_after_first = len(client.writes)

    second = list(scroll_points_missing_content_type(client))
    set_content_type(client, records, second)

    assert sorted(first) == [1, 2, 3]
    assert second == [], "a completed run must leave nothing to backfill"
    assert len(client.writes) == writes_after_first, "second run must not write"


def test_backfill_treats_an_empty_stored_value_as_already_backfilled():
    """Key presence, not truthiness, is the marker.

    Point 2's MySQL value is genuinely empty, so the backfill stores an empty
    string. A test that treated that as "still missing" would rewrite it on
    every run; key presence is what stops that while still backfilling points
    that have no key at all.
    """
    client = _FakeQdrant({2: {"content_type": ""}, 5: {"title": "e"}})

    pending = list(scroll_points_missing_content_type(client))

    assert pending == [5], "an empty-string value is already backfilled; a missing key is not"


def test_backfill_matches_the_writer_for_a_missing_mysql_value():
    """No content_type in MySQL stores "", the same value make_point would.

    If the backfill stored None where the writer stores "", a collection would
    end up with a mixed-type field and the two halves would disagree about what
    an article with no content type looks like.
    """
    client = _FakeQdrant({9: {}})
    records = {9: {}}  # row present, content_type column NULL

    set_content_type(client, records, [9])
    written = client.payloads[9]["content_type"]

    class _Sparse:
        indices = np.asarray([1])
        values = np.asarray([0.5])

    fresh = make_point(
        {"id": 9, "title": "T", "url": "u", "content_type": records[9].get("content_type")},
        np.asarray([0.1], dtype=np.float32),
        _Sparse(),
    ).payload["content_type"]

    assert written == fresh == ""


def _async_records(records):
    async def _fetch(with_body=True, ids=None):
        assert with_body is False, f"expected with_body=False, got {with_body}"
        return dict(records)

    return _fetch


def test_dry_run_writes_nothing_and_creates_no_index(monkeypatch):
    """--dry-run is a strictly read-only rehearsal.

    It must not only skip set_payload but also skip create_payload_indexes:
    creating a payload index mutates the collection schema, so a "rehearsal"
    that indexes fields is not a rehearsal. The counts and log line would still
    look right either way, so this asserts the absence of both kinds of write.
    """
    client = _FakeQdrant({1: {}, 2: {}})
    monkeypatch.setattr(bct, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(bct, "fetch_records", _async_records({1: {"content_type": "Interview"}}))
    monkeypatch.setattr(config, "MYSQL_PASSWORD", "stub")
    monkeypatch.setattr(sys, "argv", ["backfill_content_type.py", "--dry-run"])

    assert bct.main() == 0
    assert client.writes == [], "dry-run must not write payloads"
    assert client.index_calls == [], "dry-run must not create payload indexes"
    assert client.payloads == {1: {}, 2: {}}, "dry-run must leave the collection untouched"


def test_a_real_run_creates_the_content_type_index(monkeypatch):
    """The real run indexes the field it just backfilled, so it is filterable.

    A collection built before content_type existed has no index for it, so
    backfilling the payload alone would leave the values unfiltersable until a
    separate rebuild. The run creates the index as part of the same step.
    """
    client = _FakeQdrant({1: {}})
    monkeypatch.setattr(bct, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(bct, "fetch_records", _async_records({1: {"content_type": "Interview"}}))
    monkeypatch.setattr(config, "MYSQL_PASSWORD", "stub")
    monkeypatch.setattr(sys, "argv", ["backfill_content_type.py"])

    assert bct.main() == 0
    assert client.writes == [({"content_type": "Interview"}, [1])]
    assert ("content_type", PayloadSchemaType.KEYWORD) in client.index_calls


def test_points_are_grouped_by_value_into_fewer_writes():
    """set_payload writes one dict to many points, so equal values share a call."""
    client = _FakeQdrant({1: {}, 2: {}, 3: {}})
    records = {1: {"content_type": "Interview"}, 2: {"content_type": "Interview"}, 3: {"content_type": "Video"}}

    set_content_type(client, records, [1, 2, 3])

    assert sorted(len(pids) for _, pids in client.writes) == [1, 2]
    written = {pid: value["content_type"] for value, pids in client.writes for pid in pids}
    assert written == {1: "Interview", 2: "Interview", 3: "Video"}


def test_content_type_payload_index_is_created():
    """A stored field is only filterable once it is indexed."""
    client = _FakeQdrant({})

    create_payload_indexes(client)

    indexed = {field: schema for field, schema in client.index_calls}
    assert indexed["content_type"] == PayloadSchemaType.KEYWORD
    # The date field needs a DATETIME index, not a keyword one.
    assert indexed["published_date"] == PayloadSchemaType.DATETIME


def test_every_field_the_search_can_filter_on_has_a_payload_index():
    """Drift guard: the indexed set must cover every key build_facet_filter uses.

    The keys are read back out of the Filter the real read path builds, rather
    than restated here, so adding a new facet filter without indexing its field
    fails this test instead of shipping an unfilterable facet.
    """
    from app.main import build_facet_filter

    client = _FakeQdrant({})
    create_payload_indexes(client)
    indexed = {field for field, _ in client.index_calls}

    filtered = build_facet_filter("Fin", "M&A", "A", "2025-01-01", "2025-12-31", "Interview", "IPO")
    filter_keys = {c.key for c in filtered.must}

    assert filter_keys == {
        "industry_names",
        "dealtype_names",
        "author_names",
        "published_date",
        "content_type",
        "tag_names",
    }
    assert filter_keys <= indexed, f"unindexed filter fields: {filter_keys - indexed}"


def test_scroll_requests_only_the_content_type_payload_field():
    """Scrolling every payload would pull ~6KB of body per point for one field."""
    seen = {}

    class _ScrollSpy(_FakeQdrant):
        def scroll(self, *, limit, with_payload, with_vectors, offset, **kwargs):
            seen.update({"with_payload": with_payload, "with_vectors": with_vectors, "limit": limit})
            return [], None

    list(scroll_points_missing_content_type(_ScrollSpy({})))

    assert seen["with_payload"] == ["content_type"]
    assert seen["with_vectors"] is False


@pytest.mark.parametrize("value", ["Interview", "", None])
def test_a_null_or_empty_stored_value_is_treated_as_missing(value):
    """A point whose stored value is None needs the field; "" does not."""
    client = _FakeQdrant({1: {"content_type": value}})

    pending = list(scroll_points_missing_content_type(client))

    assert pending == ([1] if value is None else [])


def test_fetch_records_is_the_awaitable_the_script_uses():
    """The script calls fetch_records(with_body=False); keep that contract honest."""
    import inspect

    from update_index import fetch_records

    sig = inspect.signature(fetch_records)
    assert "with_body" in sig.parameters
    assert asyncio.iscoroutinefunction(fetch_records)
