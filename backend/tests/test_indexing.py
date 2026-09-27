import pytest
import update_index
from update_index import fingerprint, load_state, save_state, sync_delta


def _rec(**overrides) -> dict:
    base = {
        "id": 1,
        "title": "Title",
        "summary": "Summary",
        "url": "https://example.com/1",
        "published_date": "2025-01-01T00:00:00+00:00",
        "category": "Deal",
        "body": "Body",
        "author_names": ["Alice"],
        "industry_names": ["Fintech"],
        "dealtype_names": ["Series A"],
        "content_type": "Interview",
    }
    base.update(overrides)
    return base


def test_fingerprint_stable_for_identical_records():
    assert fingerprint(_rec()) == fingerprint(_rec())


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "summary",
        "url",
        "published_date",
        "category",
        "body",
        "author_names",
        "industry_names",
        "dealtype_names",
        "content_type",
    ],
)
def test_fingerprint_sensitive_to_each_field(field):
    base = _rec()
    changed = dict(base)
    value = base[field]
    changed[field] = value + ["extra"] if isinstance(value, list) else f"{value}x"
    assert fingerprint(base) != fingerprint(changed), f"fingerprint should change when {field} changes"


def _payload_rec(**overrides) -> dict:
    """A record shaped like the one make_point actually stores."""
    rec = _rec(**overrides)
    rec.setdefault("content_type", "Interview")
    return rec


def test_fingerprint_sensitive_to_a_content_type_only_change():
    """A content_type edit alone must be visible to the incremental indexer.

    ``category`` does not cover this: record_from_row sets category to
    dealtype_names when those exist, so on any row that HAS dealtype_names a
    content_type-only change leaves the fingerprint identical and the indexed
    payload goes stale forever with nothing logged.
    """
    before = _payload_rec(content_type="Interview")
    after = _payload_rec(content_type="Video")

    assert before["category"] == after["category"], "precondition: category is unchanged"
    assert fingerprint(before) != fingerprint(after)


def test_fingerprint_covers_every_field_the_payload_stores():
    """Every key make_point writes must be covered by the change fingerprint.

    The payload key set and the fingerprint field list are two halves of one
    contract: a field stored in the payload but absent from the fingerprint is
    silently frozen at whatever value it had when it was first indexed. The
    keys are read from make_point itself so the two cannot drift apart.
    """
    from _common import make_point

    class _Arr:
        def __init__(self, v):
            self.v = v

        def tolist(self):
            return self.v

    class _Sparse:
        indices = _Arr([1])
        values = _Arr([0.5])

    point = make_point(
        _payload_rec(body="Body", summary="Summary"),
        _Arr([0.1, 0.2]),
        _Sparse(),
    )

    assert set(point.payload) == set(_payload_rec().keys()) - {"id"}, (
        "fixture must supply exactly the fields the payload stores"
    )

    baseline = fingerprint(_payload_rec())

    for key in point.payload:
        original = _payload_rec()[key]
        mutated = original + ["MUTATED"] if isinstance(original, list) else "MUTATED"
        changed = fingerprint(_payload_rec(**{key: mutated}))
        assert changed != baseline, f"fingerprint ignores {key!r}: a change to it is never re-indexed"


def _state_for(records: dict[int, dict]) -> dict:
    return {
        "updated_at": None,
        "fingerprints": {str(i): fingerprint(r) for i, r in records.items()},
    }


def test_sync_delta_new_changed_deleted():
    rec1 = _rec()
    rec2 = _rec(id=2, title="Second")
    state = _state_for({1: rec1, 2: rec2})
    state["fingerprints"]["4"] = "deadbeef"

    records = {
        1: rec1,
        2: _rec(id=2, title="Second EDITED"),
        3: _rec(id=3, title="Third"),
    }

    new, changed, deleted = sync_delta(state, records)
    assert new == {3}
    assert changed == {2}
    assert deleted == {4}


def test_sync_delta_unchanged_means_empty_sets():
    records = {1: _rec(), 2: _rec(id=2, title="Second")}
    state = _state_for(records)
    new, changed, deleted = sync_delta(state, records)
    assert new == set()
    assert changed == set()
    assert deleted == set()


def test_sync_delta_empty_state_is_all_new():
    records = {1: _rec(), 2: _rec(id=2, title="Second")}
    new, changed, deleted = sync_delta({"fingerprints": {}}, records)
    assert new == {1, 2}
    assert changed == set()
    assert deleted == set()


def test_load_state_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(update_index, "STATE_PATH", str(tmp_path / "missing.json"))
    assert load_state() == {"updated_at": None, "fingerprints": {}}


def test_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(update_index, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(update_index, "STATE_PATH", str(tmp_path / "index_state.json"))

    state = {
        "updated_at": "2026-08-13T00:00:00+00:00",
        "fingerprints": {"1": fingerprint(_rec()), "2": fingerprint(_rec(id=2, title="Two"))},
    }
    save_state(state)
    assert load_state() == state
