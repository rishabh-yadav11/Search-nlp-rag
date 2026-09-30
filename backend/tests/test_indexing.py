import hashlib

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
        "tag_names": ["IPO"],
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
        "tag_names",
    ],
)
def test_fingerprint_sensitive_to_each_field(field):
    base = _rec()
    changed = dict(base)
    value = base[field]
    changed[field] = value + ["extra"] if isinstance(value, list) else f"{value}x"
    assert fingerprint(base) != fingerprint(changed), f"fingerprint should change when {field} changes"


def _payload_rec(**overrides) -> dict:
    """A record shaped like the one make_point stores (i.e. _rec plus content_type)."""
    return _rec(**overrides)


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


def _legacy_fingerprint(rec: dict) -> str:
    """The fingerprint as the PREVIOUS 9-field version computed it.

    Spelled out field by field rather than derived from ``fingerprint`` so this
    stays a genuine reconstruction of the old state file: deriving it from the
    current function would make the test pass no matter what the code does.
    """
    raw = "|".join(
        [
            rec.get("title") or "",
            rec.get("summary") or "",
            rec.get("url") or "",
            rec.get("published_date") or "",
            rec.get("category") or "",
            rec.get("body") or "",
            ",".join(rec.get("author_names") or []),
            ",".join(rec.get("industry_names") or []),
            ",".join(rec.get("dealtype_names") or []),
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def test_adding_content_type_to_the_fingerprint_requeues_the_whole_corpus():
    """Pin the one-time consequence instead of leaving it to a comment.

    Adding a term to fingerprint() re-hashes every record, so a state file
    written by the previous version matches NOTHING and the first sync after
    this ships re-embeds the entire corpus. This is the behaviour the operator
    note and the start-up WARNING describe; asserting it here is what stops the
    note from quietly becoming false.
    """
    records = {i: _payload_rec(id=i, title=f"Article {i}") for i in range(1, 6)}
    legacy_state = {
        "updated_at": "2026-08-13T00:00:00+00:00",
        "fingerprints": {str(i): _legacy_fingerprint(r) for i, r in records.items()},
    }

    # Sanity: the legacy hash is genuinely the old scheme, not the current one.
    assert legacy_state["fingerprints"]["1"] != fingerprint(records[1])

    new, changed, deleted = sync_delta(legacy_state, records)

    assert new == set(), "no rows were added, so nothing should be 'new'"
    assert deleted == set(), "no rows were removed, so nothing should be 'deleted'"
    assert changed == set(records), (
        "every stored fingerprint predates the content_type term, so the whole "
        "corpus is re-queued for re-embedding; this is the one-time cost the "
        "operator note and --init remedy exist to avoid"
    )


def test_init_reseed_clears_the_requeue():
    """The --init remedy: re-seeding from current rows leaves nothing to embed.

    --init stores fingerprints computed by the CURRENT function, so after it the
    corpus is no longer re-queued. This is the whole reason the operator remedy
    works, so it is asserted rather than assumed.
    """
    records = {i: _payload_rec(id=i, title=f"Article {i}") for i in range(1, 6)}
    legacy_state = {
        "updated_at": None,
        "fingerprints": {str(i): _legacy_fingerprint(r) for i, r in records.items()},
    }

    # What --init writes.
    reseeded = {"updated_at": None, "fingerprints": {str(i): fingerprint(r) for i, r in records.items()}}

    # Contrast: the legacy state re-queues, the re-seeded one does not. That
    # difference is the entire value of the --init remedy.
    assert sync_delta(legacy_state, records)[1] == set(records)

    new, changed, deleted = sync_delta(reseeded, records)

    assert (new, changed, deleted) == (set(), set(), set())


def _run_main(monkeypatch, tmp_path, state, records):
    """Drive update_index.main() with the filesystem, DB and Qdrant faked out."""
    import asyncio

    monkeypatch.setattr(update_index, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(update_index, "LOCK_PATH", str(tmp_path / "update.lock"))
    monkeypatch.setattr(update_index, "STATE_PATH", str(tmp_path / "index_state.json"))
    monkeypatch.setattr(update_index, "reconcile", lambda state, records: True)
    monkeypatch.setattr(update_index, "load_state", lambda: state)
    monkeypatch.setattr(update_index.sys, "argv", ["update_index.py"])

    async def _fetch(with_body=True, ids=None):
        return dict(records)

    monkeypatch.setattr(update_index, "fetch_records", _fetch)
    applied = {}
    monkeypatch.setattr(
        update_index,
        "apply_delta",
        lambda recs, new, changed, deleted, state: applied.update(new=new, changed=changed),
    )

    asyncio.run(update_index.main())
    return applied


def test_main_warns_and_names_init_when_every_row_reads_as_changed(monkeypatch, tmp_path, capsys):
    """The operator-facing payoff of the re-queue: the run says so in its log.

    A fingerprint-scheme change is indistinguishable from "every row really was
    edited", and the cost is a full re-embed. Without a log line the operator
    only finds out from a bill, so main() must name --init when it sees this
    signature.
    """
    records = {i: _payload_rec(id=i, title=f"Article {i}") for i in range(1, 4)}
    legacy = {
        "updated_at": None,
        "fingerprints": {str(i): _legacy_fingerprint(r) for i, r in records.items()},
    }

    applied = _run_main(monkeypatch, tmp_path, legacy, records)

    out = capsys.readouterr().out
    assert "WARNING" in out, "the all-changed signature must be warned about"
    assert "--init" in out, "the warning must name the remedy"
    assert applied["changed"] == set(records), "precondition: the corpus really was re-queued"


def test_main_does_not_warn_on_a_normal_partial_delta(monkeypatch, tmp_path, capsys):
    """The warning must not fire on ordinary incremental work.

    A scheduled run where a few rows changed is the normal case; warning there
    would train operators to ignore the line that matters.
    """
    records = {i: _payload_rec(id=i, title=f"Article {i}") for i in range(1, 4)}
    state = {"updated_at": None, "fingerprints": {str(i): fingerprint(r) for i, r in records.items()}}
    records[2] = _payload_rec(id=2, title="Article 2 EDITED")

    applied = _run_main(monkeypatch, tmp_path, state, records)

    out = capsys.readouterr().out
    assert "WARNING: all" not in out, "a partial delta is normal and must not warn"
    assert applied["changed"] == {2}


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
