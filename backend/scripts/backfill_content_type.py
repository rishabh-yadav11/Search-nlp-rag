"""One-off backfill: populate the `content_type` payload field for Qdrant points
that were indexed before any write path stored it.

`content_type` was produced by record_from_row and requested on every search
read (`main._PAYLOAD_FIELDS` -> `SourceArticle.content_type`, plus the live
facet vocabulary), but no write path persisted it, so the feature was dead end
to end: every content-type filter matched nothing and the facet vocabulary was
empty. make_point now stores it, so new and updated points carry the field.
This script fixes the points that are ALREADY in the collection, in place, with
no re-embedding and no collection rebuild — dense/sparse vectors are untouched
because the payload is metadata only.

Idempotent by stored VALUE, not key presence: a point is skipped when its
payload's `content_type` is not None. So the empty string an article with no
content type in MySQL legitimately gets is left alone and never rewritten, and a
second run after a complete first run writes nothing. A stored JSON `null` is
treated as missing and re-written to "" — that is deliberate, since make_point
never stores null and normalising it is what keeps the KEYWORD-indexed field
single-typed.

Safe to rehearse: `--dry-run` reports what would change and touches nothing —
it writes no payload AND creates no payload index (indexing is a schema change,
so a rehearsal that indexes fields would not be a rehearsal).

Usage:
    python scripts/backfill_content_type.py --dry-run   # rehearse
    python scripts/backfill_content_type.py             # apply
"""
import asyncio
import os
import sys
import traceback
from collections.abc import Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import create_payload_indexes, log
from qdrant_client import QdrantClient
from update_index import fetch_records

from app.config import config

# How many points to scroll per page, and how many to write per set_payload.
# Bounding these keeps the working set small on a large collection.
PAGE_SIZE = 500
BATCH_SIZE = 200


def scroll_points_missing_content_type(client: QdrantClient) -> Iterator[int]:
    """Yield point IDs whose payload has no `content_type` key, one page at a
    time, so the caller never holds the full id list in memory."""
    next_offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=PAGE_SIZE,
            with_payload=["content_type"],
            with_vectors=False,
            offset=next_offset,
        )
        for p in pts:
            if (p.payload or {}).get("content_type") is None:
                yield p.id
        if next_offset is None:
            break


def set_content_type(
    client: QdrantClient,
    records: dict[int, dict],
    batch: list[int],
    dry_run: bool = False,
):
    """Set each point's own content_type, grouped by value.

    set_payload applies the same payload dict to many points at once, so points
    sharing a value go in a single call instead of one call per point. The
    stored value is normalised the same way make_point normalises it, so a
    backfilled point and a freshly written one are indistinguishable.
    """
    by_type: dict[str, list[int]] = {}
    for pid in batch:
        value = (records.get(pid) or {}).get("content_type") or ""
        by_type.setdefault(value, []).append(pid)
    if dry_run:
        return
    for value, pids in by_type.items():
        client.set_payload(
            collection_name=config.QDRANT_COLLECTION,
            payload={"content_type": value},
            points=pids,
            wait=True,
        )


def main():
    dry_run = "--dry-run" in sys.argv
    if not config.MYSQL_PASSWORD:
        log("ERROR: MYSQL_PASSWORD not set; refusing to run (would fetch nothing)")
        return 1

    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    try:
        # A stored field is not filterable until it is indexed, and a collection
        # built before this field existed has no index for it. Creating the
        # indexes here means the backfilled values are immediately usable as a
        # content-type filter, with no separate rebuild step.
        # Skipped on --dry-run: creating an index mutates the collection, and a
        # rehearsal that silently changes the schema is worse than no rehearsal.
        if dry_run:
            log("dry-run: no payload written and no payload index created")
        else:
            create_payload_indexes(client)

        # with_body=False: only the content_type column is needed, and body is
        # the largest column in the table.
        records = asyncio.run(fetch_records(with_body=False))
        log(f"fetched {len(records)} MySQL rows for id->content_type mapping")

        total_scrolled = 0
        eligible = 0
        updated = 0
        batch: list[int] = []
        for pid in scroll_points_missing_content_type(client):
            total_scrolled += 1
            if pid in records:
                eligible += 1
                batch.append(pid)
            else:
                log(f"WARNING: id {pid} not in MySQL (skipping)")
            if len(batch) >= BATCH_SIZE:
                set_content_type(client, records, batch, dry_run=dry_run)
                updated += len(batch)
                batch = []
                if updated % (BATCH_SIZE * 25) == 0:
                    log(f"progress: {updated}/{eligible} processed")

        if batch:
            set_content_type(client, records, batch, dry_run=dry_run)
            updated += len(batch)

        if total_scrolled == 0:
            log("nothing to backfill — every point already has a content_type")
            return 0

        log(
            f"scrolled: {total_scrolled} points without content_type; "
            f"{eligible} eligible for update",
        )
    except Exception:
        log(f"ERROR: batch failed:\n{traceback.format_exc()}")
        log("STOPPED — fix the error and rerun (script is idempotent)")
        return 1
    finally:
        client.close()

    verb = "would set" if dry_run else "set"
    log(f"done: {verb} content_type on {updated} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
