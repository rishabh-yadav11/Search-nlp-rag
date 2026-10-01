"""One-off backfill: populate the `content_type` payload field for Qdrant points
indexed before any write path stored it.

The field was produced on read but persisted by nobody, so every content-type
filter matched nothing. New points carry it; this fixes the points already in the
collection, in place, with no re-embedding -- the payload is metadata only.

Idempotent by stored VALUE, not key presence: a point is skipped when its
`content_type` is not None, so the empty string an article with no content type
legitimately gets is left alone. A stored JSON `null` is treated as missing and
rewritten to "", keeping the KEYWORD-indexed field single-typed.

``--dry-run`` reports what would change and touches nothing: it writes no payload
AND creates no payload index, since indexing is itself a schema change.
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

# Bounded pages/batches keep the working set small on a large collection.
PAGE_SIZE = 500
BATCH_SIZE = 200


def scroll_points_missing_content_type(client: QdrantClient) -> Iterator[int]:
    """Yield point IDs whose payload has no `content_type`, one page at a time."""
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

    set_payload applies one payload dict to many points, so points sharing a
    value go in a single call. The value is normalised exactly as make_point
    normalises it, so a backfilled point and a freshly written one are
    indistinguishable.
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

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
    try:
        # A stored field is not filterable until indexed, and a collection built
        # before this field existed has no index for it -- so index here, making
        # the backfilled values usable as a filter with no separate rebuild.
        if dry_run:
            log("dry-run: no payload written and no payload index created")
        else:
            create_payload_indexes(client)

        # with_body=False: body is the largest column and is not needed here.
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
