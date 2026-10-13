
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

PAGE_SIZE = 500
BATCH_SIZE = 200


def scroll_points_missing_content_type(client: QdrantClient) -> Iterator[int]:

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



        if dry_run:
            log("dry-run: no payload written and no payload index created")
        else:
            create_payload_indexes(client)


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
