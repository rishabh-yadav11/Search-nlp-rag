"""One-off backfill: populate the `summary` payload field for Qdrant points
indexed before summaries were stored.

Reads the same MySQL rows as the indexer, scrolls Qdrant for points with an empty
summary, and sets the payload in small batches. Idempotent; safe to rerun.
"""
import asyncio
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections.abc import Iterator

from _common import log
from qdrant_client import QdrantClient
from update_index import fetch_records

from app.config import config

BATCH_SIZE = 200


def scroll_empty_points(client: QdrantClient) -> Iterator[int]:
    """Yield point IDs whose summary payload is empty/absent, one page at a time."""
    next_offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=500,
            with_payload=["summary"],
            with_vectors=False,
            offset=next_offset,
        )
        for p in pts:
            s = (p.payload or {}).get("summary") or ""
            if not s.strip():
                yield p.id
        if next_offset is None:
            break


def set_summary(client: QdrantClient, records: dict[int, dict], batch: list[int]):
    """Set each point's own summary.

    set_payload applies one payload dict to many points, so points sharing a
    summary value go in a single call.
    """
    by_summary: dict[str, list[int]] = {}
    for pid in batch:
        s = records.get(pid, {}).get("summary") or ""
        by_summary.setdefault(s, []).append(pid)
    for summary, pids in by_summary.items():
        client.set_payload(
            collection_name=config.QDRANT_COLLECTION,
            payload={"summary": summary},
            points=pids,
            wait=True,
        )


def main():
    if not config.MYSQL_PASSWORD:
        log("ERROR: MYSQL_PASSWORD not set; refusing to run (would fetch nothing)")
        return 1

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
    try:
        records = asyncio.run(fetch_records())
        log(f"fetched {len(records)} MySQL rows for id->summary mapping")

        total_scrolled = 0
        eligible = 0
        updated = 0
        batch: list[int] = []
        for pid in scroll_empty_points(client):
            total_scrolled += 1
            if pid in records:
                eligible += 1
                batch.append(pid)
            else:
                log(f"WARNING: id {pid} not in MySQL (skipping)")
            if len(batch) >= BATCH_SIZE:
                set_summary(client=client, records=records, batch=batch)
                updated += len(batch)
                batch = []
                if updated % (BATCH_SIZE * 25) == 0 or updated == eligible:
                    log(f"progress: {updated}/{eligible} updated")

        if batch:
            set_summary(client=client, records=records, batch=batch)
            updated += len(batch)

        if total_scrolled == 0:
            log("nothing to backfill")
            return 0

        log(f"scrolled: {total_scrolled} points with empty summary; {eligible} eligible for update")
    except Exception:
        log(f"ERROR: batch failed:\n{traceback.format_exc()}")
        log("STOPPED — fix the error and rerun (script is idempotent)")
        return 1
    finally:
        client.close()

    log(f"done: set summary on {updated} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
