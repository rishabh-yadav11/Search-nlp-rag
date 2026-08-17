"""One-off backfill: populate the `summary` payload field for Qdrant points
that were indexed before summaries were stored (empty summary). Reads the same
MySQL rows as the indexer, scrolls Qdrant for points with an empty summary,
and sets the payload in small batches with wait=True, using a fresh client per
batch so a long-lived connection can't die mid-run. Idempotent; safe to rerun.
"""
import asyncio
import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from update_index import fetch_records

from app.config import config

BATCH_SIZE = 200


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def scroll_empty_points() -> list[int]:
    """All point IDs whose summary payload is empty/absent."""
    ids: list[int] = []
    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
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
                ids.append(p.id)
        if next_offset is None:
            break
    return ids


def set_summary(records: dict[int, dict], batch: list[int]):
    """Set each point's own summary. set_payload applies one dict to many
    points, so a single client+per-point calls keep the payloads distinct."""
    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    for pid in batch:
        client.set_payload(
            collection_name=config.QDRANT_COLLECTION,
            payload={"summary": records.get(pid, {}).get("summary") or ""},
            points=[pid],
            wait=True,
        )


def main():
    if not config.MYSQL_PASSWORD:
        log("ERROR: MYSQL_PASSWORD not set; refusing to run (would fetch nothing)")
        return 1

    empty_ids = scroll_empty_points()
    log(f"scrolled: {len(empty_ids)} points with empty summary")

    if not empty_ids:
        log("nothing to backfill")
        return 0

    records = asyncio.run(fetch_records())
    log(f"fetched {len(records)} MySQL rows for id->summary mapping")

    batches = [empty_ids[i : i + BATCH_SIZE] for i in range(0, len(empty_ids), BATCH_SIZE)]
    updated = 0
    for n, batch in enumerate(batches, 1):
        missing = [i for i in batch if i not in records]
        if missing:
            log(f"WARNING: batch {n} has {len(missing)} ids not in MySQL (skipping them)")
            batch = [i for i in batch if i in records]
        if not batch:
            continue
        try:
            set_summary(records=records, batch=batch)
            updated += len(batch)
        except Exception as e:
            log(f"ERROR: batch {n} (ids {batch[0]}..{batch[-1]}) failed: {e}")
            log("STOPPED — fix the error and rerun (script is idempotent)")
            return 1
        if n % 25 == 0 or n == len(batches):
            log(f"progress: {updated}/{len(empty_ids)} ({n}/{len(batches)} batches)")

    log(f"done: set summary on {updated} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
