"""In-place backfill: store the WHOLE article body in Qdrant payloads and
recompute the sparse (BM25) vector from the full text, for points that were
indexed under the old body cap (6000 chars).

Reads all MySQL rows (whole bodies now that BODY_CHAR_LIMIT/EMBED_CHAR_LIMIT
are raised), scrolls Qdrant for the currently stored body, and for every point
whose stored body differs from the freshly fetched one, sets the new body
payload and updates the sparse vector. Dense vectors are untouched (the dense
embedder ignores body by design), so this is a lightweight in-place update — no
collection rebuild, no re-embedding of unchanged articles.

Idempotent: rerunning after a completed run finds nothing to change. After it
completes, run 'python scripts/update_index.py --init' to re-seed the delta
fingerprints (they include body), so future incremental runs don't flag these
rows as changed.

Usage:
    python scripts/backfill_body.py
"""
import asyncio
import os
import sys
import time
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from update_index import fetch_records

from app.config import config
from app.index_text import compose_sparse_text

BATCH_SIZE = 100


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def scroll_stored_bodies() -> dict[int, str]:
    """id -> currently stored `body` payload for every point."""
    bodies: dict[int, str] = {}
    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    next_offset = None
    while True:
        pts, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=500,
            with_payload=["body"],
            with_vectors=False,
            offset=next_offset,
        )
        for p in pts:
            bodies[p.id] = (p.payload or {}).get("body") or ""
        if next_offset is None:
            break
    return bodies


def main():
    if not config.MYSQL_PASSWORD:
        log("ERROR: MYSQL_PASSWORD not set; refusing to run (would fetch nothing)")
        return 1

    start = time.perf_counter()
    records = asyncio.run(fetch_records())
    log(f"fetched {len(records)} MySQL rows in {time.perf_counter() - start:.1f}s")

    stored = scroll_stored_bodies()
    log(f"scrolled {len(stored)} stored bodies")

    affected = [i for i, rec in records.items() if stored.get(i) != rec["body"]]
    log(f"points needing a body update: {len(affected)}")

    if not affected:
        log("nothing to do")
        return 0

    log(f"loading sparse model {config.SPARSE_MODEL}...")
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

    batches = [affected[i : i + BATCH_SIZE] for i in range(0, len(affected), BATCH_SIZE)]
    updated = 0
    for n, batch in enumerate(batches, 1):
        client = QdrantClient(url=config.QDRANT_URL, timeout=60)
        try:
            sparse_texts = [compose_sparse_text(records[i]) for i in batch]
            svecs = list(sparse_model.embed(sparse_texts))
            vector_points = [
                PointStruct(
                    id=i,
                    vector={
                        "sparse": SparseVector(
                            indices=s.indices.tolist(),
                            values=s.values.tolist(),
                        )
                    },
                )
                for i, s in zip(batch, svecs)
            ]
            client.update_vectors(
                collection_name=config.QDRANT_COLLECTION,
                points=vector_points,
                wait=True,
            )
            for i in batch:
                client.set_payload(
                    collection_name=config.QDRANT_COLLECTION,
                    payload={"body": records[i]["body"]},
                    points=[i],
                    wait=True,
                )
        except Exception as e:
            log(f"ERROR: batch {n} (ids {batch[0]}..{batch[-1]}) failed: {e}")
            log("STOPPED — fix the error and rerun (script is idempotent)")
            return 1
        updated += len(batch)
        if n % 10 == 0 or n == len(batches):
            log(f"progress: {updated}/{len(affected)} ({n}/{len(batches)} batches)")

    log(f"done: updated body + sparse vector on {updated} points in "
        f"{time.perf_counter() - start:.1f}s")
    log("next: python scripts/update_index.py --init  (re-seed delta fingerprints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())