"""In-place backfill: store the article body (truncated to BODY_CHAR_LIMIT) in
Qdrant payloads and recompute the sparse (BM25) vector from that text, for
points whose stored body differs from the freshly fetched, consistently
truncated one.

Reads all MySQL rows, truncates each body to BODY_CHAR_LIMIT (the same cap every
index path uses), scrolls Qdrant for the currently stored body, and for every
point whose stored body differs from the freshly fetched one, sets the new body
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
from qdrant_client.models import PointVectors, SparseVector
from update_index import fetch_records

from app.config import config
from app.index_text import compose_sparse_text

BATCH_SIZE = 100


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def scroll_stored_bodies(client: QdrantClient) -> dict[int, str]:
    """id -> currently stored `body` payload for every point."""
    bodies: dict[int, str] = {}
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

    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    try:
        stored = scroll_stored_bodies(client)
        log(f"scrolled {len(stored)} stored bodies")

        # Only update points that already exist in Qdrant (this is an in-place
        # backfill). Rows whose id is absent from Qdrant are out of scope here
        # (they get seeded by the normal index path), so excluding them prevents
        # a massive unintended backfill when Qdrant is empty or partially built.
        affected = [
            i
            for i, rec in records.items()
            if i in stored and stored[i] != rec["body"]
        ]
        log(f"points needing a body update: {len(affected)}")

        if not affected:
            log("nothing to do")
            return 0

        log(f"loading sparse model {config.SPARSE_MODEL}...")
        sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

        batches = [affected[i : i + BATCH_SIZE] for i in range(0, len(affected), BATCH_SIZE)]
        updated = 0
        for n, batch in enumerate(batches, 1):
            sparse_texts = [compose_sparse_text(records[i]) for i in batch]
            svecs = list(sparse_model.embed(sparse_texts))
            vector_points = [
                PointVectors(
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
                    payload={"body": records[i]["body"][: config.BODY_CHAR_LIMIT]},
                    points=[i],
                    wait=True,
                )
            updated += len(batch)
            if n % 10 == 0 or n == len(batches):
                log(f"progress: {updated}/{len(affected)} ({n}/{len(batches)} batches)")
    finally:
        client.close()

    log(f"done: updated body + sparse vector on {updated} points in "
        f"{time.perf_counter() - start:.1f}s")
    log("next: python scripts/update_index.py --init  (re-seed delta fingerprints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())