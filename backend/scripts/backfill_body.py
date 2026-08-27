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

import aiomysql
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointVectors, SparseVector
from update_index import EXTERNAL_URL_SQL, record_from_row

from app.config import config
from app.index_text import compose_sparse_text

# How many Qdrant points to scroll / MySQL rows to fetch per page. Bounding this
# keeps the working set in memory small even on huge collections/tables.
PAGE_SIZE = 500
VECTOR_BATCH = 100


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


async def make_pool():
    return await aiomysql.create_pool(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE,
        autocommit=True,
        minsize=1,
        maxsize=3,
    )


async def fetch_records_by_ids(pool, ids: list[int]) -> dict[int, dict]:
    """Published rows for the given ids, mapped to the canonical record.

    Only the requested slice of the table is queried (chunked by the caller via
    repeated PAGE_SIZE-sized id lists), so the whole table is never materialized
    in memory at once.
    """
    if not ids:
        return {}
    placeholders = ",".join(["%s"] * len(ids))
    query = f"""
        SELECT
            feid,
            title,
            summary,
            body,
            slug,
            {EXTERNAL_URL_SQL} AS ext_url,
            publish,
            content_type,
            author_names,
            industry_names,
            dealtype_names
        FROM {config.MYSQL_TABLE}
        WHERE status = 1 AND feid IN ({placeholders})
    """
    records: dict[int, dict] = {}
    async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
        await cur.execute(query, list(ids))
        async for row in cur:
            rec = record_from_row(row)
            records[rec["id"]] = rec
    return records


def main():
    if not config.MYSQL_PASSWORD:
        log("ERROR: MYSQL_PASSWORD not set; refusing to run (would fetch nothing)")
        return 1

    start = time.perf_counter()
    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    log(f"loading sparse model {config.SPARSE_MODEL}...")
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

    async def run():
        pool = await make_pool()
        try:
            # Page through Qdrant (bounded memory) and resolve each page's MySQL
            # rows on demand, so neither the collection nor the table is ever
            # loaded whole.
            next_offset = None
            total = 0
            updated = 0
            while True:
                pts, next_offset = client.scroll(
                    collection_name=config.QDRANT_COLLECTION,
                    limit=PAGE_SIZE,
                    with_payload=["body"],
                    with_vectors=False,
                    offset=next_offset,
                )
                total += len(pts)
                if pts:
                    ids = [p.id for p in pts]
                    stored = {
                        p.id: (p.payload or {}).get("body") or "" for p in pts
                    }
                    records = await fetch_records_by_ids(pool, ids)
                    # Only update points that already exist in Qdrant (this is an
                    # in-place backfill). Rows whose id is absent from Qdrant are
                    # out of scope here (they get seeded by the normal index
                    # path), so excluding them prevents a massive unintended
                    # backfill when Qdrant is empty or partially built.
                    affected = [
                        i
                        for i in ids
                        if i in records and stored[i] != records[i]["body"]
                    ]
                    if affected:
                        for b0 in range(0, len(affected), VECTOR_BATCH):
                            chunk = affected[b0 : b0 + VECTOR_BATCH]
                            sparse_texts = [
                                compose_sparse_text(records[i]) for i in chunk
                            ]
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
                                for i, s in zip(chunk, svecs)
                            ]
                            client.update_vectors(
                                collection_name=config.QDRANT_COLLECTION,
                                points=vector_points,
                                wait=True,
                            )
                            # One bulk payload upload for the whole chunk (no
                            # per-point wait=True round-trips).
                            client.upload_payload(
                                collection_name=config.QDRANT_COLLECTION,
                                payload=(
                                    (
                                        i,
                                        {
                                            "body": records[i]["body"][
                                                : config.BODY_CHAR_LIMIT
                                            ]
                                        },
                                    )
                                    for i in chunk
                                ),
                                batch_size=VECTOR_BATCH,
                                wait=True,
                            )
                        updated += len(affected)
                        log(
                            f"progress: {updated} updated so far "
                            f"({len(affected)} in this page)"
                        )
                if next_offset is None:
                    break
            log(f"scrolled {total} stored points")
            return updated
        finally:
            pool.close()
            await pool.wait_closed()

    try:
        updated = asyncio.run(run())
    finally:
        client.close()

    if updated == 0:
        log("nothing to do")
        log(f"done in {time.perf_counter() - start:.1f}s")
        return 0

    log(
        f"done: updated body + sparse vector on {updated} points in "
        f"{time.perf_counter() - start:.1f}s"
    )
    log("next: python scripts/update_index.py --init  (re-seed delta fingerprints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())