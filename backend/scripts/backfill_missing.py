"""Backfill: bring Qdrant in sync with the DB by adding missing points.

When the collection silently loses points (e.g. 60k vs 67k rows), the
incremental sync's reconcile warns but does not re-add them (they are not
"new" relative to its state). This script scrolls all Qdrant point ids, diffs
against the published DB rows, and embeds + upserts only the missing articles
(a surgical fix — no full rebuild, so the ~30-60 min reindex is avoided).

Run from `backend/` with the venv python (needs MySQL + Qdrant reachable):

    ./venv/bin/python scripts/backfill_missing.py

Safe to re-run: idempotent (only upserts ids absent from Qdrant). Does not
remove extra points; run `update_index.py --init` + reconcile to inspect extras.
"""
import asyncio
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

sys.path.insert(0, os.path.abspath("."))

import aiomysql

from app.config import config
from app.index_text import EXTERNAL_URL_SQL, compose_dense_text, compose_sparse_text, record_from_row


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


async def fetch_records() -> dict[int, dict]:
    pool = await aiomysql.create_pool(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT,
        user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE, autocommit=True, minsize=1, maxsize=3,
    )
    query = f"""
        SELECT feid, title, summary, body, slug, {EXTERNAL_URL_SQL} AS ext_url,
               publish, content_type, author_names, industry_names, dealtype_names
        FROM {config.MYSQL_TABLE} WHERE status = 1
    """
    records: dict[int, dict] = {}
    try:
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query)
            async for row in cur:
                rec = record_from_row(row)
                records[rec["id"]] = rec
    finally:
        pool.close()
        await pool.wait_closed()
    return records


def qdrant_ids() -> set[int]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=config.QDRANT_URL, timeout=30)
    try:
        ids: set[int] = set()
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=500, offset=offset, with_payload=False, with_vectors=False,
            )
            ids.update(p.id for p in batch)
            if offset is None:
                break
        return ids
    finally:
        client.close()


def backfill(records: dict[int, dict], missing: list[int]):
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)
    batch_size = config.EMBED_BATCH_SIZE

    def build_point(rec, dvec, svec):
        return PointStruct(
            id=rec["id"],
            vector={
                "dense": dvec.tolist(),
                "sparse": SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
            },
            payload={
                "title": rec["title"],
                "url": rec["url"],
                "published_date": rec.get("published_date"),
                "category": rec.get("category"),
                "summary": rec.get("summary") or "",
                "body": (rec.get("body") or "")[: config.BODY_CHAR_LIMIT],
                "author_names": rec.get("author_names") or [],
                "industry_names": rec.get("industry_names") or [],
                "dealtype_names": rec.get("dealtype_names") or [],
            },
        )

    executor = ThreadPoolExecutor(max_workers=config.INDEXER_WORKERS)
    pending: deque = deque()
    done = 0

    def submit(batch):
        dense_texts = [compose_dense_text(r) for r in batch]
        sparse_texts = [compose_sparse_text(r) for r in batch]
        pending.append((batch, executor.submit(
            lambda dt, st: (model.encode(dt, batch_size=len(dt), normalize_embeddings=True, show_progress_bar=False),
                            list(sparse_model.embed(st))),
            dense_texts, sparse_texts)))

    def upsert_one():
        nonlocal done
        batch, future = pending.popleft()
        dense_vecs, sparse_vecs = future.result()
        points = [build_point(r, d, s) for r, d, s in zip(batch, dense_vecs, sparse_vecs)]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=True)
        done += len(points)
        log(f"upserted {len(points)} points (running total {done})")

    try:
        for start in range(0, len(missing), batch_size):
            batch = [records[i] for i in missing[start:start + batch_size]]
            submit(batch)
            while len(pending) >= config.INDEXER_WORKERS:
                upsert_one()
        while pending:
            upsert_one()
    finally:
        executor.shutdown(wait=False)
        client.close()


def reconcile(records: dict[int, dict]):
    ids = qdrant_ids()
    db_ids = set(records)
    missing = db_ids - ids
    extra = ids - db_ids
    log(f"reconcile: Qdrant {len(ids)} points vs DB {len(db_ids)} rows "
        f"({len(missing)} missing, {len(extra)} extra)")
    return missing, extra


def main() -> None:
    start = time.perf_counter()
    records = asyncio.run(fetch_records())
    log(f"fetched {len(records)} published rows")
    ids = qdrant_ids()
    missing = sorted(set(records) - ids)
    extra = sorted(ids - set(records))
    log(f"Qdrant has {len(ids)} points; DB has {len(records)} rows; "
        f"{len(missing)} missing, {len(extra)} extra")
    if not missing:
        log("nothing to backfill")
    else:
        log(f"backfilling {len(missing)} missing articles...")
        backfill(records, missing)
    reconcile(records)
    log(f"done in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
