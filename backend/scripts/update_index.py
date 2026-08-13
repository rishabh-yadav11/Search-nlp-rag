"""
Incremental sync of the MySQL source table into the Qdrant index.

Detects NEW, CHANGED, and DELETED articles since the last run and applies the
delta to Qdrant without touching the collection schema, so a running API is
unaffected (Qdrant handles concurrent reads/writes; points are only appended
or removed, never bulk-recreated).

State lives in data/index_state.json: {last_id, updated_at, fingerprints}.
A fingerprint is the md5 of the *indexed* row values (cleaned title, summary,
url, published_date, category), so any edit that would change the payload or
the embedded text is caught.

Usage:
    python scripts/update_index.py            # scheduled run (safe no-op when current)
    python scripts/update_index.py --init     # seed state from current DB rows (no embedding)
"""
import asyncio
import fcntl
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import aiomysql

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import config
from app.index_text import compose_index_text, normalize_date, split_names
from fetch_data import clean

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_PATH = os.path.join(DATA_DIR, "index_state.json")
LOCK_PATH = os.path.join(DATA_DIR, "update.lock")

_EXT = "COALESCE(NULLIF(external_url, ''), NULLIF(canonical_url, ''))"


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_id": 0, "updated_at": None, "fingerprints": {}}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def fingerprint(rec: dict) -> str:
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


async def fetch_records() -> dict[int, dict]:
    """All published rows, mapped to the canonical indexed record."""
    pool = await aiomysql.create_pool(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE,
        autocommit=True,
        minsize=1,
        maxsize=3,
    )
    query = f"""
        SELECT
            feid,
            title,
            summary,
            body,
            slug,
            {_EXT} AS ext_url,
            publish,
            content_type,
            author_names,
            industry_names,
            dealtype_names
        FROM {config.MYSQL_TABLE}
        WHERE status = 1
    """
    records: dict[int, dict] = {}
    try:
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query)
                async for row in cur:
                    rec = {
                        "id": row["feid"],
                        "title": clean(row["title"]),
                        "summary": clean(row["summary"]),
                        "body": clean(row["body"])[: config.BODY_CHAR_LIMIT],
                        "url": row["ext_url"] or f"https://www.vccircle.com/{row['slug'] or row['feid']}",
                        "published_date": normalize_date(row["publish"]),
                        "category": (row["dealtype_names"] or row["content_type"] or "").strip(),
                        "author_names": split_names(row["author_names"]),
                        "industry_names": split_names(row["industry_names"]),
                        "dealtype_names": split_names(row["dealtype_names"]),
                    }
                    records[rec["id"]] = rec
    finally:
        pool.close()
        await pool.wait_closed()
    return records


def sync_delta(state: dict, records: dict[int, dict]):
    state_fps = state.setdefault("fingerprints", {})
    state_ids = set(int(k) for k in state_fps)
    db_ids = set(records)

    new = db_ids - state_ids
    changed = {i for i in db_ids & state_ids if state_fps[str(i)] != fingerprint(records[i])}
    deleted = state_ids - db_ids
    return new, changed, deleted


def apply_delta(records: dict[int, dict], new: set, changed: set, deleted: set, state: dict):
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(url=config.QDRANT_URL, timeout=60)

    if deleted:
        client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=sorted(deleted),
            wait=True,
        )
        for i in deleted:
            state["fingerprints"].pop(str(i), None)
        log(f"deleted {len(deleted)} points")

    to_index = new | changed
    if not to_index:
        return

    log(f"loading models (indexing {len(to_index)} rows)...")
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

    state_fps = state.setdefault("fingerprints", {})
    last_id = state.get("last_id", 0)
    to_index = sorted(to_index)
    batch_size = config.EMBED_BATCH_SIZE

    def encode_batch(texts):
        dense_vecs = model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_vecs = list(sparse_model.embed(texts))
        return dense_vecs, sparse_vecs

    def build_point(rec, dvec, svec):
        return PointStruct(
            id=rec["id"],
            vector={
                "dense": dvec.tolist(),
                "sparse": SparseVector(
                    indices=svec.indices.tolist(),
                    values=svec.values.tolist(),
                ),
            },
            payload={
                "title": rec["title"],
                "url": rec["url"],
                "published_date": rec.get("published_date"),
                "category": rec.get("category"),
                "body": (rec.get("body") or "")[: config.BODY_CHAR_LIMIT],
                "author_names": rec.get("author_names") or [],
                "industry_names": rec.get("industry_names") or [],
                "dealtype_names": rec.get("dealtype_names") or [],
            },
        )

    executor = ThreadPoolExecutor(max_workers=config.INDEXER_WORKERS)
    pending = deque()

    def submit(batch):
        pending.append((batch, executor.submit(encode_batch, [compose_index_text(r) for r in batch])))

    def upsert_one():
        nonlocal last_id
        batch, future = pending.popleft()
        dense_vecs, sparse_vecs = future.result()
        points = [build_point(r, dvec, svec) for r, dvec, svec in zip(batch, dense_vecs, sparse_vecs)]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=False)

        for r in batch:
            state_fps[str(r["id"])] = fingerprint(r)
        last_id = max(last_id, max(r["id"] for r in batch))
        state["last_id"] = last_id
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        log(f"upserted {len(batch)} points (last_id={last_id})")

    try:
        for start in range(0, len(to_index), batch_size):
            batch = [records[i] for i in to_index[start : start + batch_size]]
            submit(batch)
            while len(pending) >= config.INDEXER_WORKERS:
                upsert_one()
        while pending:
            upsert_one()
    finally:
        executor.shutdown(wait=False)


def verify_state(state: dict, records: dict[int, dict]):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=config.QDRANT_URL, timeout=30)
    try:
        info = client.get_collection(config.QDRANT_COLLECTION)
        count = info.points_count
        if count != len(records):
            log(f"WARNING: Qdrant has {count} points but DB has {len(records)} rows")
        else:
            log(f"collection '{config.QDRANT_COLLECTION}' matches DB rows ({count})")
    except Exception as e:
        log(f"WARNING: could not inspect collection: {e}")


async def main():
    if "--init" in sys.argv:
        do_init = True
    else:
        do_init = False

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        lock_fd = open(LOCK_PATH, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another update run is in progress; skipping")
        return

    start = time.perf_counter()
    state = load_state()

    if not do_init and not state.get("fingerprints"):
        log("no state found — run 'python scripts/update_index.py --init' after a full build first")
        return

    records = await fetch_records()
    log(f"fetched {len(records)} published rows")

    if do_init:
        state_fps = {str(i): fingerprint(records[i]) for i in records}
        state = {
            "last_id": max(records) if records else 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprints": state_fps,
        }
        save_state(state)
        log(f"seeded {len(state_fps)} fingerprints (last_id={state['last_id']})")
        verify_state(state, records)
        return

    new, changed, deleted = sync_delta(state, records)
    log(
        f"delta: {len(new)} new, {len(changed)} changed, {len(deleted)} deleted, "
        f"{len(records) - len(new) - len(changed) - len(deleted)} unchanged"
    )

    if not new and not changed and not deleted:
        log(f"index current ({time.perf_counter() - start:.2f}s, models not loaded)")
        return

    apply_delta(records, new, changed, deleted, state)
    log(f"done in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())