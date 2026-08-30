"""
Incremental sync of the MySQL source table into the Qdrant index.

Detects NEW, CHANGED, and DELETED articles since the last run and applies the
delta to Qdrant without touching the collection schema, so a running API is
unaffected (Qdrant handles concurrent reads/writes; points are only appended
or removed, never bulk-recreated).

State lives in data/index_state.json: {updated_at, fingerprints}.
A fingerprint is the md5 of the *indexed* row values (title, summary, url,
published_date, category, body and the facet lists), so any edit that would
change the payload or the embedded text is caught.

Durability & reconciliation:
  * Upserts are acknowledged (wait=True); state fingerprints are
    updated only after a successful upsert, so a failed batch is retried from
    the previous state on the next run.
  * reconcile() scrolls all point IDs in the collection and compares them to
    the DB row id set after every run (normal and --init), logging a WARNING
    with sample missing/extra IDs on any mismatch.

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
from datetime import UTC, datetime

import aiomysql

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config
from app.index_text import EXTERNAL_URL_SQL, compose_dense_text, compose_sparse_text, record_from_row

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_PATH = os.path.join(DATA_DIR, "index_state.json")
LOCK_PATH = os.path.join(DATA_DIR, "update.lock")


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"updated_at": None, "fingerprints": {}}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def fingerprint(rec: dict, include_body: bool = True) -> str:
    body = (rec.get("body") or "") if include_body else ""
    raw = "|".join(
        [
            rec.get("title") or "",
            rec.get("summary") or "",
            rec.get("url") or "",
            rec.get("published_date") or "",
            rec.get("category") or "",
            body,
            ",".join(rec.get("author_names") or []),
            ",".join(rec.get("industry_names") or []),
            ",".join(rec.get("dealtype_names") or []),
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


async def fetch_records(
    with_body: bool = True, ids: list[int] | None = None
) -> dict[int, dict]:
    """Published rows mapped to the canonical indexed record.

    `with_body=False` omits the (potentially large) body column for callers that
    only need ids/metadata (e.g. delta detection), and `ids` restricts the
    result set to a specific set of article ids (used to fetch full bodies only
    for the rows that actually need (re)indexing).
    """
    pool = await aiomysql.create_pool(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE,
        autocommit=True,
        minsize=1,
        maxsize=3,
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
    )
    body_select = "body," if with_body else ""
    where = "WHERE status = 1"
    params: list = []
    if ids is not None:
        placeholders = ",".join(["%s"] * len(ids))
        where = f"WHERE status = 1 AND feid IN ({placeholders})"
        params = list(ids)
    query = f"""
        SELECT
            feid,
            title,
            summary,
            {body_select}
            slug,
            {EXTERNAL_URL_SQL} AS ext_url,
            publish,
            content_type,
            author_names,
            industry_names,
            dealtype_names
        FROM {config.MYSQL_TABLE}
        {where}
    """
    records: dict[int, dict] = {}
    try:
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, params)
            async for row in cur:
                if not with_body:
                    row["body"] = ""
                rec = record_from_row(row)
                records[rec["id"]] = rec
    finally:
        pool.close()
        await pool.wait_closed()
    return records


def sync_delta(state: dict, records: dict[int, dict], include_body: bool = True):
    state_fps = state.setdefault("fingerprints", {})
    state_ids: set[int] = set()
    for k in state_fps:
        try:
            state_ids.add(int(k))
        except (ValueError, TypeError):
            continue
    db_ids = set(records)

    new = db_ids - state_ids
    changed = {
        i
        for i in db_ids & state_ids
        if state_fps.get(str(i)) != fingerprint(records[i], include_body=include_body)
    }
    deleted = state_ids - db_ids
    return new, changed, deleted


def apply_delta(records: dict[int, dict], new: set, changed: set, deleted: set, state: dict):
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector
    from sentence_transformers import SentenceTransformer

    client = None
    try:
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
            save_state(state)

        to_index = new | changed
        if not to_index:
            return

        log(f"loading models (indexing {len(to_index)} rows)...")
        model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
        sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

        state_fps = state["fingerprints"]
        to_index = sorted(to_index)
        batch_size = config.EMBED_BATCH_SIZE

        def encode_batch(dense_texts, sparse_texts):
            dense_vecs = model.encode(
                dense_texts,
                batch_size=len(dense_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sparse_vecs = list(sparse_model.embed(sparse_texts))
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
                    "summary": rec.get("summary") or "",
                    "body": (rec.get("body") or "")[: config.BODY_CHAR_LIMIT],
                    "author_names": rec.get("author_names") or [],
                    "industry_names": rec.get("industry_names") or [],
                    "dealtype_names": rec.get("dealtype_names") or [],
                },
            )

        max_workers = max(1, config.INDEXER_WORKERS)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        pending = deque()

        def submit(batch):
            dense_texts = [compose_dense_text(r) for r in batch]
            sparse_texts = [compose_sparse_text(r) for r in batch]
            pending.append(
                (batch, executor.submit(encode_batch, dense_texts, sparse_texts))
            )

        def drain_one():
            """Gather one pending future, upsert it, and surface per-batch errors.

            A failed batch is logged and skipped (its state stays unchanged so it
            is retried next run) rather than aborting the whole index update.
            """
            batch, future = pending.popleft()
            try:
                dense_vecs, sparse_vecs = future.result()
            except Exception as e:
                log(
                    f"ERROR: encoding batch ending at id {batch[-1]['id']} failed; "
                    f"skipped (will be retried next run): {e}",
                )
                return
            points = [
                build_point(r, dvec, svec)
                for r, dvec, svec in zip(batch, dense_vecs, sparse_vecs)
            ]
            try:
                client.upsert(
                    collection_name=config.QDRANT_COLLECTION, points=points, wait=True
                )
            except Exception as e:
                log(
                    f"ERROR: upsert of batch ending at id {batch[-1]['id']} failed; "
                    f"state NOT updated (batch will be retried next run): {e}",
                )
                return

            for r in batch:
                state_fps[str(r["id"])] = fingerprint(r)
            state["updated_at"] = datetime.now(UTC).isoformat()
            save_state(state)
            log(f"upserted {len(batch)} points (max_id={batch[-1]['id']})")

        try:
            for start in range(0, len(to_index), batch_size):
                batch = [records[i] for i in to_index[start : start + batch_size]]
                submit(batch)
                # Bound in-flight work to the worker count so memory and
                # concurrency stay bounded even when encoding outpaces upserts.
                while len(pending) >= max_workers:
                    drain_one()
            while pending:
                drain_one()
        finally:
            # Wait for any still-running workers so no futures are dropped.
            executor.shutdown(wait=True)
    finally:
        if client is not None:
            client.close()


def reconcile(state: dict, records: dict[int, dict]) -> bool:
    """Compare the full Qdrant point-id set to the DB row id set.

    Scrolls the whole collection in batches of 500 with payloads and vectors
    disabled (bounded, fast). Logs a WARNING listing the count mismatch plus
    sample missing/extra IDs. Returns True when in sync, False on any mismatch.
    Never raises.
    """
    from qdrant_client import QdrantClient

    client = None
    try:
        client = QdrantClient(url=config.QDRANT_URL, timeout=30)
        point_ids = set()
        next_offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=500,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.update(str(p.id) for p in batch)
            if next_offset is None:
                break

        db_ids = {str(i) for i in records}
        missing = db_ids - point_ids
        extra = point_ids - db_ids
        if missing or extra:
            log(
                f"WARNING: reconcile mismatch — Qdrant has {len(point_ids)} points "
                f"but DB has {len(db_ids)} rows ({len(missing)} missing, {len(extra)} extra)",
            )
            for i in sorted(missing)[:10]:
                log(f"  MISSING in Qdrant: {i}")
            for i in sorted(extra)[:10]:
                log(f"  EXTRA in Qdrant (not in DB): {i}")
            return False
        log(f"reconcile OK: Qdrant {len(point_ids)} points match DB {len(db_ids)} rows")
        return True
    except Exception as e:
        log(f"WARNING: reconcile failed: {e}")
        return False
    finally:
        if client is not None:
            client.close()


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

    # Fetch the FULL indexed record (including body) for every published row.
    # Body must be in the fingerprint so body-only edits are detected as
    # changes; this also derives the indexing set (to_index) and the reconcile
    # check from the SAME fetch, eliminating the TOCTOU race where a row
    # unpublished between two fetches would KeyError.
    records = await fetch_records(with_body=True)
    log(f"fetched {len(records)} published rows (with body)")

    if do_init:
        state_fps = {str(i): fingerprint(records[i]) for i in records}
        state = {
            "updated_at": datetime.now(UTC).isoformat(),
            "fingerprints": state_fps,
        }
        save_state(state)
        log(f"seeded {len(state_fps)} fingerprints")
        return 0 if reconcile(state, records) else 1

    new, changed, deleted = sync_delta(state, records)
    log(
        f"delta: {len(new)} new, {len(changed)} changed, {len(deleted)} deleted, "
        f"{len(records) - len(new) - len(changed) - len(deleted)} unchanged"
    )

    if not new and not changed and not deleted:
        log(f"index current ({time.perf_counter() - start:.2f}s, models not loaded)")
        return 0 if reconcile(state, records) else 1

    # apply_delta derives its indexing set from `records`, the same fetch used
    # to build the points, so a row can never be selected for indexing but
    # missing from the index source (no KeyError / TOCTOU).
    apply_delta(records, new, changed, deleted, state)
    log(f"done in {time.perf_counter() - start:.2f}s")
    return 0 if reconcile(state, records) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))