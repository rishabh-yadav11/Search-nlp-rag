"""
Reads data/articles.jsonl, embeds in batches (dense + BM25 sparse for hybrid
search), and upserts into Qdrant. Checkpoints progress so a crash/interrupt
can resume without re-embedding everything.

Dense encoding and upserting are pipelined across thread pool workers
(INDEXER_WORKERS) so encode of a later batch overlaps upsert of an earlier one.

Durability & backups:
  * Upserts are acknowledged (wait=True) and the checkpoint is advanced only
    after a successful upsert, so a partially-written batch is re-processed
    from the last saved checkpoint on the next run.
  * Before the collection is deleted/recreated (incompatible schema), a
    best-effort snapshot backup is taken via qdrant_backup.make_backup();
    a backup failure only logs a WARNING and does not abort the build.
  * A versioned build collection + alias switch was deliberately NOT
    implemented: the backup-first approach above is simpler, carries no risk of
    breaking a live collection/alias, and update_index.py keeps operating on
    config.QDRANT_COLLECTION as-is.

Usage:
    python scripts/build_index.py
"""
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import models as qmodels
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import config
from app.index_text import compose_dense_text, compose_sparse_text

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "articles.jsonl")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", ".checkpoint")


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return int(f.read().strip() or 0)
    return 0


def save_checkpoint(line_num: int):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(line_num))


def backup_collection_best_effort(client: QdrantClient):
    """Snapshot the collection + local artifacts before a destructive change.

    Best-effort: a failure only logs a WARNING and the build continues, so a
    transient Qdrant outage cannot block a rebuild. No-op when the collection
    does not exist (nothing to protect).
    """
    try:
        from qdrant_backup import make_backup

        existing = [c.name for c in client.get_collections().collections]
        if config.QDRANT_COLLECTION not in existing:
            return
        make_backup(client, config.QDRANT_COLLECTION)
    except Exception as e:
        log(f"WARNING: best-effort backup before collection change failed: {e}")


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        backup_collection_best_effort(client)
        create_collection(client)
        return

    info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
    # Introspection is tolerant: some qdrant server/client combinations don't
    # surface sparse_vectors_config/hnsw_config on the returned model. When we
    # can't confirm the existing schema, recreate it so vectors match config.
    vectors = info.config.params.vectors
    dense_size = vectors.get("dense").size if isinstance(vectors, dict) else vectors.size
    dim_matches = dense_size == config.EMBED_DIM

    sparse_cfg = getattr(info.config, "sparse_vectors_config", None)
    if sparse_cfg is not None:
        sparse_cfg = sparse_cfg.sparse if hasattr(sparse_cfg, "sparse") else sparse_cfg.get("sparse")
    needs_idf = bool(sparse_cfg and sparse_cfg.modifier == Modifier.IDF)

    if needs_idf and dim_matches:
        print(f"Collection '{config.QDRANT_COLLECTION}' already exists, resuming upserts into it")
        return

    print(
        f"Collection '{config.QDRANT_COLLECTION}' exists but is incompatible "
        f"(dense size={dense_size} vs {config.EMBED_DIM}, sparse modifier idf={needs_idf}). "
        f"Deleting and recreating it so indexed vectors match the current config.",
    )
    backup_collection_best_effort(client)
    client.delete_collection(collection_name=config.QDRANT_COLLECTION)
    create_collection(client)


def create_collection(client: QdrantClient):
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={"dense": VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        hnsw_config=qmodels.HnswConfigDiff(m=32, ef_construct=256),
    )
    client.create_payload_index(config.QDRANT_COLLECTION, "category", PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.QDRANT_COLLECTION, "published_date", PayloadSchemaType.DATETIME)
    client.create_payload_index(config.QDRANT_COLLECTION, "author_names", PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.QDRANT_COLLECTION, "industry_names", PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.QDRANT_COLLECTION, "dealtype_names", PayloadSchemaType.KEYWORD)
    print(f"Created collection '{config.QDRANT_COLLECTION}'")


def to_point(row: dict, dvec, svec) -> PointStruct:
    return PointStruct(
        id=row["id"],
        vector={
            "dense": dvec.tolist(),
            "sparse": SparseVector(indices=svec.indices.tolist(), values=svec.values.tolist()),
        },
        payload={
            "title": row["title"],
            "url": row["url"],
            "published_date": row.get("published_date"),
            "category": row.get("category"),
            "summary": (row.get("summary") or ""),
            "body": (row.get("body") or "")[: config.BODY_CHAR_LIMIT],
            "author_names": row.get("author_names") or [],
            "industry_names": row.get("industry_names") or [],
            "dealtype_names": row.get("dealtype_names") or [],
        },
    )


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data file at {DATA_PATH} — run scripts/fetch_data.py first.")
        return

    print(f"Loading dense embedding model {config.EMBED_MODEL} on {config.EMBED_DEVICE}...")
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    print(f"Loading sparse embedding model {config.SPARSE_MODEL}...")
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    ensure_collection(client)

    start_line = load_checkpoint()

    with open(DATA_PATH) as f:
        lines = f.readlines()
    total = len(lines)
    print(f"{total} articles in dataset, resuming from line {start_line}")

    def encode_batch(dense_texts, sparse_texts):
        dense_vecs = model.encode(
            dense_texts,
            batch_size=len(dense_texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_vecs = list(sparse_model.embed(sparse_texts))
        return dense_vecs, sparse_vecs

    batch_rows, dense_texts, sparse_texts = [], [], []
    pending = deque()  # (end_line, future of encode_batch)

    def submit_batch(end_line: int):
        pending.append((end_line, executor.submit(encode_batch, dense_texts, sparse_texts)))

    def upsert_and_checkpoint():
        if not pending:
            return
        end_line, future = pending.popleft()
        rows = batch_frames.pop(end_line)
        dense_vecs, sparse_vecs = future.result()
        points = [to_point(row, dvec, svec) for row, dvec, svec in zip(rows, dense_vecs, sparse_vecs)]
        try:
            client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=True)
        except Exception as e:
            log(
                f"ERROR: upsert of batch ending line {end_line} failed; checkpoint NOT "
                f"advanced (batch will be retried from the last saved checkpoint): {e}",
            )
            raise
        save_checkpoint(end_line)
        pbar.update(len(rows))

    batch_frames = {}
    executor = ThreadPoolExecutor(max_workers=config.INDEXER_WORKERS)
    pbar = tqdm(total=total - start_line, desc="Embedding + indexing")

    try:
        for i, line in enumerate(lines):
            if i < start_line:
                continue
            row = json.loads(line)
            batch_rows.append(row)
            dense_texts.append(compose_dense_text(row))
            sparse_texts.append(compose_sparse_text(row))

            if len(batch_rows) >= config.EMBED_BATCH_SIZE:
                batch_frames[i + 1] = batch_rows
                submit_batch(i + 1)
                batch_rows, dense_texts, sparse_texts = [], [], []
                while len(pending) >= config.INDEXER_WORKERS:
                    upsert_and_checkpoint()

        if batch_rows:
            batch_frames[len(lines)] = batch_rows
            submit_batch(len(lines))

        while pending:
            upsert_and_checkpoint()
    finally:
        executor.shutdown(wait=False)

    pbar.close()
    processed = total - start_line
    try:
        info = client.get_collection(config.QDRANT_COLLECTION)
        count = info.points_count or 0
        if count < processed:
            log(
                f"WARNING: collection '{config.QDRANT_COLLECTION}' has {count} points "
                f"but {processed} lines were processed",
            )
        else:
            log(f"verified: {count} points in collection >= {processed} lines processed")
    except Exception as e:
        log(f"WARNING: could not verify points_count: {e}")
    print("Index build complete.")


if __name__ == "__main__":
    main()