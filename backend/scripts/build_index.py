"""
Reads data/articles.jsonl, embeds in batches (dense + BM25 sparse for hybrid
search), and upserts into Qdrant. Checkpoints progress so a crash/interrupt
can resume without re-embedding everything.

Dense encoding and upserting are pipelined across thread pool workers
(INDEXER_WORKERS) so encode of a later batch overlaps upsert of an earlier one.

Usage:
    python scripts/build_index.py
"""
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer
from qdrant_client.models import models as qmodels
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import config
from app.index_text import compose_index_text

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "articles.jsonl")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", ".checkpoint")


def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return int(f.read().strip() or 0)
    return 0


def save_checkpoint(line_num: int):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(line_num))


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
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

    def encode_batch(batch_texts):
        dense_vecs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_vecs = list(sparse_model.embed(batch_texts))
        return dense_vecs, sparse_vecs

    batch_rows, batch_texts = [], []
    pending = deque()  # (end_line, future of encode_batch)

    def submit_batch(end_line: int):
        pending.append((end_line, executor.submit(encode_batch, batch_texts)))

    def upsert_and_checkpoint():
        if not pending:
            return
        end_line, future = pending.popleft()
        dense_vecs, sparse_vecs = future.result()
        points = [to_point(row, dvec, svec) for row, dvec, svec in zip(batch_frames[end_line], dense_vecs, sparse_vecs)]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=False)
        save_checkpoint(end_line)
        pbar.update(len(batch_frames[end_line]))

    batch_frames = {}
    executor = ThreadPoolExecutor(max_workers=config.INDEXER_WORKERS)
    pbar = tqdm(total=total - start_line, desc="Embedding + indexing")

    try:
        for i, line in enumerate(lines):
            if i < start_line:
                continue
            row = json.loads(line)
            batch_rows.append(row)
            batch_texts.append(compose_index_text(row))

            if len(batch_rows) >= config.EMBED_BATCH_SIZE:
                batch_frames[i + 1] = batch_rows
                submit_batch(i + 1)
                batch_rows, batch_texts = [], []
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
    print("Index build complete.")


if __name__ == "__main__":
    main()