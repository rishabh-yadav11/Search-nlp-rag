"""
Reads data/articles.jsonl, embeds in batches (dense + BM25 sparse for hybrid
search), and upserts into Qdrant. Checkpoints progress so a crash/interrupt
can resume without re-embedding everything.

Usage:
    python scripts/build_index.py
"""
import json
import os
import sys

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
    sparse_cfg = info.config.sparse_vectors_config.get("sparse")
    needs_idf = sparse_cfg is not None and sparse_cfg.modifier == Modifier.IDF
    if not needs_idf:
        print(
            f"Collection '{config.QDRANT_COLLECTION}' exists with a mismatched sparse "
            f"configuration (expected modifier=idf). Deleting and recreating it so "
            f"indexed vectors share the fastembed BM25 vocab with queries.",
        )
        client.delete_collection(collection_name=config.QDRANT_COLLECTION, wait=True)
        create_collection(client)
    else:
        print(f"Collection '{config.QDRANT_COLLECTION}' already exists, resuming upserts into it")


def create_collection(client: QdrantClient):
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={"dense": VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128),
    )
    client.create_payload_index(config.QDRANT_COLLECTION, "category", PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.QDRANT_COLLECTION, "published_date", PayloadSchemaType.DATETIME)
    print(f"Created collection '{config.QDRANT_COLLECTION}'")


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

    batch_rows, batch_texts = [], []

    def flush(line_num: int):
        if not batch_rows:
            return
        dense_vecs = model.encode(
            batch_texts,
            batch_size=config.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sparse_vecs = list(sparse_model.embed(batch_texts))
        points = []
        for row, dvec, svec in zip(batch_rows, dense_vecs, sparse_vecs):
            points.append(
                PointStruct(
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
                    },
                )
            )
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=False)
        save_checkpoint(line_num)

    pbar = tqdm(total=total - start_line, desc="Embedding + indexing")
    for i, line in enumerate(lines):
        if i < start_line:
            continue
        row = json.loads(line)
        text = f"{row['title']}. {row.get('summary') or ''}".strip()
        batch_rows.append(row)
        batch_texts.append(text)

        if len(batch_rows) >= config.EMBED_BATCH_SIZE:
            flush(i + 1)
            pbar.update(len(batch_rows))
            batch_rows, batch_texts = [], []

    flush(total)
    pbar.update(len(batch_rows))
    pbar.close()
    print("Index build complete.")


if __name__ == "__main__":
    main()
