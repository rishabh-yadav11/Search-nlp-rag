"""Embed data/articles.jsonl into Qdrant; the resume checkpoint advances only after an
acknowledged upsert, so an interrupted run restarts from the last durable batch."""
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import models as qmodels
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import create_payload_indexes, log, make_point

from app.config import config
from app.index_text import compose_dense_text, compose_sparse_text

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "articles.jsonl")
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", ".checkpoint")


def load_checkpoint() -> tuple[int, int]:
    """Return (resume_line, skipped); `skipped` is stored with the position so a resume
    neither loses nor double-counts the malformed lines."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = f.read().strip()
        if data:
            try:
                obj = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                # Legacy plain-integer checkpoint (old code wrote a bare number).
                try:
                    return int(data), 0
                except (TypeError, ValueError):
                    return 0, 0
            if isinstance(obj, int):
                return obj, 0
            if isinstance(obj, dict):
                return int(obj.get("line", 0)), int(obj.get("skipped", 0))
            return 0, 0
    return 0, 0


def save_checkpoint(line_num: int, skipped: int):
    tmp_path = f"{CHECKPOINT_PATH}.tmp"
    with open(tmp_path, "w") as f:
        f.write(json.dumps({"line": line_num, "skipped": skipped}))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, CHECKPOINT_PATH)


def backup_collection_best_effort(client: QdrantClient):
    """Best-effort snapshot: a failure only warns, so a transient Qdrant outage cannot block a rebuild."""
    try:
        from qdrant_backup import make_backup

        existing = [c.name for c in client.get_collections().collections]
        if config.QDRANT_COLLECTION not in existing:
            return
        make_backup(client, config.QDRANT_COLLECTION)
    except Exception as e:
        log(f"WARNING: best-effort backup before collection change failed: {e}")


def ensure_collection(client: QdrantClient) -> bool:
    """True if the collection was (re)created empty this run; the caller MUST then reset
    the checkpoint, which refers to a collection that no longer exists."""
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        backup_collection_best_effort(client)
        create_collection(client)
        return True

    info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
    # Rebuild keys on dense size + sparse idf modifier; a schema we cannot confirm counts as incompatible.
    vectors = info.config.params.vectors
    dense_size = vectors.get("dense").size if isinstance(vectors, dict) else vectors.size
    dim_matches = dense_size == config.EMBED_DIM

    sparse_cfg = getattr(info.config, "sparse_vectors_config", None)
    modifier = None
    if sparse_cfg is not None:
        inner = sparse_cfg.sparse if hasattr(sparse_cfg, "sparse") else sparse_cfg.get("sparse")
        modifier = inner.get("modifier") if isinstance(inner, dict) else getattr(inner, "modifier", None)
    needs_idf = modifier is not None and modifier == Modifier.IDF

    if needs_idf and dim_matches:
        print(f"Collection '{config.QDRANT_COLLECTION}' already exists, resuming upserts into it")
        return False

    print(
        f"Collection '{config.QDRANT_COLLECTION}' exists but is incompatible "
        f"(dense size={dense_size} vs {config.EMBED_DIM}, sparse modifier idf={needs_idf}). "
        f"Deleting and recreating it so indexed vectors match the current config.",
    )
    backup_collection_best_effort(client)
    client.delete_collection(collection_name=config.QDRANT_COLLECTION)
    create_collection(client)
    return True


def create_collection(client: QdrantClient):
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={"dense": VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        hnsw_config=qmodels.HnswConfigDiff(m=32, ef_construct=256),
    )
    create_payload_indexes(client)
    print(f"Created collection '{config.QDRANT_COLLECTION}'")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data file at {DATA_PATH} — run scripts/fetch_data.py first.")
        return

    # Imported lazily: sentence_transformers pulls in torch, which the schema and checkpoint helpers must not require.
    from sentence_transformers import SentenceTransformer

    print(f"Loading dense embedding model {config.EMBED_MODEL} on {config.EMBED_DEVICE}...")
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    print(f"Loading sparse embedding model {config.SPARSE_MODEL}...")
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL)

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
    recreated = ensure_collection(client)

    # A resumed build skips create_collection, so re-index payloads here or a newly added field stays unfilterable.
    try:
        create_payload_indexes(client)
    except Exception as e:
        log(f"WARNING: could not create payload indexes: {e}")

    if recreated:
        save_checkpoint(0, 0)
        print(
            "Collection was (re)created empty this run — embed checkpoint reset to 0 so "
            "the ENTIRE dataset is re-indexed (the old checkpoint referred to a "
            "collection that no longer exists).",
        )

    start_line, skipped = load_checkpoint()

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
    pending = deque()

    def submit_batch(end_line: int):
        pending.append((end_line, executor.submit(encode_batch, dense_texts, sparse_texts)))

    def upsert_and_checkpoint():
        if not pending:
            return
        end_line, future = pending.popleft()
        rows = batch_frames.pop(end_line)
        dense_vecs, sparse_vecs = future.result()
        # Both vectors go on the same point: a sparse vector without its dense counterpart is unsearchable.
        points = [make_point(row, dvec, svec) for row, dvec, svec in zip(rows, dense_vecs, sparse_vecs)]
        try:
            client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=True)
        except Exception as e:
            log(
                f"ERROR: upsert of batch ending line {end_line} failed; checkpoint NOT "
                f"advanced (batch will be retried from the last saved checkpoint): {e}",
            )
            raise
        save_checkpoint(end_line, skipped)
        pbar.update(len(rows))

    batch_frames = {}
    executor = ThreadPoolExecutor(max_workers=config.INDEXER_WORKERS)
    pbar = tqdm(total=total - start_line, desc="Embedding + indexing")

    try:
        for i, line in enumerate(lines):
            if i < start_line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                log(f"Skipping malformed jsonl line {i + 1}: not valid JSON; build continues.")
                continue
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
    if skipped:
        log(
            f"Build finished with {skipped} malformed jsonl line(s) skipped. "
            f"Verify the source dataset if completeness matters.",
        )
    try:
        info = client.get_collection(config.QDRANT_COLLECTION)
        count = info.points_count or 0
        expected = total - skipped
        if count < expected:
            log(
                f"ERROR: collection '{config.QDRANT_COLLECTION}' has {count} points "
                f"but the valid dataset has {expected} articles ({expected - count} "
                f"missing). The index is INCOMPLETE — investigate before going live.",
            )
        elif count > total:
            log(
                f"WARNING: collection '{config.QDRANT_COLLECTION}' has {count} points "
                f"which EXCEEDS the {total}-row dataset (possible stale points from a "
                f"previous schema). Consider a clean rebuild.",
            )
        else:
            log(
                f"verified: {count} points in collection match the {expected} valid "
                f"dataset articles (build complete).",
            )
    except Exception as e:
        log(f"WARNING: could not verify points_count: {e}")
    print("Index build complete.")


if __name__ == "__main__":
    main()