"""
Removes the entire index so the next run starts from zero.

Drops the Qdrant collection and deletes the local data artifacts
(articles.jsonl, build checkpoint, incremental index state). The embedding
model cache, venv, and .env are left untouched.

Usage:
    python scripts/reset_index.py            # interactive confirmation
    python scripts/reset_index.py --yes      # skip confirmation
    python scripts/reset_index.py --keep-data  # drop collection only
"""
import os
import socket
import sys
from datetime import datetime, timezone

from qdrant_client import QdrantClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import config

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")

DATA_FILES = [
    os.path.join(DATA_DIR, "articles.jsonl"),
    os.path.join(DATA_DIR, ".checkpoint"),
    os.path.join(DATA_DIR, "index_state.json"),
    os.path.join(DATA_DIR, "update.lock"),
]

API_PORT = int(os.getenv("API_PORT", "8001"))


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def api_is_live() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", API_PORT), timeout=1):
            return True
    except OSError:
        return False


def main():
    keep_data = "--keep-data" in sys.argv
    assume_yes = "--yes" in sys.argv

    targets = [f"Qdrant collection '{config.QDRANT_COLLECTION}'"]
    if not keep_data:
        targets += [os.path.relpath(p, BACKEND_DIR) for p in DATA_FILES]

    print("This will permanently remove:")
    for t in targets:
        print(f"  - {t}")
    print(f"Backend API on port {API_PORT} must be stopped or will 500 until the index is rebuilt.")

    if not assume_yes:
        reply = input("Type 'yes' to continue: ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return

    if api_is_live():
        log(f"WARNING: something is listening on port {API_PORT} — live queries will fail after this")

    client = QdrantClient(url=config.QDRANT_URL, timeout=30)
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION in existing:
        client.delete_collection(collection_name=config.QDRANT_COLLECTION)
        log(f"dropped collection '{config.QDRANT_COLLECTION}'")
    else:
        log(f"collection '{config.QDRANT_COLLECTION}' did not exist")

    if not keep_data:
        for p in DATA_FILES:
            if os.path.exists(p):
                os.remove(p)
                log(f"removed {os.path.relpath(p, BACKEND_DIR)}")

    print("\nNext steps to rebuild from zero:")
    print("  1. python scripts/fetch_data.py")
    print("  2. python scripts/build_index.py")
    print("  3. python scripts/update_index.py --init")
    print("  4. (re)start the API")


if __name__ == "__main__":
    main()