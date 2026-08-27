"""
Removes the entire index so the next run starts from zero.

Drops the Qdrant collection and deletes the local data artifacts
(articles.jsonl, build checkpoint, incremental index state). The embedding
model cache, venv, and .env are left untouched.

Safety: before deleting anything, a snapshot backup is taken
via qdrant_backup.make_backup() (Qdrant collection snapshot + copies of
articles.jsonl/index_state.json under backend/backups/). Deletion only proceeds
after the snapshot succeeds, unless the explicit --skip-backup flag is passed
(dangerous). Run scripts/backup_qdrant.py to back up without resetting.

Usage:
    python scripts/reset_index.py            # interactive confirmation
    python scripts/reset_index.py --yes      # skip confirmation
    python scripts/reset_index.py --keep-data  # drop collection only
    python scripts/reset_index.py --skip-backup  # delete WITHOUT a backup (dangerous)
"""
import os
import socket
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_backup import log, make_backup
from qdrant_client import QdrantClient

from app.config import config

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")

DATA_FILES = [
    os.path.join(DATA_DIR, "articles.jsonl"),
    os.path.join(DATA_DIR, ".checkpoint"),
    os.path.join(DATA_DIR, "index_state.json"),
    os.path.join(DATA_DIR, "update.lock"),
]

def _parse_api_port(default: str = "8001") -> int:
    raw = os.getenv("API_PORT", default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


API_PORT = _parse_api_port()


def api_is_live() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", API_PORT), timeout=1):
            return True
    except OSError:
        return False


def main():
    keep_data = "--keep-data" in sys.argv
    assume_yes = "--yes" in sys.argv
    skip_backup = "--skip-backup" in sys.argv

    targets = [f"Qdrant collection '{config.QDRANT_COLLECTION}'"]
    if not keep_data:
        targets += [os.path.relpath(p, BACKEND_DIR) for p in DATA_FILES]

    print("=" * 70)
    print("  WARNING: This will PERMANENTLY delete:")
    for t in targets:
        print(f"    - {t}")
    print(f"  Backend API on port {API_PORT} must be stopped or will 500 until the index is rebuilt.")
    print("=" * 70)

    if not skip_backup:
        print("A snapshot backup of the collection and local artifacts will be taken")
        print("under backend/backups/ BEFORE anything is deleted.")
    else:
        print("!!! --skip-backup set: NO backup will be taken. Data will be unrecoverable.")

    if not assume_yes:
        reply = input("Type 'yes' to continue: ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return

    if api_is_live():
        log(f"WARNING: something is listening on port {API_PORT} — live queries will fail after this")

    client = QdrantClient(url=config.QDRANT_URL, timeout=30)
    try:
        existing = [c.name for c in client.get_collections().collections]

        if not skip_backup:
            if config.QDRANT_COLLECTION in existing:
                dest, snapshot_ok = make_backup(client, config.QDRANT_COLLECTION)
                if not snapshot_ok:
                    log(
                        f"ERROR: snapshot for '{config.QDRANT_COLLECTION}' could not be created; "
                        f"aborting reset to avoid irreversible loss",
                    )
                    log("Re-run with --skip-backup to force deletion without a backup.")
                    return
                log(f"backup before reset: {dest}")
            else:
                # Collection is missing: still attempt a backup of any local artifacts
                # so the reset leaves a recoverable record instead of silently doing nothing.
                dest, snapshot_ok = make_backup(client, config.QDRANT_COLLECTION)
                if dest is None:
                    log(
                        f"collection '{config.QDRANT_COLLECTION}' does not exist and no local "
                        f"artifacts were available to back up — nothing to snapshot; proceeding",
                    )
                else:
                    log(
                        f"collection '{config.QDRANT_COLLECTION}' does not exist; backed up "
                        f"available local artifacts only before reset: {dest}",
                    )

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
    finally:
        client.close()

    print("\nNext steps to rebuild from zero:")
    print("  1. python scripts/fetch_data.py")
    print("  2. python scripts/build_index.py")
    print("  3. python scripts/update_index.py --init")
    print("  4. (re)start the API")


if __name__ == "__main__":
    main()