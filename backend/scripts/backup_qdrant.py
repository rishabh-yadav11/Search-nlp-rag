"""
Create a Qdrant collection snapshot backup and copy local artifacts
(audit item #5).

A backup directory ``backend/backups/<collection>-<timestamp>`` is created
containing the Qdrant collection snapshot (downloaded from the server) plus
copies of ``data/articles.jsonl`` and ``data/index_state.json`` (if present).
Retention keeps only the most recent ``BACKUP_RETENTION`` (default 5) backups.

The snapshot download is optional/safe: if ``create_snapshot`` or the download
fails, the error is logged but the local artifacts are still copied.

Usage:
    python scripts/backup_qdrant.py            # create a snapshot backup
    python scripts/backup_qdrant.py --prune-only   # only enforce retention
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_backup import log, make_backup, prune_backups
from qdrant_client import QdrantClient

from app.config import config


def main():
    client = QdrantClient(url=config.QDRANT_URL, timeout=30)

    if "--prune-only" in sys.argv:
        removed = prune_backups(config.QDRANT_COLLECTION)
        if removed:
            for r in removed:
                log(f"pruned {r}")
        else:
            log(f"no backups to prune for '{config.QDRANT_COLLECTION}'")
        return

    dest, snapshot_ok = make_backup(client, config.QDRANT_COLLECTION)
    if dest is None:
        log("backup failed — nothing was written")
        sys.exit(1)
    log(f"backup complete: {dest} (snapshot {'ok' if snapshot_ok else 'failed but artifacts copied'})")


if __name__ == "__main__":
    main()
