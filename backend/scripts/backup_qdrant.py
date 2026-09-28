"""
Create a Qdrant collection snapshot backup and copy local artifacts.

A backup directory ``backend/backups/<collection>-<timestamp>`` is created
containing the Qdrant collection snapshot (downloaded from the server) plus
copies of ``data/articles.jsonl`` and ``data/index_state.json`` (if present).
Retention keeps only the most recent ``BACKUP_RETENTION`` (default 5) backups.

The snapshot download is what makes a backup usable: if ``create_snapshot`` or
the download fails, the error is logged and the exit status is 1 even though
the local artifacts may still have been copied — a directory without a verified
local snapshot is not a backup of the collection.

Exit status: 0 only when a verified local snapshot archive was written, or
when ``--prune-only`` was requested (which writes no snapshot by design); 1 on
any failure.

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
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=30)
    try:
        if "--prune-only" in sys.argv:
            removed = prune_backups(config.QDRANT_COLLECTION)
            if removed:
                for r in removed:
                    log(f"pruned {r}")
            else:
                log(f"no backups to prune for '{config.QDRANT_COLLECTION}'")
            return

        backup = make_backup(client, config.QDRANT_COLLECTION)
        if backup.dest is None:
            log("backup FAILED — nothing was written")
            sys.exit(1)
        if not backup.snapshot_ok:
            log(
                f"backup FAILED — no verified local snapshot was written for "
                f"'{config.QDRANT_COLLECTION}'; {backup.dest} holds only the local "
                f"artifacts {backup.artifacts}"
                + (f" and snapshot {backup.snapshot_name} exists only inside the Qdrant container" if backup.snapshot_name else ""),
            )
            sys.exit(1)
        log(f"backup complete: {backup.dest}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
