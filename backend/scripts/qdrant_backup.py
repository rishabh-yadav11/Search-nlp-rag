"""
Shared Qdrant snapshot/backup helpers.

A backup is a directory under ``backend/backups/`` named
``<collection>-<UTC-YYYYmmdd-HHMMSS-mmmmmm>`` containing:

  * the Qdrant collection snapshot file downloaded from the server (``.snapshot``)
  * copies of ``data/articles.jsonl`` and ``data/index_state.json`` (if present)

``create_snapshot`` itself is authoritative: the server-side snapshot is the
durable artifact. Downloading it locally is *optional/safe* — if the download
fails we log a clear error but still copy the local artifacts, so a backup
directory is never left useless by a transient network hiccup.

Used by:

  * ``backup_qdrant.py``      — CLI entry point
  * ``reset_index.py``        — hard-gates deletion on a successful snapshot
  * ``build_index.py``        — best-effort backup before a schema recreate

Only the most recent ``BACKUP_RETENTION`` (default 5) backups per collection are
kept; older ones are pruned by ``prune_backups``.
"""
import os
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import urlparse

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUPS_DIR = os.path.join(BACKEND_DIR, "backups")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

TS_RE = re.compile(r"^\d{8}-\d{6}-\d{6}$")


def log(msg: str):
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}", flush=True)


def _parse_retention() -> int:
    raw = os.getenv("BACKUP_RETENTION", "5")
    try:
        return int(raw)
    except (TypeError, ValueError):
        log(f"WARNING: invalid BACKUP_RETENTION='{raw}', falling back to 5")
        return 5


RETENTION = _parse_retention()

LOCAL_ARTIFACTS = ["articles.jsonl", "index_state.json"]


def new_backup_dir(collection_name: str) -> str:
    """Create and return the backup dir ``backend/backups/<collection>-<ts>``."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    dest = os.path.join(BACKUPS_DIR, f"{collection_name}-{ts}")
    os.makedirs(dest, exist_ok=True)
    return dest


def _snapshot_download_url(collection_name: str, snapshot_name: str) -> str:
    if BACKEND_DIR not in sys.path:
        sys.path.append(BACKEND_DIR)
    from app.config import config

    base = config.QDRANT_URL.rstrip("/")
    return f"{base}/collections/{collection_name}/snapshots/{snapshot_name}"


def _redact_url(url: str) -> str:
    """Return the URL with any userinfo (credentials/token) stripped."""
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            return parsed._replace(netloc=netloc).geturl()
    except ValueError:
        pass
    return url


@contextmanager
def _open_url(url: str, **kwargs):
    import requests

    resp = requests.get(url, **kwargs)
    try:
        yield resp
    finally:
        resp.close()


def create_and_download_snapshot(client, collection_name: str, dest_dir: str):
    """Create a server-side collection snapshot and try to download it locally.

    Returns the snapshot file name (already downloaded) on success. Returns the
    snapshot name even when only the *download* failed (the server-side snapshot
    is durable), and ``None`` when the snapshot could not be created.
    """
    try:
        snap = client.create_snapshot(collection_name=collection_name, wait=True)
    except Exception as e:
        log(f"ERROR: could not create snapshot for collection '{collection_name}': {e}")
        return None
    if snap is None:
        log(f"ERROR: create_snapshot returned no snapshot for collection '{collection_name}'")
        return None

    name = snap.name
    try:
        url = _snapshot_download_url(collection_name, name)
        log(f"downloading snapshot '{name}' from {_redact_url(url)}")
        with _open_url(url, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            dest = os.path.join(dest_dir, name)
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        log(f"saved snapshot to {os.path.relpath(dest, BACKEND_DIR)}")
    except Exception as e:
        log(
            f"WARNING: snapshot '{name}' was created server-side but could not be "
            f"downloaded locally: {e}",
        )
    return name


def copy_local_artifacts(dest_dir: str) -> list:
    """Copy data/articles.jsonl and data/index_state.json into dest_dir if present."""
    copied = []
    for artifact in LOCAL_ARTIFACTS:
        src = os.path.join(DATA_DIR, artifact)
        if os.path.exists(src):
            dst = os.path.join(dest_dir, artifact)
            shutil.copy2(src, dst)
            copied.append(artifact)
    return copied


def backup_dirs(collection_name: str) -> list:
    """Existing backup directories for a collection, oldest first.

    Sorted by the backup's sortable creation timestamp encoded in the
    directory name (``<collection>-<UTC-YYYYmmdd-HHMMSS-mmmmmm>``) rather than
    directory mtime, which can be altered by copies/restores.
    """
    prefix = f"{collection_name}-"
    if not os.path.isdir(BACKUPS_DIR):
        return []
    dirs = [
        os.path.join(BACKUPS_DIR, d)
        for d in os.listdir(BACKUPS_DIR)
        if d.startswith(prefix)
        and TS_RE.match(d[len(prefix):]) is not None
        and os.path.isdir(os.path.join(BACKUPS_DIR, d))
    ]
    return sorted(dirs, key=os.path.basename)


def prune_backups(collection_name: str, retention: int | None = None) -> list:
    """Keep only the newest ``retention`` backups; return paths that were removed."""
    retention = retention if retention is not None else RETENTION
    if retention <= 0:
        return []
    dirs = backup_dirs(collection_name)
    removed = []
    for d in dirs[:-retention]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d)
    return removed


def make_backup(client, collection_name: str):
    """Create one backup (snapshot + local artifacts) and enforce retention.

    Returns ``(dest_dir, snapshot_ok)`` where ``dest_dir`` is ``None`` when
    nothing could be backed up at all.
    """
    dest = new_backup_dir(collection_name)
    snapshot_ok = create_and_download_snapshot(client, collection_name, dest) is not None
    copied = copy_local_artifacts(dest)

    if not snapshot_ok and not copied:
        shutil.rmtree(dest, ignore_errors=True)
        log(f"ERROR: no snapshot and no local artifacts to back up for '{collection_name}'")
        return None, False

    log(
        f"backup written to {os.path.relpath(dest, BACKEND_DIR)} "
        f"(snapshot={'ok' if snapshot_ok else 'FAILED'}, artifacts={copied or 'none'})",
    )
    for removed in prune_backups(collection_name):
        log(f"pruned old backup {os.path.relpath(removed, BACKUPS_DIR)}")
    return dest, snapshot_ok
