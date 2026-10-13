
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from typing import NamedTuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import log, redact_url

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUPS_DIR = os.path.join(BACKEND_DIR, "backups")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

TS_RE = re.compile(r"^\d{8}-\d{6}-\d{6}$")

DOWNLOAD_TIMEOUT = 300
DOWNLOAD_CHUNK = 1 << 20


class SnapshotResult(NamedTuple):


    name: str | None
    ok: bool
    local_path: str | None
    detail: str

    @property
    def server_side_only(self) -> bool:

        return self.name is not None and not self.ok


class BackupResult(NamedTuple):


    dest: str | None
    snapshot_ok: bool
    snapshot_name: str | None
    artifacts: list


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


def _download_to(url: str, dest: str, timeout: int = DOWNLOAD_TIMEOUT) -> None:

    req = urllib.request.Request(url, method="GET")
    expected = None
    written = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        header = resp.headers.get("Content-Length")
        if header is not None:
            try:
                expected = int(header)
            except ValueError:
                expected = None
        else:
            log(
                f"WARNING: {redact_url(url)} sent no Content-Length; this download "
                f"cannot be checked for truncation, only for being a readable archive",
            )
        while True:
            chunk = resp.read(DOWNLOAD_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    if expected is not None and written != expected:
        raise OSError(f"download truncated: received {written} of {expected} bytes")


def _local_snapshot_is_valid(path: str) -> tuple[bool, str]:

    if not os.path.isfile(path):
        return False, f"no file at {path}"
    size = os.path.getsize(path)
    if size == 0:
        return False, f"{path} is empty (0 bytes)"
    try:
        with tarfile.open(path, "r:*") as tf:
            members = sum(1 for _ in tf)
    except (tarfile.TarError, OSError, EOFError) as e:
        return False, f"{path} is not a readable snapshot archive ({e})"
    if members == 0:
        return False, f"{path} is an empty archive (0 entries)"
    return True, f"{size} bytes, {members} entries"


def create_and_download_snapshot(client, collection_name: str, dest_dir: str) -> SnapshotResult:

    try:
        snap = client.create_snapshot(collection_name=collection_name, wait=True)
    except Exception as e:
        log(f"ERROR: could not create snapshot for collection '{collection_name}': {e}")
        return SnapshotResult(None, False, None, f"snapshot creation failed: {e}")
    if snap is None:
        log(f"ERROR: create_snapshot returned no snapshot for collection '{collection_name}'")
        return SnapshotResult(None, False, None, "create_snapshot returned no snapshot")

    name = snap.name
    url = None
    dest = os.path.join(dest_dir, name)
    try:
        url = _snapshot_download_url(collection_name, name)
        log(f"downloading snapshot '{name}' from {redact_url(url)}")
        _download_to(url, dest)
    except Exception as e:
        err = str(e).replace(url, redact_url(url)) if url else str(e)


        if os.path.exists(dest):
            os.remove(dest)
        log(
            f"ERROR: snapshot '{name}' was created server-side but NO local copy was "
            f"written ({err}); it survives only inside the Qdrant container and is "
            f"destroyed by a container recreate or `docker rm` — this is not a backup",
        )
        return SnapshotResult(name, False, None, err)

    ok, detail = _local_snapshot_is_valid(dest)
    if not ok:


        os.remove(dest)
        log(f"ERROR: downloaded snapshot '{name}' failed verification: {detail}")
        return SnapshotResult(name, False, None, detail)

    log(f"saved verified snapshot to {os.path.relpath(dest, BACKEND_DIR)} ({detail})")
    return SnapshotResult(name, True, dest, detail)


def copy_local_artifacts(dest_dir: str) -> list:

    copied = []
    for artifact in LOCAL_ARTIFACTS:
        src = os.path.join(DATA_DIR, artifact)
        if os.path.exists(src):
            dst = os.path.join(dest_dir, artifact)
            shutil.copy2(src, dst)
            copied.append(artifact)
    return copied


def backup_dirs(collection_name: str) -> list:

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

    retention = retention if retention is not None else RETENTION
    if retention <= 0:
        return []
    dirs = backup_dirs(collection_name)
    removed = []
    for d in dirs[:-retention]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d)
    return removed


def make_backup(client, collection_name: str) -> BackupResult:

    dest = new_backup_dir(collection_name)
    result = create_and_download_snapshot(client, collection_name, dest)
    copied = copy_local_artifacts(dest)

    if not result.ok and not copied:
        shutil.rmtree(dest, ignore_errors=True)
        log(
            f"ERROR: no verified snapshot and no local artifacts to back up for "
            f"'{collection_name}'; the backup directory was removed",
        )
        outcome = BackupResult(None, False, result.name, copied)
    elif not result.ok:
        log(
            f"ERROR: backup for '{collection_name}' is INCOMPLETE — the collection "
            f"snapshot was not written to disk ({result.detail}); only the local "
            f"artifacts {copied} are in this directory",
        )
        outcome = BackupResult(dest, False, result.name, copied)
    else:
        log(
            f"backup written to {os.path.relpath(dest, BACKEND_DIR)} "
            f"(snapshot=ok, artifacts={copied or 'none'})",
        )
        outcome = BackupResult(dest, True, result.name, copied)




    for removed in prune_backups(collection_name):
        log(f"pruned old backup {os.path.relpath(removed, BACKUPS_DIR)}")
    return outcome
