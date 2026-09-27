"""
Shared Qdrant snapshot/backup helpers.

A backup is a directory under ``backend/backups/`` named
``<collection>-<UTC-YYYYmmdd-HHMMSS-mmmmmm>`` containing:

  * the Qdrant collection snapshot file downloaded from the server (``.snapshot``)
  * copies of ``data/articles.jsonl`` and ``data/index_state.json`` (if present)

The *local* copy of the snapshot is the only durable artifact: a server-side
snapshot lives in the Qdrant container's local storage and is destroyed by a
container recreate, a ``docker rm``, or the port rebind in ``setup.sh``. A
backup therefore only counts as successful when a non-empty, readable snapshot
archive was written to disk and verified — ``reset_index.py`` hard-gates an
irreversible delete on that signal, and a green-looking but empty backup would
turn a declared-safe rebuild into unrecoverable data loss.

Snapshot download uses ``urllib.request`` from the stdlib rather than
``requests``, which was never declared in ``requirements.txt`` and so was only
present in a venv as an unpinned transitive artifact: on a clean
``pip install -r requirements.txt`` the download raised ``ImportError``, which
the old blanket ``except Exception`` swallowed into a success report.

Used by:

  * ``backup_qdrant.py``      — CLI entry point
  * ``reset_index.py``        — hard-gates deletion on a verified local snapshot
  * ``build_index.py``        — best-effort backup before a schema recreate

Only the most recent ``BACKUP_RETENTION`` (default 5) backups per collection are
kept; older ones are pruned by ``prune_backups``.
"""
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from typing import NamedTuple
from urllib.parse import urlparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import log

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUPS_DIR = os.path.join(BACKEND_DIR, "backups")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

TS_RE = re.compile(r"^\d{8}-\d{6}-\d{6}$")

DOWNLOAD_TIMEOUT = 300
DOWNLOAD_CHUNK = 1 << 20


class SnapshotResult(NamedTuple):
    """Outcome of one create-snapshot + download attempt.

    ``name`` is the server-side snapshot name when Qdrant created one, so an
    operator can still retrieve it from the container. ``ok`` — and only
    ``ok`` — means a verified local copy exists; ``local_path`` is then set.
    ``detail`` is a human-readable reason, always populated when ``ok`` is
    false, for the operator-facing log line.
    """

    name: str | None
    ok: bool
    local_path: str | None
    detail: str

    @property
    def server_side_only(self) -> bool:
        """A snapshot exists in the container but nothing durable was written."""
        return self.name is not None and not self.ok


class BackupResult(NamedTuple):
    """Outcome of ``make_backup``.

    ``dest`` is the backup directory (``None`` when nothing at all could be
    backed up). ``snapshot_ok`` means a verified local snapshot archive exists;
    ``snapshot_name`` is set whenever the server-side snapshot was created, and
    ``artifacts`` lists the local data files copied.
    """

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
    """Return the URL with userinfo and query-param secrets stripped.

    On any parse failure, returns a safe placeholder rather than the original
    URL so a malformed (and potentially secret-bearing) value cannot leak.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return parsed._replace(netloc=netloc, query="").geturl()
    except ValueError:
        return "<redacted>"


def _download_to(url: str, dest: str, timeout: int = DOWNLOAD_TIMEOUT) -> None:
    """Stream ``url`` to ``dest`` using only the stdlib.

    ``urllib.request.urlopen`` follows 3xx redirects through the default
    opener and raises ``HTTPError`` on any non-2xx status (the equivalent of
    ``requests``' ``raise_for_status()``), and applies ``timeout`` to both the
    connect and each socket read. The body is streamed in chunks rather than
    read whole, so a multi-gigabyte snapshot never lands in memory.

    The bytes received are counted and compared with the advertised
    ``Content-Length``. This is not optional: ``http.client`` returns a short
    read as plain ``b""`` and closes the connection rather than raising
    ``IncompleteRead`` (only an un-sized ``read()`` does), so
    ``shutil.copyfileobj`` would happily write a truncated snapshot to disk and
    report success — the exact silent-data-loss shape this module exists to
    prevent.

    The count above is the *only* truncation defence. A response served without
    a ``Content-Length`` (a close-delimited body) cannot be checked at all, and
    ``_local_snapshot_is_valid`` does **not** make up for it: a half-delivered
    tar can still parse as a complete archive, so nothing detects a truncated
    un-sized response. Qdrant is expected to send ``Content-Length`` for
    snapshot downloads, but that was not verifiable against a live Qdrant
    here, so it is treated as an assumption rather than a guarantee: the
    un-sized case logs a WARNING at the point it happens, so an operator is
    never misled about what was actually verified.

    Nothing here needs ``requests``, which is not a declared dependency.
    """
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
                f"WARNING: {_redact_url(url)} sent no Content-Length; this download "
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
    """Check that ``path`` is a readable, non-empty Qdrant snapshot archive.

    A Qdrant collection snapshot is a tar archive — the Qdrant documentation
    states "Snapshots are tar archive files"
    (https://qdrant.tech/documentation/snapshots/) — so this walks the member
    headers. A zero-byte file and a file of unrelated bytes both fail here and
    are reported as failures rather than as a backup. This check is *not* a
    truncation defence: a half-delivered tar can still parse cleanly, so
    ``_download_to``'s ``Content-Length`` count is what catches truncation.
    Returns ``(ok, detail)`` where ``detail`` is a human-readable size
    summary on success and the reason on failure.
    """
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
    """Create a server-side collection snapshot and download it locally.

    Returns a :class:`SnapshotResult` whose ``ok`` flag is true only when a
    verified local archive was written to ``dest_dir``. A snapshot that exists
    solely in the Qdrant container is reported as ``ok=False`` with the
    server-side ``name`` populated, because that artifact does not survive a
    container recreate and must not be treated as a backup.
    """
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
        log(f"downloading snapshot '{name}' from {_redact_url(url)}")
        _download_to(url, dest)
    except Exception as e:
        err = str(e).replace(url, _redact_url(url)) if url else str(e)
        # A half-written file is not a backup either: remove it so the backup
        # directory can never be mistaken for holding a snapshot.
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
        # A file that exists but does not verify is worse than none: a restore
        # would pick it up. Remove it so only verified artifacts remain.
        os.remove(dest)
        log(f"ERROR: downloaded snapshot '{name}' failed verification: {detail}")
        return SnapshotResult(name, False, None, detail)

    log(f"saved verified snapshot to {os.path.relpath(dest, BACKEND_DIR)} ({detail})")
    return SnapshotResult(name, True, dest, detail)


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


def make_backup(client, collection_name: str) -> BackupResult:
    """Create one backup (snapshot + local artifacts) and enforce retention.

    Returns a :class:`BackupResult` whose ``snapshot_ok`` is true only when a
    verified local snapshot archive exists. ``dest`` is ``None`` when nothing
    at all could be backed up; a non-``None`` ``dest`` with ``snapshot_ok``
    false means the directory holds local artifacts but no collection snapshot
    and is therefore not a usable backup of the collection.
    """
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

    # Retention runs on every path, including a failed backup: a run of
    # failing backups creates a directory each time, and those directories are
    # exactly what would fill backend/backups/ if pruning only ran on success.
    for removed in prune_backups(collection_name):
        log(f"pruned old backup {os.path.relpath(removed, BACKUPS_DIR)}")
    return outcome
