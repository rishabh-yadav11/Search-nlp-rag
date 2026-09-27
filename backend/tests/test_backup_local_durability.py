"""A backup only counts when a verified snapshot archive is on local disk.

``qdrant_backup.py`` used to download snapshots with ``requests`` — which was
never declared in ``requirements.txt`` — and swallowed the resulting failure in
a blanket ``except Exception`` that still reported success. ``reset_index.py``
then used that success as its hard gate before irreversibly dropping the
collection, so a backup that wrote nothing still read as a green backup and the
delete went ahead.

These tests pin the fixed contract:

  * success is reported only when a non-empty, readable archive exists on disk,
  * a server-side snapshot that was never downloaded is a *failure*, reported
    loudly, with no leftover partial file,
  * ``reset_index`` refuses to delete the collection and exits non-zero when no
    usable backup exists, and deletes it (exit 0) when one does.
"""
import io
import os
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import qdrant_backup
import reset_index
from qdrant_backup import make_backup

SNAPSHOT_NAME = "20260927-120000-000001.snapshot"


class _FakeSnapshot:
    def __init__(self, name=SNAPSHOT_NAME):
        self.name = name


class _FakeClient:
    """Qdrant client stub that can create a snapshot or fail to."""

    def __init__(self, snap=None, raises=None):
        self._snap = _FakeSnapshot() if snap is None else snap
        self._raises = raises

    def create_snapshot(self, collection_name, wait=False):
        if self._raises is not None:
            raise self._raises
        return self._snap

    def close(self):
        pass


def _tar_bytes(entries=("collection/meta.json", "collection/0/segment.bin")):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for entry in entries:
            payload = b"payload-for-" + entry.encode()
            info = tarfile.TarInfo(name=entry)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


@pytest.fixture
def backups_dir(tmp_path, monkeypatch):
    """Isolate BACKUPS_DIR/DATA_DIR so a test never touches the real tree."""
    backups = tmp_path / "backups"
    data = tmp_path / "data"
    backups.mkdir()
    data.mkdir()
    monkeypatch.setattr(qdrant_backup, "BACKUPS_DIR", str(backups))
    monkeypatch.setattr(qdrant_backup, "DATA_DIR", str(data))
    return backups


def _write_articles(data_dir, name="articles.jsonl"):
    (data_dir / name).write_text('{"id": 1}\n')


class _SnapshotServer:
    """Real HTTP server so the stdlib download path is exercised end to end."""

    def __init__(self, body):
        self.body = body
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence stderr noise
                pass

            def do_GET(self):
                if self.path.startswith("/redirect"):
                    self.send_response(302)
                    self.send_header("Location", "/collections/c/snapshots/s")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path.startswith("/truncated"):
                    # Promise more bytes than are sent, then hang up mid-body.
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(outer.body) + 500))
                    self.end_headers()
                    self.wfile.write(outer.body[: len(outer.body) // 2])
                    self.close_connection = True
                elif self.path.startswith("/stall"):
                    # Announce a body, then stall without ever sending it.
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(outer.body)))
                    self.end_headers()
                    time.sleep(3)
                elif self.path.startswith("/unsized"):
                    # No Content-Length: the body is delimited by the close.
                    self.send_response(200)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(outer.body)
                    self.close_connection = True
                elif self.path.startswith("/collections/"):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(outer.body)))
                    self.end_headers()
                    self.wfile.write(outer.body)
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base(self):
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def snapshot_server():
    servers = []

    def _make(body):
        srv = _SnapshotServer(body)
        servers.append(srv)
        return srv

    yield _make
    for srv in servers:
        srv.close()


# --- the success signal really means "a verifiable local archive exists" ---


@pytest.mark.parametrize("route", ["/collections/c/snapshots/s", "/redirect"])
def test_real_download_through_stdlib_is_reported_as_success(
    tmp_path, monkeypatch, snapshot_server, capsys, route
):
    """No ``requests`` anywhere: urllib streams, follows a 302, writes a real
    archive, and only then is the backup reported as ok."""
    body = _tar_bytes()
    srv = snapshot_server(body)
    url = srv.base + route
    monkeypatch.setattr(qdrant_backup, "_snapshot_download_url", lambda *a: url)

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(_FakeSnapshot("s")), "c", str(tmp_path)
    )

    assert result.ok is True
    assert result.name == "s"
    assert result.local_path == str(tmp_path / "s")
    written = (tmp_path / "s").read_bytes()
    assert written == body and len(written) > 0
    with tarfile.open(result.local_path, "r:*") as tf:
        assert len(tf.getnames()) == 2
    assert "saved verified snapshot" in capsys.readouterr().out


def test_truncated_transfer_is_not_reported_as_a_backup(tmp_path, monkeypatch, snapshot_server, capsys):
    """A body that stops mid-stream must fail loudly and leave no file behind."""
    srv = snapshot_server(_tar_bytes())
    monkeypatch.setattr(
        qdrant_backup, "_snapshot_download_url", lambda *a: srv.base + "/truncated"
    )

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(_FakeSnapshot("s")), "c", str(tmp_path)
    )

    assert result.ok is False
    assert result.local_path is None
    assert os.listdir(tmp_path) == [], "a partial download must not be left in the backup dir"
    assert "ERROR" in capsys.readouterr().out


def test_http_error_status_is_not_reported_as_a_backup(tmp_path, monkeypatch, snapshot_server):
    srv = snapshot_server(_tar_bytes())
    monkeypatch.setattr(qdrant_backup, "_snapshot_download_url", lambda *a: srv.base + "/nope")

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(_FakeSnapshot("s")), "c", str(tmp_path)
    )

    assert result.ok is False
    assert "404" in result.detail
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(b"", id="zero-byte"),
        pytest.param(b"not a tar at all", id="junk-bytes"),
        pytest.param(_tar_bytes()[:120], id="truncated-archive"),
    ],
)
def test_non_archive_download_is_rejected_and_removed(
    tmp_path, monkeypatch, corrupt, capsys
):
    """Non-empty is not enough — the file must actually be a readable archive."""

    def _fake_download(url, dest, timeout=None):
        with open(dest, "wb") as f:
            f.write(corrupt)

    monkeypatch.setattr(qdrant_backup, "_download_to", _fake_download)

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(_FakeSnapshot("s")), "c", str(tmp_path)
    )

    assert result.ok is False
    assert result.local_path is None
    assert os.listdir(tmp_path) == [], "an unverifiable snapshot file must be removed"
    assert "failed verification" in capsys.readouterr().out


def test_unverifiable_download_warns_that_it_could_not_be_length_checked(
    tmp_path, monkeypatch, snapshot_server, capsys
):
    """A response with no Content-Length cannot be checked for truncation, and
    the archive check does not make up for it. The operator must be told, rather
    than being left to assume the file was length-verified."""
    srv = snapshot_server(_tar_bytes())
    monkeypatch.setattr(
        qdrant_backup, "_snapshot_download_url", lambda *a: srv.base + "/unsized"
    )

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(_FakeSnapshot("s")), "c", str(tmp_path)
    )

    out = capsys.readouterr().out
    assert "sent no Content-Length" in out
    assert "cannot be checked for truncation" in out
    assert result.ok is True  # the archive is intact here, so it is accepted


def test_entirely_empty_archive_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda url, dest, timeout=None: _write_bytes(dest, _tar_bytes(entries=())),
    )

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(_FakeSnapshot("s")), "c", str(tmp_path)
    )

    assert result.ok is False


# --- the failure signal reaches every caller ---


def test_failed_download_reports_failure_and_names_the_server_side_snapshot(
    tmp_path, monkeypatch, capsys
):
    """The exact historical failure: the download blew up (here, the ImportError
    a clean `pip install -r requirements.txt` produced) but the backup must not
    claim success just because Qdrant created a snapshot."""
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'requests'")),
    )

    result = qdrant_backup.create_and_download_snapshot(
        _FakeClient(), "c", str(tmp_path)
    )

    assert result.ok is False
    assert result.name == SNAPSHOT_NAME, "operator still needs the container-side name"
    assert result.local_path is None
    assert result.server_side_only is True
    assert "No module named 'requests'" in result.detail
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "NO local copy was written" in out
    assert os.listdir(tmp_path) == []


def test_make_backup_with_no_artifacts_and_failed_snapshot_writes_nothing(
    backups_dir, monkeypatch
):
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError("connection reset")),
    )

    result = make_backup(_FakeClient(), "c")

    assert result.dest is None
    assert result.snapshot_ok is False
    assert list(backups_dir.iterdir()) == [], "an unusable backup dir must not be left behind"


def test_make_backup_reports_failure_when_only_local_artifacts_were_copied(
    backups_dir, monkeypatch, capsys
):
    """This is the value reset_index gates on: a directory exists, but it holds
    no collection snapshot, so it must not read as a successful backup."""
    _write_articles(backups_dir.parent / "data")
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError("connection reset")),
    )

    result = make_backup(_FakeClient(), "c")

    assert result.dest is not None
    assert result.snapshot_ok is False, "local artifacts alone are not a collection backup"
    assert result.snapshot_name == SNAPSHOT_NAME
    assert result.artifacts == ["articles.jsonl"]
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "INCOMPLETE" in out
    assert "snapshot=ok" not in out


def test_make_backup_reports_success_only_with_a_verified_archive(backups_dir, monkeypatch):
    def _fake_download(url, dest, timeout=None):
        with open(dest, "wb") as f:
            f.write(_tar_bytes())

    monkeypatch.setattr(qdrant_backup, "_download_to", _fake_download)

    result = make_backup(_FakeClient(), "c")

    assert result.snapshot_ok is True
    assert result.dest is not None
    snapshot = os.path.join(result.dest, SNAPSHOT_NAME)
    assert os.path.getsize(snapshot) > 0
    with tarfile.open(snapshot, "r:*") as tf:
        assert len(tf.getnames()) == 2

def test_retention_is_enforced_even_when_every_backup_fails(backups_dir, monkeypatch):
    """A run of failing backups creates a directory each time; if pruning only
    ran on success, backend/backups/ would grow without bound on exactly the
    runs an operator is most likely to be repeating."""
    for i in range(6):
        stale = backups_dir / f"c-2026010{i}-000000-000000"
        stale.mkdir()
        (stale / "marker").write_text("old")
    monkeypatch.setattr(qdrant_backup, "RETENTION", 2)
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError("connection reset")),
    )

    for _ in range(3):
        make_backup(_FakeClient(), "c")

    remaining = sorted(p.name for p in backups_dir.iterdir())
    assert len(remaining) == 2, remaining
    assert not (backups_dir / "c-20260100-000000-000000").exists()


def test_download_honours_the_timeout(snapshot_server, tmp_path):
    """A stalled server must not hang a backup forever."""
    srv = snapshot_server(_tar_bytes())
    dest = str(tmp_path / "s.snapshot")

    with pytest.raises(TimeoutError):
        qdrant_backup._download_to(srv.base + "/stall", dest, timeout=0.5)


def test_snapshot_creation_failure_is_reported_as_failure(tmp_path):
    client = _FakeClient(raises=RuntimeError("qdrant down"))

    result = qdrant_backup.create_and_download_snapshot(client, "c", str(tmp_path))

    assert result.ok is False
    assert result.name is None
    assert result.server_side_only is False
    assert "qdrant down" in result.detail


# --- reset_index's destructive gate keys on the local artifact ---


class _FakeQdrant:
    def __init__(self, collections, calls):
        self._collections = collections
        self._calls = calls

    def get_collections(self):
        return type("R", (), {"collections": [type("C", (), {"name": n})() for n in self._collections]})()

    def create_snapshot(self, collection_name, wait=False):
        # A real server-side snapshot, so a test using the real make_backup gets
        # as far as the download instead of failing at the create step.
        return _FakeSnapshot()

    def delete_collection(self, collection_name):
        self._calls.append(collection_name)

    def close(self):
        pass


@pytest.fixture
def reset_env(monkeypatch):
    """Drive reset_index.main() against a recording fake: no real Qdrant, no real
    data files, no real port probe."""
    from app.config import config

    calls: list = []
    client = _FakeQdrant([config.QDRANT_COLLECTION], calls)
    monkeypatch.setattr(reset_index, "QdrantClient", lambda **kw: client)
    monkeypatch.setattr(reset_index, "api_is_live", lambda: False)
    monkeypatch.setattr(reset_index, "DATA_FILES", [])
    monkeypatch.setattr(sys, "argv", ["reset_index.py", "--yes"])
    return calls


def test_reset_refuses_to_delete_when_no_local_snapshot_was_written(
    reset_env, monkeypatch, tmp_path, capsys
):
    dest = str(tmp_path / "backup")
    monkeypatch.setattr(
        reset_index,
        "make_backup",
        lambda client, coll: qdrant_backup.BackupResult(dest, False, SNAPSHOT_NAME, ["articles.jsonl"]),
    )

    rc = reset_index.main()

    assert reset_env == [], "the collection must not be dropped without a usable backup"
    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "aborting reset to avoid irreversible loss" in out


def test_reset_drops_the_collection_when_a_verified_backup_exists(
    reset_env, monkeypatch, tmp_path, capsys
):
    from app.config import config

    dest = str(tmp_path / "backup")
    monkeypatch.setattr(
        reset_index,
        "make_backup",
        lambda client, coll: qdrant_backup.BackupResult(dest, True, SNAPSHOT_NAME, []),
    )

    rc = reset_index.main()

    assert reset_env == [config.QDRANT_COLLECTION]
    assert rc == 0
    assert f"backup before reset: {dest}" in capsys.readouterr().out


def test_reset_end_to_end_with_a_failed_download_refuses_to_delete(
    reset_env, monkeypatch, tmp_path, capsys
):
    """End to end through the real make_backup, with a client that *does* create a
    server-side snapshot: force the download to fail the way the undeclared
    `requests` import did, and check the reset still refuses to delete."""
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'requests'")),
    )
    monkeypatch.setattr(qdrant_backup, "BACKUPS_DIR", str(tmp_path / "backups"))
    (tmp_path / "backups").mkdir()
    monkeypatch.setattr(qdrant_backup, "DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "articles.jsonl").write_text('{"id": 1}\n')
    monkeypatch.setattr(reset_index, "make_backup", make_backup)

    rc = reset_index.main()

    assert reset_env == []
    assert rc == 1
    out = capsys.readouterr().out
    # The download itself failed, rather than the run dying somewhere earlier.
    assert "NO local copy was written" in out
    assert "No module named 'requests'" in out
    assert "aborting reset to avoid irreversible loss" in out


def test_skip_backup_still_deletes(reset_env, monkeypatch, capsys):
    """The documented escape hatch is unchanged: --skip-backup deletes and the
    gate is not consulted."""
    from app.config import config

    monkeypatch.setattr(sys, "argv", ["reset_index.py", "--yes", "--skip-backup"])

    def _boom(client, coll):
        raise AssertionError("--skip-backup must not take a backup")

    monkeypatch.setattr(reset_index, "make_backup", _boom)
    rc = reset_index.main()

    assert reset_env == [config.QDRANT_COLLECTION]
    assert rc == 0


def test_reset_exits_nonzero_when_qdrant_is_unreachable(reset_env, monkeypatch):
    """An unreachable Qdrant means the reset did not run; the exit status has to
    say so, otherwise a wrapper or CI step reads a failure as success."""

    class _Dead:
        def get_collections(self):
            raise OSError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(reset_index, "QdrantClient", lambda **kw: _Dead())

    assert reset_index.main() == 1
    assert reset_env == []


def test_reset_exits_nonzero_when_the_operator_declines(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["reset_index.py"])
    monkeypatch.setattr("builtins.input", lambda *a: "no")

    def _boom(*a, **k):
        raise AssertionError("a declined reset must not touch Qdrant")

    monkeypatch.setattr(reset_index, "QdrantClient", _boom)

    assert reset_index.main() == 1


def test_backup_cli_exits_nonzero_when_no_snapshot_landed_on_disk(
    backups_dir, monkeypatch, capsys
):
    """backup_qdrant.py is what cron runs; it must not report success for a
    backup that wrote no snapshot."""
    import backup_qdrant

    _write_articles(backups_dir.parent / "data")
    monkeypatch.setattr(sys, "argv", ["backup_qdrant.py"])
    monkeypatch.setattr(backup_qdrant, "QdrantClient", lambda **kw: _FakeClient())
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'requests'")),
    )

    with pytest.raises(SystemExit) as exc:
        backup_qdrant.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "backup FAILED" in out
    assert "backup complete" not in out


def test_backup_cli_exits_nonzero_when_nothing_at_all_was_written(backups_dir, monkeypatch, capsys):
    """The other half of the CLI contract: no snapshot and no local artifacts,
    so make_backup removes the directory and returns dest=None. Nothing was
    written, so the exit status must not read as success either.

    The sibling test above writes an articles.jsonl, so it takes the
    `not backup.snapshot_ok` branch; this one deliberately reaches the earlier
    `backup.dest is None` branch instead."""
    import backup_qdrant

    data_dir = backups_dir.parent / "data"
    assert os.listdir(data_dir) == [], "this test must run with no local artifacts to copy"
    monkeypatch.setattr(sys, "argv", ["backup_qdrant.py"])
    monkeypatch.setattr(
        backup_qdrant, "QdrantClient", lambda **kw: _FakeClient(raises=OSError("refused"))
    )

    with pytest.raises(SystemExit) as exc:
        backup_qdrant.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "nothing was written" in out
    assert "backup complete" not in out
    assert list(backups_dir.iterdir()) == [], "the unusable backup dir must be gone"


def test_backup_cli_prune_only_exits_zero_without_writing_a_snapshot(backups_dir, monkeypatch, capsys):
    """--prune-only takes no backup, so the snapshot exit-status contract does
    not apply to it: it must still succeed, and must not touch Qdrant."""
    import backup_qdrant

    created = []

    class _Recording:
        def create_snapshot(self, collection_name, wait=False):
            created.append(collection_name)
            return _FakeSnapshot()

        def close(self):
            pass

    monkeypatch.setattr(sys, "argv", ["backup_qdrant.py", "--prune-only"])
    monkeypatch.setattr(backup_qdrant, "QdrantClient", lambda **kw: _Recording())

    assert backup_qdrant.main() is None  # no SystemExit: the script exits 0
    assert created == [], "--prune-only must not create a snapshot"
    assert list(backups_dir.iterdir()) == []
    assert "no backups to prune" in capsys.readouterr().out


def test_backup_cli_returns_cleanly_after_a_verified_backup(backups_dir, monkeypatch, capsys):
    import backup_qdrant

    monkeypatch.setattr(sys, "argv", ["backup_qdrant.py"])
    monkeypatch.setattr(backup_qdrant, "QdrantClient", lambda **kw: _FakeClient())
    monkeypatch.setattr(
        qdrant_backup,
        "_download_to",
        lambda url, dest, timeout=None: _write_bytes(dest, _tar_bytes()),
    )

    assert backup_qdrant.main() is None  # no SystemExit: the script exits 0
    assert "backup complete" in capsys.readouterr().out


# --- the scripts must not need an undeclared dependency ---


NO_REQUESTS_IMPORT = """
import sys


class _Block:
    def find_module(self, name, path=None):
        if name == "requests" or name.startswith("requests."):
            raise ImportError("requests is not installed")
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "requests" or name.startswith("requests."):
            raise ImportError("requests is not installed")
        return None


sys.meta_path.insert(0, _Block())
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import tarfile
import io

import qdrant_backup
import backup_qdrant  # noqa: F401

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    info = tarfile.TarInfo(name="c/0")
    info.size = 3
    tf.addfile(info, io.BytesIO(b"abc"))

# The stdlib download path works with `requests` unimportable.
import http.server
import threading

body = buf.getvalue()


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
import tempfile
import os

dest = os.path.join(tempfile.mkdtemp(), "s.snapshot")
qdrant_backup._download_to("http://127.0.0.1:%d/x" % port, dest, timeout=10)
with tarfile.open(dest, "r:*") as tf:
    assert tf.getnames() == ["c/0"], tf.getnames()
srv.shutdown()
assert "requests" not in sys.modules
print("OK")
"""


def test_scripts_import_and_download_with_requests_unavailable(tmp_path):
    """A clean `pip install -r requirements.txt` has no `requests`. The ops
    scripts must still import and download a snapshot in that environment."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", NO_REQUESTS_IMPORT],
        check=False,
        cwd=backend_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout
