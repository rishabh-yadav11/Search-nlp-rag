"""Data paths must be absolute and validated, or startup must fail loudly (#294).

The defect: every data location defaulted to a path relative to the process
working directory (`data/chat.db`, `data/auth.db`, `data/query_vocab.json.gz`),
and the stores' `os.makedirs(..., exist_ok=True)` + `sqlite3.connect` happily
opened a BRAND-NEW EMPTY database at whatever directory the process happened to
be started from. pm2 was configured with a hardcoded CWD, so any checkout at
another path turned the deployment into "all conversations and users are gone"
with no error at any level.

These tests pin the two halves of the fix:

1. the configured values are absolute and anchored to the backend root, so the
   working directory cannot change their meaning; and
2. a location that cannot be used is a hard startup failure, not an empty DB.

Hermeticity note: `app.config` calls `load_dotenv()` at import, and
python-dotenv's `find_dotenv()` walks up from the CALLING FILE, not the CWD --
so `monkeypatch.chdir` alone would leave a developer's real `.env` feeding these
tests. The fixture below therefore owns the environment explicitly AND
neutralises `load_dotenv` itself for the duration of each reload.
"""

import asyncio
import importlib
import os
import re
from pathlib import Path

import pytest

import app.config as config_module
from app.config import BACKEND_ROOT, _data_path, _env_bool, ensure_data_paths_ready

# The env vars that steer the knobs under test, cleared so a developer's or the
# deploy box's real .env cannot decide any answer here.
_DATA_PATH_ENV = (
    "CHAT_DB_PATH",
    "AUTH_DB_PATH",
    "QUERY_FIX_VOCAB_PATH",
    "RERANK_ONNX_DIR",
)

DATA_PATH_KNOBS = _DATA_PATH_ENV

BOOL_KNOBS = (
    "ENABLE_QUERY_EXPANSION",
    "ENABLE_ENTITY_BOOST",
    "ENABLE_WEAK_FALLBACK",
    "ENABLE_QUERY_FIX",
    "ENABLE_DIVERSITY",
    "ENABLE_CLICK_BOOST",
    "ENABLE_BODY_RESCUE",
    "ENABLE_RECOMMENDATIONS",
)


@pytest.fixture
def load_config(monkeypatch):
    """Return a callable that reloads `app.config` under a controlled environment.

    `Config` computes every knob in the class body at import time, so
    `monkeypatch.setattr(config, "CHAT_DB_PATH", ...)` would only prove the
    attribute is assignable -- not what the real parsing code produces. Only a
    reload exercises the actual `os.getenv` path, so that is what this does.

    `load_dotenv` is neutralised for the reload because the module calls it at
    import and it resolves its file from the caller's location, i.e. regardless
    of the environment set here. The original module-level `config` object is
    restored afterwards so the rest of the suite sees exactly what it saw before.
    """
    saved_config = config_module.config
    for name in _DATA_PATH_ENV:
        monkeypatch.delenv(name, raising=False)

    def _load(env: dict[str, str]):
        monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(config_module).config

    yield _load
    config_module.config = saved_config


@pytest.mark.parametrize("knob", DATA_PATH_KNOBS)
def test_shipped_relative_default_resolves_to_the_backend_root(load_config, knob):
    """The `data/...` default in `.env.example` must resolve under the backend dir.

    It must stay *accepted* (every existing deploy .env uses it), but the
    resolved value is anchored to BACKEND_ROOT, not left relative.
    """
    cfg = load_config({})
    resolved = Path(getattr(cfg, knob))
    assert resolved.is_absolute(), (
        f"{knob} resolved to {str(resolved)!r}, which is still relative; a relative "
        "data path is interpreted against whatever directory the process started in"
    )
    assert resolved == BACKEND_ROOT / "data" / Path(str(resolved)).name


@pytest.mark.parametrize("knob", DATA_PATH_KNOBS)
def test_resolution_does_not_depend_on_the_working_directory(
    load_config, knob, tmp_path, monkeypatch
):
    """Two different CWDs must produce the identical path.

    This is the property the bug actually lacked. Before the fix the knob was
    returned verbatim, so the file the app opened was a function of the launch
    directory -- which is how a pm2 CWD mismatch produced a second, empty DB.
    """
    elsewhere = tmp_path / "some-other-directory"
    elsewhere.mkdir()

    monkeypatch.chdir(BACKEND_ROOT)
    from_backend = getattr(load_config({}), knob)
    monkeypatch.chdir(elsewhere)
    from_elsewhere = getattr(load_config({}), knob)

    assert from_backend == from_elsewhere, (
        f"{knob} depends on the working directory: {from_backend!r} from the backend "
        f"dir vs {from_elsewhere!r} from {str(elsewhere)!r}"
    )


def test_absolute_value_is_kept_verbatim(load_config):
    """A deployment that mounts its own volume must not be relocated."""
    cfg = load_config({"CHAT_DB_PATH": "/srv/vccircle/data/chat.db"})
    assert cfg.CHAT_DB_PATH == "/srv/vccircle/data/chat.db"


def test_quoted_and_padded_path_value_is_normalised(load_config):
    """`CHAT_DB_PATH=" data/chat.db "` means the path, quotes and padding aside.

    That spelling is what an editor or a copy-paste produces; taking it
    literally would create a directory whose name contains quote characters.
    """
    cfg = load_config({"CHAT_DB_PATH": '  "data/chat.db"  '})
    assert cfg.CHAT_DB_PATH == str(BACKEND_ROOT / "data" / "chat.db")


def test_home_relative_path_is_expanded(load_config):
    """`~/data/chat.db` means the operator's home, not a literal `~` directory."""
    monkey = os.environ  # only used to read HOME deterministically below
    home = monkey.get("HOME", "")
    if not home:
        pytest.skip("HOME is not set; there is nothing to expand")
    cfg = load_config({"AUTH_DB_PATH": "~/data/auth.db"})
    assert cfg.AUTH_DB_PATH == str(Path(home) / "data" / "auth.db")
    assert "~" not in cfg.AUTH_DB_PATH


def test_blank_path_value_is_rejected_at_config_load(load_config):
    """`CHAT_DB_PATH=` (set but empty) is a typo, not "use the default"."""
    with pytest.raises(ValueError, match="CHAT_DB_PATH"):
        load_config({"CHAT_DB_PATH": "   "})


def test_data_path_helper_never_returns_a_relative_path(monkeypatch):
    """`_data_path` is absolute by construction, whatever it is handed."""
    monkeypatch.delenv("SOME_PATH_KNOB", raising=False)
    for raw, expected in (
        ("data/x.db", BACKEND_ROOT / "data" / "x.db"),
        ("/abs/x.db", Path("/abs/x.db")),
    ):
        monkeypatch.setenv("SOME_PATH_KNOB", raw)
        resolved = _data_path("SOME_PATH_KNOB", "unused-default")
        assert Path(resolved).is_absolute()
        assert Path(resolved) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("on", True),
        ("ON", True),
        ("YES", True),
        ("1", True),
        ("  true  ", True),
        ('"true"', True),
        ("false", False),
        ("FALSE", False),
        ("off", False),
        ("No", False),
        ("0", False),
        ("  false  ", False),
    ],
)
def test_boolean_knob_accepts_every_reasonable_spelling(monkeypatch, raw, expected):
    """`TRUE`, `True` and `on` all mean true.

    Before the fix the parser was
    `os.getenv(X, "true").lower() in ("1", "true", "yes")`, so `TRUE` and `on`
    silently read as FALSE: a whole feature turned off with no signal anywhere.
    The default is set to the opposite of `expected` so a parser that ignored
    the value entirely would also fail this.
    """
    monkeypatch.delenv("SOME_TOGGLE", raising=False)
    monkeypatch.setenv("SOME_TOGGLE", raw)
    assert _env_bool("SOME_TOGGLE", not expected) is expected


def test_unrecognised_boolean_value_raises_instead_of_defaulting(monkeypatch):
    """`ENABLE_X=treu` must not silently read as "feature disabled"."""
    monkeypatch.setenv("SOME_TOGGLE", "treu")
    with pytest.raises(ValueError, match="SOME_TOGGLE"):
        _env_bool("SOME_TOGGLE", True)


def test_unset_boolean_knob_keeps_its_default(monkeypatch):
    """An absent variable is a legitimate "not configured", not a typo."""
    monkeypatch.delenv("SOME_TOGGLE", raising=False)
    assert _env_bool("SOME_TOGGLE", True) is True
    monkeypatch.setenv("SOME_TOGGLE", "   ")
    assert _env_bool("SOME_TOGGLE", False) is False


def test_truthy_spellings_match_the_setup_sh_auth_trust_warning():
    """The accepted true-spellings must be exactly the ones setup.sh warns about.

    `setup.sh services` prints a security warning when
    `AUTH_TRUST_X_FORWARDED_FOR` is a forced True, and its regex comment claims
    it covers "exactly the ones config._env_tristate reads". That claim is
    load-bearing: a forced True leaves X-Forwarded-For trusted from ANY peer
    (issue #245), and the setup warning is the only signal for it. So if the
    truthy set is widened and the regex is not, a value like `=y` forces header
    trust with no warning anywhere.
    """
    setup_sh = (Path(__file__).resolve().parents[2] / "setup.sh").read_text()
    regex = re.search(
        r"grep -qiE '\^AUTH_TRUST_X_FORWARDED_FOR=(?P<body>[^']*)'", setup_sh
    )
    assert regex is not None, (
        "could not find the AUTH_TRUST_X_FORWARDED_FOR spellings regex in setup.sh; "
        "this guard must be updated to follow it"
    )
    # Strip the POSIX character classes first: their names ("space") sit inside
    # brackets and are not spellings.
    body = re.sub(r"\[\[:?[^]]*\]\]", "", regex.group("body"))
    guarded = frozenset(re.findall(r"[a-z0-9]+", body))
    assert config_module._TRUE_VALUES == guarded, (
        f"config accepts {sorted(config_module._TRUE_VALUES)} as true but setup.sh "
        f"only warns for {sorted(guarded)}; a value in the difference forces "
        "X-Forwarded-For trust from any peer with no setup warning"
    )


@pytest.mark.parametrize("knob", BOOL_KNOBS)
def test_every_boolean_knob_goes_through_the_validating_parser(load_config, knob):
    """A typo in any shipped ENABLE_* knob is a boot failure, not a silent off.

    Asserted through the real class body: a knob left on the old inline
    `.lower() in ("1", "true", "yes")` expression cannot pass this.
    """
    with pytest.raises(ValueError, match=knob):
        load_config({knob: "definitely-not-a-bool"})


@pytest.mark.parametrize("knob", BOOL_KNOBS)
def test_boolean_knob_still_defaults_to_enabled_when_unset(load_config, knob):
    """The control for the test above: an absent knob keeps its shipped default."""
    cfg = load_config({})
    assert getattr(cfg, knob) is True, f"{knob} no longer defaults to enabled"


class _Cfg:
    """Stand-in exposing only the attributes `ensure_data_paths_ready` reads.

    Deliberately has no RERANK_ONNX_DIR: that knob is inert (the ONNX backend
    was removed) and requiring its directory would fail a perfectly healthy
    deploy. Its absence here is the assertion.
    """

    def __init__(self, chat: str, auth: str, vocab: str):
        self.CHAT_DB_PATH = chat
        self.AUTH_DB_PATH = auth
        self.QUERY_FIX_VOCAB_PATH = vocab


def test_usable_data_paths_pass_validation(tmp_path):
    """The control case: writable, creatable locations are accepted."""
    cfg = _Cfg(
        str(tmp_path / "chat" / "chat.db"),
        str(tmp_path / "auth" / "auth.db"),
        str(tmp_path / "vocab" / "query_vocab.json.gz"),
    )
    ensure_data_paths_ready(cfg)
    # The directories it created are the ones the stores will open files in.
    assert (tmp_path / "chat").is_dir()
    assert (tmp_path / "auth").is_dir()
    assert (tmp_path / "vocab").is_dir()


def test_unwritable_data_directory_fails_loudly_instead_of_creating_an_empty_db(tmp_path):
    """A read-only parent must stop the boot, not yield a fresh empty database.

    This is the exact production symptom: `os.makedirs(..., exist_ok=True)`
    succeeds against an existing directory, `sqlite3.connect` then succeeds
    too, and the app goes on serving a brand-new database as if it had no
    history at all.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # readable + traversable, not writable
    if os.access(locked, os.W_OK):  # running as a user who ignores the mode
        pytest.skip("cannot make a directory unwritable for this user")
    try:
        cfg = _Cfg(
            str(locked / "chat.db"),
            str(tmp_path / "auth.db"),
            str(tmp_path / "query_vocab.json.gz"),
        )
        with pytest.raises(RuntimeError) as excinfo:
            ensure_data_paths_ready(cfg)
        message = str(excinfo.value)
        assert "CHAT_DB_PATH" in message
        assert str(locked) in message
        assert "EMPTY" in message, (
            "the failure must say what starting anyway would cost, not merely that "
            f"something failed: {message!r}"
        )
    finally:
        locked.chmod(0o700)


def test_data_path_through_a_regular_file_fails_naming_the_knob(tmp_path):
    """A path whose parent is a regular file cannot be created -- say which knob."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    cfg = _Cfg(
        str(blocker / "nested" / "chat.db"),
        str(tmp_path / "auth.db"),
        str(tmp_path / "query_vocab.json.gz"),
    )
    with pytest.raises(RuntimeError, match="CHAT_DB_PATH"):
        ensure_data_paths_ready(cfg)


def test_auth_db_location_is_validated_too(tmp_path):
    """Not just the chat DB: a bad AUTH_DB_PATH is the same silent-empty-DB bug."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    cfg = _Cfg(
        str(tmp_path / "chat.db"),
        str(blocker / "auth.db"),
        str(tmp_path / "query_vocab.json.gz"),
    )
    with pytest.raises(RuntimeError, match="AUTH_DB_PATH"):
        ensure_data_paths_ready(cfg)


def test_missing_vocab_file_is_not_a_startup_failure(tmp_path):
    """The vocab file is optional by design; only its directory must be usable.

    `init_fixer` already treats a missing vocabulary as a no-op, so failing
    startup on a missing file would break a deploy that never built one.
    """
    cfg = _Cfg(
        str(tmp_path / "chat.db"),
        str(tmp_path / "auth.db"),
        str(tmp_path / "no-such-dir" / "query_vocab.json.gz"),
    )
    ensure_data_paths_ready(cfg)
    assert not (tmp_path / "no-such-dir" / "query_vocab.json.gz").exists()
    assert (tmp_path / "no-such-dir").is_dir()


def test_startup_lifespan_refuses_to_boot_on_an_unusable_data_location(monkeypatch, tmp_path):
    """The real lifespan must abort on a bad data dir, before any store opens.

    A validation function nobody calls fixes nothing, and a call placed after
    `ChatStore.connect` is too late -- the empty database already exists by
    then. So this drives the actual startup context manager with a data
    location that cannot be used and asserts the store was never constructed.
    """
    from app import chat as chat_module
    from app import main

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(main.config, "CHAT_DB_PATH", str(blocker / "nested" / "chat.db"))

    constructed = []

    class _Tripwire:
        def __init__(self, path):
            constructed.append(path)

    monkeypatch.setattr(chat_module, "ChatStore", _Tripwire)

    async def _enter():
        async with main.lifespan(main.app):
            pytest.fail("lifespan started up despite an unusable CHAT_DB_PATH")

    with pytest.raises(RuntimeError, match="CHAT_DB_PATH"):
        asyncio.run(_enter())
    assert constructed == [], "a store was constructed despite the failed validation"


def test_lifespan_is_still_an_async_context_manager():
    """Guard the wiring itself: a lifespan must be decorated to run at all.

    FastAPI only invokes the hook when it is an async context manager; as a
    bare coroutine it is never awaited, so the whole startup path -- including
    the data-location validation -- would be silently dead code.
    """
    from app import main

    assert hasattr(main.lifespan, "__wrapped__"), (
        "app.main.lifespan is not wrapped by @asynccontextmanager, so FastAPI "
        "never runs it and the startup data-path validation never executes"
    )
