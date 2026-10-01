"""The app's log records must actually be emitted.

Before the fix, nothing in the app configured logging. Uvicorn's worker applies
``uvicorn.config.LOGGING_CONFIG``, which leaves the ROOT logger at WARNING with
no handlers and hangs a single handler off the ``uvicorn`` logger with
``propagate: False``. Every app module logger is a child of root, so it inherited
WARNING, owned no handler, and ``logger.info(...)`` was discarded: the boot
lines an operator actually needs never appeared, with no signal they were lost.

The tests here pin the three properties the fix delivers, and the one cost it
must not pay:

* an INFO record from a module logger is emitted, under the real startup path
  (``app.main``, in a fresh interpreter -- not a hand-assembled logger tree);
* a WARNING is written exactly once, and uvicorn keeps its own handler, so
  nothing is doubled and nothing of uvicorn's is clobbered;
* the level is an operator knob, while the root logger's level stays put so
  third-party INFO output is not switched on in production.

Every test that needs a known starting point asks for ``unconfigured_logging``
first: ``app.main`` configures logging at import, so whichever test imported it
earlier in the session would otherwise decide the state this module sees.
"""

import importlib
import logging
import logging.config
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
from uvicorn.config import LOGGING_CONFIG

from app import config as app_config
from app import logging_config

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Undo every global logging change a test in this module makes.

    The app installs a handler on the root logger and sets levels on the app
    loggers -- process-wide, by design. A test that reconfigures logging would
    otherwise hand its levels and handlers to the rest of the session.
    """
    root = logging.getLogger()
    root_state = (root.level, list(root.handlers))
    saved = {
        name: (log.level, list(log.handlers), log.propagate, log.disabled)
        for name, log in list(logging.Logger.manager.loggerDict.items())
        if isinstance(log, logging.Logger)
    }
    yield
    for name, log in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(log, logging.Logger):
            continue
        state = saved.get(name)
        if state is None:
            log.setLevel(logging.NOTSET)
            log.handlers = []
            log.propagate = True
            log.disabled = False
        else:
            level, handlers, propagate, disabled = state
            log.setLevel(level)
            log.handlers = handlers
            log.propagate = propagate
            log.disabled = disabled
    root.setLevel(root_state[0])
    root.handlers = root_state[1]


@pytest.fixture
def unconfigured_logging():
    """The logging state the app booted in before the fix: no handler of the
    app's own, the app loggers at their inherited level and root at WARNING --
    what uvicorn leaves behind."""
    handler = logging_config.installed_handler()
    if handler is not None:
        logging.getLogger().removeHandler(handler)
    for name in logging_config.APP_LOGGERS:
        app_logger = logging.getLogger(name)
        app_logger.setLevel(logging.NOTSET)
        app_logger.propagate = True
    logging.getLogger().setLevel(logging.WARNING)


def _configure(level: str = "INFO") -> logging.Handler:
    """Configure logging at ``level`` and return the handler it installed."""
    assert logging_config.configure_logging(level) == level.upper()
    handler = logging_config.installed_handler()
    assert handler is not None
    return handler


def _startup_in_a_fresh_process(body: str, env_extra: dict[str, str] | None = None) -> str:
    """Run ``body`` in a fresh interpreter that has imported app.main; return stderr.

    A subprocess, because that is the only way to see the real startup path:
    ``app.main`` is imported once per process, so an in-process check would
    report whatever an earlier test left behind. ``LOG_LEVEL`` is controlled here
    and ``.env`` loading is switched off, so no ambient file decides the level.
    """
    env = dict(os.environ)
    env.pop("LOG_LEVEL", None)
    env.update(env_extra or {})
    env["PYTHON_DOTENV_DISABLED"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", f"import app.main\n{body}"],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stderr


def _from(caplog, name: str) -> list[logging.LogRecord]:
    """The captured records of one logger, ignoring the config's own line."""
    return [record for record in caplog.records if record.name == name]


class TestTheBug:
    def test_a_module_logger_info_is_dropped_while_unconfigured(self, caplog, unconfigured_logging):
        """The defect itself: the record is never created, so nothing sees it."""
        assert logging_config.installed_handler() is None
        assert logging.getLogger("chat").getEffectiveLevel() == logging.WARNING

        caplog.clear()
        logging.getLogger("chat").info("purged 3 expired conversation(s)")

        assert caplog.records == []

    def test_uvicorn_worker_config_is_what_the_app_boots_into(self, unconfigured_logging):
        """The premise of the fix, stated so a uvicorn upgrade cannot hide it.

        If a future uvicorn gives root a handler and an INFO level, this premise
        is false and the other tests here would be measuring something the
        running app no longer needs.
        """
        logging.config.dictConfig(LOGGING_CONFIG)
        uvicorn_logger = logging.getLogger("uvicorn")

        assert logging.getLogger().level == logging.WARNING
        assert uvicorn_logger.handlers and uvicorn_logger.propagate is False
        assert logging.getLogger("chat").getEffectiveLevel() == logging.WARNING


class TestRecordsAreEmitted:
    def test_a_module_logger_info_is_emitted_once_configured(self, caplog, unconfigured_logging):
        caplog.clear()
        _configure()

        logging.getLogger("chat").info("purged 3 expired conversation(s)")

        records = _from(caplog, "chat")
        assert [r.levelno for r in records] == [logging.INFO], caplog.records
        assert [r.getMessage() for r in records] == ["purged 3 expired conversation(s)"]

    def test_a_dotted_module_logger_info_is_emitted(self, caplog, unconfigured_logging):
        """The ``logging.getLogger(__name__)`` modules log as ``app.<module>``."""
        caplog.clear()
        _configure()

        logging.getLogger("app.recommender").info("reranked 8 candidates")

        assert [r.getMessage() for r in _from(caplog, "app.recommender")] == ["reranked 8 candidates"]

    def test_the_record_is_written_with_timestamp_level_and_name(self, unconfigured_logging):
        """Operators grep these lines, so the app's own format is pinned: the
        bare message lastResort printed is barely better than no line at all."""
        _configure()
        formatter = logging_config.installed_handler().formatter
        record = logging.LogRecord("chat", logging.INFO, __file__, 1, "purged 3 expired conversations", None, None)

        formatted = formatter.format(record)

        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO chat: purged 3 expired conversations",
            formatted,
        ), formatted


class TestNoDuplication:
    def test_a_warning_is_written_by_one_handler_and_not_by_last_resort(self, monkeypatch, unconfigured_logging):
        """One line on the app's stderr per record, not two.

        The old path wrote WARNING through ``logging.lastResort``; the fix adds a
        real handler, which is found first and makes lastResort unreachable. A
        second copy of the app's handler on root -- which a repeat call that
        stacked one instead of reusing it would produce -- would put the same
        line on stderr twice.
        """
        handler = _configure()
        _configure()  # a second call must reuse the handler, not stack another
        assert logging_config.app_handlers() == [handler]
        assert logging.getLogger("chat").handlers == []

        via_last_resort = []

        class _FallbackProbe(logging.Handler):
            def emit(self, record):
                via_last_resort.append(record.getMessage())

        monkeypatch.setattr(logging, "lastResort", _FallbackProbe())
        written = []
        monkeypatch.setattr(handler, "emit", lambda record: written.append(record.getMessage()))

        logging.getLogger("chat").warning("purged 2 expired token(s)")

        assert written == ["purged 2 expired token(s)"]
        assert via_last_resort == []

    def test_uvicorn_keeps_its_own_handler_and_its_output_is_not_duplicated(self, monkeypatch, unconfigured_logging):
        logging.config.dictConfig(LOGGING_CONFIG)
        uvicorn_handler = logging.getLogger("uvicorn").handlers[0]

        app_handler = _configure()

        assert logging.getLogger("uvicorn").handlers == [uvicorn_handler]
        assert logging.getLogger("uvicorn").propagate is False
        assert logging.getLogger("uvicorn").level == logging.INFO

        uvicorn_written, app_written = [], []
        monkeypatch.setattr(uvicorn_handler, "emit", lambda record: uvicorn_written.append(record.getMessage()))
        monkeypatch.setattr(app_handler, "emit", lambda record: app_written.append(record.getMessage()))

        logging.getLogger("uvicorn.error").info("Application startup complete.")

        assert uvicorn_written == ["Application startup complete."]
        assert app_written == []

    def test_third_party_loggers_are_not_switched_on(self, unconfigured_logging):
        """The cost this fix must not pay.

        Every third-party logger is a child of root too, so raising ROOT to INFO
        would switch on httpx/openai's per-request output in production. The
        app's level belongs on the app's loggers and on its handler.
        """
        _configure("DEBUG")

        assert logging.getLogger("chat").getEffectiveLevel() == logging.DEBUG
        assert logging.getLogger().level == logging.WARNING
        for name in ("httpx", "openai", "some_library_the_app_does_not_own"):
            assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING, name


class TestTheLevelIsAnOperatorKnob:
    def test_the_level_comes_from_the_log_level_knob(self, monkeypatch, caplog, unconfigured_logging):
        monkeypatch.setattr(app_config.config, "LOG_LEVEL", "WARNING")
        caplog.clear()

        assert logging_config.configure_logging() == "WARNING"

        logging.getLogger("chat").info("dropped again")
        logging.getLogger("chat").warning("kept")
        assert [r.getMessage() for r in _from(caplog, "chat")] == ["kept"]

    def test_the_level_argument_wins_and_is_case_insensitive(self, caplog, unconfigured_logging):
        caplog.clear()

        assert logging_config.configure_logging("debug") == "DEBUG"

        assert logging.getLogger("chat").getEffectiveLevel() == logging.DEBUG
        logging.getLogger("chat").debug("spelled in lower case")
        assert [r.getMessage() for r in _from(caplog, "chat")] == ["spelled in lower case"]

    def test_an_unrecognised_level_falls_back_to_info_and_says_so(self, caplog, unconfigured_logging):
        caplog.clear()

        applied = logging_config.configure_logging("LOUD")

        assert applied == "INFO"
        assert logging.getLogger("chat").getEffectiveLevel() == logging.INFO
        complaints = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("LOG_LEVEL='LOUD'" in message for message in complaints), complaints

    def test_not_set_is_refused_because_it_would_silently_re_drop_the_records(self, unconfigured_logging):
        """NOTSET means "inherit", i.e. root's WARNING: the pre-fix behaviour."""
        assert logging_config.configure_logging("NOTSET") == "INFO"
        assert logging.getLogger("chat").getEffectiveLevel() == logging.INFO


class TestRealStartupPath:
    def test_importing_app_main_emits_module_logger_info_at_the_default_level(self):
        stderr = _startup_in_a_fresh_process(
            "import logging, sys\n"
            "print('EFFECTIVE', logging.getLevelName(logging.getLogger('chat').getEffectiveLevel()),"
            " file=sys.stderr)\n"
            "logging.getLogger('chat').info('bootstrapped admin account ops@example.com')\n"
        )

        assert "EFFECTIVE INFO" in stderr
        assert "INFO chat: bootstrapped admin account ops@example.com" in stderr
        assert stderr.count("chat: bootstrapped admin account ops@example.com") == 1

    def test_the_env_var_raises_the_level_of_a_real_process(self):
        stderr = _startup_in_a_fresh_process(
            "import logging, sys\n"
            "print('EFFECTIVE', logging.getLevelName(logging.getLogger('chat').getEffectiveLevel()),"
            " file=sys.stderr)\n"
            "logging.getLogger('chat').debug('third leg timing')\n"
            "logging.getLogger('httpx').info('third party noise')\n",
            env_extra={"LOG_LEVEL": "debug"},
        )

        assert "EFFECTIVE DEBUG" in stderr
        assert "DEBUG chat: third leg timing" in stderr
        assert "third party noise" not in stderr


class TestEveryAppLoggerIsCovered:
    def test_each_module_logger_in_the_app_is_configured(self, unconfigured_logging):
        """A new module-level ``logger`` that APP_LOGGERS does not name goes quiet.

        Every module in the ``app`` package is imported and its ``logger``
        checked. The app's loggers are bare names ("chat", "auth", ...) that
        share no parent below root, so a new one is not covered by accident.
        """
        _configure()
        import app

        found = {}
        for info in pkgutil.iter_modules(app.__path__):
            module = importlib.import_module(f"app.{info.name}")
            module_logger = getattr(module, "logger", None)
            if isinstance(module_logger, logging.Logger):
                found[module_logger.name] = module_logger

        assert len(found) >= 10, f"expected the app's module loggers, found {sorted(found)}"
        for name, module_logger in found.items():
            covered = name in logging_config.APP_LOGGERS or name.startswith("app.")
            assert covered, f"{name} is a bare-named logger missing from app.logging_config.APP_LOGGERS"
            assert module_logger.getEffectiveLevel() == logging.INFO, name
            assert module_logger.propagate is True, name
