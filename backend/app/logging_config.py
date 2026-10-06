"""Make the app's own log records actually reach stderr.

Uvicorn's worker applies ``uvicorn.config.LOGGING_CONFIG`` before the app is
imported, and that config leaves the ROOT logger at WARNING with no handlers,
hanging a single ``DefaultHandler`` off the ``uvicorn`` logger with
``propagate: False``. Every app module logger is a child of root, so it inherits
WARNING, owns no handler, and falls through to ``logging.lastResort``: an INFO
call is silently discarded while WARNING survives.

``configure_logging`` is called once from ``app.main`` at import, which happens
before FastAPI's lifespan runs, so operator-visible events from boot and from the
background loops are emitted under the real startup path.

What is deliberately NOT done, and why:

* **The root LOGGER's level is never changed.** The handler carries the app's
  level; raising the root level to INFO would switch on every third-party logger
  (httpx, openai, ...) and their per-request output in production too.
* **Only the app's own loggers get the app's level**, so third-party loggers
  keep the root level they have today. ``APP_LOGGERS`` is the list; a module that
  grows a new bare-named module-level ``logger`` must be added there or it goes
  quiet again.
* **Existing handlers are left alone.** ``logging.config.dictConfig`` in its
  default (non-incremental) mode removes every handler already installed on root
  and calls ``logging.shutdown`` on the process-wide handler list, clobbering
  handlers this process never asked to touch (a test harness's capture handler,
  an embedding application's). The ``uvicorn`` logger is never named or
  reconfigured, so its handler and ``propagate: False`` survive and its output is
  written once, never duplicated by the app's handler.
"""

import logging
import sys

from app.config import config

logger = logging.getLogger(__name__)

# The levels an operator may set through LOG_LEVEL. NOTSET is deliberately not
# offered: it means "inherit", handing these loggers the root level (WARNING
# under uvicorn) and silently dropping the records this module exists to emit.
VALID_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

# Every logger the app package creates. "app" covers the modules logging under
# their own dotted name via ``getLogger(__name__)``; the rest are bare names,
# which is why they need naming one by one.
APP_LOGGERS = (
    "app",
    "analytics",
    "auth",
    "cache",
    "chat",
    "close_guard",
    "cost_budget",
    "diversity",
    "encoders",
    "health",
    "llm",
    "query_fix",
    "reranker",
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Marks the handler this module owns, so a repeat call updates that one handler
# instead of stacking a second copy of the app's output.
_APP_HANDLER_ATTR = "_vccircle_app_handler"


def _resolve_level(requested: str) -> tuple[int, str | None]:
    """Return ``(level number, complaint or None)`` for a requested level name.

    An unrecognised value resolves to INFO with a complaint instead of raising:
    a typo in an operator's .env must not take the API down, but it must not
    pass for a working setting either.
    """
    name = (requested or "").strip().upper()
    if name in VALID_LEVELS:
        return logging.getLevelNamesMapping()[name], None
    complaint = f"LOG_LEVEL={requested!r} is not one of {', '.join(VALID_LEVELS)}; logging at INFO instead"
    return logging.INFO, complaint


def _app_handler(level: int) -> logging.Handler:
    """Return this module's single root handler, at ``level``.

    Root keeps whatever level it already had: the handler's level is what admits
    the app's INFO records, and a record reaches it only after the emitting
    logger's own level check has passed.
    """
    existing = installed_handler()
    if existing is not None:
        existing.setLevel(level)
        return existing
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(level)
    setattr(handler, _APP_HANDLER_ATTR, True)
    logging.getLogger().addHandler(handler)
    return handler


def app_handlers() -> list[logging.Handler]:
    """Every root handler this module owns: one, unless a second was stacked."""
    return [handler for handler in logging.getLogger().handlers if getattr(handler, _APP_HANDLER_ATTR, False)]


def installed_handler() -> logging.Handler | None:
    """The root handler this module installed, or None if it has not run."""
    handlers = app_handlers()
    return handlers[0] if handlers else None


def configure_logging(level: str | None = None) -> str:
    """Install the app's stderr handler and set the app loggers' level.

    ``level`` overrides ``config.LOG_LEVEL`` (i.e. the ``LOG_LEVEL`` env var);
    both accept CRITICAL, ERROR, WARNING, INFO or DEBUG, case-insensitively.
    Returns the level name actually applied. Safe to call more than once: the
    second call reuses the handler it installed the first time.
    """
    requested = config.LOG_LEVEL if level is None else level
    levelno, complaint = _resolve_level(requested)
    _app_handler(levelno)
    for name in APP_LOGGERS:
        app_logger = logging.getLogger(name)
        app_logger.setLevel(levelno)
        app_logger.propagate = True
    if complaint:
        # Emitted after the handler exists, so the operator actually sees it.
        logger.warning(complaint)
    applied = logging.getLevelName(levelno)
    logger.info("app loggers at %s, %d logger(s) writing to stderr", applied, len(APP_LOGGERS))
    return applied
