

import logging
import sys

from app.config import config

logger = logging.getLogger(__name__)




VALID_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")




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



_APP_HANDLER_ATTR = "_vccircle_app_handler"


def _resolve_level(requested: str) -> tuple[int, str | None]:

    name = (requested or "").strip().upper()
    if name in VALID_LEVELS:
        return logging.getLevelNamesMapping()[name], None
    complaint = f"LOG_LEVEL={requested!r} is not one of {', '.join(VALID_LEVELS)}; logging at INFO instead"
    return logging.INFO, complaint


def _app_handler(level: int) -> logging.Handler:

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

    return [handler for handler in logging.getLogger().handlers if getattr(handler, _APP_HANDLER_ATTR, False)]


def installed_handler() -> logging.Handler | None:

    handlers = app_handlers()
    return handlers[0] if handlers else None


def configure_logging(level: str | None = None) -> str:

    requested = config.LOG_LEVEL if level is None else level
    levelno, complaint = _resolve_level(requested)
    _app_handler(levelno)
    for name in APP_LOGGERS:
        app_logger = logging.getLogger(name)
        app_logger.setLevel(levelno)
        app_logger.propagate = True
    if complaint:

        logger.warning(complaint)
    applied = logging.getLevelName(levelno)
    logger.info("app loggers at %s, %d logger(s) writing to stderr", applied, len(APP_LOGGERS))
    return applied
