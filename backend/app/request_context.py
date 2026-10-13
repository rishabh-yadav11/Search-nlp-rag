

import logging
import re
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LEN = 64



NO_REQUEST_ID = "-"

_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def is_valid_request_id(value: object) -> bool:

    if not isinstance(value, str):
        return False
    if not value or len(value) > REQUEST_ID_MAX_LEN or value == NO_REQUEST_ID:
        return False
    return _REQUEST_ID_RE.match(value) is not None


def resolve_request_id(raw: object) -> str:

    if isinstance(raw, str) and is_valid_request_id(raw):
        return raw
    return uuid.uuid4().hex


def current_request_id() -> str:

    return _request_id_var.get() or NO_REQUEST_ID


@contextmanager
def bound_request_id(value: str) -> Iterator[None]:

    token = _request_id_var.set(value)
    try:
        yield
    finally:
        _request_id_var.reset(token)


def scope_request_id(scope: MutableMapping[str, Any], *, assign: str | None = None) -> str:

    state = scope.setdefault("state", {})
    stored = state.get("request_id")
    if isinstance(stored, str) and is_valid_request_id(stored):
        return stored
    if assign is None:
        return ""
    state["request_id"] = assign
    return assign


class RequestIdFilter(logging.Filter):


    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        return True
