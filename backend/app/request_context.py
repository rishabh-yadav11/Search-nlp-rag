"""Per-request correlation id: resolution, propagation and log stamping.

One inbound or generated id follows a request end to end: it is echoed on the
response as ``X-Request-ID``, carried in a ContextVar for every log record
emitted downstream, and handed to the top-level exception handler so a bare
``500 Internal Server Error`` can be traced back to the request that caused it.

The inbound header is honoured ONLY when it matches ``[A-Za-z0-9._-]{1,64}``.
That restriction is a log-injection / header-spoofing guard, not cosmetics: a
value carrying ``\\n`` would otherwise be echoed verbatim into a response header
and stamped into a log line, letting a caller forge a whole log record (or
smuggle a second header) with a single request.
"""

import logging
import re
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LEN = 64
# Stands in for "no id bound", so a formatter can interpolate the field
# unconditionally. It is a placeholder, NOT an id: `scope_request_id` returns
# the empty string instead so a caller can tell "absent" from "present".
NO_REQUEST_ID = "-"

_REQUEST_ID_RE = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def is_valid_request_id(value: object) -> bool:
    """True only for a str of 1..64 chars drawn from ``[A-Za-z0-9._-]``.

    Non-strings, empty strings, anything with whitespace/CR/LF/colon and
    anything longer than 64 chars are rejected -- see the module docstring.
    """
    if not isinstance(value, str):
        return False
    if not value or len(value) > REQUEST_ID_MAX_LEN:
        return False
    return _REQUEST_ID_RE.match(value) is not None


def resolve_request_id(raw: object) -> str:
    """Honour a valid inbound id, otherwise mint a fresh 32-char hex one.

    The rejected value is never echoed anywhere -- not in the response header,
    not in the 500 body -- so a hostile header can neither spoof a correlation
    id nor reach a log line.
    """
    if isinstance(raw, str) and is_valid_request_id(raw):
        return raw
    return uuid.uuid4().hex


def current_request_id() -> str:
    """The id bound to the running request, or ``NO_REQUEST_ID`` outside one.

    Never returns ``None``: a logging filter calls this for every record in the
    process, including ones emitted outside any request.
    """
    return _request_id_var.get() or NO_REQUEST_ID


@contextmanager
def bound_request_id(value: str) -> Iterator[None]:
    """Bind ``value`` for the duration of the block.

    The previous value is restored in a ``finally``, so a request that raises
    cannot leak its id into the next one handled on the same event loop.
    """
    token = _request_id_var.set(value)
    try:
        yield
    finally:
        _request_id_var.reset(token)


def scope_request_id(scope: MutableMapping[str, Any], *, assign: str | None = None) -> str:
    """Read (and optionally seed) the id cached in ``scope["state"]``.

    ``scope["state"]`` survives the ContextVar reset: by the time Starlette's
    ``ServerErrorMiddleware`` -- which sits OUTSIDE the user middleware stack and
    therefore outside the ``bound_request_id`` block -- invokes the ``Exception``
    handler, the id is only still available here.

    Returns ``""`` when no id is stored, so a caller can write the idiomatic
    ``scope_request_id(scope) or resolve_request_id(None)``. It does NOT return
    ``NO_REQUEST_ID``: ``"-"`` is truthy and would pass for a real id.
    """
    state = scope.setdefault("state", {})
    stored = state.get("request_id")
    if isinstance(stored, str) and is_valid_request_id(stored):
        return stored
    if assign is None:
        return ""
    state["request_id"] = assign
    return assign


class RequestIdFilter(logging.Filter):
    """Stamp ``record.request_id`` unless the record already carries one.

    An explicit ``extra={"request_id": ...}`` wins, so a caller that knows
    better than the ambient ContextVar (a background job correlating several
    requests, say) is never overwritten. Returns True for every record: the
    field is decoration, and dropping records would hide real errors.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        return True
