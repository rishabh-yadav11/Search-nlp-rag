"""Request-id middleware, access log and the top-level exception handlers.

Starlette's stack is ``ServerErrorMiddleware -> user middlewares ->
ExceptionMiddleware -> router``, and this module's wiring depends on that shape:

* A handler registered for ``Exception`` is invoked by ``ServerErrorMiddleware``,
  which sits OUTSIDE the user middleware stack. The response it builds therefore
  never passes through :meth:`RequestIdMiddleware.__call__`'s ``send`` wrapper and
  has to carry ``X-Request-ID`` itself -- and the ContextVar has already been
  reset by the middleware's ``finally`` by then, so the handler reads the id from
  ``scope["state"]`` instead.
* A handler registered for ``RequestValidationError`` is invoked by
  ``ExceptionMiddleware``, INSIDE the stack, so its response is stamped by the
  middleware like any other.

The middleware is raw ASGI on purpose. ``BaseHTTPMiddleware`` runs downstream code
in a separate task that does not inherit the caller's contextvars (so nothing
downstream could ever see the bound id) and buffers a ``StreamingResponse``,
which breaks the SSE chat stream.
"""

import logging
import sys
import time
from collections.abc import Mapping
from typing import Any

from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.request_context import (
    NO_REQUEST_ID,
    REQUEST_ID_HEADER,
    RequestIdFilter,
    bound_request_id,
    resolve_request_id,
    scope_request_id,
)

access_logger = logging.getLogger("app.access")
logger = logging.getLogger("app.observability")

# The access line is logged at INFO, and this explicit level is what carries it
# to the root handler: `isEnabledFor` stops walking the hierarchy here, so the
# root logger's own WARNING default never suppresses the record, while every
# other module keeps its effective level and does not flood. Handler levels --
# not ancestor logger levels -- decide what a propagated record is rendered at,
# and install_root_log_handler's handler is at INFO.
access_logger.setLevel(logging.INFO)

# Raw ASGI header name, pre-lowercased: comparing bytes avoids decoding every
# header on the way in and on the way out.
_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] %(message)s"


def _inbound_request_id(scope: Scope) -> str | None:
    """The raw inbound ``X-Request-ID`` value, or None.

    Read straight from the raw bytes rather than through Starlette's ``Headers``:
    the value must reach ``is_valid_request_id`` unmodified, so a CR/LF forgery
    is rejected instead of normalised away. ``latin-1`` is ASGI's header charset
    and never fails; any byte outside the allowed set stays outside it and is
    rejected as invalid.
    """
    for name, value in scope.get("headers") or ():
        if name.lower() == _REQUEST_ID_HEADER_BYTES:
            return value.decode("latin-1")
    return None


def _with_request_id(headers: list[tuple[bytes, bytes]], request_id: str) -> list[tuple[bytes, bytes]]:
    """The response headers with ``X-Request-ID`` set to ``request_id``.

    Appended, never substituted over the whole list: CORS and TrustedHost
    headers must survive. An existing entry is replaced rather than duplicated,
    so a handler that sets the header itself does not produce two.
    """
    kept = [pair for pair in headers if pair[0].lower() != _REQUEST_ID_HEADER_BYTES]
    kept.append((_REQUEST_ID_HEADER_BYTES, request_id.encode("latin-1")))
    return kept


def _user_id(scope: Scope) -> str:
    """The authenticated user id on this scope, or ``NO_REQUEST_ID``."""
    user = scope.get("user")
    if isinstance(user, Mapping):
        value = user.get("id")
    else:
        value = getattr(user, "id", None)
    return str(value) if value else NO_REQUEST_ID


class RequestIdMiddleware:
    """Resolve and propagate the request id, then log one access line.

    Deliberately does NOT convert an exception into a response: the registered
    ``Exception`` handler owns the 500, and swallowing the error here would also
    hide the traceback from gunicorn's own error logging.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # lifespan/websocket traffic carries no request id and must not be
            # given one: a contextvar bound here would outlive nothing useful.
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(_inbound_request_id(scope))
        scope_request_id(scope, assign=request_id)
        started_at = time.perf_counter()
        status_code: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
                # Must happen while http.response.start is still mutable: once
                # the body has started, adding a header is a protocol violation
                # (and a StreamingResponse starts it immediately).
                message = {
                    **message,
                    "headers": _with_request_id(message.get("headers") or [], request_id),
                }
            await send(message)

        try:
            with bound_request_id(request_id):
                await self.app(scope, receive, send_with_request_id)
        finally:
            # `extra` rather than the ContextVar: the id is a local here because
            # an exception unwinding through this `finally` has already reset
            # the ContextVar, and the RequestIdFilter never overwrites a field
            # the caller set. Without it the most frequent record in the
            # process -- this one -- would render its `request_id` field as the
            # "no id" placeholder while carrying the real id only in the text.
            access_logger.info(
                "%s %s -> %s in %.1fms request_id=%s user_id=%s",
                scope.get("method", "-"),
                scope.get("path", "-"),
                500 if status_code is None else status_code,
                (time.perf_counter() - started_at) * 1000,
                request_id,
                _user_id(scope),
                extra={"request_id": request_id},
            )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer any unhandled exception with an opaque 500 that carries the id.

    The id is repeated in the message text, not left to the ``request_id``
    field: this record can reach a log with no ``request_id``-aware formatter
    (an operator's own ``dictConfig``, or the ``logging.lastResort`` handler that
    renders ``"%(message)s"`` alone when the root logger has none), and the id in
    the text is what makes the line findable by grep either way.
    """
    request_id = scope_request_id(request.scope) or resolve_request_id(None)
    with bound_request_id(request_id):
        logger.error(
            "Unhandled exception serving %s %s (request_id=%s)",
            request.method,
            request.url.path,
            request_id,
            exc_info=exc,
        )
    # The body carries the id (for the caller to quote in a report) and nothing
    # else: no str(exc), no exception class, no traceback, because this response
    # reaches an unauthenticated caller on any failure, including a crash whose
    # message may quote a DSN, a file path or a token.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Answer a body/query validation failure with the standard 422 shape.

    FastAPI's default handler returns ``exc.errors()`` verbatim, and for pydantic
    failures that carries ``ctx`` (the raw exception object) and ``input`` (an
    arbitrary Python value). Neither is reliably JSON-serialisable -- a
    non-serialisable one turns a 422 into a 500 -- and both can quote values the
    caller never sent. Only the four descriptive string fields are kept; the
    existing ``detail``-is-a-list-of-errors contract clients already parse is
    unchanged.
    """
    request_id = scope_request_id(request.scope) or resolve_request_id(None)
    errors: list[dict[str, Any]] = [
        {key: error[key] for key in ("type", "loc", "msg", "url") if key in error} for error in exc.errors()
    ]
    # INFO, not ERROR: a rejected request is the caller's problem. Raising an
    # ERROR record for every 422 would bury the genuine server faults.
    logger.info(
        "Rejected invalid request %s %s (request_id=%s): %s error(s)",
        request.method,
        request.url.path,
        request_id,
        len(errors),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


def _stamps_request_id(handler: logging.Handler) -> bool:
    """True if this handler's formatter already renders the request id."""
    fmt = getattr(handler.formatter, "_fmt", None)
    return bool(fmt) and "request_id" in fmt


def install_root_log_handler() -> None:
    """Attach the request-id-stamping handler to the ROOT logger, once.

    Idempotent: if a handler whose formatter renders ``request_id`` is already
    attached -- ours, or an operator's own ``dictConfig`` -- nothing is added, so
    calling this twice cannot duplicate every line in the process.

    The handler is on the ROOT logger (and the ``RequestIdFilter`` is on the
    HANDLER, not on a logger) because that is the only way a record from any
    module -- app code, redis, qdrant, httpx -- is stamped: filters attached to a
    logger do not run for records that merely propagate up to it. The filter
    guarantees ``request_id`` is present on every record, so the formatter can
    never raise ``internal error in logging``.

    The root logger's own level is deliberately left alone. Modules here log at
    WARNING precisely because INFO is dropped today (see the "TrustedHost allowed
    hosts" line in app.main), and dropping the root level to make the access line
    visible would flood production with them. The access line is instead logged
    at INFO with ``app.access`` pinned to INFO (see above) and the handler set to
    INFO, so it is still rendered in a default gunicorn/uvicorn deployment, where
    the root logger stays at WARNING -- and no level is claimed here that the
    code does not use.
    """
    root = logging.getLogger()
    if any(_stamps_request_id(handler) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(RequestIdFilter())
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
