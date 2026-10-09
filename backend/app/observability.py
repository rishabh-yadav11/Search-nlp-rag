"""Request-id middleware, access log and the top-level exception handlers.

``app.logging_config.configure_logging`` is the single owner of the app's root
handler, so this module never installs one of its own (two root handlers means
every record is written twice) and never sets a logger level (the next
``configure_logging()`` call would undo it). It adds a *filter* to the handler
``configure_logging()`` installed; because that handler's format has no
``%(request_id)s`` field, every line written here repeats the id in the message
text, which stays greppable however a deployment formats its logs.

Starlette's stack is ``ServerErrorMiddleware -> user middlewares ->
ExceptionMiddleware -> router``:

* An ``Exception`` handler is invoked by ``ServerErrorMiddleware``, OUTSIDE the
  user middleware, so its response never passes the middleware's ``send`` wrapper
  and must carry ``X-Request-ID`` itself -- reading the id from
  ``scope["state"]``, since the ContextVar has been reset by then.
* A ``RequestValidationError`` handler is invoked by ``ExceptionMiddleware``,
  INSIDE the stack, so its response is stamped by the middleware like any other.

Raw ASGI on purpose: ``BaseHTTPMiddleware`` runs downstream code in a separate
task that does not inherit the caller's contextvars, and buffers a
``StreamingResponse``, which breaks the SSE chat stream.
"""

import logging
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

# The access line is logged at INFO with no level of its own: the `app` logger it
# inherits from already sits at the operator's LOG_LEVEL, and pinning one here
# would be undone by the next configure_logging() call.

# Raw ASGI header name, pre-lowercased: byte compare avoids decoding every header.
_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")

def _inbound_request_id(scope: Scope) -> str | None:
    """The raw inbound ``X-Request-ID`` value, or None.

    Read from the raw bytes, not Starlette's ``Headers``: it must reach
    ``is_valid_request_id`` unmodified so a CR/LF forgery is rejected instead of
    normalised away. ``latin-1`` is ASGI's header charset and never fails.
    """
    for name, value in scope.get("headers") or ():
        if name.lower() == _REQUEST_ID_HEADER_BYTES:
            return value.decode("latin-1")
    return None


def _with_request_id(headers: list[tuple[bytes, bytes]], request_id: str) -> list[tuple[bytes, bytes]]:
    """The response headers with ``X-Request-ID`` set to ``request_id``.

    Appended, never substituted over the whole list, so CORS and TrustedHost
    headers survive; an existing entry is replaced rather than duplicated.
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
    ``Exception`` handler owns the 500, and swallowing it here would hide the
    traceback from gunicorn's own error logging.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # lifespan/websocket traffic carries no request id and must not get one.
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
                # Only while http.response.start is still mutable: adding a header
                # after the body starts is a protocol violation.
                message = {
                    **message,
                    "headers": _with_request_id(message.get("headers") or [], request_id),
                }
            await send(message)

        try:
            with bound_request_id(request_id):
                await self.app(scope, receive, send_with_request_id)
        finally:
            duration = (time.perf_counter() - started_at)
            # `extra` rather than the ContextVar: an exception unwinding through
            # this `finally` has already reset the ContextVar, and RequestIdFilter
            # never overwrites a caller-set field -- without it this record, the
            # most frequent one in the process, would render the "no id" placeholder.
            access_logger.info(
                "%s %s -> %s in %.1fms request_id=%s user_id=%s",
                scope.get("method", "-"),
                scope.get("path", "-"),
                500 if status_code is None else status_code,
                duration * 1000,
                request_id,
                _user_id(scope),
                extra={"request_id": request_id},
            )
            # Prometheus hook stays in this middleware so EVERY route gets counted
            # once, regardless of handler outcome. Imported lazily below the access
            # log: /metrics itself passes through this same middleware, and eagerly
            # importing metrics here would fight the router-import ordering in main.
            try:
                from app.metrics import inc_http_request

                inc_http_request(
                    scope.get("method", "-"),
                    scope.get("path", "-"),
                    500 if status_code is None else status_code,
                    duration,
                )
            except Exception:
                logger.debug("prometheus request hook failed", exc_info=True)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer any unhandled exception with an opaque 500 that carries the id.

    The id is repeated in the message text, not left to the ``request_id`` field:
    this record can reach a log with no ``request_id``-aware formatter (an
    operator's own ``dictConfig``, or ``logging.lastResort``), and the text form
    is findable by grep either way.
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
    # else: no str(exc), class or traceback, since this reaches an unauthenticated
    # caller and a crash message may quote a DSN, a file path or a token.
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Answer a body/query validation failure with the standard 422 shape.

    FastAPI's default handler returns ``exc.errors()`` verbatim, which for pydantic
    failures carries ``ctx`` (the raw exception object) and ``input`` (an
    arbitrary value): neither is reliably JSON-serialisable -- a non-serialisable
    one turns a 422 into a 500 -- and both can quote values the caller never sent.
    Only the four descriptive string fields are kept.
    """
    request_id = scope_request_id(request.scope) or resolve_request_id(None)
    errors: list[dict[str, Any]] = [
        {key: error[key] for key in ("type", "loc", "msg", "url") if key in error} for error in exc.errors()
    ]
    # INFO, not ERROR: a rejected request is the caller's problem, and an ERROR
    # record per 422 would bury the genuine server faults.
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


def attach_request_id_filter() -> logging.Handler | None:
    """Stamp the request id onto every record the app's own handler renders.

    ``configure_logging()`` owns the app's root handler and exposes it through
    ``installed_handler()``; the RequestIdFilter goes on THAT handler. A second
    root handler would write every record twice, and a *logger*-level filter is
    worse: ``Logger.callHandlers`` consults a handler's filters only on the path
    to that handler, so it never runs for records that merely propagate through.

    A filter rather than a format field, because the format is
    ``configure_logging()``'s to change; what the filter guarantees regardless of
    the format is that ``record.request_id`` exists on every record for any
    consumer that reads records rather than rendered text.

    MUST be called after ``configure_logging()`` has run: ``installed_handler()``
    returns None before that, and this deliberately creates nothing in that case
    rather than installing a handler of its own. ``app.main`` calls it from the
    middleware wiring block, below the ``configure_logging()`` call at its top.

    Idempotent. Returns that handler, or None when logging is not configured.
    """
    from app.logging_config import installed_handler

    handler = installed_handler()
    if handler is None:
        return None
    if not any(isinstance(existing, RequestIdFilter) for existing in handler.filters):
        handler.addFilter(RequestIdFilter())
    return handler
