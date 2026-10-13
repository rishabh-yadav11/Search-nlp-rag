

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






_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")

def _inbound_request_id(scope: Scope) -> str | None:

    for name, value in scope.get("headers") or ():
        if name.lower() == _REQUEST_ID_HEADER_BYTES:
            return value.decode("latin-1")
    return None


def _with_request_id(headers: list[tuple[bytes, bytes]], request_id: str) -> list[tuple[bytes, bytes]]:

    kept = [pair for pair in headers if pair[0].lower() != _REQUEST_ID_HEADER_BYTES]
    kept.append((_REQUEST_ID_HEADER_BYTES, request_id.encode("latin-1")))
    return kept


def _user_id(scope: Scope) -> str:

    user = scope.get("user")
    if isinstance(user, Mapping):
        value = user.get("id")
    else:
        value = getattr(user, "id", None)
    return str(value) if value else NO_REQUEST_ID


class RequestIdMiddleware:


    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":

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

    request_id = scope_request_id(request.scope) or resolve_request_id(None)
    with bound_request_id(request_id):
        logger.error(
            "Unhandled exception serving %s %s (request_id=%s)",
            request.method,
            request.url.path,
            request_id,
            exc_info=exc,
        )



    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:

    request_id = scope_request_id(request.scope) or resolve_request_id(None)
    errors: list[dict[str, Any]] = [
        {key: error[key] for key in ("type", "loc", "msg", "url") if key in error} for error in exc.errors()
    ]


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

    from app.logging_config import installed_handler

    handler = installed_handler()
    if handler is None:
        return None
    if not any(isinstance(existing, RequestIdFilter) for existing in handler.filters):
        handler.addFilter(RequestIdFilter())
    return handler
