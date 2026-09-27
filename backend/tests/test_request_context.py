"""Tests for the per-request correlation id: middleware, log stamping and the
top-level exception handlers.

The apps under test are built locally (middleware + handlers, no lifespan) so the
cases stay hermetic and independent of ``app.main``'s global state -- with two
deliberate exceptions: ``test_real_app_search_qdrant_outage_is_500_with_a_request_id``
and ``test_real_app_registers_the_middleware_and_stamps_a_response`` drive the
REAL ``app.main.app``, so a regression that drops the wiring from ``main.py`` is
caught here rather than in production.
"""

import logging
import re

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from starlette.responses import StreamingResponse

from app.observability import (
    RequestIdMiddleware,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.request_context import (
    NO_REQUEST_ID,
    REQUEST_ID_HEADER,
    RequestIdFilter,
    current_request_id,
    is_valid_request_id,
)

_GENERATED_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


class _Boom(RuntimeError):
    """A stand-in for the Qdrant/Redis outage a real request can hit."""


_OUTAGE = "qdrant: connection refused at 10.0.0.7:6333"


def _build_app() -> FastAPI:
    """A local app wired exactly as app.main wires the real one."""
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    return app


class _ListHandler(logging.Handler):
    """Collects records instead of rendering them."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _stamps_request_id(handler: logging.Handler) -> bool:
    fmt = getattr(handler.formatter, "_fmt", None)
    return bool(fmt) and "request_id" in fmt


@pytest.fixture(autouse=True)
def _detach_ambient_root_handler():
    """Hide the handler app.main installs on the root logger.

    Importing app.main calls install_root_log_handler(), which attaches a stderr
    handler for the rest of the session. It is detached here so these cases see
    only the records they asked for and the run's output stays clean, then put
    back so no sibling test inherits a changed logging setup.
    """
    root = logging.getLogger()
    saved = list(root.handlers)
    for handler in saved:
        if _stamps_request_id(handler):
            root.removeHandler(handler)
    try:
        yield
    finally:
        for handler in saved:
            if handler not in root.handlers:
                root.addHandler(handler)


@pytest.fixture
def log():
    """Capture every record the request produces, stamped exactly as in prod.

    The RequestIdFilter sits on the handler (as it does in production) rather
    than on a logger, so a record only carries `request_id` because the filter
    that ships in app.observability put it there.

    The root logger's level is deliberately NOT lowered here: leaving it at the
    WARNING it carries under gunicorn/uvicorn is what makes the access-record
    assertions real rather than self-fulfilling -- the INFO access line has to
    reach the handler on the strength of `app.access`'s own level alone.
    """
    handler = _ListHandler()
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)


def _errors(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if r.levelno >= logging.ERROR]


def test_dependency_failure_is_500_with_the_id_and_leaks_nothing(log):
    app = _build_app()

    @app.get("/boom")
    async def boom():
        raise _Boom(_OUTAGE)

    resp = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert resp.status_code == 500
    rid = resp.headers[REQUEST_ID_HEADER]
    assert resp.json() == {"detail": "Internal server error", "request_id": rid}

    errors = _errors(log)
    assert len(errors) == 1
    assert rid in errors[0].getMessage()
    assert "/boom" in errors[0].getMessage()
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[0] is _Boom

    assert _OUTAGE not in resp.text
    assert "Boom" not in resp.text
    assert "Traceback" not in resp.text


class _VectorOnlyCache:
    """Cache stand-in that answers the query-vector key and misses everything
    else, so /search skips the encoders and goes straight to Qdrant."""

    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        if key.startswith("vec:"):
            return {"dense": [0.1] * 8, "si": [1], "sv": [0.5]}
        return self.store.get(key)

    async def set(self, key, value, ttl=None):
        self.store[key] = value


class _BoomQdrant:
    async def query_points(self, **kwargs):
        raise _Boom(_OUTAGE)


class _FakeRateRedis:
    """In-memory limiter store: /search fails CLOSED with 503 when the limiter's
    Redis is unreachable, which would make an outage test pass for the wrong
    reason."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    async def set(self, key, value, nx=False, ex=None):
        return True

    async def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]


def test_real_app_search_qdrant_outage_is_500_with_a_request_id(log, monkeypatch):
    from app import auth, main

    monkeypatch.setattr(auth, "_rate_client", _FakeRateRedis())
    monkeypatch.setattr(main, "cache", _VectorOnlyCache())
    # state is a dict, so the qdrant client is swapped with setitem (the
    # dict-aware form of setattr, and the one monkeypatch can undo).
    monkeypatch.setitem(main.state, "qdrant", _BoomQdrant())

    async def fake_record_search(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "record_search", fake_record_search)

    resp = TestClient(main.app, raise_server_exceptions=False).get("/search", params={"q": "fintech funding"})

    assert resp.status_code == 500
    rid = resp.headers[REQUEST_ID_HEADER]
    assert resp.json()["request_id"] == rid
    errors = _errors(log)
    assert len(errors) == 1
    assert rid in errors[0].getMessage()
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[0] is _Boom
    assert _OUTAGE not in resp.text
    assert "Boom" not in resp.text


def test_http_exception_is_not_swallowed(log):
    app = _build_app()

    @app.get("/teapot")
    async def teapot():
        raise HTTPException(status_code=418, detail="teapot")

    client = TestClient(app, raise_server_exceptions=False)

    missing = client.get("/nope")
    assert missing.status_code == 404
    assert missing.headers[REQUEST_ID_HEADER]

    raised = client.get("/teapot")
    assert raised.status_code == 418
    assert raised.json()["detail"] == "teapot"
    assert raised.headers[REQUEST_ID_HEADER]

    assert _errors(log) == []


class _Payload(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _reject(cls, value: str) -> str:
        if value == "bad":
            # A raw exception object lands in the pydantic error's `ctx`, which
            # is what makes the stock 422 handler unserialisable.
            raise ValueError("sentinel-validator-message")
        return value


def test_request_validation_error_stays_a_422(log):
    app = _build_app()

    @app.post("/things")
    async def things(payload: _Payload):
        return {"ok": True}

    resp = TestClient(app, raise_server_exceptions=False).post("/things", json={"name": "bad"})

    assert resp.status_code == 422
    rid = resp.headers[REQUEST_ID_HEADER]
    body = resp.json()
    assert body["request_id"] == rid
    assert isinstance(body["detail"], list) and body["detail"]
    first = body["detail"][0]
    assert first["loc"][-1] == "name"
    # `ctx` holds the raw exception object and `input` the rejected value;
    # neither is guaranteed JSON-serialisable, so neither is echoed.
    assert set(first) <= {"type", "loc", "msg", "url"}
    assert "ctx" not in first and "input" not in first
    assert _errors(log) == []


def test_inbound_request_id_is_honoured_on_the_500_path(log):
    app = _build_app()

    @app.get("/boom")
    async def boom():
        raise _Boom(_OUTAGE)

    resp = TestClient(app, raise_server_exceptions=False).get("/boom", headers={REQUEST_ID_HEADER: "my-id-123"})

    assert resp.status_code == 500
    assert resp.headers[REQUEST_ID_HEADER] == "my-id-123"
    assert resp.json()["request_id"] == "my-id-123"
    assert "my-id-123" in _errors(log)[0].getMessage()


def test_absent_request_id_is_generated_and_echoed(log):
    app = _build_app()

    @app.get("/boom")
    async def boom():
        raise _Boom(_OUTAGE)

    resp = TestClient(app, raise_server_exceptions=False).get("/boom")

    rid = resp.headers[REQUEST_ID_HEADER]
    assert _GENERATED_ID_RE.match(rid)
    assert resp.json()["request_id"] == rid


@pytest.mark.parametrize("hostile", ["a" * 200, "x\nFAKE ERROR forged", "has space", ""])
def test_hostile_inbound_request_id_is_replaced_and_never_echoed(log, hostile):
    assert not is_valid_request_id(hostile)
    app = _build_app()

    @app.get("/boom")
    async def boom():
        raise _Boom(_OUTAGE)

    resp = TestClient(app, raise_server_exceptions=False).get("/boom", headers={REQUEST_ID_HEADER: hostile})

    assert resp.status_code == 500
    rid = resp.headers[REQUEST_ID_HEADER]
    assert _GENERATED_ID_RE.match(rid)
    assert rid != hostile
    if hostile:
        assert hostile not in resp.text
        assert hostile not in resp.headers.get(REQUEST_ID_HEADER, "")
        assert hostile not in _errors(log)[0].getMessage()


def _log_deeply() -> None:
    logging.getLogger("app.deep").error("deep handler failed")


def _log_deeper() -> None:
    _log_deeply()


def test_record_from_deep_in_the_handler_carries_the_same_id(log):
    app = _build_app()

    @app.get("/deep")
    async def deep():
        _log_deeper()
        return {"ok": True}

    resp = TestClient(app, raise_server_exceptions=False).get("/deep")

    rid = resp.headers[REQUEST_ID_HEADER]
    deep_records = [r for r in log if r.name == "app.deep"]
    assert len(deep_records) == 1
    assert deep_records[0].request_id == rid


def test_a_normal_200_logs_exactly_one_access_record(log):
    app = _build_app()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    resp = TestClient(app, raise_server_exceptions=False).get("/ok")

    assert resp.status_code == 200
    access = [r for r in log if r.name == "app.access"]
    assert len(access) == 1
    message = access[0].getMessage()
    assert "GET" in message
    assert "/ok" in message
    assert "200" in message
    assert re.search(r"in \d+\.\d+ms", message)
    assert resp.headers[REQUEST_ID_HEADER] in message
    assert _errors(log) == []


def test_no_contextvar_leak_between_requests(log):
    app = _build_app()

    @app.get("/leaky")
    async def leaky():
        _log_deeper()
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    first = client.get("/leaky", headers={REQUEST_ID_HEADER: "first-id"})
    log.clear()
    second = client.get("/leaky", headers={REQUEST_ID_HEADER: "second-id"})

    assert first.headers[REQUEST_ID_HEADER] == "first-id"
    assert second.headers[REQUEST_ID_HEADER] == "second-id"
    ours = [r for r in log if r.name.startswith("app.")]
    assert ours
    # The ContextVar is already reset when the access line is emitted from the
    # middleware's `finally`, so that record's *field* is NO_REQUEST_ID -- which
    # is exactly why the access line repeats the id in its message. What matters
    # is that no record is stamped with, or names, the previous request's id.
    assert {r.request_id for r in ours} <= {"second-id", NO_REQUEST_ID}
    assert "first-id" not in "".join(r.getMessage() for r in log)
    for record in ours:
        assert record.request_id == "second-id" or "second-id" in record.getMessage()
    assert current_request_id() == NO_REQUEST_ID


def test_streaming_response_keeps_every_chunk_and_the_id(log):
    app = _build_app()
    chunks = ["data: one\n\n", "data: two\n\n", "data: three\n\n"]

    @app.get("/stream")
    async def stream():
        async def body():
            for chunk in chunks:
                yield chunk

        return StreamingResponse(body(), media_type="text/event-stream")

    resp = TestClient(app, raise_server_exceptions=False).get("/stream")

    assert resp.status_code == 200
    assert resp.text == "".join(chunks)
    assert _GENERATED_ID_RE.match(resp.headers[REQUEST_ID_HEADER])
    assert len(resp.headers.get_list(REQUEST_ID_HEADER)) == 1
    access = [r for r in log if r.name == "app.access"]
    assert len(access) == 1
    assert "200" in access[0].getMessage()
    assert _errors(log) == []


def test_real_app_registers_the_middleware_and_stamps_a_response():
    """main.py's own wiring, not a locally rebuilt copy of it.

    The other cases build their app from the same three calls, so a regression
    that changed the wiring itself -- dropping the middleware, or registering a
    catch-all over HTTPException -- would leave every one of them green. This
    pins the real app instead: the Exception handler must be ours, and
    HTTPException must still be Starlette's own (a catch-all there would turn a
    deliberate 404/401/429 into an opaque 500 and log a server fault for what
    is a client error).
    """
    from fastapi import applications as fastapi_applications
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from app import main

    assert any(m.cls is RequestIdMiddleware for m in main.app.user_middleware)
    assert main.app.exception_handlers.get(Exception) is unhandled_exception_handler
    assert main.app.exception_handlers.get(RequestValidationError) is validation_exception_handler
    assert main.app.exception_handlers[StarletteHTTPException] is fastapi_applications.http_exception_handler

    client = TestClient(main.app, raise_server_exceptions=False)
    resp = client.get("/live")

    assert resp.status_code == 200
    assert _GENERATED_ID_RE.match(resp.headers[REQUEST_ID_HEADER])

    # A route that does not exist is a client error and has to survive the
    # catch-all untouched -- 404, not 500, and correlated like any response.
    missing = client.get("/definitely-not-a-route")
    assert missing.status_code == 404
    assert _GENERATED_ID_RE.match(missing.headers[REQUEST_ID_HEADER])
