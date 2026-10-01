"""Per-request correlation id: middleware, log stamping and the top-level exception handlers."""

import logging
import os
import re
import subprocess
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from starlette.responses import StreamingResponse

from app.logging_config import configure_logging, installed_handler
from app.observability import (
    RequestIdMiddleware,
    attach_request_id_filter,
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

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log():
    """Capture records; `configure_logging()` is process-wide, so its state is restored."""
    root = logging.getLogger()
    root_state = (root.level, list(root.handlers))
    loggers = {
        name: (log_.level, log_.propagate)
        for name, log_ in list(logging.Logger.manager.loggerDict.items())
        if isinstance(log_, logging.Logger)
    }
    configure_logging()
    handler = _ListHandler()
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        for stale in [h for h in root.handlers if h not in root_state[1]]:
            root.removeHandler(stale)
        root.setLevel(root_state[0])
        for name, (level, propagate) in loggers.items():
            log_ = logging.getLogger(name)
            log_.setLevel(level)
            log_.propagate = propagate


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
    """Answers only the query-vector key, so /search skips the encoders and calls Qdrant."""

    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        if key.startswith("vec:"):
            return {"dense": [0.1] * 8, "si": [1], "sv": [0.5]}
        return self.store.get(key)

    async def get_many(self, keys):
        """Mirrors HybridCache.get_many: positional results, one per key."""
        return [await self.get(key) for key in keys]

    async def set(self, key, value, ttl=None):
        self.store[key] = value


class _BoomQdrant:
    async def query_points(self, **kwargs):
        raise _Boom(_OUTAGE)


class _FakeRateRedis:
    """Must look reachable: /search fails CLOSED with 503 when the limiter's Redis is down, masking the Qdrant outage."""

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
            # A raw exception in `ctx` makes the stock 422 handler unserialisable.
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


@pytest.mark.parametrize(
    "hostile",
    [
        "a" * 200,
        "x\nFAKE ERROR forged",
        "has space",
        "",
        # Bare newline/CR: the cases above also carry spaces, so these pin log forgery alone.
        "x\n",
        "x\rFORGED",
    ],
)
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
    # Emitted from the middleware's `finally`, after the ContextVar reset, so the id is passed explicitly.
    assert access[0].request_id == resp.headers[REQUEST_ID_HEADER]
    assert _errors(log) == []


def test_the_no_id_placeholder_is_not_usable_as_a_real_id(log):
    """`-` is inside the allowed character class, so the placeholder needs an explicit refusal."""
    assert not is_valid_request_id(NO_REQUEST_ID)
    app = _build_app()

    @app.get("/boom")
    async def boom():
        raise _Boom(_OUTAGE)

    resp = TestClient(app, raise_server_exceptions=False).get("/boom", headers={REQUEST_ID_HEADER: NO_REQUEST_ID})

    rid = resp.headers[REQUEST_ID_HEADER]
    assert _GENERATED_ID_RE.match(rid)
    assert rid != NO_REQUEST_ID
    assert resp.json()["request_id"] == rid
    assert NO_REQUEST_ID not in _errors(log)[0].getMessage()


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
    assert {r.request_id for r in ours} == {"second-id"}
    assert "first-id" not in "".join(r.getMessage() for r in log)
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


def test_the_request_id_filter_extends_the_app_handler_and_never_stacks_one(log):
    """Correlation extends `configure_logging()`'s root handler; a second one logs every record twice."""
    before = list(logging.getLogger().handlers)

    handler = attach_request_id_filter()
    again = attach_request_id_filter()

    assert handler is again is installed_handler()
    assert list(logging.getLogger().handlers) == before
    assert handler in before
    assert sum(isinstance(f, RequestIdFilter) for f in handler.filters) == 1


def test_the_id_survives_into_the_line_the_app_handler_actually_renders(log):
    """The app format string renders no field, so the id must reach the line via the message text."""
    app = _build_app()

    @app.get("/boom")
    async def boom():
        raise _Boom(_OUTAGE)

    resp = TestClient(app, raise_server_exceptions=False).get("/boom")

    rid = resp.headers[REQUEST_ID_HEADER]
    handler = installed_handler()
    assert handler is not None
    rendered = "".join(handler.formatter.format(r) for r in log)
    assert rid in rendered
    assert "Traceback" in rendered


_PROBE = (
    "import logging\n"
    "from app import main  # the real import path: configure_logging, then the wiring\n"
    "from app.logging_config import installed_handler\n"
    "from app.request_context import RequestIdFilter\n"
    "h = installed_handler()\n"
    "print('HANDLER', h is not None)\n"
    "print('FILTERS', sum(isinstance(f, RequestIdFilter) for f in h.filters) if h else -1)\n"
    "print('ON_ROOT', bool(h) and h in logging.getLogger().handlers)\n"
    "print('APP_HANDLERS', len([x for x in logging.getLogger().handlers\n"
    "                            if getattr(x, '_vccircle_app_handler', False)]))\n"
)


def test_the_real_import_path_actually_attaches_the_filter():
    """The attach must happen after `configure_logging()`, so the probe runs in a fresh interpreter."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
        ],
        check=False,
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "HANDLER True" in proc.stdout, proc.stdout
    assert "FILTERS 1" in proc.stdout, proc.stdout
    assert "ON_ROOT True" in proc.stdout, proc.stdout
    assert "APP_HANDLERS 1" in proc.stdout, proc.stdout


def test_real_app_registers_the_middleware_and_stamps_a_response():
    """main.py's own wiring, which `_build_app` only mirrors: HTTPException must stay Starlette's or a 404/401/429 becomes a 500."""
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

    missing = client.get("/definitely-not-a-route")
    assert missing.status_code == 404
    assert _GENERATED_ID_RE.match(missing.headers[REQUEST_ID_HEADER])
