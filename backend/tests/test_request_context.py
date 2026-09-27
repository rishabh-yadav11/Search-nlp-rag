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
    """Collects records instead of rendering them."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log():
    """Capture every record the request produces, stamped exactly as in prod.

    #293's `configure_logging()` is what makes an INFO app record exist at all,
    so it is called here rather than relied on from whatever a previous test
    happened to import -- `app` lands at the app level, `app.access` and
    `app.observability` with it, and the root logger stays at WARNING. It is a
    process-wide change, so the previous state of every logger and of root is
    snapshotted and put back afterwards.

    The RequestIdFilter is attached to the capture handler exactly as
    `attach_request_id_filter` attaches it to the app's real one, so a record
    only carries `request_id` because the filter that ships in
    app.observability put it there.
    """
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


@pytest.mark.parametrize(
    "hostile",
    [
        "a" * 200,
        "x\nFAKE ERROR forged",
        "has space",
        "",
        # Newline/CR on their own, with NO other disqualifying character: the
        # cases above are rejected partly for their spaces, so they would still
        # pass if someone widened the character class to admit CR/LF. These two
        # are the log-forgery guard, tested by the guard alone.
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
    # The field the shipped formatter interpolates, not just the text: the
    # access line is emitted from the middleware's `finally`, where the
    # ContextVar is already reset, so the id has to be passed explicitly or
    # every access line would render the "no id" placeholder.
    assert access[0].request_id == resp.headers[REQUEST_ID_HEADER]
    assert _errors(log) == []


def test_the_no_id_placeholder_is_not_usable_as_a_real_id(log):
    """NO_REQUEST_ID means "no id bound" everywhere it is rendered.

    A hyphen is inside the allowed character class, so the class alone does not
    exclude it: without an explicit refusal a caller could label a live request
    `-` and make it indistinguishable from a record emitted outside any request.
    """
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
    # The ContextVar is already reset when the access line is emitted from the
    # middleware's `finally`, so that record carries its id explicitly. What
    # matters is that no record is stamped with, or names, the previous
    # request's id.
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
    """#293's `configure_logging()` is the single owner of the root handler.

    Correlation must extend that handler, not install a second one: two root
    handlers write every record in the process twice, which is exactly the
    single-write property #293 pins. So the filter goes on the handler
    `installed_handler()` returns, and calling this twice still attaches one.
    """
    before = list(logging.getLogger().handlers)

    handler = attach_request_id_filter()
    again = attach_request_id_filter()

    assert handler is again is installed_handler()
    # The root handler list is byte-for-byte what it was: this added a filter,
    # not a handler, and it is a filter on the one #293 already installed.
    assert list(logging.getLogger().handlers) == before
    assert handler in before
    assert sum(isinstance(f, RequestIdFilter) for f in handler.filters) == 1


def test_the_id_survives_into_the_line_the_app_handler_actually_renders(log):
    """The filter sets a field; #293's format string renders no field.

    What makes the id greppable in a shipped deployment is therefore the message
    text, so this renders a captured record through the real handler's own
    formatter and looks for the id in the output -- not on the record.
    """
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
    assert "Traceback" in rendered  # the traceback reaches the same line

# The script the fresh-interpreter ordering check runs: importing app.main is what
# calls configure_logging() and then the correlation wiring, in that order.
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
    """The ordering constraint, pinned: the attach must happen AFTER #293 runs.

    `configure_logging()` is called from `app.main` near the top of that module
    and `installed_handler()` returns None before it, so a call site that
    drifted above it would leave the filter permanently unattached and no app
    line would carry an id -- silently, because every other case here would
    still pass. That is the failure this turns into a red test.

    Run in a fresh interpreter, deliberately: whether the filter is attached
    depends on the ORDER of two module-level calls at import, and this test
    session's own fixtures (mine and #293's) add and remove root handlers. An
    in-process assertion would read whatever the previously-run test left
    behind instead of the real import path. This is the same
    fresh-interpreter technique `tests/test_api_surface_hardening.py` and
    `tests/test_logging_config.py` already use for import-time behaviour.
    """
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
    # And still exactly one app handler: a second one would double every line.
    assert "APP_HANDLERS 1" in proc.stdout, proc.stdout


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
