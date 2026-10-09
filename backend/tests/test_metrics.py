"""Tests for the Prometheus metrics module (``app.metrics``).

The router is mounted on a bare FastAPI app so this suite proves the exposition
works independently of ``app.main`` (which the integrator wires up separately).
Assertions are made against the exact exposition text ``generate_latest`` emits
-- the same bytes Prometheus scrapes -- so they exercise the real output rather
than an in-process representation. Each numeric assertion compares against a
baseline captured inside the same test, so ordering between tests never matters.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import health as health_module
from app.metrics import (
    KNOWN_ROUTE_TEMPLATES,
    OTHER_ROUTE,
    generate_latest,
    inc_http_request,
    inc_llm,
    inc_search,
    registry,
    route_label,
    router,
    set_state_provider,
)


@pytest.fixture()
def metrics_client() -> TestClient:
    """A bare FastAPI app with ONLY the metrics router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _body() -> str:
    return generate_latest(registry).decode()


def _count(metric_line_regex: str) -> float:
    """Exact sample-line counts matching a full ``name{...} value`` sub-line.

    ``metric_line_regex`` must already be an anchored regex with a capture group
    for the value, e.g. ``http_requests_total\\{[^}]*\\}\\s+([0-9.]+)``.
    """
    total = 0.0
    for match in re.finditer(metric_line_regex, _body()):
        total += float(match.group(1))
    return total


# --- exposition endpoint ----------------------------------------------------


def test_metrics_endpoint_returns_prometheus_text(metrics_client) -> None:
    resp = metrics_client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "# HELP http_requests_total" in body
    assert "# TYPE http_requests_total counter" in body
    # The default self-collectors (process/GC) are present per the contract.
    assert "process_start_time_seconds" in body
    assert "python_gc_objects_collected_total" in body


def test_metrics_endpoint_lists_core_families(metrics_client) -> None:
    body = metrics_client.get("/metrics").text
    for family in (
        "http_requests_total",
        "http_request_duration_seconds",
        "search_requests_total",
        "search_duration_seconds",
        "llm_request_total",
        "llm_duration_seconds",
        "redis_available",
        "qdrant_available",
        "health_checks",
    ):
        assert f"# HELP {family}" in body
        assert f"# TYPE {family}" in body


def test_metrics_endpoint_requires_no_auth(metrics_client) -> None:
    """/metrics is mounted with no dependencies, so an anonymous scrape works."""
    assert metrics_client.get("/metrics").status_code == 200


# --- per-family hooks --------------------------------------------------------


def test_inc_http_request_records_bounded_labels() -> None:
    def total(mid: str, pth: str, st: str) -> float:
        label_block = f"method=\"{re.escape(mid)}\",path=\"{re.escape(pth)}\",status=\"{re.escape(st)}\""
        pattern = f"http_requests_total\\{{{label_block}\\}}\\s+([0-9.]+)"
        return _count(pattern)

    start_search = total("GET", "/search", "200")
    start_post = total("POST", "/api/chat/sessions/{session_id}/messages", "200")
    start_other = total("GET", OTHER_ROUTE, "404")

    inc_http_request("GET", "/search", 200, 0.123)
    inc_http_request("POST", "/api/chat/sessions/sess_42/messages", 200, 0.5)
    inc_http_request("GET", "/totally/unknown/oracle", 404, 0.01)

    assert total("GET", "/search", "200") == start_search + 1
    assert total("POST", "/api/chat/sessions/{session_id}/messages", "200") == start_post + 1
    assert total("GET", OTHER_ROUTE, "404") == start_other + 1


def test_route_label_cardinality_bound() -> None:
    """Unknown paths always collapse to OTHER, so the series set never grows."""
    labels = {
        route_label("/search"),
        route_label("/api/auth/users/abc123/tokens/revoke"),
        route_label("/recommend/similar/999"),
        route_label("/api/chat/sessions/x"),
        route_label("/api/chat/sessions/x/messages/stream"),
        route_label("/ready/deep"),
        route_label("/metrics"),
    }
    assert labels == {
        "/search",
        "/api/auth/users/{user_id}/tokens/revoke",
        "/recommend/similar/{article_id}",
        "/api/chat/sessions/{session_id}",
        "/api/chat/sessions/{session_id}/messages/stream",
        "/ready/deep",
        "/metrics",
    }
    # Unknown route (including one that leaks a query string) -> OTHER.
    assert route_label("/definitely/not/a/route") == OTHER_ROUTE
    assert route_label("/search?q=hello&page=2") == "/search"
    assert route_label("") == OTHER_ROUTE


def test_inc_search_outcomes_bounded() -> None:
    start_ok = _count(r'search_requests_total\{outcome="ok"\}\s+([0-9.]+)')
    start_zero = _count(r'search_requests_total\{outcome="zero"\}\s+([0-9.]+)')
    # No series for other outcomes yet.
    assert _count(r'search_requests_total\{outcome="weak"\}\s+([0-9.]+)') == 0
    assert _count(r'search_requests_total\{outcome="bogus"\}\s+([0-9.]+)') == 0

    inc_search("ok", 0.7)
    inc_search("zero", 0.2)
    inc_search("bogus", 0.3)  # must collapse to "ok", not create a new series
    assert _count(r'search_requests_total\{outcome="ok"\}\s+([0-9.]+)') == start_ok + 2
    assert _count(r'search_requests_total\{outcome="zero"\}\s+([0-9.]+)') == start_zero + 1
    assert _count(r'search_requests_total\{outcome="weak"\}\s+([0-9.]+)') == 0
    assert _count(r'search_requests_total\{outcome="bogus"\}\s+([0-9.]+)') == 0


def test_inc_llm_status_bounded() -> None:
    start_ok = _count(r'llm_request_total\{status="ok"\}\s+([0-9.]+)')
    start_error = _count(r'llm_request_total\{status="error"\}\s+([0-9.]+)')
    # No series for other statuses yet.
    assert _count(r'llm_request_total\{status="weird"\}\s+([0-9.]+)') == 0

    inc_llm("ok", 1.2)
    inc_llm("error", 0.1)
    inc_llm("weird", 0.2)  # must collapse to "error"
    assert _count(r'llm_request_total\{status="ok"\}\s+([0-9.]+)') == start_ok + 1
    assert _count(r'llm_request_total\{status="error"\}\s+([0-9.]+)') == start_error + 2
    assert _count(r'llm_request_total\{status="weird"\}\s+([0-9.]+)') == 0


def test_route_label_full_known_set_stable() -> None:
    """Every known template maps onto itself (round-trip), none becomes other."""
    for template in KNOWN_ROUTE_TEMPLATES:
        assert route_label(template) == template, template


def test_refresh_dependency_gauges_without_provider_is_safe(metrics_client) -> None:
    """No state provider -> /metrics succeeds; qdrant/models gauges are 0.

    ``redis`` is probed from the app's own configured client (health._redis_status),
    which an earlier app-booting test may have faked to healthy, so this test
    asserts it is a clean 0/1 boolean rather than a fixed value. ``llm`` is read
    from config only (no state needed), and conftest configures a healthy-looking
    key, so it reports 1 exactly as the real app would with the same key.
    """
    # The test's premise is "no provider": integration wires main.set_state_provider
    # at import, and an earlier app-booting test populates main.state, so without
    # an explicit reset here qdrant/models would report 1.0 from the real app's
    # state and this test would fail with a value it never set. Restore the premise.
    set_state_provider(None)
    resp = metrics_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    checks = re.findall(r'health_checks\{check="(\w+)"\}\s+([0-9.]+)', body)
    by_check = {name: float(val) for name, val in checks}
    assert by_check["qdrant"] == 0.0
    assert by_check["models"] == 0.0
    assert by_check["redis"] in (0.0, 1.0)
    assert by_check["llm"] == (1.0 if health_module._llm_status()[0] else 0.0)
