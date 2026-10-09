"""Prometheus metrics registry, ``GET /metrics``, and the instrumentation hooks.

This module is the single place the process registers Prometheus metric families,
so no two callers can create a second copy of a family under the same name. It is
self-contained: it defines the ``router`` (``GET /metrics``) the integrator mounts,
plus the instrumentation hooks the app (``app.main`` / ``app.observability`` /
``app.chat``) call at runtime. Nothing here imports ``app.main`` at module scope,
so mounting this router on a bare FastAPI app (as the tests do) never pulls in
torch/Qdrant/Redis/clients.

The default ``prometheus_client.REGISTRY`` is used as-is so the exposition carries
the standard process / GC / platform self-metrics (which the default registry
already registers at import) alongside our own families.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

# The default registry carries the process/GC/platform self-metrics already.
registry = REGISTRY

# ---------------------------------------------------------------------------
# Bounded route labels
# ---------------------------------------------------------------------------
# The cardinality guard, not a routing table. The ``path`` label may only take
# values from this bounded set (plus OTHER_ROUTE), so an attacker cannot grow
# the metric series by requesting arbitrary URLs. Query strings are never part
# of a label: a request "path" is route-template-shaped before it ever reaches
# the label setter.
OTHER_ROUTE = "other"

# Every route template the application serves today. Templates (the literal
# ``{placeholder}`` form, as FastAPI routes them) are the bounded vocabulary;
# concrete request paths that match a template collapse onto that template.
KNOWN_ROUTE_TEMPLATES: frozenset[str] = frozenset(
    {
        # health.py (no prefix)
        "/health",
        "/live",
        "/ready",
        "/readyz",
        "/ready/deep",
        # main.py
        "/search",
        "/facets",
        "/analytics/click",
        "/analytics/summary",
        "/analytics/chat",
        "/analytics/users",
        "/recommend/interaction",
        "/recommend/similar/{article_id}",
        "/recommend/similar/batch",
        "/recommend/for-you",
        "/recommend/trending",
        # auth.py (prefix /api/auth)
        "/api/auth/signup",
        "/api/auth/login",
        "/api/auth/me",
        "/api/auth/logout",
        "/api/auth/change-password",
        "/api/auth/users",
        "/api/auth/users/{user_id}",
        "/api/auth/users/{user_id}/tokens/revoke",
        "/api/auth/service-tokens",
        "/api/auth/service-tokens/revoke",
        # chat.py (prefix /api/chat)
        "/api/chat/sessions",
        "/api/chat/sessions/{session_id}",
        "/api/chat/sessions/{session_id}/messages",
        "/api/chat/sessions/{session_id}/messages/stream",
        "/api/chat/sessions/{session_id}/messages/{message_id}/rating",
        "/api/chat/usage",
        # this router
        "/metrics",
    }
)

_route_param = re.compile(r"\{[^{}]+\}")


def _template_regex(template: str) -> re.Pattern[str]:
    # re.escape THEN substitute would double-escape the braces; substitute first
    # so only real static text is escaped.
    parts = _route_param.split(template)
    pattern = "[^/]+".join(re.escape(part) for part in parts)
    return re.compile("^" + pattern + "$")


_ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (t, _template_regex(t)) for t in sorted(KNOWN_ROUTE_TEMPLATES)
)


def route_label(path: str) -> str:
    """Collapse a request path onto a bounded label.

    ``path`` is expected to be the route path from the ASGI scope (no query
    string), but a query string is stripped defensively anyway. An exact match
    against a known template wins; otherwise the concrete path is matched
    against the known templates (so ``/api/chat/sessions/123`` collapses to
    ``/api/chat/sessions/{session_id}``); anything unmatched -- unknown URLs,
    404s, path-traversal probes -- becomes ``other``, keeping series cardinality
    fixed regardless of what arbitrary paths a client requests.
    """
    path = path.split("?", 1)[0]
    for template, pattern in _ROUTE_PATTERNS:
        if pattern.match(path):
            return template
    return OTHER_ROUTE


def register_route_template(template: str) -> None:
    """Extend the bounded template vocabulary.

    The integrator calls this for any route added after import (e.g. from
    ``app.routes``) so its label stays a template rather than collapsing to
    ``other``. Calling it is optional -- unknown routes are already handled by
    the ``other`` label, the hook just makes them informative.
    """
    global KNOWN_ROUTE_TEMPLATES, _ROUTE_PATTERNS
    if not isinstance(template, str):
        return
    if template in KNOWN_ROUTE_TEMPLATES:
        return
    KNOWN_ROUTE_TEMPLATES = KNOWN_ROUTE_TEMPLATES | {template}
    _ROUTE_PATTERNS = tuple(sorted((t, _template_regex(t)) for t in KNOWN_ROUTE_TEMPLATES))


# ---------------------------------------------------------------------------
# Metric families
# ---------------------------------------------------------------------------
http_requests_total: Counter = Counter(
    "http_requests_total",
    "Total HTTP requests served, by method, bounded route label and status code.",
    ("method", "path", "status"),
)
http_request_duration_seconds: Histogram = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds, by method and bounded route label.",
    ("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
search_requests_total: Counter = Counter(
    "search_requests_total",
    "Search requests by outcome class (zero | weak | ok).",
    ("outcome",),
)
search_duration_seconds: Histogram = Histogram(
    "search_duration_seconds",
    "Search request latency in seconds (full retrieval+rerank turn).",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
llm_requests_total: Counter = Counter(
    "llm_request_total",
    "LLM call attempts by status (ok | error).",
    ("status",),
)
llm_duration_seconds: Histogram = Histogram(
    "llm_duration_seconds",
    "LLM call duration in seconds.",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
redis_available: Gauge = Gauge(
    "redis_available",
    "1 when the readiness Redis client is reachable, else 0.",
)
qdrant_available: Gauge = Gauge(
    "qdrant_available",
    "1 when the Qdrant collection is reachable, else 0.",
)
health_checks: Gauge = Gauge(
    "health_checks",
    "Readiness verdict per check (1 = ok, 0 = failing), from app/health.py.",
    ("check",),
)

# ---------------------------------------------------------------------------
# Instrumentation hooks (documented, stable surface for the app to call)
# ---------------------------------------------------------------------------


def inc_http_request(method: str, path: str, status: int, dur: float) -> None:
    """Record one HTTP request (Duration in SECONDS).

    ``path`` is bucketized via :func:`route_label` before it touches any label,
    so the ``path`` label set stays bounded -- this is the cardinality guard the
    RequestIdMiddleware-level probe relies on. ``dur`` must be seconds; the
    histogram unit is seconds.
    """
    label = route_label(path)
    http_requests_total.labels(method=method, path=label, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, path=label).observe(dur)


def inc_search(outcome: str, dur: float) -> None:
    """Record one search/retrieval turn (Duration in SECONDS).

    ``outcome`` is one of ``zero`` (no results), ``weak`` (weak results,
    answered with a fallback note) or ``ok``. Anything else is mapped to ``ok``
    so an unknown outcome can never create series either.
    """
    outcome = outcome if outcome in ("zero", "weak", "ok") else "ok"
    search_requests_total.labels(outcome=outcome).inc()
    search_duration_seconds.observe(dur)


def inc_llm(status: str, dur: float) -> None:
    """Record one LLM call (Duration in SECONDS). ``status`` is ``ok``/``error``."""
    status = status if status in ("ok", "error") else "error"
    llm_requests_total.labels(status=status).inc()
    llm_duration_seconds.observe(dur)


# ---------------------------------------------------------------------------
# Readiness gauges, reusing app/health.py's own probe functions
# ---------------------------------------------------------------------------
# state_provider returns the app's ``main.state`` dict, or None. main.py calls
# set_state_provider when it mounts the router; the /metrics endpoint then reads
# health/redis/qdrant from the app's OWN configured clients on every scrape,
# which is the loopback scrape contract (no separate background task needed).
_state_provider: Callable[[], dict | None] | None = None


def set_state_provider(provider: Callable[[], dict | None]) -> None:
    global _state_provider
    _state_provider = provider


async def refresh_dependency_gauges() -> None:
    """Update the health/redis/qdrant gauges from app/health.py's probes.

    Best-effort: any probe failure leaves the previous gauge value in place and
    is logged, so a wrecked dependency never turns the /metrics scrape into a
    500 (which itself is a series with the same bounded label set).
    """
    provider = _state_provider
    state = provider() if provider is not None else None
    health = importlib.import_module("app.health")

    redis_ok = False
    try:
        redis_ok, _cache_mode = await health._redis_status()
    except Exception:
        logger.warning("redis readiness probe for metrics failed", exc_info=True)
    redis_available.set(1 if redis_ok else 0)

    qdrant_ok = False
    try:
        qdrant_ok = await health._qdrant_ok(state) if state is not None else False
    except Exception:
        logger.warning("qdrant readiness probe for metrics failed", exc_info=True)
    qdrant_available.set(1 if qdrant_ok else 0)

    try:
        models_ok = health._models_ok(state) if state is not None else False
    except Exception:  # noqa: BLE001 - a probe failure keeps the last value, never breaks /metrics
        models_ok = False
    try:
        llm_ok, _reason = health._llm_status()
    except Exception:  # noqa: BLE001 - a probe failure keeps the last value, never breaks /metrics
        llm_ok = False

    for check, ok in (
        ("qdrant", qdrant_ok),
        ("models", models_ok),
        ("redis", redis_ok),
        ("llm", llm_ok),
    ):
        health_checks.labels(check=check).set(1 if ok else 0)


@router.get("/metrics")
async def metrics() -> Response:
    """Expose the registry in Prometheus text format.

    Deliberately NO authentication and NO rate limit: this is scraped by
    Prometheus on loopback (127.0.0.1:8001/metrics), and putting an auth wall on
    the scrape target would either defeat the scraper or leak every route
    through a login-of-last-resort. The port stays loopback (setup.sh binds
    gunicorn to 127.0.0.1) so it is not reachable from the network.
    """
    await refresh_dependency_gauges()
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
