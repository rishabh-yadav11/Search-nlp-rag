

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


registry = REGISTRY









OTHER_ROUTE = "other"




KNOWN_ROUTE_TEMPLATES: frozenset[str] = frozenset(
    {

        "/health",
        "/live",
        "/ready",
        "/readyz",
        "/ready/deep",

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

        "/api/chat/sessions",
        "/api/chat/sessions/{session_id}",
        "/api/chat/sessions/{session_id}/messages",
        "/api/chat/sessions/{session_id}/messages/stream",
        "/api/chat/sessions/{session_id}/messages/{message_id}/rating",
        "/api/chat/usage",

        "/metrics",
    }
)

_route_param = re.compile(r"\{[^{}]+\}")


def _template_regex(template: str) -> re.Pattern[str]:


    parts = _route_param.split(template)
    pattern = "[^/]+".join(re.escape(part) for part in parts)
    return re.compile("^" + pattern + "$")


_ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (t, _template_regex(t)) for t in sorted(KNOWN_ROUTE_TEMPLATES)
)


def route_label(path: str) -> str:

    path = path.split("?", 1)[0]
    for template, pattern in _ROUTE_PATTERNS:
        if pattern.match(path):
            return template
    return OTHER_ROUTE


def register_route_template(template: str) -> None:

    global KNOWN_ROUTE_TEMPLATES, _ROUTE_PATTERNS
    if not isinstance(template, str):
        return
    if template in KNOWN_ROUTE_TEMPLATES:
        return
    KNOWN_ROUTE_TEMPLATES = KNOWN_ROUTE_TEMPLATES | {template}
    _ROUTE_PATTERNS = tuple(sorted((t, _template_regex(t)) for t in KNOWN_ROUTE_TEMPLATES))





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






def inc_http_request(method: str, path: str, status: int, dur: float) -> None:

    label = route_label(path)
    http_requests_total.labels(method=method, path=label, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, path=label).observe(dur)


def inc_search(outcome: str, dur: float) -> None:

    outcome = outcome if outcome in ("zero", "weak", "ok") else "ok"
    search_requests_total.labels(outcome=outcome).inc()
    search_duration_seconds.observe(dur)


def inc_llm(status: str, dur: float) -> None:

    status = status if status in ("ok", "error") else "error"
    llm_requests_total.labels(status=status).inc()
    llm_duration_seconds.observe(dur)









_state_provider: Callable[[], dict | None] | None = None


def set_state_provider(provider: Callable[[], dict | None]) -> None:
    global _state_provider
    _state_provider = provider


async def refresh_dependency_gauges() -> None:

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

    await refresh_dependency_gauges()
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
