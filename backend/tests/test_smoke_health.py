"""Smoke tests for the health/readiness probes over the real app.

The session TestClient reports a loopback peer (see conftest), which is exactly
what /ready/deep requires, so all four probes are reachable with the fakes
reporting healthy.
"""

from __future__ import annotations


def test_health_ok(app_client) -> None:
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_live_ok(app_client) -> None:
    resp = app_client.get("/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_ok_with_fakes(app_client) -> None:
    resp = app_client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    checks = body["checks"]
    assert checks["qdrant"]["ok"] is True
    assert checks["models"]["ok"] is True
    assert checks["redis"]["ok"] is True
    assert checks["llm"]["ok"] is True


def test_readyz_alias(app_client) -> None:
    resp = app_client.get("/readyz")
    assert resp.status_code == 200


def test_ready_deep_reachable_from_loopback(app_client) -> None:
    resp = app_client.get("/ready/deep")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_unknown_route_404(app_client) -> None:
    assert app_client.get("/definitely/not/a/route").status_code == 404
