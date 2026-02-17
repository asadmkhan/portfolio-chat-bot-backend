import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.v1.health import router as health_router
from app.main import app


def _registered_paths() -> set[str]:
    return {route.path for route in app.routes}


def test_portfolio_routes_are_registered() -> None:
    paths = _registered_paths()

    assert "/v1/health" in paths
    assert "/v1/chat/stream" in paths
    assert "/v1/chat/feedback" in paths
    assert "/v1/actions/calendar/book" in paths
    assert "/v1/actions/calendar/availability" in paths
    assert "/v1/analytics/summary" in paths
    assert "/v1/analytics/latest" in paths
    assert "/v1/analytics/top-questions" in paths
    assert "/v1/analytics/feedback" in paths


def test_tools_and_recruiter_routes_are_removed() -> None:
    paths = _registered_paths()

    assert not any(path.startswith("/v1/tools") for path in paths)
    assert not any(path.startswith("/v1/recruiter") for path in paths)


def test_health_endpoint_returns_healthy() -> None:
    test_app = FastAPI()
    test_app.include_router(health_router, prefix="/v1")
    client = TestClient(test_app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
