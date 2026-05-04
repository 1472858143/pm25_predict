from __future__ import annotations

from fastapi.testclient import TestClient


def test_horizon_metrics_returns_rows(client: TestClient) -> None:
    response = client.get("/api/horizon-metrics/lstm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "lstm"
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["horizon"] == 1


def test_horizon_metrics_unknown_model_404(client: TestClient) -> None:
    response = client.get("/api/horizon-metrics/ghost")
    assert response.status_code == 404
