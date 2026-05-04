from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_default_returns_first_window_first_start(client: TestClient) -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["window"] == "window_720h_to_72h"
    assert payload["start"] == "start_2026_03_01_0000"
    assert payload["predict_start"] == "2026-03-01 00:00:00+08:00"
    names = sorted(m["model_name"] for m in payload["models"])
    assert names == ["attention_lstm", "lstm"]
    assert payload["missing_models"] == []


def test_metrics_returns_numeric_fields(client: TestClient) -> None:
    response = client.get("/api/metrics")
    payload = response.json()
    lstm = next(m for m in payload["models"] if m["model_name"] == "lstm")
    assert lstm["RMSE"] == 31.0
    assert lstm["bias"] == -10.0


def test_metrics_unknown_window_returns_404(client: TestClient) -> None:
    response = client.get("/api/metrics", params={"window": "window_999h_to_99h"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_metrics_window_with_no_starts_returns_404(client: TestClient) -> None:
    response = client.get("/api/metrics", params={"window": "window_168h_to_72h"})
    assert response.status_code == 404
    assert "no starts" in response.json()["detail"].lower()


def test_metrics_skips_corrupt_model(client: TestClient, output_root) -> None:
    target = (
        output_root
        / "window_720h_to_72h"
        / "predictions"
        / "start_2026_03_01_0000"
        / "lstm"
        / "metrics.json"
    )
    target.write_text("not json", encoding="utf-8")
    response = client.get("/api/metrics")
    assert response.status_code == 500
