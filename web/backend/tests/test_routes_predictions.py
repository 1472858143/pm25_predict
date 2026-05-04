from __future__ import annotations

from fastapi.testclient import TestClient


def test_predictions_aggregate_returns_arrays(client: TestClient) -> None:
    response = client.get("/api/predictions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["window"] == "window_720h_to_72h"
    assert payload["horizons"] == [1, 2, 3]
    assert len(payload["timestamps"]) == 3
    assert len(payload["y_true"]) == 3
    assert set(payload["predictions"].keys()) == {"lstm", "attention_lstm"}
    assert len(payload["predictions"]["lstm"]) == 3


def test_predictions_aggregate_y_true_values(client: TestClient) -> None:
    response = client.get("/api/predictions")
    payload = response.json()
    assert payload["y_true"] == [101.0, 102.0, 103.0]


def test_model_predictions_returns_rows(client: TestClient) -> None:
    response = client.get("/api/predictions/lstm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "lstm"
    assert len(payload["rows"]) == 3
    first = payload["rows"][0]
    assert first["horizon"] == 1
    assert first["y_true"] == 101.0
    assert "abs_error" in first


def test_model_predictions_unknown_model_404(client: TestClient) -> None:
    response = client.get("/api/predictions/ghost_model")
    assert response.status_code == 404


def test_predictions_pass_window_and_start_params(client: TestClient) -> None:
    response = client.get(
        "/api/predictions",
        params={"window": "window_720h_to_72h", "start": "start_2026_03_01_0000"},
    )
    assert response.status_code == 200
    assert response.json()["start"] == "start_2026_03_01_0000"
