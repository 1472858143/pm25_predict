from __future__ import annotations

from fastapi.testclient import TestClient


def test_windows_returns_both_entries(client: TestClient) -> None:
    response = client.get("/api/windows")
    assert response.status_code == 200
    payload = response.json()
    names = [w["name"] for w in payload["windows"]]
    assert "window_720h_to_72h" in names
    assert "window_168h_to_72h" in names


def test_windows_includes_input_output_and_starts(client: TestClient) -> None:
    response = client.get("/api/windows")
    payload = response.json()
    full = next(w for w in payload["windows"] if w["name"] == "window_720h_to_72h")
    assert full["input_window"] == 720
    assert full["output_window"] == 72
    assert full["starts"] == ["start_2026_03_01_0000"]


def test_windows_returns_empty_when_root_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "nope"))
    from importlib import reload

    from app import main as main_module

    reload(main_module)
    client = TestClient(main_module.app)
    response = client.get("/api/windows")
    assert response.status_code == 200
    assert response.json() == {"windows": []}
