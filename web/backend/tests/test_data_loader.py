from __future__ import annotations

from pathlib import Path

from app.data_loader import (
    list_models,
    list_starts,
    list_windows,
    load_horizon_metrics_csv,
    load_metrics,
    load_predictions_csv,
    resolve_predict_start,
    start_exists,
    window_exists,
)


def test_list_windows_returns_both_windows(output_root: Path) -> None:
    windows = list_windows(output_root)
    names = [w["name"] for w in windows]
    assert "window_720h_to_72h" in names
    assert "window_168h_to_72h" in names
    full = next(w for w in windows if w["name"] == "window_720h_to_72h")
    assert full["input_window"] == 720
    assert full["output_window"] == 72
    assert full["starts"] == ["start_2026_03_01_0000"]
    empty = next(w for w in windows if w["name"] == "window_168h_to_72h")
    assert empty["starts"] == []


def test_list_windows_handles_missing_root(tmp_path: Path) -> None:
    assert list_windows(tmp_path / "does_not_exist") == []


def test_list_starts_excludes_unrelated_dirs(output_root: Path) -> None:
    window_dir = output_root / "window_720h_to_72h"
    (window_dir / "predictions" / "not_a_start").mkdir(parents=True, exist_ok=True)
    starts = list_starts(window_dir)
    assert starts == ["start_2026_03_01_0000"]


def test_list_models_returns_sorted_names(output_root: Path) -> None:
    models = list_models(output_root, "window_720h_to_72h", "start_2026_03_01_0000")
    assert models == ["attention_lstm", "lstm"]


def test_load_metrics_returns_dict(output_root: Path) -> None:
    metrics = load_metrics(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "lstm")
    assert metrics is not None
    assert metrics["RMSE"] == 31.0
    assert metrics["R2"] == 0.3


def test_load_metrics_missing_returns_none(output_root: Path) -> None:
    assert load_metrics(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "ghost") is None


def test_load_predictions_csv_parses_rows(output_root: Path) -> None:
    rows = load_predictions_csv(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "lstm")
    assert len(rows) == 3
    assert rows[0]["model_name"] == "lstm"
    assert rows[0]["horizon"] == "1"


def test_load_horizon_metrics_csv_parses_rows(output_root: Path) -> None:
    rows = load_horizon_metrics_csv(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "lstm")
    assert len(rows) == 3
    assert rows[0]["horizon"] == "1"


def test_resolve_predict_start_returns_canonical_string(output_root: Path) -> None:
    value = resolve_predict_start(output_root, "window_720h_to_72h", "start_2026_03_01_0000")
    assert value == "2026-03-01 00:00:00+08:00"


def test_window_exists_and_start_exists(output_root: Path) -> None:
    assert window_exists(output_root, "window_720h_to_72h")
    assert not window_exists(output_root, "window_999h_to_99h")
    assert start_exists(output_root, "window_720h_to_72h", "start_2026_03_01_0000")
    assert not start_exists(output_root, "window_720h_to_72h", "start_9999_99_99_9999")
