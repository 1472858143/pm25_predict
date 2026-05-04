"""Build a minimal outputs/ directory under a given root for tests."""
from __future__ import annotations

import json
from pathlib import Path

WINDOW = "window_720h_to_72h"
START = "start_2026_03_01_0000"
PREDICT_START = "2026-03-01 00:00:00+08:00"
MODELS = ("lstm", "attention_lstm")

PREDICTIONS_HEADER = (
    "model_name,sample_id,origin_timestamp,target_end_timestamp,timestamp,"
    "horizon,y_true,y_pred_model,y_pred,error,abs_error,relative_error"
)

HORIZON_HEADER = "horizon,RMSE,MAE,MAPE,SMAPE,R2,bias"


def _metrics_for(model_name: str) -> dict[str, float]:
    base = 30.0 if model_name == "lstm" else 25.0
    return {
        "RMSE": base + 1.0,
        "MAE": base,
        "MAPE": base + 5.0,
        "SMAPE": base + 2.0,
        "R2": 0.5 if model_name == "attention_lstm" else 0.3,
        "bias": -10.0 if model_name == "lstm" else -5.0,
    }


def _prediction_rows(model_name: str) -> list[str]:
    rows: list[str] = []
    for horizon in (1, 2, 3):
        y_true = 100.0 + horizon
        y_pred_model = y_true + 5.0 if model_name == "lstm" else y_true + 2.0
        y_pred = y_pred_model - 1.0
        error = y_pred - y_true
        rows.append(
            f"{model_name},0,{PREDICT_START},2026-03-03 23:00:00+08:00,"
            f"2026-03-01 0{horizon - 1}:00:00+08:00,{horizon},{y_true},"
            f"{y_pred_model},{y_pred},{error},{abs(error)},{abs(error) / y_true}"
        )
    return rows


def _horizon_rows(model_name: str) -> list[str]:
    rows: list[str] = []
    for horizon in (1, 2, 3):
        delta = float(horizon)
        rows.append(f"{horizon},{delta + 1},{delta},{delta + 5},{delta + 2},0.0,{delta}")
    return rows


def build_fixture_outputs(root: Path) -> Path:
    """Create the minimal fixture under root (which is treated as OUTPUT_ROOT)."""
    pred_dir = root / WINDOW / "predictions" / START
    for model in MODELS:
        model_path = pred_dir / model
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "metrics.json").write_text(
            json.dumps(_metrics_for(model)), encoding="utf-8"
        )
        (model_path / "prediction_summary.json").write_text(
            json.dumps(
                {
                    "model_name": model,
                    "predict_start": PREDICT_START,
                    "metrics": _metrics_for(model),
                }
            ),
            encoding="utf-8",
        )
        predictions_csv = "\n".join([PREDICTIONS_HEADER, *_prediction_rows(model)]) + "\n"
        (model_path / "predictions.csv").write_text(predictions_csv, encoding="utf-8")
        horizon_csv = "\n".join([HORIZON_HEADER, *_horizon_rows(model)]) + "\n"
        (model_path / "horizon_metrics.csv").write_text(horizon_csv, encoding="utf-8")
    # Add a second window with no predictions, to validate empty-starts handling.
    (root / "window_168h_to_72h" / "data").mkdir(parents=True, exist_ok=True)
    return root
