from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pm25_forecast.utils.metrics import regression_metrics
from pm25_forecast.utils.plotting import plot_error_curve, plot_prediction_curve, plot_scatter, write_plot_status


PREDICTION_COLUMNS = [
    "model_name",
    "sample_id",
    "origin_timestamp",
    "target_end_timestamp",
    "timestamp",
    "horizon",
    "y_true",
    "y_pred_model",
    "y_pred",
    "error",
    "abs_error",
    "relative_error",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _flatten_strings(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=str).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def _target_array(values: Any, sample_count: int, output_window: int) -> np.ndarray:
    array = np.asarray(values, dtype=str)
    expected = sample_count * output_window
    if array.size != expected:
        raise ValueError(
            f"timestamps_target size mismatch: expected {expected} values for "
            f"{sample_count} samples x {output_window} horizons, got {array.size}."
        )
    return array.reshape(sample_count, output_window)


def _as_2d(name: str, values: Any, sample_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.size % sample_count != 0:
            raise ValueError(f"{name} length {array.size} cannot be reshaped for {sample_count} samples.")
        return array.reshape(sample_count, array.size // sample_count)
    if array.ndim == 2:
        if array.shape[0] != sample_count:
            raise ValueError(f"{name} sample count mismatch: {array.shape[0]} != {sample_count}.")
        return array
    raise ValueError(f"{name} must be 1D or 2D, got {array.ndim}D.")


def build_predictions_frame(
    *,
    model_name: str,
    y_true: Any,
    y_pred_model: Any,
    y_pred: Any,
    timestamps_start: Any,
    timestamps_end: Any,
    timestamps_target: Any,
) -> pd.DataFrame:
    start = _flatten_strings("timestamps_start", timestamps_start)
    end = _flatten_strings("timestamps_end", timestamps_end)
    if start.shape != end.shape:
        raise ValueError(f"timestamps_start and timestamps_end shape mismatch: {start.shape} != {end.shape}.")

    sample_count = int(start.size)
    true_2d = _as_2d("y_true", y_true, sample_count)
    pred_model_2d = _as_2d("y_pred_model", y_pred_model, sample_count)
    pred_2d = _as_2d("y_pred", y_pred, sample_count)
    if true_2d.shape != pred_model_2d.shape:
        raise ValueError(f"y_true and y_pred_model shape mismatch: {true_2d.shape} != {pred_model_2d.shape}.")
    if true_2d.shape != pred_2d.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true_2d.shape} != {pred_2d.shape}.")

    output_window = int(true_2d.shape[1])
    target = _target_array(timestamps_target, sample_count, output_window)
    y_true_flat = true_2d.reshape(-1)
    y_pred_model_flat = pred_model_2d.reshape(-1)
    y_pred_flat = pred_2d.reshape(-1)
    error = y_pred_flat - y_true_flat
    abs_error = np.abs(error)
    relative_error = abs_error / np.maximum(np.abs(y_true_flat), 1.0)

    frame = pd.DataFrame(
        {
            "model_name": np.repeat(str(model_name), sample_count * output_window),
            "sample_id": np.repeat(np.arange(sample_count), output_window),
            "origin_timestamp": np.repeat(start, output_window),
            "target_end_timestamp": np.repeat(end, output_window),
            "timestamp": target.reshape(-1),
            "horizon": np.tile(np.arange(1, output_window + 1), sample_count),
            "y_true": y_true_flat,
            "y_pred_model": y_pred_model_flat,
            "y_pred": y_pred_flat,
            "error": error,
            "abs_error": abs_error,
            "relative_error": relative_error,
        }
    )
    return frame[PREDICTION_COLUMNS]


def _stage_metric_ranges(output_window: int) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for start_horizon in range(1, int(output_window) + 1, 24):
        end_horizon = min(start_horizon + 23, int(output_window))
        ranges[f"h{start_horizon:03d}_{end_horizon:03d}"] = (start_horizon, end_horizon)
    return ranges


def _prediction_plot_title(model_name: str, plot_kind: str, output_window: int, predict_start: Any) -> str:
    suffixes = {
        "prediction": "Prediction",
        "error": "Prediction Error",
        "scatter": "True vs Predicted",
    }
    if plot_kind not in suffixes:
        supported = ", ".join(suffixes)
        raise ValueError(f"Unsupported prediction plot kind: {plot_kind}. Supported kinds: {supported}")
    return f"{model_name} {int(output_window)}h PM2.5 {suffixes[plot_kind]} ({predict_start})"


def _horizon_metrics(predictions: pd.DataFrame, output_window: int) -> list[dict[str, float | int | None]]:
    metrics: list[dict[str, float | int | None]] = []
    for horizon in range(1, int(output_window) + 1):
        mask = predictions["horizon"] == horizon
        metrics.append(
            {
                "horizon": horizon,
                **regression_metrics(predictions.loc[mask, "y_true"], predictions.loc[mask, "y_pred"]),
            }
        )
    return metrics


def _stage_metrics(predictions: pd.DataFrame, output_window: int) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    for name, (start_horizon, end_horizon) in _stage_metric_ranges(output_window).items():
        mask = (predictions["horizon"] >= start_horizon) & (predictions["horizon"] <= end_horizon)
        metrics[name] = regression_metrics(predictions.loc[mask, "y_true"], predictions.loc[mask, "y_pred"])
    return metrics


def write_prediction_outputs(
    *,
    predictions: pd.DataFrame,
    output_dir: str | Path,
    model_name: str,
    model_path: str | Path | None,
    calibration_path: str | Path | None,
    calibration_applied: bool,
    calibration_method: str | None,
    device: str,
    predict_start: Any,
) -> dict[str, Any]:
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    output = Path(output_dir)
    plots_dir = output / "plots"
    output.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frame = predictions.loc[:, PREDICTION_COLUMNS].copy()
    if frame.empty:
        raise ValueError("predictions cannot be empty.")
    output_window = int(frame["horizon"].max())
    forecast_sample_count = int(frame["sample_id"].nunique())

    metrics = regression_metrics(frame["y_true"], frame["y_pred"])
    model_raw_metrics = regression_metrics(frame["y_true"], frame["y_pred_model"])
    horizon_metrics = _horizon_metrics(frame, output_window)
    stage_metrics = _stage_metrics(frame, output_window)
    stage_metric_rows = [{"stage": name, **values} for name, values in stage_metrics.items()]

    frame.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "metrics_model_raw.json", model_raw_metrics)
    _write_json(output / "horizon_metrics.json", horizon_metrics)
    pd.DataFrame(horizon_metrics).to_csv(output / "horizon_metrics.csv", index=False, encoding="utf-8")
    _write_json(output / "stage_metrics.json", stage_metrics)
    pd.DataFrame(stage_metric_rows).to_csv(output / "stage_metrics.csv", index=False, encoding="utf-8")

    plot_status = {
        "prediction_curve": plot_prediction_curve(
            frame,
            plots_dir / "prediction_curve.png",
            _prediction_plot_title(model_name, "prediction", output_window, predict_start),
        ),
        "error_curve": plot_error_curve(
            frame,
            plots_dir / "error_curve.png",
            _prediction_plot_title(model_name, "error", output_window, predict_start),
        ),
        "scatter": plot_scatter(
            frame,
            plots_dir / "scatter.png",
            _prediction_plot_title(model_name, "scatter", output_window, predict_start),
        ),
    }
    write_plot_status(output / "plot_status.md", plot_status)

    summary = {
        "model_name": str(model_name),
        "prediction_dir": str(output),
        "model_path": None if model_path is None else str(model_path),
        "calibration_path": None if calibration_path is None else str(calibration_path),
        "calibration_applied": bool(calibration_applied),
        "calibration_method": calibration_method,
        "device": str(device),
        "sample_count": int(len(frame)),
        "forecast_sample_count": forecast_sample_count,
        "output_window": output_window,
        "predict_start": str(predict_start),
        "metrics": metrics,
        "model_raw_metrics": model_raw_metrics,
    }
    _write_json(output / "prediction_summary.json", summary)
    return summary
