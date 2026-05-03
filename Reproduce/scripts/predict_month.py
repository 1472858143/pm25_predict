from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Reproduce.models.lstm_one_step import LSTMConfig, build_model, require_torch
from Reproduce.utils.calibration import apply_calibration
from Reproduce.utils.data_utils import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    experiment_name,
    load_scaler,
    prepare_data_bundle,
    parse_predict_start,
    read_json,
    safe_timestamp_label,
    write_json,
)
from Reproduce.utils.metrics import regression_metrics
from Reproduce.utils.plotting import (
    plot_error_curve,
    plot_loss_curve,
    plot_prediction_curve,
    plot_scatter,
    write_plot_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one direct multi-output window with a trained LSTM.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Reproduction outputs root.")
    parser.add_argument("--input-window", type=int, default=720)
    parser.add_argument("--output-window", type=int, default=24)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START, help="Prediction start timestamp.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-path", default=None, help="Path to model checkpoint. Defaults to best-validation-loss model if available.")
    parser.add_argument("--calibration-path", default=None, help="Optional fitted calibration JSON path.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable fitted prediction calibration.")
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate prepared data before predicting.")
    return parser.parse_args()


def select_device(torch: Any, requested: str) -> Any:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(exp_dir: Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    best_val = exp_dir / "model_best_val_loss.pt"
    if best_val.exists():
        return best_val
    best = exp_dir / "model_best_train_loss.pt"
    if best.exists():
        return best
    return exp_dir / "model.pt"


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    if args.prepare_data:
        prepare_data_bundle(
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    exp_dir = Path(args.output_root) / experiment_name(args.input_window, args.output_window)
    bundle_path = exp_dir / "data" / "windows.npz"
    if not bundle_path.exists():
        prepare_data_bundle(
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )
    bundle = np.load(bundle_path, allow_pickle=True)
    X_predict = bundle["X_predict"].astype(np.float32)
    y_predict_raw = bundle["y_predict_raw"].astype(np.float32)
    timestamps_start = bundle["timestamps_predict_start"].astype(str)
    timestamps_end = bundle["timestamps_predict_end"].astype(str)
    timestamps_target = bundle["timestamps_predict_target"].astype(str)
    if len(X_predict) != 1:
        raise ValueError(f"Expected one prediction sample, got {len(X_predict)}.")

    device = select_device(torch, args.device)
    model = build_model(
        LSTMConfig(
            input_size=int(X_predict.shape[-1]),
            output_size=int(y_predict_raw.shape[-1]) if y_predict_raw.ndim > 1 else 1,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        )
    ).to(device)
    ckpt = checkpoint_path(exp_dir, args.model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    predictions_scaled: list[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(X_predict), batch_size):
            batch = torch.from_numpy(X_predict[start : start + batch_size]).to(device)
            pred = model(batch).detach().cpu().numpy()
            predictions_scaled.append(pred)
    y_pred_scaled = np.concatenate(predictions_scaled)

    scaler = load_scaler(exp_dir)
    y_pred_model = scaler.inverse_column(y_pred_scaled, TARGET_COLUMN)
    calibration_applied = False
    calibration_path = Path(args.calibration_path) if args.calibration_path else exp_dir / "calibration.json"
    calibration: dict[str, Any] | None = None
    if not bool(args.no_calibration) and calibration_path.exists():
        calibration = read_json(calibration_path)
        y_pred = apply_calibration(y_pred_model, calibration)
        calibration_applied = calibration.get("method") not in {None, "none"}
    else:
        y_pred = y_pred_model
    y_true = y_predict_raw
    if y_pred_model.ndim == 1:
        y_pred_model = y_pred_model.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    output_window = int(y_true.shape[1])
    sample_ids = np.repeat(np.arange(len(y_true)), output_window)
    horizons = np.tile(np.arange(1, output_window + 1), len(y_true))
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "origin_timestamp": np.repeat(timestamps_start, output_window),
            "target_end_timestamp": np.repeat(timestamps_end, output_window),
            "timestamp": timestamps_target.reshape(-1),
            "horizon": horizons,
            "y_true": y_true.reshape(-1),
            "y_pred_model": y_pred_model.reshape(-1),
            "y_pred": y_pred.reshape(-1),
        }
    )
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["relative_error"] = predictions["abs_error"] / np.maximum(predictions["y_true"].abs(), 1.0)

    metrics = regression_metrics(predictions["y_true"], predictions["y_pred"])
    model_metrics = regression_metrics(predictions["y_true"], predictions["y_pred_model"])
    horizon_metrics = [
        {
            "horizon": horizon,
            **regression_metrics(
                predictions.loc[predictions["horizon"] == horizon, "y_true"],
                predictions.loc[predictions["horizon"] == horizon, "y_pred"],
            ),
        }
        for horizon in range(1, output_window + 1)
    ]
    stage_ranges = {
        "h01_08": (1, min(8, output_window)),
        "h09_16": (9, min(16, output_window)),
        "h17_24": (17, output_window),
    }
    stage_metrics = {}
    for stage_name, (start_h, end_h) in stage_ranges.items():
        if start_h <= end_h:
            mask = (predictions["horizon"] >= start_h) & (predictions["horizon"] <= end_h)
            stage_metrics[stage_name] = regression_metrics(
                predictions.loc[mask, "y_true"],
                predictions.loc[mask, "y_pred"],
            )

    prediction_start = parse_predict_start(args.predict_start)
    window_dir = exp_dir / f"start_{safe_timestamp_label(prediction_start)}"
    plots_dir = window_dir / "plots"
    window_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(window_dir / "predictions.csv", index=False, encoding="utf-8")
    predictions.to_csv(exp_dir / "predictions.csv", index=False, encoding="utf-8")
    write_json(window_dir / "metrics.json", metrics)
    write_json(exp_dir / "metrics.json", metrics)
    write_json(window_dir / "metrics_model_raw.json", model_metrics)
    write_json(window_dir / "horizon_metrics.json", horizon_metrics)
    pd.DataFrame(horizon_metrics).to_csv(window_dir / "horizon_metrics.csv", index=False, encoding="utf-8")
    write_json(window_dir / "stage_metrics.json", stage_metrics)
    pd.DataFrame([{"stage": name, **values} for name, values in stage_metrics.items()]).to_csv(
        window_dir / "stage_metrics.csv", index=False, encoding="utf-8"
    )

    plot_status: dict[str, bool] = {}
    plot_status["prediction_curve"] = plot_prediction_curve(
        predictions,
        plots_dir / "prediction_curve.png",
        f"LSTM 24h PM2.5 Prediction ({args.predict_start})",
    )
    plot_status["error_curve"] = plot_error_curve(
        predictions,
        plots_dir / "error_curve.png",
        f"LSTM 24h PM2.5 Prediction Error ({args.predict_start})",
    )
    plot_status["scatter"] = plot_scatter(
        predictions,
        plots_dir / "scatter.png",
        f"LSTM 24h PM2.5 True vs Predicted ({args.predict_start})",
    )
    history_path = exp_dir / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        plot_status["loss_curve"] = plot_loss_curve(history, exp_dir / "plots" / "loss_curve.png")
    write_plot_status(window_dir / "plot_status.md", plot_status)

    summary = {
        "experiment_dir": str(exp_dir),
        "prediction_dir": str(window_dir),
        "model_path": str(ckpt),
        "calibration_path": str(calibration_path) if calibration_path.exists() else None,
        "calibration_applied": bool(calibration_applied),
        "calibration_method": None if calibration is None else calibration.get("method"),
        "device": str(device),
        "sample_count": int(len(predictions)),
        "forecast_sample_count": int(len(y_true)),
        "output_window": int(output_window),
        "predict_start": str(prediction_start),
        "metrics": metrics,
        "model_raw_metrics": model_metrics,
    }
    write_json(window_dir / "prediction_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_prediction(args)
    print(f"Predicted samples: {summary['sample_count']}")
    print(f"Output dir: {summary['prediction_dir']}")
    print(f"Metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
