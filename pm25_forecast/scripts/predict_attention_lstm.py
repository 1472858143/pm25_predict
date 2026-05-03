from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.models.attention_lstm import AttentionConfig, build_model
from pm25_forecast.models.lstm_one_step import require_torch
from pm25_forecast.utils.calibration import apply_calibration
from pm25_forecast.utils.data_utils import (
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    load_scaler,
    prepare_data_bundle,
    parse_predict_start,
    read_json,
)
from pm25_forecast.utils.paths import model_dir, prediction_dir, window_experiment_dir
from pm25_forecast.utils.prediction_io import build_predictions_frame, write_prediction_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one direct multi-output window with a trained AttentionLSTM.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Forecasting outputs root.")
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START, help="Prediction start timestamp.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention-heads", type=int, default=4, help="Number of attention heads.")
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


def checkpoint_path(attn_dir: Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    best_val = attn_dir / "model_best_val_loss.pt"
    if best_val.exists():
        return best_val
    best = attn_dir / "model_best_train_loss.pt"
    if best.exists():
        return best
    return attn_dir / "model.pt"


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    if args.prepare_data:
        prepare_data_bundle(
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    attn_dir = model_dir(window_dir, "attention_lstm")
    bundle_path = window_dir / "data" / "windows.npz"
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
        AttentionConfig(
            input_size=int(X_predict.shape[-1]),
            output_size=int(y_predict_raw.shape[-1]) if y_predict_raw.ndim > 1 else 1,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            num_heads=int(args.attention_heads),
        )
    ).to(device)
    ckpt = checkpoint_path(attn_dir, args.model_path)
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

    scaler = load_scaler(window_dir)
    y_pred_model = scaler.inverse_column(y_pred_scaled, TARGET_COLUMN)
    calibration_applied = False
    calibration_path = Path(args.calibration_path) if args.calibration_path else attn_dir / "calibration.json"
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
    predictions = build_predictions_frame(
        model_name="attention_lstm",
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred=y_pred,
        timestamps_start=timestamps_start,
        timestamps_end=timestamps_end,
        timestamps_target=timestamps_target,
    )

    prediction_start = parse_predict_start(args.predict_start)
    prediction_output_dir = prediction_dir(window_dir, args.predict_start, "attention_lstm")
    summary = write_prediction_outputs(
        predictions=predictions,
        output_dir=prediction_output_dir,
        model_name="attention_lstm",
        model_path=ckpt,
        calibration_path=calibration_path if calibration_path.exists() else None,
        calibration_applied=calibration_applied,
        calibration_method=None if calibration is None else calibration.get("method"),
        device=str(device),
        predict_start=args.predict_start,
    )
    summary["experiment_dir"] = str(window_dir)
    summary["predict_start"] = str(prediction_start)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_prediction(args)
    print(f"Predicted samples: {summary['sample_count']}")
    print(f"Output dir: {summary['prediction_dir']}")
    print(f"Metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
