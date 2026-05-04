from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.models.attention_lstm_seq2seq import Seq2SeqConfig, build_seq2seq_model
from pm25_forecast.models.lstm_one_step import require_torch
from pm25_forecast.scripts.train_lstm import select_device
from pm25_forecast.utils.calibration import apply_calibration
from pm25_forecast.utils.data_utils import (
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    FeatureMinMaxScaler,
    parse_predict_start,
    read_json,
)
from pm25_forecast.utils.paths import model_dir, prediction_dir, window_experiment_dir
from pm25_forecast.utils.prediction_io import build_predictions_frame, write_prediction_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with AttentionLSTMSeq2Seq.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    p.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    p.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--encoder-num-layers", type=int, default=2)
    p.add_argument("--decoder-num-layers", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def checkpoint_path(out_dir: Path) -> Path:
    for name in ("model_best_val_loss.pt", "model.pt"):
        candidate = out_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {out_dir}")


def _history_columns(data_config: dict[str, Any]) -> list[str]:
    return list(data_config.get("feature_columns_history") or data_config["feature_columns_full"])


def _load_state_dict(torch, ckpt: Path, device):
    try:
        return torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt, map_location=device)


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    out_dir = model_dir(window_dir, "attention_lstm_seq2seq")
    pred_dir = prediction_dir(window_dir, args.predict_start, "attention_lstm_seq2seq")
    pred_dir.mkdir(parents=True, exist_ok=True)

    data_config = read_json(window_dir / "data" / "data_config.json")
    history_cols = _history_columns(data_config)
    pm25_idx = history_cols.index(TARGET_COLUMN)
    with np.load(window_dir / "data" / "windows.npz", allow_pickle=True) as bundle:
        X_predict_full = bundle["X_predict_full"].astype(np.float32)
        X_predict_future = bundle["X_predict_future"].astype(np.float32)
        y_predict_raw = bundle["y_predict_raw"].astype(np.float32)
        timestamps_predict_target = bundle["timestamps_predict_target"][0].copy()
        timestamps_predict_start = str(bundle["timestamps_predict_start"][0])
        timestamps_predict_end = str(bundle["timestamps_predict_end"][0])
    first_pm25 = X_predict_full[:, -1, pm25_idx : pm25_idx + 1]

    device = select_device(torch, args.device)
    cfg = Seq2SeqConfig(
        input_size_history=int(X_predict_full.shape[-1]),
        input_size_future=int(X_predict_future.shape[-1]),
        hidden_size=int(args.hidden_size),
        encoder_num_layers=int(args.encoder_num_layers),
        decoder_num_layers=int(args.decoder_num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        output_window=int(args.output_window),
    )
    model = build_seq2seq_model(cfg).to(device)
    ckpt = checkpoint_path(out_dir)
    model.load_state_dict(_load_state_dict(torch, ckpt, device))
    model.eval()

    with torch.no_grad():
        hist = torch.from_numpy(X_predict_full).to(device)
        fut = torch.from_numpy(X_predict_future).to(device)
        first = torch.from_numpy(first_pm25).to(device)
        pred_scaled = model(hist, fut, first, teacher_forcing_targets=None, teacher_forcing_prob=0.0)
        pred_scaled_np = pred_scaled.cpu().numpy()

    scaler_full = FeatureMinMaxScaler.from_dict(read_json(window_dir / "data" / "scaler_full.json"))
    target_min = scaler_full.data_min[history_cols.index(TARGET_COLUMN)]
    target_scale = scaler_full.scale[history_cols.index(TARGET_COLUMN)]
    pred_raw_model = pred_scaled_np * target_scale + target_min

    calibration_path = out_dir / "calibration.json"
    calibration = read_json(calibration_path) if calibration_path.exists() else {"method": "none"}
    pred_raw = apply_calibration(pred_raw_model, calibration)
    calibration_applied = calibration.get("method", "none") != "none"

    predictions = build_predictions_frame(
        model_name="attention_lstm_seq2seq",
        y_true=y_predict_raw,
        y_pred_model=pred_raw_model,
        y_pred=pred_raw,
        timestamps_start=[timestamps_predict_start],
        timestamps_end=[timestamps_predict_end],
        timestamps_target=timestamps_predict_target.reshape(1, -1),
    )

    summary = write_prediction_outputs(
        predictions=predictions,
        output_dir=pred_dir,
        model_name="attention_lstm_seq2seq",
        model_path=ckpt,
        calibration_path=calibration_path if calibration_path.exists() else None,
        calibration_applied=calibration_applied,
        calibration_method=calibration.get("method", "none"),
        device=str(device),
        predict_start=args.predict_start,
    )
    summary["experiment_dir"] = str(window_dir)
    summary["predict_start"] = str(parse_predict_start(args.predict_start))
    return summary


def main() -> None:
    run_prediction(parse_args())


if __name__ == "__main__":
    main()
