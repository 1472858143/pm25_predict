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

from pm25_forecast.models.attention_lstm_seq2seq import Seq2SeqConfig, build_seq2seq_model
from pm25_forecast.models.lstm_one_step import require_torch
from pm25_forecast.scripts.train_lstm import select_device
from pm25_forecast.utils.calibration import apply_calibration
from pm25_forecast.utils.data_utils import (
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    TARGET_COLUMN,
    FeatureMinMaxScaler,
    build_enriched_features,
    fill_missing_values,
    load_beijing_data,
    read_json,
    write_json,
)
from pm25_forecast.utils.metrics import regression_metrics
from pm25_forecast.utils.paths import model_dir, window_experiment_dir


def generate_origin_timestamps(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    output_window: int,
    stride: int,
) -> list[pd.Timestamp]:
    if eval_end <= eval_start:
        return []
    horizon_delta = pd.Timedelta(hours=int(output_window))
    last_allowed = eval_end - horizon_delta
    if last_allowed < eval_start:
        return []
    origins: list[pd.Timestamp] = []
    current = eval_start
    while current < last_allowed:
        origins.append(current)
        current = current + pd.Timedelta(hours=int(stride))
    return origins


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling-origin evaluation.")
    p.add_argument("--model", default="attention_lstm_seq2seq")
    p.add_argument("--data-path", default=None, help="Optional override for the input CSV.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    p.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    p.add_argument("--eval-start", required=True)
    p.add_argument("--eval-end", required=True)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--encoder-num-layers", type=int, default=2)
    p.add_argument("--decoder-num-layers", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-origins", type=int, default=None)
    return p.parse_args()


def _history_columns(data_config: dict[str, Any]) -> list[str]:
    return list(data_config.get("feature_columns_history") or data_config["feature_columns_full"])


def _load_state_dict(torch, ckpt: Path, device):
    try:
        return torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt, map_location=device)


def _ensure_tz(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.tz is None:
        return timestamp.tz_localize("Asia/Shanghai")
    return timestamp


def build_origin_inputs(
    enriched: pd.DataFrame,
    history_cols: list[str],
    future_cols: list[str],
    scaler_full: FeatureMinMaxScaler,
    scaler_future: FeatureMinMaxScaler,
    origin_ts: pd.Timestamp,
    input_window: int,
    output_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ts = pd.to_datetime(enriched["timestamp"])
    matches = ts == origin_ts
    if not matches.any():
        return None
    origin_pos = int(np.where(matches)[0][0])
    if origin_pos < int(input_window):
        return None
    if origin_pos + int(output_window) > len(enriched):
        return None
    history_slice = enriched.iloc[origin_pos - int(input_window) : origin_pos]
    future_slice = enriched.iloc[origin_pos : origin_pos + int(output_window)]
    history_full = scaler_full.transform(history_slice[history_cols])
    future_arr = scaler_future.transform(future_slice[future_cols])
    y_true = future_slice[TARGET_COLUMN].to_numpy(dtype=float)
    return (
        history_full.astype(np.float32),
        future_arr.astype(np.float32),
        y_true.astype(np.float32),
        np.asarray(future_slice["timestamp"].astype(str), dtype=object),
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.model != "attention_lstm_seq2seq":
        raise NotImplementedError(f"Rolling eval for {args.model} not implemented")
    torch, _ = require_torch()
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    out_dir = model_dir(window_dir, args.model)
    data_config = read_json(window_dir / "data" / "data_config.json")
    history_cols = _history_columns(data_config)
    future_cols = list(data_config["feature_columns_future"])
    scaler_full = FeatureMinMaxScaler.from_dict(read_json(window_dir / "data" / "scaler_full.json"))
    scaler_future = FeatureMinMaxScaler.from_dict(read_json(window_dir / "data" / "scaler_future.json"))

    csv_path = Path(args.data_path) if args.data_path else Path(data_config["data_path"])
    frame = fill_missing_values(load_beijing_data(csv_path))
    enriched = build_enriched_features(frame, drop_warmup=False)

    eval_start = _ensure_tz(pd.Timestamp(args.eval_start))
    eval_end = _ensure_tz(pd.Timestamp(args.eval_end))
    origins = generate_origin_timestamps(eval_start, eval_end, int(args.output_window), int(args.stride))
    if args.max_origins is not None:
        origins = origins[: int(args.max_origins)]
    if not origins:
        raise ValueError("No origins generated; check eval-start/eval-end/stride.")

    device = select_device(torch, args.device)
    cfg = Seq2SeqConfig(
        input_size_history=len(history_cols),
        input_size_future=len(future_cols),
        hidden_size=int(args.hidden_size),
        encoder_num_layers=int(args.encoder_num_layers),
        decoder_num_layers=int(args.decoder_num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        output_window=int(args.output_window),
    )
    model = build_seq2seq_model(cfg).to(device)
    ckpt = out_dir / "model_best_val_loss.pt"
    if not ckpt.exists():
        ckpt = out_dir / "model.pt"
    model.load_state_dict(_load_state_dict(torch, ckpt, device))
    model.eval()

    calibration_path = out_dir / "calibration.json"
    calibration = read_json(calibration_path) if calibration_path.exists() else {"method": "none"}
    pm25_idx_history = history_cols.index(TARGET_COLUMN)
    target_min = scaler_full.data_min[pm25_idx_history]
    target_scale = scaler_full.scale[pm25_idx_history]

    per_origin: list[dict[str, Any]] = []
    all_pred_per_step: list[np.ndarray] = []
    all_true_per_step: list[np.ndarray] = []
    skipped: list[str] = []
    with torch.no_grad():
        for origin_ts in origins:
            inputs = build_origin_inputs(
                enriched,
                history_cols,
                future_cols,
                scaler_full,
                scaler_future,
                origin_ts,
                int(args.input_window),
                int(args.output_window),
            )
            if inputs is None:
                skipped.append(str(origin_ts))
                continue
            history_full, future_arr, y_true, _target_timestamps = inputs
            first_pm25 = history_full[-1:, pm25_idx_history : pm25_idx_history + 1]
            hist = torch.from_numpy(history_full[None, :, :]).to(device)
            fut = torch.from_numpy(future_arr[None, :, :]).to(device)
            first = torch.from_numpy(first_pm25).to(device)
            pred_scaled = model(hist, fut, first, teacher_forcing_targets=None, teacher_forcing_prob=0.0).cpu().numpy()[0]
            pred_raw = pred_scaled * target_scale + target_min
            pred_calibrated = apply_calibration(pred_raw.reshape(1, -1), calibration)[0]
            metrics = regression_metrics(y_true.reshape(1, -1), pred_calibrated.reshape(1, -1))
            metrics["origin"] = str(origin_ts)
            per_origin.append(metrics)
            all_pred_per_step.append(pred_calibrated)
            all_true_per_step.append(y_true)

    if not per_origin:
        raise RuntimeError("No origins produced predictions; check input window availability.")

    df = pd.DataFrame(per_origin)
    eval_dir = (
        window_dir
        / "evaluations"
        / f"rolling_{eval_start.strftime('%Y%m%d_%H%M')}_{eval_end.strftime('%Y%m%d_%H%M')}_{args.model}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(eval_dir / "per_origin_metrics.csv", index=False, encoding="utf-8")

    pred_stack = np.stack(all_pred_per_step, axis=0)
    true_stack = np.stack(all_true_per_step, axis=0)
    horizon_rows = []
    for h in range(int(args.output_window)):
        m = regression_metrics(true_stack[:, h : h + 1], pred_stack[:, h : h + 1])
        m["horizon"] = int(h + 1)
        horizon_rows.append(m)
    pd.DataFrame(horizon_rows)[["horizon", "RMSE", "MAE", "MAPE", "SMAPE", "R2", "bias"]].to_csv(
        eval_dir / "horizon_metrics.csv",
        index=False,
        encoding="utf-8",
    )

    aggregate = {
        "model": args.model,
        "eval_start": str(eval_start),
        "eval_end": str(eval_end),
        "stride": int(args.stride),
        "n_origins_attempted": len(origins),
        "n_origins_evaluated": len(per_origin),
        "n_origins_skipped": len(skipped),
        "skipped_origins": skipped,
        "R2_mean": float(df["R2"].mean()),
        "R2_median": float(df["R2"].median()),
        "R2_p25": float(df["R2"].quantile(0.25)),
        "R2_p75": float(df["R2"].quantile(0.75)),
        "R2_min": float(df["R2"].min()),
        "R2_max": float(df["R2"].max()),
        "RMSE_mean": float(df["RMSE"].mean()),
        "MAE_mean": float(df["MAE"].mean()),
        "bias_mean": float(df["bias"].mean()),
    }
    write_json(eval_dir / "aggregate.json", aggregate)
    print(
        f"Rolling eval done: n={aggregate['n_origins_evaluated']} "
        f"R2_mean={aggregate['R2_mean']:.4f} R2_median={aggregate['R2_median']:.4f}"
    )
    return aggregate


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
