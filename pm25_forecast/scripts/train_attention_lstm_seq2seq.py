from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.models.attention_lstm_seq2seq import Seq2SeqConfig, build_seq2seq_model
from pm25_forecast.models.lstm_one_step import require_torch
from pm25_forecast.scripts.train_lstm import select_device
from pm25_forecast.utils.calibration import (
    apply_calibration,
    fit_horizon_isotonic_calibration,
    fit_horizon_linear_calibration,
)
from pm25_forecast.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    FeatureMinMaxScaler,
    prepare_data_bundle,
    read_json,
    write_json,
)
from pm25_forecast.utils.metrics import regression_metrics
from pm25_forecast.utils.paths import model_dir, window_experiment_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AttentionLSTMSeq2Seq.")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    p.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    p.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--encoder-num-layers", type=int, default=2)
    p.add_argument("--decoder-num-layers", type=int, default=1)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduled-sampling-decay-end", type=int, default=20)
    p.add_argument("--scheduled-sampling-min-prob", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--calibration", default="none", choices=["none", "horizon_linear", "horizon_isotonic"])
    p.add_argument("--calibration-fit", default="train", choices=["train", "validation"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--prepare-data", action="store_true")
    return p.parse_args()


def teacher_forcing_prob(epoch: int, decay_end: int, min_prob: float, decay_start: int = 5) -> float:
    if epoch <= decay_start:
        return 1.0
    if epoch >= decay_end:
        return float(min_prob)
    span = max(1, int(decay_end) - int(decay_start))
    progress = (epoch - decay_start) / span
    return float(1.0 - progress * (1.0 - float(min_prob)))


def resolve_paths(output_root: str | Path, input_window: int, output_window: int) -> dict[str, Path]:
    window_dir = window_experiment_dir(output_root, input_window, output_window)
    out_dir = model_dir(window_dir, "attention_lstm_seq2seq")
    return {
        "window_dir": window_dir,
        "out_dir": out_dir,
        "data_config_path": window_dir / "data" / "data_config.json",
        "bundle_path": window_dir / "data" / "windows.npz",
    }


def _history_columns(data_config: dict[str, Any]) -> list[str]:
    return list(data_config.get("feature_columns_history") or data_config["feature_columns_full"])


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    torch, _ = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    if args.prepare_data:
        prepare_data_bundle(
            data_path=args.data_path,
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    paths = resolve_paths(args.output_root, args.input_window, args.output_window)
    if not paths["data_config_path"].exists() or not paths["bundle_path"].exists():
        prepare_data_bundle(
            data_path=args.data_path,
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    data_config = read_json(paths["data_config_path"])
    with np.load(paths["bundle_path"], allow_pickle=True) as bundle:
        if int(bundle["bundle_version"]) < 2:
            raise RuntimeError("Bundle is v1; re-run prepare_data to upgrade to v2.")
        X_train_full = bundle["X_train_full"].astype(np.float32)
        X_train_future = bundle["X_train_future"].astype(np.float32)
        y_train = bundle["y_train"].astype(np.float32)
        y_train_raw = bundle["y_train_raw"].astype(np.float32)
        X_val_full = bundle["X_validation_full"].astype(np.float32)
        X_val_future = bundle["X_validation_future"].astype(np.float32)
        y_val = bundle["y_validation"].astype(np.float32)
        y_val_raw = bundle["y_validation_raw"].astype(np.float32)

    if len(X_train_full) == 0 or len(X_val_full) == 0:
        raise ValueError("Insufficient training or validation samples.")

    history_cols = _history_columns(data_config)
    pm25_index_history = history_cols.index(TARGET_COLUMN)
    first_pm25_train = X_train_full[:, -1, pm25_index_history : pm25_index_history + 1]
    first_pm25_val = X_val_full[:, -1, pm25_index_history : pm25_index_history + 1]

    device = select_device(torch, args.device)
    cfg = Seq2SeqConfig(
        input_size_history=int(X_train_full.shape[-1]),
        input_size_future=int(X_train_future.shape[-1]),
        output_size=1,
        hidden_size=int(args.hidden_size),
        encoder_num_layers=int(args.encoder_num_layers),
        decoder_num_layers=int(args.decoder_num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        output_window=int(args.output_window),
    )
    model = build_seq2seq_model(cfg).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_full),
        torch.from_numpy(X_train_future),
        torch.from_numpy(first_pm25_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    val_ds = TensorDataset(
        torch.from_numpy(X_val_full),
        torch.from_numpy(X_val_future),
        torch.from_numpy(first_pm25_val),
        torch.from_numpy(y_val),
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=int(args.lr_patience),
        factor=float(args.lr_factor),
        min_lr=1e-6,
    )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_val_state = None
    best_val_epoch = 0
    patience_counter = 0
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    epoch = 0

    for epoch in range(1, int(args.epochs) + 1):
        tf_prob = teacher_forcing_prob(
            epoch,
            int(args.scheduled_sampling_decay_end),
            float(args.scheduled_sampling_min_prob),
        )
        model.train()
        train_losses: list[float] = []
        for hist, fut, first, target in train_loader:
            hist = hist.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            first = first.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(hist, fut, first, teacher_forcing_targets=target, teacher_forcing_prob=tf_prob)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for hist, fut, first, target in val_loader:
                hist = hist.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                first = first.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(hist, fut, first, teacher_forcing_targets=None, teacher_forcing_prob=0.0)
                val_losses.append(float(torch.mean((pred - target) ** 2).cpu().item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "teacher_forcing_prob": tf_prob,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_val_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_val_loss={best_val_loss:.6f} best_epoch={best_val_epoch} tf_prob={tf_prob:.3f} lr={lr:.2e}"
        )
        if patience_counter >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch}")
            break

    if best_val_state is not None:
        model.load_state_dict(best_val_state)
    model_path = paths["out_dir"] / "model.pt"
    best_path = paths["out_dir"] / "model_best_val_loss.pt"
    torch.save(model.state_dict(), model_path)
    torch.save(best_val_state if best_val_state is not None else model.state_dict(), best_path)

    calibration_path = paths["out_dir"] / "calibration.json"
    calibration_method = str(args.calibration)
    if calibration_method != "none":
        scaler_full = FeatureMinMaxScaler.from_dict(read_json(paths["window_dir"] / "data" / "scaler_full.json"))
        target_min = scaler_full.data_min[history_cols.index(TARGET_COLUMN)]
        target_scale = scaler_full.scale[history_cols.index(TARGET_COLUMN)]

        if str(args.calibration_fit) == "train":
            cal_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False)
            cal_true_raw = y_train_raw
        else:
            cal_loader = val_loader
            cal_true_raw = y_val_raw

        preds_scaled: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for hist, fut, first, _target in cal_loader:
                hist = hist.to(device, non_blocking=True)
                fut = fut.to(device, non_blocking=True)
                first = first.to(device, non_blocking=True)
                pred = model(hist, fut, first, teacher_forcing_targets=None, teacher_forcing_prob=0.0)
                preds_scaled.append(pred.cpu().numpy())
        cal_pred_scaled = np.concatenate(preds_scaled, axis=0)
        cal_pred_raw = cal_pred_scaled * target_scale + target_min

        if calibration_method == "horizon_linear":
            calibration = fit_horizon_linear_calibration(cal_true_raw, cal_pred_raw)
        else:
            calibration = fit_horizon_isotonic_calibration(cal_true_raw, cal_pred_raw)
        cal_pred_calibrated = apply_calibration(cal_pred_raw, calibration)
        calibration["fit_data"] = str(args.calibration_fit)
        calibration["metrics_before"] = regression_metrics(cal_true_raw, cal_pred_raw)
        calibration["metrics_after"] = regression_metrics(cal_true_raw, cal_pred_calibrated)
        write_json(calibration_path, calibration)
        print(f"Calibration after metrics: {calibration['metrics_after']}")
    else:
        write_json(calibration_path, {"method": "none", "fit_data": "none"})

    pd.DataFrame(history).to_csv(paths["out_dir"] / "training_history.csv", index=False, encoding="utf-8")
    training_config = {
        "data_config": data_config,
        "model": {
            "input_size_history": cfg.input_size_history,
            "input_size_future": cfg.input_size_future,
            "hidden_size": cfg.hidden_size,
            "encoder_num_layers": cfg.encoder_num_layers,
            "decoder_num_layers": cfg.decoder_num_layers,
            "num_heads": cfg.num_heads,
            "dropout": cfg.dropout,
            "output_window": cfg.output_window,
        },
        "training": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "epochs_run": epoch,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "scheduled_sampling_decay_end": int(args.scheduled_sampling_decay_end),
            "scheduled_sampling_min_prob": float(args.scheduled_sampling_min_prob),
            "best_validation_loss": float(best_val_loss),
            "best_validation_epoch": int(best_val_epoch),
            "elapsed_seconds": float(time.time() - start_time),
            "device": str(device),
            "model_path": str(model_path),
            "best_model_path": str(best_path),
            "calibration_method": calibration_method,
            "seed": int(args.seed),
        },
    }
    write_json(paths["out_dir"] / "training_config.json", training_config)
    return {
        "model_name": "attention_lstm_seq2seq",
        "model_dir": str(paths["out_dir"]),
        "model_path": str(model_path),
        "best_model_path": str(best_path),
        "training_config_path": str(paths["out_dir"] / "training_config.json"),
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
