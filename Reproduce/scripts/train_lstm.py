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

from Reproduce.models.lstm_one_step import LSTMConfig, build_model, require_torch
from Reproduce.utils.calibration import apply_calibration, fit_horizon_linear_calibration
from Reproduce.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    TARGET_COLUMN,
    experiment_name,
    load_scaler,
    prepare_data_bundle,
    read_json,
    write_json,
)
from Reproduce.utils.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct multi-output LSTM for PM2.5 reproduction.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input CSV path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Reproduction outputs root.")
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW, help="Historical input window in hours.")
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW, help="Direct forecast horizon in hours.")
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START, help="Prediction start timestamp.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", default="weighted_huber", choices=["mse", "weighted_mse", "weighted_huber"])
    parser.add_argument("--peak-quantile", type=float, default=0.75, help="Train-set quantile used as high-PM2.5 threshold.")
    parser.add_argument("--extreme-quantile", type=float, default=0.90, help="Train-set quantile used as extreme-PM2.5 threshold.")
    parser.add_argument("--peak-threshold", type=float, default=None, help="Optional absolute PM2.5 threshold for peak weighting.")
    parser.add_argument("--extreme-threshold", type=float, default=None, help="Optional absolute PM2.5 threshold for stronger weighting.")
    parser.add_argument("--peak-weight", type=float, default=3.0)
    parser.add_argument("--extreme-weight", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=0.05, help="Huber delta in scaled PM2.5 space.")
    parser.add_argument("--variance-penalty", type=float, default=0.05, help="Penalty for batch-level prediction variance shrinkage.")
    parser.add_argument("--calibration", default="horizon_linear", choices=["none", "horizon_linear"])
    parser.add_argument("--calibration-fit", default="train", choices=["train", "validation"])
    parser.add_argument("--calibration-slope-min", type=float, default=0.5)
    parser.add_argument("--calibration-slope-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate prepared window data before training.")
    return parser.parse_args()


def select_device(torch: Any, requested: str) -> Any:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_peak_thresholds(
    y_train_raw: np.ndarray,
    peak_threshold: float | None,
    extreme_threshold: float | None,
    peak_quantile: float,
    extreme_quantile: float,
) -> tuple[float, float]:
    flattened = np.asarray(y_train_raw, dtype=float).reshape(-1)
    if flattened.size == 0:
        raise ValueError("Cannot compute peak thresholds on empty y_train_raw.")
    resolved_peak = float(peak_threshold) if peak_threshold is not None else float(np.quantile(flattened, peak_quantile))
    resolved_extreme = (
        float(extreme_threshold) if extreme_threshold is not None else float(np.quantile(flattened, extreme_quantile))
    )
    if resolved_extreme < resolved_peak:
        resolved_extreme = resolved_peak
    return resolved_peak, resolved_extreme


def build_target_weights(
    y_raw: np.ndarray,
    peak_threshold: float,
    extreme_threshold: float,
    peak_weight: float,
    extreme_weight: float,
) -> np.ndarray:
    target = np.asarray(y_raw, dtype=float)
    weights = np.ones_like(target, dtype=np.float32)
    weights = np.where(target >= float(peak_threshold), float(peak_weight), weights)
    weights = np.where(target >= float(extreme_threshold), float(extreme_weight), weights)
    mean_weight = float(np.mean(weights))
    if mean_weight > 1e-12:
        weights = weights / mean_weight
    return weights.astype(np.float32)


def loss_value(
    torch: Any,
    pred: Any,
    target: Any,
    weights: Any,
    loss_name: str,
    huber_delta: float,
    variance_penalty: float,
) -> Any:
    error = pred - target
    if loss_name == "weighted_huber":
        delta = max(float(huber_delta), 1e-8)
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        base_loss = 0.5 * quadratic.pow(2) / delta + linear
        base_loss = base_loss * weights
    elif loss_name == "weighted_mse":
        base_loss = error.pow(2) * weights
    else:
        base_loss = error.pow(2)

    loss = base_loss.mean()
    if float(variance_penalty) > 0.0 and int(pred.shape[0]) > 1:
        pred_std = torch.std(pred, dim=0, unbiased=False)
        target_std = torch.std(target, dim=0, unbiased=False)
        loss = loss + float(variance_penalty) * torch.mean((pred_std - target_std).pow(2))
    return loss


def collect_model_predictions(torch: Any, model: Any, loader: Any, device: Any) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_X = batch[0].to(device, non_blocking=True)
            pred = model(batch_X).detach().cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


def main() -> None:
    args = parse_args()
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

    exp_dir = Path(args.output_root) / experiment_name(args.input_window, args.output_window)
    data_config_path = exp_dir / "data" / "data_config.json"
    bundle_path = exp_dir / "data" / "windows.npz"
    if not data_config_path.exists() or not bundle_path.exists():
        prepare_data_bundle(
            data_path=args.data_path,
            output_root=args.output_root,
            input_window=args.input_window,
            output_window=args.output_window,
            predict_start=args.predict_start,
        )

    data_config = read_json(data_config_path)
    bundle = np.load(bundle_path, allow_pickle=True)
    X_train = bundle["X_train"].astype(np.float32)
    y_train = bundle["y_train"].astype(np.float32)
    y_train_raw = bundle["y_train_raw"].astype(np.float32)
    X_validation = bundle["X_validation"].astype(np.float32)
    y_validation = bundle["y_validation"].astype(np.float32)
    y_validation_raw = bundle["y_validation_raw"].astype(np.float32)
    if len(X_train) == 0:
        raise ValueError("No training samples were prepared.")
    if len(X_validation) == 0:
        raise ValueError("No validation samples were prepared.")

    peak_threshold, extreme_threshold = resolve_peak_thresholds(
        y_train_raw=y_train_raw,
        peak_threshold=args.peak_threshold,
        extreme_threshold=args.extreme_threshold,
        peak_quantile=float(args.peak_quantile),
        extreme_quantile=float(args.extreme_quantile),
    )
    train_weights = build_target_weights(
        y_train_raw,
        peak_threshold=peak_threshold,
        extreme_threshold=extreme_threshold,
        peak_weight=float(args.peak_weight),
        extreme_weight=float(args.extreme_weight),
    )
    validation_weights = build_target_weights(
        y_validation_raw,
        peak_threshold=peak_threshold,
        extreme_threshold=extreme_threshold,
        peak_weight=float(args.peak_weight),
        extreme_weight=float(args.extreme_weight),
    )

    device = select_device(torch, args.device)
    model = build_model(
        LSTMConfig(
            input_size=int(X_train.shape[-1]),
            output_size=int(y_train.shape[-1]) if y_train.ndim > 1 else 1,
            hidden_size=int(args.hidden_size),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
        )
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(train_weights))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    train_eval_loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    validation_dataset = TensorDataset(
        torch.from_numpy(X_validation),
        torch.from_numpy(y_validation),
        torch.from_numpy(validation_weights),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    history: list[dict[str, float | int]] = []
    best_train_loss = float("inf")
    best_train_state = None
    best_val_loss = float("inf")
    best_val_state = None
    best_val_epoch = 0
    exp_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    print(
        f"loss={args.loss} "
        f"peak_threshold={peak_threshold:.3f} "
        f"extreme_threshold={extreme_threshold:.3f} "
        f"variance_penalty={float(args.variance_penalty):.4f}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_X, batch_y, batch_weight in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_X)
            loss = loss_value(
                torch,
                pred,
                batch_y,
                batch_weight,
                loss_name=str(args.loss),
                huber_delta=float(args.huber_delta),
                variance_penalty=float(args.variance_penalty),
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses))
        model.eval()
        val_losses: list[float] = []
        val_sse = 0.0
        val_abs_error = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_weight in validation_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_weight = batch_weight.to(device, non_blocking=True)
                pred = model(batch_X)
                val_loss = loss_value(
                    torch,
                    pred,
                    batch_y,
                    batch_weight,
                    loss_name=str(args.loss),
                    huber_delta=float(args.huber_delta),
                    variance_penalty=float(args.variance_penalty),
                )
                val_losses.append(float(val_loss.detach().cpu().item()))
                diff = pred - batch_y
                val_sse += float(torch.sum(diff.pow(2)).detach().cpu().item())
                val_abs_error += float(torch.sum(torch.abs(diff)).detach().cpu().item())
                val_count += int(batch_y.numel())
        validation_loss = float(np.mean(val_losses))
        validation_rmse_scaled = float(np.sqrt(val_sse / max(val_count, 1)))
        validation_mae_scaled = float(val_abs_error / max(val_count, 1))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_rmse_scaled": validation_rmse_scaled,
                "validation_mae_scaled": validation_mae_scaled,
            }
        )
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_val_epoch = epoch
            best_val_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.8f} "
            f"val_loss={validation_loss:.8f} "
            f"val_rmse_scaled={validation_rmse_scaled:.8f} "
            f"best_val_loss={best_val_loss:.8f} "
            f"best_val_epoch={best_val_epoch}"
        )

    model_path = exp_dir / "model.pt"
    best_model_path = exp_dir / "model_best_train_loss.pt"
    best_val_model_path = exp_dir / "model_best_val_loss.pt"
    torch.save(model.state_dict(), model_path)
    if best_train_state is not None:
        torch.save(best_train_state, best_model_path)
    if best_val_state is not None:
        torch.save(best_val_state, best_val_model_path)

    calibration_path = exp_dir / "calibration.json"
    calibration_metrics_before = None
    calibration_metrics_after = None
    if str(args.calibration) == "horizon_linear":
        scaler = load_scaler(exp_dir)
        if best_val_state is not None:
            model.load_state_dict(best_val_state)
        calibration_loader = train_eval_loader if str(args.calibration_fit) == "train" else validation_loader
        calibration_true_raw = y_train_raw if str(args.calibration_fit) == "train" else y_validation_raw
        calibration_pred_scaled = collect_model_predictions(torch, model, calibration_loader, device)
        calibration_pred_raw = scaler.inverse_column(calibration_pred_scaled, TARGET_COLUMN)
        calibration = fit_horizon_linear_calibration(
            y_true=calibration_true_raw,
            y_pred=calibration_pred_raw,
            slope_min=float(args.calibration_slope_min),
            slope_max=float(args.calibration_slope_max),
            clip_min=0.0,
        )
        calibration_pred_calibrated = apply_calibration(calibration_pred_raw, calibration)
        calibration_metrics_before = regression_metrics(calibration_true_raw, calibration_pred_raw)
        calibration_metrics_after = regression_metrics(calibration_true_raw, calibration_pred_calibrated)
        calibration["fit_data"] = str(args.calibration_fit)
        calibration["source_model_path"] = str(best_val_model_path)
        calibration["metrics_before"] = calibration_metrics_before
        calibration["metrics_after"] = calibration_metrics_after
        write_json(calibration_path, calibration)
        pd.DataFrame(calibration["horizon_stats"]).to_csv(
            exp_dir / "calibration_horizon_stats.csv",
            index=False,
            encoding="utf-8",
        )
        print(
            f"Calibration {args.calibration_fit} metrics: "
            f"before={calibration_metrics_before} "
            f"after={calibration_metrics_after}"
        )
    else:
        write_json(
            calibration_path,
            {
                "method": "none",
                "fit_data": "none",
                "source_model_path": str(best_val_model_path),
            },
        )

    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False, encoding="utf-8")
    training_config = {
        "data_config": data_config,
        "model": {
            "input_size": int(X_train.shape[-1]),
            "output_size": int(y_train.shape[-1]) if y_train.ndim > 1 else 1,
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
        },
        "loss": {
            "name": str(args.loss),
            "peak_threshold": float(peak_threshold),
            "extreme_threshold": float(extreme_threshold),
            "peak_quantile": float(args.peak_quantile),
            "extreme_quantile": float(args.extreme_quantile),
            "peak_weight": float(args.peak_weight),
            "extreme_weight": float(args.extreme_weight),
            "huber_delta": float(args.huber_delta),
            "variance_penalty": float(args.variance_penalty),
        },
        "calibration": {
            "method": str(args.calibration),
            "fit_data": str(args.calibration_fit),
            "path": str(calibration_path),
            "slope_min": float(args.calibration_slope_min),
            "slope_max": float(args.calibration_slope_max),
            "metrics_before": calibration_metrics_before,
            "metrics_after": calibration_metrics_after,
        },
        "training": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(device),
            "best_train_loss": float(best_train_loss),
            "best_validation_loss": float(best_val_loss),
            "best_validation_epoch": int(best_val_epoch),
            "elapsed_seconds": float(time.time() - start_time),
            "model_path": str(model_path),
            "best_train_loss_model_path": str(best_model_path),
            "best_validation_loss_model_path": str(best_val_model_path),
        },
    }
    write_json(exp_dir / "training_config.json", training_config)
    print(f"Saved model: {model_path}")
    print(f"Saved best-train-loss model: {best_model_path}")
    print(f"Saved best-validation-loss model: {best_val_model_path}")
    print(f"Saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
