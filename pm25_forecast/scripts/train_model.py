from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.models.statistical_models import (
    load_train_pm25_series,
    save_statistical_model,
    train_arima_model,
    train_sarima_model,
)
from pm25_forecast.models.tree_models import save_tree_model, train_random_forest_model, train_xgboost_model
from pm25_forecast.scripts import train_lstm
from pm25_forecast.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    prepare_data_bundle,
    read_json,
    write_json,
)
from pm25_forecast.utils.paths import SUPPORTED_MODEL_NAMES, model_dir, validate_model_name, window_experiment_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one PM2.5 forecasting model.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", default="weighted_huber", choices=["mse", "weighted_mse", "weighted_huber"])
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--extreme-threshold", type=float, default=None)
    parser.add_argument("--peak-weight", type=float, default=3.0)
    parser.add_argument("--extreme-weight", type=float, default=5.0)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--variance-penalty", type=float, default=0.05)
    parser.add_argument("--calibration", default="horizon_linear", choices=["none", "horizon_linear"])
    parser.add_argument("--calibration-fit", default="train", choices=["train", "validation"])
    parser.add_argument("--calibration-slope-min", type=float, default=0.5)
    parser.add_argument("--calibration-slope-max", type=float, default=3.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--arima-order", nargs=3, type=int, default=[2, 1, 2])
    parser.add_argument("--sarima-order", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--sarima-seasonal-order", nargs=4, type=int, default=[1, 0, 1, 24])
    return parser


def _ensure_data(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if args.prepare_data:
        prepare_data_bundle(args.data_path, args.output_root, args.input_window, args.output_window, args.predict_start)
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    data_config_path = window_dir / "data" / "data_config.json"
    bundle_path = window_dir / "data" / "windows.npz"
    if not data_config_path.exists() or not bundle_path.exists():
        prepare_data_bundle(args.data_path, args.output_root, args.input_window, args.output_window, args.predict_start)
    bundle = np.load(bundle_path, allow_pickle=True)
    return window_dir, read_json(data_config_path), {key: bundle[key] for key in bundle.files}


def train_non_lstm(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    model_name = validate_model_name(args.model)
    window_dir, data_config, bundle = _ensure_data(args)
    out_dir = model_dir(window_dir, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"

    if model_name == "random_forest":
        model = train_random_forest_model(
            bundle["X_train"],
            bundle["y_train_raw"],
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        save_tree_model(model, model_path)
    elif model_name == "xgboost":
        model = train_xgboost_model(
            bundle["X_train"],
            bundle["y_train_raw"],
            n_estimators=args.n_estimators,
            max_depth=6 if args.max_depth is None else args.max_depth,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        save_tree_model(model, model_path)
    elif model_name == "arima":
        model = train_arima_model(load_train_pm25_series(data_config), tuple(args.arima_order))
        save_statistical_model(model, model_path)
    elif model_name == "sarima":
        model = train_sarima_model(
            load_train_pm25_series(data_config),
            tuple(args.sarima_order),
            tuple(args.sarima_seasonal_order),
        )
        save_statistical_model(model, model_path)
    else:
        raise ValueError(f"Unsupported non-LSTM model: {model_name}")

    config = {
        "model_name": model_name,
        "model_path": str(model_path),
        "data_config": data_config,
        "training": {
            "seed": int(args.seed),
            "elapsed_seconds": float(time.time() - start),
        },
    }
    write_json(out_dir / "training_config.json", config)
    return config


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if validate_model_name(args.model) == "lstm":
        return train_lstm.run_training(args)
    return train_non_lstm(args)


def main() -> None:
    summary = run_training(build_arg_parser().parse_args())
    print(f"Trained model: {summary['model_name']}")
    print(f"Model path: {summary['model_path']}")


if __name__ == "__main__":
    main()
