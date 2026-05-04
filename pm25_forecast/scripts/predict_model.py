from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.models.statistical_models import forecast_statistical_model, load_statistical_model
from pm25_forecast.models.tree_models import load_tree_model, predict_tree_model
from pm25_forecast.scripts import predict_attention_lstm_seq2seq, predict_month
from pm25_forecast.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    prepare_data_bundle,
)
from pm25_forecast.utils.paths import SUPPORTED_MODEL_NAMES, model_dir, prediction_dir, validate_model_name, window_experiment_dir
from pm25_forecast.utils.prediction_io import build_predictions_frame, write_prediction_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one PM2.5 window with one trained model.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--encoder-num-layers", type=int, default=2)
    parser.add_argument("--decoder-num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--no-calibration", action="store_true")
    return parser


def _ensure_prediction_bundle(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.prepare_data:
        prepare_data_bundle(args.data_path, args.output_root, args.input_window, args.output_window, args.predict_start)
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    bundle_path = window_dir / "data" / "windows.npz"
    if not bundle_path.exists():
        prepare_data_bundle(args.data_path, args.output_root, args.input_window, args.output_window, args.predict_start)
    bundle = np.load(bundle_path, allow_pickle=True)
    return window_dir, {key: bundle[key] for key in bundle.files}


def _predict_non_lstm(args: argparse.Namespace) -> dict[str, Any]:
    model_name = validate_model_name(args.model)
    window_dir, bundle = _ensure_prediction_bundle(args)
    X_predict = bundle["X_predict"].astype(np.float32)
    y_true = bundle["y_predict_raw"].astype(np.float32)
    timestamps_start = bundle["timestamps_predict_start"].astype(str)
    timestamps_end = bundle["timestamps_predict_end"].astype(str)
    timestamps_target = bundle["timestamps_predict_target"].astype(str)

    out_model_dir = model_dir(window_dir, model_name)
    model_path = Path(args.model_path) if args.model_path else out_model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name in {"random_forest", "xgboost"}:
        model = load_tree_model(model_path)
        y_pred_model = predict_tree_model(model, X_predict)
    else:
        model = load_statistical_model(model_path)
        output_window = int(y_true.shape[1]) if y_true.ndim > 1 else int(args.output_window)
        y_pred_model = forecast_statistical_model(model, output_window)
    y_pred = y_pred_model

    predictions = build_predictions_frame(
        model_name=model_name,
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred=y_pred,
        timestamps_start=timestamps_start,
        timestamps_end=timestamps_end,
        timestamps_target=timestamps_target,
    )
    out_dir = prediction_dir(window_dir, args.predict_start, model_name)
    return write_prediction_outputs(
        predictions=predictions,
        output_dir=out_dir,
        model_name=model_name,
        model_path=model_path,
        calibration_path=None,
        calibration_applied=False,
        calibration_method=None,
        device="cpu",
        predict_start=args.predict_start,
    )


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    model_name = validate_model_name(args.model)
    if model_name == "lstm":
        return predict_month.run_prediction(args)
    if model_name == "attention_lstm":
        from pm25_forecast.scripts import predict_attention_lstm
        return predict_attention_lstm.run_prediction(args)
    if model_name == "attention_lstm_seq2seq":
        return predict_attention_lstm_seq2seq.run_prediction(args)
    return _predict_non_lstm(args)


def main() -> None:
    summary = run_prediction(build_arg_parser().parse_args())
    print(f"Predicted samples: {summary['sample_count']}")
    print(f"Output dir: {summary['prediction_dir']}")
    print(f"Metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
