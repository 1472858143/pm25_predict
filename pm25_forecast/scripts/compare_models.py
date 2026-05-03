from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.utils.data_utils import (
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    write_json,
)
from pm25_forecast.utils.metrics import regression_metrics
from pm25_forecast.utils.paths import (
    SUPPORTED_MODEL_NAMES,
    comparison_dir,
    prediction_dir,
    validate_model_name,
    window_experiment_dir,
)


def compare_existing_predictions(experiment_dir: str | Path, predict_start: str, models: Iterable[str]) -> Path:
    exp_dir = Path(experiment_dir)
    frames = []
    metrics_rows = []
    for raw_model_name in models:
        model_name = validate_model_name(raw_model_name)
        pred_dir = prediction_dir(exp_dir, predict_start, model_name)
        pred_path = pred_dir / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {model_name}: {pred_path}")
        frame = pd.read_csv(pred_path)
        if "model_name" not in frame.columns:
            frame.insert(0, "model_name", model_name)
        frames.append(frame)
        metrics_rows.append(
            {
                "model_name": model_name,
                **regression_metrics(frame["y_true"], frame["y_pred"]),
                "prediction_dir": str(pred_dir),
            }
        )

    out_dir = comparison_dir(exp_dir, predict_start)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8")
    write_json(out_dir / "model_metrics.json", metrics_rows)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "all_predictions.csv", index=False, encoding="utf-8")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare existing model prediction outputs.")
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODEL_NAMES), choices=SUPPORTED_MODEL_NAMES)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    exp_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    out_dir = compare_existing_predictions(exp_dir, args.predict_start, args.models)
    print(f"Comparison dir: {out_dir}")


if __name__ == "__main__":
    main()
