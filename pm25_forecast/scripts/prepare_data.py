from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.utils.data_utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
    prepare_data_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare direct multi-step PM2.5 forecasting data.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input CSV path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Forecasting outputs root.")
    parser.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW, help="Historical input window in hours.")
    parser.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW, help="Direct forecast horizon in hours.")
    parser.add_argument("--predict-start", default=DEFAULT_PREDICT_START, help="Prediction start timestamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = prepare_data_bundle(
        data_path=args.data_path,
        output_root=args.output_root,
        input_window=args.input_window,
        output_window=args.output_window,
        predict_start=args.predict_start,
    )
    print(f"Prepared data for {config['experiment_name']}")
    print(f"Bundle: {config['bundle_path']}")
    print(f"Train period: {config['train_period']['start']} -> {config['train_period']['end']}")
    print(f"Validation period: {config['validation_period']['start']} -> {config['validation_period']['end']}")
    print(f"Prediction window: {config['prediction_window']['start']} -> {config['prediction_window']['end']}")
    print(f"Shapes: {config['sample_shapes']}")


if __name__ == "__main__":
    main()
