from __future__ import annotations

from pathlib import Path

from pm25_forecast.utils.data_utils import parse_predict_start, safe_timestamp_label


SUPPORTED_MODEL_NAMES = ("lstm", "attention_lstm", "xgboost", "random_forest", "arima", "sarima")


def validate_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or model_name not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported}")
    return model_name


def window_experiment_name(input_window: int, output_window: int) -> str:
    return f"window_{int(input_window)}h_to_{int(output_window)}h"


def window_experiment_dir(output_root: str | Path, input_window: int, output_window: int) -> Path:
    return Path(output_root) / window_experiment_name(input_window, output_window)


def data_dir(experiment_dir: str | Path) -> Path:
    return Path(experiment_dir) / "data"


def model_dir(experiment_dir: str | Path, model_name: str) -> Path:
    return Path(experiment_dir) / "models" / validate_model_name(model_name)


def start_dir_name(predict_start: str) -> str:
    return f"start_{safe_timestamp_label(parse_predict_start(predict_start))}"


def prediction_root_dir(experiment_dir: str | Path, predict_start: str) -> Path:
    return Path(experiment_dir) / "predictions" / start_dir_name(predict_start)


def prediction_dir(experiment_dir: str | Path, predict_start: str, model_name: str) -> Path:
    return prediction_root_dir(experiment_dir, predict_start) / validate_model_name(model_name)


def comparison_dir(experiment_dir: str | Path, predict_start: str) -> Path:
    return Path(experiment_dir) / "comparisons" / start_dir_name(predict_start)
