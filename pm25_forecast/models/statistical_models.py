from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

from pm25_forecast.utils.data_utils import TARGET_COLUMN, fill_missing_values, load_beijing_data


def load_train_pm25_series(data_config: dict[str, Any]) -> np.ndarray:
    data_path = Path(data_config["data_path"])
    train_end = pd.Timestamp(data_config["train_period"]["end"])
    frame = fill_missing_values(load_beijing_data(data_path))
    train_frame = frame.loc[frame["timestamp"] <= train_end].copy()
    if train_frame.empty:
        raise ValueError("Training PM2.5 series is empty for statistical model.")
    return train_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)


def train_arima_model(series: Any, order: tuple[int, int, int] = (2, 1, 2)) -> Any:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for arima. Install statsmodels in the active environment.") from exc
    values = np.asarray(series, dtype=float)
    model = ARIMA(values, order=tuple(int(value) for value in order))
    return model.fit()


def train_sarima_model(
    series: Any,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
) -> Any:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for sarima. Install statsmodels in the active environment.") from exc
    values = np.asarray(series, dtype=float)
    model = SARIMAX(
        values,
        order=tuple(int(value) for value in order),
        seasonal_order=tuple(int(value) for value in seasonal_order),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_statistical_model(model: Any, output_window: int) -> np.ndarray:
    forecast = model.forecast(steps=int(output_window))
    values = np.asarray(forecast, dtype=np.float32).reshape(1, int(output_window))
    return np.maximum(values, 0.0)


def save_statistical_model(model: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(model, fh)


def load_statistical_model(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)
