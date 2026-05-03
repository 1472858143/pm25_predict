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


def train_sarima_auto(
    series: Any,
    seasonal_period: int = 24,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    max_P: int = 2,
    max_D: int = 1,
    max_Q: int = 2,
) -> tuple[Any, dict[str, Any]]:
    try:
        from pmdarima import auto_arima
    except ImportError as exc:
        raise RuntimeError("pmdarima is required for auto SARIMA. Run: pip install pmdarima") from exc
    values = np.asarray(series, dtype=float)
    model = auto_arima(
        values,
        start_p=1, max_p=int(max_p),
        d=None, max_d=int(max_d),
        start_q=1, max_q=int(max_q),
        start_P=1, max_P=int(max_P),
        D=None, max_D=int(max_D),
        start_Q=1, max_Q=int(max_Q),
        m=int(seasonal_period),
        seasonal=True,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        enforce_stationarity=False,
        enforce_invertibility=False,
        information_criterion="aic",
    )
    order = model.order
    seasonal_order = model.seasonal_order
    info: dict[str, Any] = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "aic": float(model.aic()),
        "seasonal_period": int(seasonal_period),
    }
    return model, info


def forecast_statistical_model(model: Any, output_window: int) -> np.ndarray:
    if hasattr(model, "forecast"):
        forecast = model.forecast(steps=int(output_window))
    else:
        forecast = model.predict(n_periods=int(output_window))
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
