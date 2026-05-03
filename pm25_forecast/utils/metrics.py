from __future__ import annotations

import math
from typing import Any

import numpy as np


def _flat(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def finite_or_none(value: float) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def regression_metrics(y_true: Any, y_pred: Any, mape_denominator_min: float = 1.0) -> dict[str, float | None]:
    true = _flat(y_true)
    pred = _flat(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true.shape} != {pred.shape}")
    if true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    error = pred - true
    abs_error = np.abs(error)
    squared_error = error**2

    rmse = float(np.sqrt(np.mean(squared_error)))
    mae = float(np.mean(abs_error))
    denominator = np.maximum(np.abs(true), float(mape_denominator_min))
    mape = float(np.mean(abs_error / denominator) * 100.0)
    smape_denominator = np.maximum((np.abs(true) + np.abs(pred)) / 2.0, float(mape_denominator_min))
    smape = float(np.mean(abs_error / smape_denominator) * 100.0)
    bias = float(np.mean(error))

    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

    return {
        "RMSE": finite_or_none(rmse),
        "MAE": finite_or_none(mae),
        "MAPE": finite_or_none(mape),
        "SMAPE": finite_or_none(smape),
        "R2": finite_or_none(r2),
        "bias": finite_or_none(bias),
    }


def monthly_stage_metrics(timestamps: Any, y_true: Any, y_pred: Any) -> dict[str, dict[str, float | None]]:
    import pandas as pd

    ts = pd.to_datetime(timestamps)
    true = _flat(y_true)
    pred = _flat(y_pred)
    if len(ts) != len(true) or len(true) != len(pred):
        raise ValueError("timestamps, y_true, and y_pred must have the same length.")

    days = ts.dt.day if hasattr(ts, "dt") else ts.day
    stages = {
        "early_month": days <= 10,
        "middle_month": (days >= 11) & (days <= 20),
        "late_month": days >= 21,
    }
    result: dict[str, dict[str, float | None]] = {}
    for name, mask in stages.items():
        mask_array = np.asarray(mask, dtype=bool)
        result[name] = regression_metrics(true[mask_array], pred[mask_array])
    return result
