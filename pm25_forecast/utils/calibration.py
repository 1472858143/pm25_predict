from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _as_2d(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}.")
    return array


def fit_horizon_linear_calibration(
    y_true: Any,
    y_pred: Any,
    slope_min: float = 0.5,
    slope_max: float = 3.0,
    clip_min: float = 0.0,
) -> dict[str, Any]:
    true = _as_2d(y_true)
    pred = _as_2d(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true.shape} != {pred.shape}")

    slopes: list[float] = []
    intercepts: list[float] = []
    horizon_stats: list[dict[str, float | int]] = []
    for horizon_index in range(true.shape[1]):
        x = pred[:, horizon_index]
        y = true[:, horizon_index]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            slope = 1.0
            intercept = 0.0
        else:
            pred_mean = float(np.mean(x))
            true_mean = float(np.mean(y))
            variance = float(np.var(x))
            if variance <= 1e-12:
                slope = 1.0
            else:
                covariance = float(np.mean((x - pred_mean) * (y - true_mean)))
                slope = covariance / variance
                slope = float(np.clip(slope, float(slope_min), float(slope_max)))
            intercept = true_mean - slope * pred_mean

        calibrated = x * slope + intercept
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        horizon_stats.append(
            {
                "horizon": horizon_index + 1,
                "sample_count": int(len(x)),
                "pred_mean": float(np.mean(x)) if len(x) else 0.0,
                "true_mean": float(np.mean(y)) if len(y) else 0.0,
                "pred_std": float(np.std(x)) if len(x) else 0.0,
                "true_std": float(np.std(y)) if len(y) else 0.0,
                "calibrated_mean": float(np.mean(calibrated)) if len(calibrated) else 0.0,
                "calibrated_std": float(np.std(calibrated)) if len(calibrated) else 0.0,
            }
        )

    return {
        "method": "horizon_linear",
        "output_window": int(true.shape[1]),
        "slope": slopes,
        "intercept": intercepts,
        "slope_min": float(slope_min),
        "slope_max": float(slope_max),
        "clip_min": float(clip_min),
        "horizon_stats": horizon_stats,
    }


def fit_horizon_isotonic_calibration(
    y_true: Any,
    y_pred: Any,
    clip_min: float = 0.0,
) -> dict[str, Any]:
    true = _as_2d(y_true)
    pred = _as_2d(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {true.shape} != {pred.shape}")

    x_thresholds: list[list[float]] = []
    y_thresholds: list[list[float]] = []
    horizon_stats: list[dict[str, float | int]] = []
    for horizon_index in range(true.shape[1]):
        x = pred[:, horizon_index]
        y = true[:, horizon_index]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            x_thresholds.append([0.0, 1.0])
            y_thresholds.append([0.0, 1.0])
        else:
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(x, y)
            x_thresholds.append([float(v) for v in iso.X_thresholds_])
            y_thresholds.append([float(v) for v in iso.y_thresholds_])
        horizon_stats.append(
            {
                "horizon": horizon_index + 1,
                "sample_count": int(len(x)),
                "pred_mean": float(np.mean(x)) if len(x) else 0.0,
                "true_mean": float(np.mean(y)) if len(y) else 0.0,
            }
        )

    return {
        "method": "horizon_isotonic",
        "output_window": int(true.shape[1]),
        "x_thresholds": x_thresholds,
        "y_thresholds": y_thresholds,
        "clip_min": float(clip_min),
        "horizon_stats": horizon_stats,
    }


def apply_calibration(y_pred: Any, calibration: dict[str, Any] | None) -> np.ndarray:
    pred = _as_2d(y_pred)
    if not calibration or calibration.get("method") in {None, "none"}:
        return pred.copy()
    method = calibration.get("method")
    if method == "horizon_linear":
        return _apply_horizon_linear(pred, calibration)
    if method == "horizon_isotonic":
        return _apply_horizon_isotonic(pred, calibration)
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_horizon_linear(pred: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    slopes = np.asarray(calibration["slope"], dtype=float).reshape(1, -1)
    intercepts = np.asarray(calibration["intercept"], dtype=float).reshape(1, -1)
    if slopes.shape[1] != pred.shape[1]:
        raise ValueError(f"Calibration horizon mismatch: {slopes.shape[1]} != {pred.shape[1]}")

    calibrated = pred * slopes + intercepts
    clip_min = calibration.get("clip_min")
    if clip_min is not None:
        calibrated = np.maximum(calibrated, float(clip_min))
    return calibrated


def _apply_horizon_isotonic(pred: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    x_thresholds = calibration["x_thresholds"]
    y_thresholds = calibration["y_thresholds"]
    if len(x_thresholds) != pred.shape[1]:
        raise ValueError(f"Calibration horizon mismatch: {len(x_thresholds)} != {pred.shape[1]}")
    out = np.empty_like(pred, dtype=float)
    for horizon_index in range(pred.shape[1]):
        xs = np.asarray(x_thresholds[horizon_index], dtype=float)
        ys = np.asarray(y_thresholds[horizon_index], dtype=float)
        out[:, horizon_index] = np.interp(pred[:, horizon_index], xs, ys)
    clip_min = calibration.get("clip_min")
    if clip_min is not None:
        out = np.maximum(out, float(clip_min))
    return out
