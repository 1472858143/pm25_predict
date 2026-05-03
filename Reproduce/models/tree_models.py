from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np


def flatten_window_features(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected X with shape [samples, input_window, features], got {array.shape}.")
    return array.reshape(array.shape[0], array.shape[1] * array.shape[2])


def train_random_forest_model(
    X_train: Any,
    y_train: Any,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for random_forest. Install scikit-learn in the active environment.") from exc
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=None if max_depth is None else int(max_depth),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    model.fit(flatten_window_features(X_train), np.asarray(y_train, dtype=np.float32))
    return model


def train_xgboost_model(
    X_train: Any,
    y_train: Any,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    try:
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("xgboost and scikit-learn are required for xgboost model training.") from exc
    base = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        objective="reg:squarederror",
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(flatten_window_features(X_train), np.asarray(y_train, dtype=np.float32))
    return model


def save_tree_model(model: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(model, fh)


def load_tree_model(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def predict_tree_model(model: Any, X_predict: Any) -> np.ndarray:
    prediction = model.predict(flatten_window_features(X_predict))
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)
    return prediction
