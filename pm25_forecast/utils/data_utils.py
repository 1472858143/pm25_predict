from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "pm25_forecast"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed_beijing.csv"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
DEFAULT_PREDICT_START = "2026-03-01 00:00:00+08:00"
DEFAULT_INPUT_WINDOW = 720
DEFAULT_OUTPUT_WINDOW = 72

RAW_COLUMN_MAP = {
    "timestamp": "timestamp",
    "pm25": "pm25",
    "temp": "temperature",
    "humidity": "humidity",
    "wind_speed": "wind_speed",
    "precipitation": "precipitation",
    "pressure": "pressure",
}

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "pressure",
    "pm25",
]
TARGET_COLUMN = "pm25"
ENRICHED_FEATURE_COLUMNS_HISTORY = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "pressure",
    "pm25",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "pm25_lag_1h",
    "pm25_lag_24h",
    "pm25_lag_168h",
    "pm25_roll24_mean",
    "pm25_roll24_max",
    "pm25_roll24_std",
]
ENRICHED_FEATURE_COLUMNS_FUTURE = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "pressure",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


@dataclass
class MinMaxState:
    columns: list[str]
    data_min: list[float]
    data_max: list[float]


class FeatureMinMaxScaler:
    def __init__(self, columns: list[str], data_min: np.ndarray, data_max: np.ndarray) -> None:
        self.columns = list(columns)
        self.data_min = np.asarray(data_min, dtype=float)
        self.data_max = np.asarray(data_max, dtype=float)
        self.scale = np.where(np.abs(self.data_max - self.data_min) <= 1e-12, 1.0, self.data_max - self.data_min)

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "FeatureMinMaxScaler":
        values = frame[columns].to_numpy(dtype=float)
        return cls(columns, np.nanmin(values, axis=0), np.nanmax(values, axis=0))

    def transform(self, frame_or_values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame_or_values, pd.DataFrame):
            values = frame_or_values[self.columns].to_numpy(dtype=float)
        else:
            values = np.asarray(frame_or_values, dtype=float)
        return (values - self.data_min) / self.scale

    def inverse_column(self, values: np.ndarray, column: str) -> np.ndarray:
        if column not in self.columns:
            raise KeyError(f"Column not found in scaler: {column}")
        index = self.columns.index(column)
        arr = np.asarray(values, dtype=float)
        return arr * self.scale[index] + self.data_min[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "data_min": [float(value) for value in self.data_min],
            "data_max": [float(value) for value in self.data_max],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureMinMaxScaler":
        return cls(list(payload["columns"]), np.asarray(payload["data_min"], dtype=float), np.asarray(payload["data_max"], dtype=float))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_name(input_window: int, output_window: int = DEFAULT_OUTPUT_WINDOW) -> str:
    return f"window_{int(input_window)}h_to_{int(output_window)}h"


def parse_predict_start(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    return timestamp


def safe_timestamp_label(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y_%m_%d_%H%M")


def load_beijing_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw = pd.read_csv(data_path)
    if "temp" not in raw.columns and "temperature" in raw.columns:
        raw = raw.rename(columns={"temperature": "temp"})
    missing = sorted(set(RAW_COLUMN_MAP) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    frame = raw.rename(columns=RAW_COLUMN_MAP)[list(RAW_COLUMN_MAP.values())].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Found invalid timestamp values.")

    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values("timestamp")
    numeric_columns = [column for column in frame.columns if column != "timestamp"]
    frame = frame.groupby("timestamp", as_index=False)[numeric_columns].mean()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def data_profile(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["timestamp"])
    diffs = timestamps.diff().dropna()
    expected_gap = pd.Timedelta(hours=1)
    gap_mask = diffs != expected_gap
    gaps = []
    for index, delta in diffs[gap_mask].items():
        gaps.append(
            {
                "previous_timestamp": str(timestamps.iloc[index - 1]),
                "next_timestamp": str(timestamps.iloc[index]),
                "gap_hours": float(delta / pd.Timedelta(hours=1)),
            }
        )

    stats: dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        values = frame[column]
        stats[column] = {
            "missing": int(values.isna().sum()),
            "min": None if values.dropna().empty else float(values.min()),
            "max": None if values.dropna().empty else float(values.max()),
            "mean": None if values.dropna().empty else float(values.mean()),
            "std": None if values.dropna().empty else float(values.std()),
        }

    return {
        "row_count": int(len(frame)),
        "start_timestamp": str(timestamps.iloc[0]) if len(timestamps) else None,
        "end_timestamp": str(timestamps.iloc[-1]) if len(timestamps) else None,
        "duplicate_timestamp_count_after_grouping": 0,
        "non_hourly_gap_count": int(gap_mask.sum()),
        "non_hourly_gaps_first_20": gaps[:20],
        "columns": list(frame.columns),
        "feature_stats": stats,
    }


def profile_to_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# 数据检查报告",
        "",
        f"- 样本数: `{profile['row_count']}`",
        f"- 起始时间: `{profile['start_timestamp']}`",
        f"- 结束时间: `{profile['end_timestamp']}`",
        f"- 非 1 小时间隔数量: `{profile['non_hourly_gap_count']}`",
        "",
        "## 字段统计",
        "",
        "| 字段 | 缺失数 | 最小值 | 最大值 | 均值 | 标准差 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for column, stats in profile["feature_stats"].items():
        lines.append(
            f"| `{column}` | `{stats['missing']}` | `{stats['min']}` | `{stats['max']}` | `{stats['mean']}` | `{stats['std']}` |"
        )
    if profile["non_hourly_gaps_first_20"]:
        lines.extend(["", "## 时间缺口前 20 条", ""])
        for gap in profile["non_hourly_gaps_first_20"]:
            lines.append(f"- `{gap['previous_timestamp']}` -> `{gap['next_timestamp']}`: `{gap['gap_hours']}` hours")
    return "\n".join(lines) + "\n"


def fill_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()
    numeric_columns = [column for column in filled.columns if column != "timestamp"]
    filled[numeric_columns] = filled[numeric_columns].interpolate(method="linear", limit_direction="both")
    if filled[numeric_columns].isna().any().any():
        raise ValueError("Missing numeric values remain after interpolation.")
    return filled


def build_enriched_features(frame: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise ValueError("frame must contain a 'timestamp' column")
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)

    pm25 = out["pm25"].astype(float)
    out["pm25_lag_1h"] = pm25.shift(1)
    out["pm25_lag_24h"] = pm25.shift(24)
    out["pm25_lag_168h"] = pm25.shift(168)
    rolled = pm25.shift(1).rolling(window=24, min_periods=24)
    out["pm25_roll24_mean"] = rolled.mean()
    out["pm25_roll24_max"] = rolled.max()
    out["pm25_roll24_std"] = rolled.std()

    if drop_warmup:
        out = out.iloc[168:].reset_index(drop=True)
    return out


def build_windows(
    normalized_features: np.ndarray,
    raw_target: np.ndarray,
    timestamps: np.ndarray,
    input_window: int,
    output_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    target_start_timestamps: list[str] = []
    target_end_timestamps: list[str] = []
    target_timestamps: list[np.ndarray] = []
    input_window = int(input_window)
    output_window = int(output_window)
    for target_index in range(input_window, len(normalized_features) - output_window + 1):
        target_slice = slice(target_index, target_index + output_window)
        X.append(normalized_features[target_index - input_window : target_index])
        y.append(raw_target[target_slice].astype(float))
        target_start_timestamps.append(str(timestamps[target_index]))
        target_end_timestamps.append(str(timestamps[target_index + output_window - 1]))
        target_timestamps.append(timestamps[target_slice].astype(object))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_start_timestamps, dtype=object),
        np.asarray(target_end_timestamps, dtype=object),
        np.asarray(target_timestamps, dtype=object),
    )


def build_windows_v2(
    normalized_full_features: np.ndarray,
    normalized_future_features: np.ndarray,
    raw_target: np.ndarray,
    timestamps: np.ndarray,
    input_window: int,
    output_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_full: list[np.ndarray] = []
    X_future: list[np.ndarray] = []
    y: list[np.ndarray] = []
    target_start_timestamps: list[str] = []
    target_end_timestamps: list[str] = []
    target_timestamps: list[np.ndarray] = []
    input_window = int(input_window)
    output_window = int(output_window)
    for target_index in range(input_window, len(normalized_full_features) - output_window + 1):
        target_slice = slice(target_index, target_index + output_window)
        X_full.append(normalized_full_features[target_index - input_window : target_index])
        X_future.append(normalized_future_features[target_slice])
        y.append(raw_target[target_slice].astype(float))
        target_start_timestamps.append(str(timestamps[target_index]))
        target_end_timestamps.append(str(timestamps[target_index + output_window - 1]))
        target_timestamps.append(timestamps[target_slice].astype(object))
    return (
        np.asarray(X_full, dtype=np.float32),
        np.asarray(X_future, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_start_timestamps, dtype=object),
        np.asarray(target_end_timestamps, dtype=object),
        np.asarray(target_timestamps, dtype=object),
    )


def prepare_data_bundle(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    input_window: int = DEFAULT_INPUT_WINDOW,
    output_window: int = DEFAULT_OUTPUT_WINDOW,
    predict_start: str = DEFAULT_PREDICT_START,
) -> dict[str, Any]:
    output_root_path = Path(output_root)
    exp_name = experiment_name(input_window, output_window)
    exp_dir = output_root_path / exp_name
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    frame = fill_missing_values(load_beijing_data(data_path))
    profile = data_profile(frame)
    write_json(output_root_path / "data_profile.json", profile)
    (output_root_path / "data_profile.md").write_text(profile_to_markdown(profile), encoding="utf-8")

    prediction_start = parse_predict_start(predict_start)
    prediction_end = prediction_start + pd.Timedelta(hours=int(output_window) - 1)
    validation_start = prediction_start - pd.DateOffset(months=3)
    train_end = validation_start - pd.Timedelta(hours=1)
    validation_end = prediction_start - pd.Timedelta(hours=1)
    train_frame = frame[frame["timestamp"] <= train_end].copy()
    validation_frame = frame[(frame["timestamp"] >= validation_start) & (frame["timestamp"] <= validation_end)].copy()
    prediction_frame = frame[(frame["timestamp"] >= prediction_start) & (frame["timestamp"] <= prediction_end)].copy()
    reserve_frame = frame[frame["timestamp"] > prediction_end].copy()
    if train_frame.empty:
        raise ValueError("Training period is empty.")
    if validation_frame.empty:
        raise ValueError("Validation period is empty.")
    if len(prediction_frame) != int(output_window):
        raise ValueError(
            f"Prediction window is incomplete: expected {output_window} rows from "
            f"{prediction_start} to {prediction_end}, got {len(prediction_frame)}."
        )

    scaler = FeatureMinMaxScaler.fit(train_frame, FEATURE_COLUMNS)
    normalized_features = scaler.transform(frame)
    raw_target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    timestamps = frame["timestamp"].astype(str).to_numpy(dtype=object)
    X_all, y_all_raw, ts_start_all, ts_end_all, ts_target_all = build_windows(
        normalized_features,
        raw_target,
        timestamps,
        input_window,
        output_window,
    )

    ts_start_index = pd.to_datetime(ts_start_all)
    ts_end_index = pd.to_datetime(ts_end_all)
    train_mask = ts_end_index <= train_end
    validation_mask = (ts_start_index >= validation_start) & (ts_end_index <= validation_end)
    predict_mask = (ts_start_index == prediction_start) & (ts_end_index == prediction_end)
    if int(np.sum(predict_mask)) != 1:
        raise ValueError(
            f"Expected exactly one prediction sample starting at {prediction_start}, "
            f"got {int(np.sum(predict_mask))}."
        )

    target_index = scaler.columns.index(TARGET_COLUMN)
    y_all_scaled = (y_all_raw - scaler.data_min[target_index]) / scaler.scale[target_index]

    bundle_path = data_dir / "windows.npz"

    enriched_frame = build_enriched_features(frame, drop_warmup=False)
    full_columns = ENRICHED_FEATURE_COLUMNS_HISTORY
    future_columns = ENRICHED_FEATURE_COLUMNS_FUTURE
    train_enriched = enriched_frame.iloc[: len(train_frame)].copy()
    full_scaler = FeatureMinMaxScaler.fit(train_enriched, full_columns)
    future_scaler = FeatureMinMaxScaler.fit(train_enriched, future_columns)
    normalized_full_features = full_scaler.transform(enriched_frame[full_columns])
    normalized_future_features = future_scaler.transform(enriched_frame[future_columns])
    normalized_full_features = np.nan_to_num(normalized_full_features, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_future_features = np.nan_to_num(normalized_future_features, nan=0.0, posinf=0.0, neginf=0.0)
    (
        X_full_all,
        X_future_all,
        y_v2_all_raw,
        ts_v2_start_all,
        ts_v2_end_all,
        ts_v2_target_all,
    ) = build_windows_v2(
        normalized_full_features,
        normalized_future_features,
        raw_target,
        timestamps,
        input_window,
        output_window,
    )
    if not (
        np.array_equal(ts_start_all, ts_v2_start_all)
        and np.array_equal(ts_end_all, ts_v2_end_all)
        and np.array_equal(ts_target_all, ts_v2_target_all)
        and np.allclose(y_all_raw, y_v2_all_raw, equal_nan=True)
    ):
        raise RuntimeError("v2 window timestamps are not aligned with legacy windows.")

    np.savez_compressed(
        bundle_path,
        bundle_version=np.asarray(2, dtype=np.int64),
        X_train=X_all[train_mask],
        X_train_full=X_full_all[train_mask],
        X_train_future=X_future_all[train_mask],
        y_train=y_all_scaled[train_mask].astype(np.float32),
        y_train_raw=y_all_raw[train_mask].astype(np.float32),
        timestamps_train_start=ts_start_all[train_mask],
        timestamps_train_end=ts_end_all[train_mask],
        timestamps_train_target=ts_target_all[train_mask],
        X_validation=X_all[validation_mask],
        X_validation_full=X_full_all[validation_mask],
        X_validation_future=X_future_all[validation_mask],
        y_validation=y_all_scaled[validation_mask].astype(np.float32),
        y_validation_raw=y_all_raw[validation_mask].astype(np.float32),
        timestamps_validation_start=ts_start_all[validation_mask],
        timestamps_validation_end=ts_end_all[validation_mask],
        timestamps_validation_target=ts_target_all[validation_mask],
        X_predict=X_all[predict_mask],
        X_predict_full=X_full_all[predict_mask],
        X_predict_future=X_future_all[predict_mask],
        y_predict=y_all_scaled[predict_mask].astype(np.float32),
        y_predict_raw=y_all_raw[predict_mask].astype(np.float32),
        timestamps_predict_start=ts_start_all[predict_mask],
        timestamps_predict_end=ts_end_all[predict_mask],
        timestamps_predict_target=ts_target_all[predict_mask],
    )

    scaler_path = data_dir / "scaler.json"
    scaler_full_path = data_dir / "scaler_full.json"
    scaler_future_path = data_dir / "scaler_future.json"
    write_json(scaler_path, scaler.to_dict())
    write_json(scaler_full_path, full_scaler.to_dict())
    write_json(scaler_future_path, future_scaler.to_dict())

    config = {
        "experiment_name": exp_name,
        "input_window": int(input_window),
        "output_window": int(output_window),
        "forecast_horizon": int(output_window),
        "prediction_strategy": "direct_multi_output",
        "predict_start": str(prediction_start),
        "predict_end": str(prediction_end),
        "data_path": str(Path(data_path)),
        "feature_columns": FEATURE_COLUMNS,
        "feature_columns_full": full_columns,
        "feature_columns_future": future_columns,
        "target_column": TARGET_COLUMN,
        "train_period": {
            "row_count": int(len(train_frame)),
            "start": str(train_frame["timestamp"].iloc[0]),
            "end": str(train_frame["timestamp"].iloc[-1]),
        },
        "validation_period": {
            "row_count": int(len(validation_frame)),
            "start": str(validation_frame["timestamp"].iloc[0]),
            "end": str(validation_frame["timestamp"].iloc[-1]),
            "purpose": "best_model_selection",
        },
        "prediction_window": {
            "row_count": int(len(prediction_frame)),
            "start": str(prediction_frame["timestamp"].iloc[0]),
            "end": str(prediction_frame["timestamp"].iloc[-1]),
        },
        "reserve_period": {
            "row_count": int(len(reserve_frame)),
            "start": None if reserve_frame.empty else str(reserve_frame["timestamp"].iloc[0]),
            "end": None if reserve_frame.empty else str(reserve_frame["timestamp"].iloc[-1]),
        },
        "bundle_path": str(bundle_path),
        "scaler_path": str(scaler_path),
        "scaler_full_path": str(scaler_full_path),
        "scaler_future_path": str(scaler_future_path),
        "bundle_version": 2,
        "sample_shapes": {
            "X_train": list(X_all[train_mask].shape),
            "X_train_full": list(X_full_all[train_mask].shape),
            "X_train_future": list(X_future_all[train_mask].shape),
            "y_train": list(y_all_scaled[train_mask].shape),
            "X_validation": list(X_all[validation_mask].shape),
            "X_validation_full": list(X_full_all[validation_mask].shape),
            "X_validation_future": list(X_future_all[validation_mask].shape),
            "y_validation": list(y_all_scaled[validation_mask].shape),
            "X_predict": list(X_all[predict_mask].shape),
            "X_predict_full": list(X_full_all[predict_mask].shape),
            "X_predict_future": list(X_future_all[predict_mask].shape),
            "y_predict": list(y_all_scaled[predict_mask].shape),
        },
    }
    write_json(data_dir / "data_config.json", config)
    return config


def load_bundle(experiment_dir: str | Path) -> dict[str, Any]:
    exp_dir = Path(experiment_dir)
    bundle = np.load(exp_dir / "data" / "windows.npz", allow_pickle=True)
    return {key: bundle[key] for key in bundle.files}


def load_scaler(experiment_dir: str | Path) -> FeatureMinMaxScaler:
    return FeatureMinMaxScaler.from_dict(read_json(Path(experiment_dir) / "data" / "scaler.json"))
