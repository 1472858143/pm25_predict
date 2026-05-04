from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class WindowInfo(BaseModel):
    name: str
    input_window: int
    output_window: int
    starts: list[str]


class WindowsResponse(BaseModel):
    windows: list[WindowInfo]


class ModelMetrics(BaseModel):
    model_name: str
    RMSE: float
    MAE: float
    MAPE: float
    SMAPE: float
    R2: float
    bias: float


class MetricsResponse(BaseModel):
    window: str
    start: str
    predict_start: str
    models: list[ModelMetrics]
    missing_models: list[str]


class PredictionsAggregateResponse(BaseModel):
    window: str
    start: str
    horizons: list[int]
    timestamps: list[str]
    y_true: list[float]
    predictions: dict[str, list[float]]
    missing_models: list[str]


class PredictionRow(BaseModel):
    sample_id: int
    origin_timestamp: str
    target_end_timestamp: str
    timestamp: str
    horizon: int
    y_true: float
    y_pred_model: float
    y_pred: float
    error: float
    abs_error: float
    relative_error: float


class ModelPredictionsResponse(BaseModel):
    window: str
    start: str
    model_name: str
    rows: list[PredictionRow]


class HorizonMetricRow(BaseModel):
    horizon: int
    RMSE: float
    MAE: float
    MAPE: float
    SMAPE: float
    R2: float
    bias: float


class HorizonMetricsResponse(BaseModel):
    window: str
    start: str
    model_name: str
    rows: list[HorizonMetricRow]
