from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_output_root
from app.data_loader import (
    list_models,
    list_windows,
    load_metrics,
    load_predictions_csv,
    resolve_predict_start,
    start_exists,
    window_exists,
)
from app.schemas import (
    MetricsResponse,
    ModelMetrics,
    ModelPredictionsResponse,
    PredictionRow,
    PredictionsAggregateResponse,
    WindowInfo,
    WindowsResponse,
)

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/windows", response_model=WindowsResponse)
def get_windows() -> WindowsResponse:
    root = get_output_root()
    raw = list_windows(root)
    return WindowsResponse(windows=[WindowInfo(**item) for item in raw])


def _resolve_window_start(window: str | None, start: str | None) -> tuple[str, str]:
    root = get_output_root()
    windows = list_windows(root)
    if not windows:
        raise HTTPException(status_code=404, detail="No windows available")
    chosen_window = window
    if chosen_window is None:
        chosen_window = next((w["name"] for w in windows if w["starts"]), windows[0]["name"])
    if not window_exists(root, chosen_window):
        raise HTTPException(status_code=404, detail=f"Window '{chosen_window}' not found")
    chosen_start = start
    if chosen_start is None:
        info = next(w for w in windows if w["name"] == chosen_window)
        if not info["starts"]:
            raise HTTPException(status_code=404, detail=f"Window '{chosen_window}' has no starts")
        chosen_start = info["starts"][0]
    if not start_exists(root, chosen_window, chosen_start):
        raise HTTPException(status_code=404, detail=f"Start '{chosen_start}' not found")
    return chosen_window, chosen_start


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(window: str | None = None, start: str | None = None) -> MetricsResponse:
    root = get_output_root()
    chosen_window, chosen_start = _resolve_window_start(window, start)
    models: list[ModelMetrics] = []
    missing: list[str] = []
    for model_name in list_models(root, chosen_window, chosen_start):
        try:
            raw = load_metrics(root, chosen_window, chosen_start, model_name)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load metrics for '{model_name}': {exc}",
            ) from exc
        if raw is None:
            missing.append(model_name)
            continue
        try:
            models.append(ModelMetrics(model_name=model_name, **raw))
        except Exception:
            missing.append(model_name)
    return MetricsResponse(
        window=chosen_window,
        start=chosen_start,
        predict_start=resolve_predict_start(root, chosen_window, chosen_start),
        models=models,
        missing_models=missing,
    )


def _coerce_float(value: str) -> float:
    return float(value)


def _coerce_int(value: str) -> int:
    return int(value)


@router.get("/predictions", response_model=PredictionsAggregateResponse)
def get_predictions_aggregate(
    window: str | None = None, start: str | None = None
) -> PredictionsAggregateResponse:
    root = get_output_root()
    chosen_window, chosen_start = _resolve_window_start(window, start)
    model_names = list_models(root, chosen_window, chosen_start)

    horizons: list[int] = []
    timestamps: list[str] = []
    y_true: list[float] = []
    predictions: dict[str, list[float]] = {}
    missing: list[str] = []
    reference_set = False

    for model_name in model_names:
        try:
            rows = load_predictions_csv(root, chosen_window, chosen_start, model_name)
        except FileNotFoundError:
            missing.append(model_name)
            continue
        if not rows:
            missing.append(model_name)
            continue
        if not reference_set:
            horizons = [_coerce_int(r["horizon"]) for r in rows]
            timestamps = [r["timestamp"] for r in rows]
            y_true = [_coerce_float(r["y_true"]) for r in rows]
            reference_set = True
        predictions[model_name] = [_coerce_float(r["y_pred"]) for r in rows]

    return PredictionsAggregateResponse(
        window=chosen_window,
        start=chosen_start,
        horizons=horizons,
        timestamps=timestamps,
        y_true=y_true,
        predictions=predictions,
        missing_models=missing,
    )


@router.get("/predictions/{model_name}", response_model=ModelPredictionsResponse)
def get_predictions_for_model(
    model_name: str, window: str | None = None, start: str | None = None
) -> ModelPredictionsResponse:
    root = get_output_root()
    chosen_window, chosen_start = _resolve_window_start(window, start)
    if model_name not in list_models(root, chosen_window, chosen_start):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found for {chosen_window}/{chosen_start}")
    rows = load_predictions_csv(root, chosen_window, chosen_start, model_name)
    parsed: list[PredictionRow] = []
    for r in rows:
        parsed.append(
            PredictionRow(
                sample_id=_coerce_int(r["sample_id"]),
                origin_timestamp=r["origin_timestamp"],
                target_end_timestamp=r["target_end_timestamp"],
                timestamp=r["timestamp"],
                horizon=_coerce_int(r["horizon"]),
                y_true=_coerce_float(r["y_true"]),
                y_pred_model=_coerce_float(r["y_pred_model"]),
                y_pred=_coerce_float(r["y_pred"]),
                error=_coerce_float(r["error"]),
                abs_error=_coerce_float(r["abs_error"]),
                relative_error=_coerce_float(r["relative_error"]),
            )
        )
    return ModelPredictionsResponse(
        window=chosen_window,
        start=chosen_start,
        model_name=model_name,
        rows=parsed,
    )
