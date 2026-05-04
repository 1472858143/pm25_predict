from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_output_root
from app.data_loader import (
    list_models,
    list_windows,
    load_metrics,
    resolve_predict_start,
    start_exists,
    window_exists,
)
from app.schemas import (
    MetricsResponse,
    ModelMetrics,
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
