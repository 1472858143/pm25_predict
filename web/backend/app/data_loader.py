from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

WINDOW_PATTERN = re.compile(r"^window_(\d+)h_to_(\d+)h$")
START_PATTERN = re.compile(r"^start_\d{4}_\d{2}_\d{2}_\d{4}$")


def list_windows(output_root: Path) -> list[dict[str, Any]]:
    """Return sorted window descriptors found under output_root."""
    if not output_root.exists():
        return []
    windows: list[dict[str, Any]] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        match = WINDOW_PATTERN.match(child.name)
        if not match:
            continue
        starts = list_starts(child)
        windows.append(
            {
                "name": child.name,
                "input_window": int(match.group(1)),
                "output_window": int(match.group(2)),
                "starts": starts,
            }
        )
    return windows


def list_starts(window_dir: Path) -> list[str]:
    """Return start IDs under window_dir/predictions/, sorted descending by name."""
    predictions_dir = window_dir / "predictions"
    if not predictions_dir.exists():
        return []
    starts = [
        child.name
        for child in predictions_dir.iterdir()
        if child.is_dir() and START_PATTERN.match(child.name)
    ]
    return sorted(starts, reverse=True)


def list_models(output_root: Path, window: str, start: str) -> list[str]:
    """Return model names that have a directory under predictions/start_*/."""
    base = output_root / window / "predictions" / start
    if not base.exists():
        return []
    return sorted(child.name for child in base.iterdir() if child.is_dir())


def model_dir(output_root: Path, window: str, start: str, model_name: str) -> Path:
    return output_root / window / "predictions" / start / model_name


def load_metrics(output_root: Path, window: str, start: str, model_name: str) -> dict[str, float] | None:
    """Read metrics.json for one model. Return None if missing."""
    path = model_dir(output_root, window, start, model_name) / "metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prediction_summary(output_root: Path, window: str, start: str, model_name: str) -> dict[str, Any] | None:
    path = model_dir(output_root, window, start, model_name) / "prediction_summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions_csv(output_root: Path, window: str, start: str, model_name: str) -> list[dict[str, str]]:
    """Read predictions.csv as a list of raw string-valued rows."""
    path = model_dir(output_root, window, start, model_name) / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_horizon_metrics_csv(output_root: Path, window: str, start: str, model_name: str) -> list[dict[str, str]]:
    path = model_dir(output_root, window, start, model_name) / "horizon_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"horizon_metrics.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_predict_start(output_root: Path, window: str, start: str) -> str:
    """Return canonical predict_start string from any model's prediction_summary.json."""
    for model_name in list_models(output_root, window, start):
        summary = load_prediction_summary(output_root, window, start, model_name)
        if summary and "predict_start" in summary:
            return str(summary["predict_start"])
    return ""


def window_exists(output_root: Path, window: str) -> bool:
    return (output_root / window).is_dir() and bool(WINDOW_PATTERN.match(window))


def start_exists(output_root: Path, window: str, start: str) -> bool:
    return (output_root / window / "predictions" / start).is_dir()
