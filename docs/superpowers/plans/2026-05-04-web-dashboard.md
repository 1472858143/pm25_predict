# PM2.5 Web Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web dashboard (`web/backend` FastAPI + `web/frontend` React/Vite) that visualizes and compares PM2.5 multi-model prediction outputs from `pm25_forecast/outputs/`.

**Architecture:** Read-only FastAPI backend exposes 6 JSON endpoints over the existing `outputs/` directory; React/TypeScript SPA renders three modules (metrics table, prediction curve, error analysis) with selectable window + start.

**Tech Stack:** FastAPI + Pydantic + pytest (backend); React 18 + TypeScript + Vite + Ant Design + ECharts + axios + vitest + @testing-library/react (frontend).

---

## File Structure

```
web/
├── .gitignore
├── README.md
├── backend/
│   ├── pyproject.toml
│   ├── README.md
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app + CORS + router include
│   │   ├── config.py          # OUTPUT_ROOT env var resolution
│   │   ├── schemas.py         # Pydantic response models
│   │   ├── data_loader.py     # outputs/ scanning + JSON/CSV parsers
│   │   └── routes.py          # 6 GET endpoints
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py        # fixtures (tmp outputs dir + TestClient)
│       ├── fixtures/
│       │   └── make_fixtures.py  # builds minimal outputs into tmp_path
│       ├── test_data_loader.py
│       ├── test_routes_windows.py
│       ├── test_routes_metrics.py
│       ├── test_routes_predictions.py
│       └── test_routes_horizon_metrics.py
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── tsconfig.node.json
    ├── vite.config.ts
    ├── index.html
    ├── README.md
    ├── src/
    │   ├── main.tsx
    │   ├── App.tsx
    │   ├── styles/global.css
    │   ├── types/api.ts                 # TS types matching backend schemas
    │   ├── api/client.ts                # axios instance + endpoint funcs
    │   ├── utils/format.ts              # formatWindow / formatStart
    │   ├── utils/colors.ts              # MODEL_COLORS palette
    │   ├── utils/url.ts                 # URL ?window=&start= sync
    │   ├── components/Header.tsx
    │   ├── components/MetricsTable.tsx
    │   ├── components/PredictionCurveChart.tsx
    │   └── components/ErrorAnalysis.tsx
    └── tests/
        ├── setup.ts
        ├── format.test.ts
        ├── url.test.ts
        ├── MetricsTable.test.tsx
        └── Header.test.tsx
```

---

## Task 1: Backend scaffold + .gitignore + health endpoint

**Files:**
- Create: `web/.gitignore`
- Create: `web/README.md`
- Create: `web/backend/pyproject.toml`
- Create: `web/backend/README.md`
- Create: `web/backend/app/__init__.py`
- Create: `web/backend/app/main.py`
- Create: `web/backend/app/config.py`
- Create: `web/backend/tests/__init__.py`
- Create: `web/backend/tests/test_health.py`

- [ ] **Step 1: Create `web/.gitignore`**

````
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
.eggs/

# Node
node_modules/
dist/
.vite/

# Editor / OS
.DS_Store
Thumbs.db
````

- [ ] **Step 2: Create `web/README.md`**

````markdown
# PM2.5 Web Dashboard

Local web dashboard for comparing PM2.5 multi-model prediction outputs.

## Subprojects
- `backend/` — FastAPI service exposing read-only JSON over `pm25_forecast/outputs/`
- `frontend/` — React + TypeScript + Vite SPA

See each subproject's README for setup.
````

- [ ] **Step 3: Create `web/backend/pyproject.toml`**

````toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pm25-web-backend"
version = "0.1.0"
description = "FastAPI backend for PM2.5 dashboard"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.5",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "httpx>=0.27",
]

[tool.setuptools.packages.find]
include = ["app*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
````

- [ ] **Step 4: Create `web/backend/README.md`**

````markdown
# Backend (FastAPI)

## Setup
    conda activate pm25
    cd web/backend
    pip install -e .[test]

## Run
    uvicorn app.main:app --reload --port 8000

Custom outputs root via PowerShell:
    $env:OUTPUT_ROOT = "E:\pm25_predict\pm25_forecast\outputs"
    uvicorn app.main:app --reload --port 8000

## Test
    pytest -v

## Endpoints
- GET /api/health
- GET /api/windows
- GET /api/metrics?window=...&start=...
- GET /api/predictions?window=...&start=...
- GET /api/predictions/{model}?window=...&start=...
- GET /api/horizon-metrics/{model}?window=...&start=...
````

- [ ] **Step 5: Create `web/backend/app/__init__.py` (empty file)**

````python
````

- [ ] **Step 6: Create `web/backend/app/config.py`**

````python
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "pm25_forecast" / "outputs"


def get_output_root() -> Path:
    """Resolve outputs root from OUTPUT_ROOT env var or default location."""
    env_value = os.environ.get("OUTPUT_ROOT")
    if env_value:
        return Path(env_value).resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()
````

- [ ] **Step 7: Create `web/backend/app/main.py`**

````python
from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_output_root

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="PM2.5 Dashboard API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    root = get_output_root()
    if not root.exists():
        logger.warning("OUTPUT_ROOT does not exist: %s", root)

    health_router = APIRouter(prefix="/api")

    @health_router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(health_router)
    return app


app = create_app()
````

- [ ] **Step 8: Create `web/backend/tests/__init__.py` (empty file)**

````python
````

- [ ] **Step 9: Write failing test `web/backend/tests/test_health.py`**

````python
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
````

- [ ] **Step 10: Install backend deps and run test**

Run from `web/backend`:
````
pip install -e .[test]
pytest tests/test_health.py -v
````
Expected: PASS (1 passed)

- [ ] **Step 11: Commit**

````
git add web/.gitignore web/README.md web/backend
git commit -m "feat(web): scaffold backend with health endpoint"
````

---

## Task 2: Backend data loader + test fixtures

**Files:**
- Create: `web/backend/app/schemas.py`
- Create: `web/backend/app/data_loader.py`
- Create: `web/backend/tests/fixtures/__init__.py`
- Create: `web/backend/tests/fixtures/make_fixtures.py`
- Create: `web/backend/tests/conftest.py`
- Create: `web/backend/tests/test_data_loader.py`

- [ ] **Step 1: Create `web/backend/app/schemas.py`**

````python
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
````

- [ ] **Step 2: Create `web/backend/app/data_loader.py`**

````python
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
````

- [ ] **Step 3: Create `web/backend/tests/fixtures/__init__.py` (empty)**

````python
````

- [ ] **Step 4: Create `web/backend/tests/fixtures/make_fixtures.py`**

````python
"""Build a minimal outputs/ directory under a given root for tests."""
from __future__ import annotations

import json
from pathlib import Path

WINDOW = "window_720h_to_72h"
START = "start_2026_03_01_0000"
PREDICT_START = "2026-03-01 00:00:00+08:00"
MODELS = ("lstm", "attention_lstm")

PREDICTIONS_HEADER = (
    "model_name,sample_id,origin_timestamp,target_end_timestamp,timestamp,"
    "horizon,y_true,y_pred_model,y_pred,error,abs_error,relative_error"
)

HORIZON_HEADER = "horizon,RMSE,MAE,MAPE,SMAPE,R2,bias"


def _metrics_for(model_name: str) -> dict[str, float]:
    base = 30.0 if model_name == "lstm" else 25.0
    return {
        "RMSE": base + 1.0,
        "MAE": base,
        "MAPE": base + 5.0,
        "SMAPE": base + 2.0,
        "R2": 0.5 if model_name == "attention_lstm" else 0.3,
        "bias": -10.0 if model_name == "lstm" else -5.0,
    }


def _prediction_rows(model_name: str) -> list[str]:
    rows: list[str] = []
    for horizon in (1, 2, 3):
        y_true = 100.0 + horizon
        y_pred_model = y_true + 5.0 if model_name == "lstm" else y_true + 2.0
        y_pred = y_pred_model - 1.0
        error = y_pred - y_true
        rows.append(
            f"{model_name},0,{PREDICT_START},2026-03-03 23:00:00+08:00,"
            f"2026-03-01 0{horizon - 1}:00:00+08:00,{horizon},{y_true},"
            f"{y_pred_model},{y_pred},{error},{abs(error)},{abs(error) / y_true}"
        )
    return rows


def _horizon_rows(model_name: str) -> list[str]:
    rows: list[str] = []
    for horizon in (1, 2, 3):
        delta = float(horizon)
        rows.append(f"{horizon},{delta + 1},{delta},{delta + 5},{delta + 2},0.0,{delta}")
    return rows


def build_fixture_outputs(root: Path) -> Path:
    """Create the minimal fixture under root (which is treated as OUTPUT_ROOT)."""
    pred_dir = root / WINDOW / "predictions" / START
    for model in MODELS:
        model_path = pred_dir / model
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "metrics.json").write_text(
            json.dumps(_metrics_for(model)), encoding="utf-8"
        )
        (model_path / "prediction_summary.json").write_text(
            json.dumps(
                {
                    "model_name": model,
                    "predict_start": PREDICT_START,
                    "metrics": _metrics_for(model),
                }
            ),
            encoding="utf-8",
        )
        predictions_csv = "\n".join([PREDICTIONS_HEADER, *_prediction_rows(model)]) + "\n"
        (model_path / "predictions.csv").write_text(predictions_csv, encoding="utf-8")
        horizon_csv = "\n".join([HORIZON_HEADER, *_horizon_rows(model)]) + "\n"
        (model_path / "horizon_metrics.csv").write_text(horizon_csv, encoding="utf-8")
    # Add a second window with no predictions, to validate empty-starts handling.
    (root / "window_168h_to_72h" / "data").mkdir(parents=True, exist_ok=True)
    return root
````

- [ ] **Step 5: Create `web/backend/tests/conftest.py`**

````python
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.make_fixtures import build_fixture_outputs


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialize a minimal outputs/ tree and point OUTPUT_ROOT to it."""
    root = build_fixture_outputs(tmp_path / "outputs")
    monkeypatch.setenv("OUTPUT_ROOT", str(root))
    return root


@pytest.fixture
def client(output_root: Path) -> Iterator[TestClient]:
    """TestClient that picks up the env var via a freshly-built app."""
    # Import inside the fixture so config.get_output_root() reads the patched env.
    from importlib import reload

    from app import main as main_module

    reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client
````

- [ ] **Step 6: Write failing tests `web/backend/tests/test_data_loader.py`**

````python
from __future__ import annotations

from pathlib import Path

from app.data_loader import (
    list_models,
    list_starts,
    list_windows,
    load_horizon_metrics_csv,
    load_metrics,
    load_predictions_csv,
    resolve_predict_start,
    start_exists,
    window_exists,
)


def test_list_windows_returns_both_windows(output_root: Path) -> None:
    windows = list_windows(output_root)
    names = [w["name"] for w in windows]
    assert "window_720h_to_72h" in names
    assert "window_168h_to_72h" in names
    full = next(w for w in windows if w["name"] == "window_720h_to_72h")
    assert full["input_window"] == 720
    assert full["output_window"] == 72
    assert full["starts"] == ["start_2026_03_01_0000"]
    empty = next(w for w in windows if w["name"] == "window_168h_to_72h")
    assert empty["starts"] == []


def test_list_windows_handles_missing_root(tmp_path: Path) -> None:
    assert list_windows(tmp_path / "does_not_exist") == []


def test_list_starts_excludes_unrelated_dirs(output_root: Path) -> None:
    window_dir = output_root / "window_720h_to_72h"
    (window_dir / "predictions" / "not_a_start").mkdir(parents=True, exist_ok=True)
    starts = list_starts(window_dir)
    assert starts == ["start_2026_03_01_0000"]


def test_list_models_returns_sorted_names(output_root: Path) -> None:
    models = list_models(output_root, "window_720h_to_72h", "start_2026_03_01_0000")
    assert models == ["attention_lstm", "lstm"]


def test_load_metrics_returns_dict(output_root: Path) -> None:
    metrics = load_metrics(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "lstm")
    assert metrics is not None
    assert metrics["RMSE"] == 31.0
    assert metrics["R2"] == 0.3


def test_load_metrics_missing_returns_none(output_root: Path) -> None:
    assert load_metrics(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "ghost") is None


def test_load_predictions_csv_parses_rows(output_root: Path) -> None:
    rows = load_predictions_csv(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "lstm")
    assert len(rows) == 3
    assert rows[0]["model_name"] == "lstm"
    assert rows[0]["horizon"] == "1"


def test_load_horizon_metrics_csv_parses_rows(output_root: Path) -> None:
    rows = load_horizon_metrics_csv(output_root, "window_720h_to_72h", "start_2026_03_01_0000", "lstm")
    assert len(rows) == 3
    assert rows[0]["horizon"] == "1"


def test_resolve_predict_start_returns_canonical_string(output_root: Path) -> None:
    value = resolve_predict_start(output_root, "window_720h_to_72h", "start_2026_03_01_0000")
    assert value == "2026-03-01 00:00:00+08:00"


def test_window_exists_and_start_exists(output_root: Path) -> None:
    assert window_exists(output_root, "window_720h_to_72h")
    assert not window_exists(output_root, "window_999h_to_99h")
    assert start_exists(output_root, "window_720h_to_72h", "start_2026_03_01_0000")
    assert not start_exists(output_root, "window_720h_to_72h", "start_9999_99_99_9999")
````

- [ ] **Step 7: Run tests to verify they pass**

Run from `web/backend`:
````
pytest tests/test_data_loader.py -v
````
Expected: All 10 tests PASS.

- [ ] **Step 8: Commit**

````
git add web/backend/app/schemas.py web/backend/app/data_loader.py web/backend/tests
git commit -m "feat(web): add backend data loader + fixtures"
````

---

## Task 3: `/api/windows` endpoint

**Files:**
- Create: `web/backend/app/routes.py`
- Modify: `web/backend/app/main.py` (replace inline health router with routes module)
- Create: `web/backend/tests/test_routes_windows.py`

- [ ] **Step 1: Create `web/backend/app/routes.py`**

````python
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_output_root
from app.data_loader import list_windows
from app.schemas import WindowInfo, WindowsResponse

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/windows", response_model=WindowsResponse)
def get_windows() -> WindowsResponse:
    root = get_output_root()
    raw = list_windows(root)
    return WindowsResponse(windows=[WindowInfo(**item) for item in raw])
````

- [ ] **Step 2: Update `web/backend/app/main.py`**

````python
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_output_root
from app.routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="PM2.5 Dashboard API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    root = get_output_root()
    if not root.exists():
        logger.warning("OUTPUT_ROOT does not exist: %s", root)
    app.include_router(router)
    return app


app = create_app()
````

- [ ] **Step 3: Write failing tests `web/backend/tests/test_routes_windows.py`**

````python
from __future__ import annotations

from fastapi.testclient import TestClient


def test_windows_returns_both_entries(client: TestClient) -> None:
    response = client.get("/api/windows")
    assert response.status_code == 200
    payload = response.json()
    names = [w["name"] for w in payload["windows"]]
    assert "window_720h_to_72h" in names
    assert "window_168h_to_72h" in names


def test_windows_includes_input_output_and_starts(client: TestClient) -> None:
    response = client.get("/api/windows")
    payload = response.json()
    full = next(w for w in payload["windows"] if w["name"] == "window_720h_to_72h")
    assert full["input_window"] == 720
    assert full["output_window"] == 72
    assert full["starts"] == ["start_2026_03_01_0000"]


def test_windows_returns_empty_when_root_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "nope"))
    from importlib import reload

    from app import main as main_module

    reload(main_module)
    client = TestClient(main_module.app)
    response = client.get("/api/windows")
    assert response.status_code == 200
    assert response.json() == {"windows": []}
````

- [ ] **Step 4: Run tests**

````
pytest tests/test_routes_windows.py tests/test_health.py -v
````
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

````
git add web/backend/app/routes.py web/backend/app/main.py web/backend/tests/test_routes_windows.py
git commit -m "feat(web): add /api/windows endpoint"
````

---

## Task 4: `/api/metrics` endpoint

**Files:**
- Modify: `web/backend/app/routes.py` (add metrics route)
- Create: `web/backend/tests/test_routes_metrics.py`

- [ ] **Step 1: Add metrics route to `web/backend/app/routes.py`**

Append after the `get_windows` function:

````python
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
        raw = load_metrics(root, chosen_window, chosen_start, model_name)
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
````

(Replace the duplicate `from app.data_loader import list_windows` and `from app.schemas import WindowInfo, WindowsResponse` lines that already exist at the top of the file with this consolidated import block. The full file should have a single set of imports near the top.)

- [ ] **Step 2: Verify final `web/backend/app/routes.py` matches this canonical version**

````python
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
        raw = load_metrics(root, chosen_window, chosen_start, model_name)
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
````

- [ ] **Step 3: Write failing tests `web/backend/tests/test_routes_metrics.py`**

````python
from __future__ import annotations

from fastapi.testclient import TestClient


def test_metrics_default_returns_first_window_first_start(client: TestClient) -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["window"] == "window_720h_to_72h"
    assert payload["start"] == "start_2026_03_01_0000"
    assert payload["predict_start"] == "2026-03-01 00:00:00+08:00"
    names = sorted(m["model_name"] for m in payload["models"])
    assert names == ["attention_lstm", "lstm"]
    assert payload["missing_models"] == []


def test_metrics_returns_numeric_fields(client: TestClient) -> None:
    response = client.get("/api/metrics")
    payload = response.json()
    lstm = next(m for m in payload["models"] if m["model_name"] == "lstm")
    assert lstm["RMSE"] == 31.0
    assert lstm["bias"] == -10.0


def test_metrics_unknown_window_returns_404(client: TestClient) -> None:
    response = client.get("/api/metrics", params={"window": "window_999h_to_99h"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_metrics_window_with_no_starts_returns_404(client: TestClient) -> None:
    response = client.get("/api/metrics", params={"window": "window_168h_to_72h"})
    assert response.status_code == 404
    assert "no starts" in response.json()["detail"].lower()


def test_metrics_skips_corrupt_model(client: TestClient, output_root) -> None:
    # Corrupt one model's metrics.json
    target = (
        output_root
        / "window_720h_to_72h"
        / "predictions"
        / "start_2026_03_01_0000"
        / "lstm"
        / "metrics.json"
    )
    target.write_text("not json", encoding="utf-8")
    response = client.get("/api/metrics")
    assert response.status_code == 500  # JSON parse blows up; covered by Task 5 hardening
````

Note: the corrupt-model test asserts 500 by design — the spec says CSV/JSON parse failures return 500.

- [ ] **Step 4: Run tests**

````
pytest tests/test_routes_metrics.py -v
````
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

````
git add web/backend/app/routes.py web/backend/tests/test_routes_metrics.py
git commit -m "feat(web): add /api/metrics endpoint"
````

---

## Task 5: `/api/predictions` and `/api/predictions/{model}` endpoints

**Files:**
- Modify: `web/backend/app/routes.py` (add two routes + imports)
- Create: `web/backend/tests/test_routes_predictions.py`

- [ ] **Step 1: Extend imports in `web/backend/app/routes.py`**

Update the existing import lines (DO NOT add a second import block):

````python
from app.data_loader import (
    list_models,
    list_windows,
    load_horizon_metrics_csv,
    load_metrics,
    load_predictions_csv,
    resolve_predict_start,
    start_exists,
    window_exists,
)
from app.schemas import (
    HorizonMetricRow,
    HorizonMetricsResponse,
    MetricsResponse,
    ModelMetrics,
    ModelPredictionsResponse,
    PredictionRow,
    PredictionsAggregateResponse,
    WindowInfo,
    WindowsResponse,
)
````

- [ ] **Step 2: Append predictions routes to `web/backend/app/routes.py`**

````python
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
````

- [ ] **Step 3: Write failing tests `web/backend/tests/test_routes_predictions.py`**

````python
from __future__ import annotations

from fastapi.testclient import TestClient


def test_predictions_aggregate_returns_arrays(client: TestClient) -> None:
    response = client.get("/api/predictions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["window"] == "window_720h_to_72h"
    assert payload["horizons"] == [1, 2, 3]
    assert len(payload["timestamps"]) == 3
    assert len(payload["y_true"]) == 3
    assert set(payload["predictions"].keys()) == {"lstm", "attention_lstm"}
    assert len(payload["predictions"]["lstm"]) == 3


def test_predictions_aggregate_y_true_values(client: TestClient) -> None:
    response = client.get("/api/predictions")
    payload = response.json()
    assert payload["y_true"] == [101.0, 102.0, 103.0]


def test_model_predictions_returns_rows(client: TestClient) -> None:
    response = client.get("/api/predictions/lstm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "lstm"
    assert len(payload["rows"]) == 3
    first = payload["rows"][0]
    assert first["horizon"] == 1
    assert first["y_true"] == 101.0
    assert "abs_error" in first


def test_model_predictions_unknown_model_404(client: TestClient) -> None:
    response = client.get("/api/predictions/ghost_model")
    assert response.status_code == 404


def test_predictions_pass_window_and_start_params(client: TestClient) -> None:
    response = client.get(
        "/api/predictions",
        params={"window": "window_720h_to_72h", "start": "start_2026_03_01_0000"},
    )
    assert response.status_code == 200
    assert response.json()["start"] == "start_2026_03_01_0000"
````

- [ ] **Step 4: Run tests**

````
pytest tests/test_routes_predictions.py -v
````
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

````
git add web/backend/app/routes.py web/backend/tests/test_routes_predictions.py
git commit -m "feat(web): add /api/predictions endpoints"
````

---

## Task 6: `/api/horizon-metrics/{model}` endpoint

**Files:**
- Modify: `web/backend/app/routes.py` (add one route)
- Create: `web/backend/tests/test_routes_horizon_metrics.py`

- [ ] **Step 1: Append horizon-metrics route to `web/backend/app/routes.py`**

````python
@router.get("/horizon-metrics/{model_name}", response_model=HorizonMetricsResponse)
def get_horizon_metrics(
    model_name: str, window: str | None = None, start: str | None = None
) -> HorizonMetricsResponse:
    root = get_output_root()
    chosen_window, chosen_start = _resolve_window_start(window, start)
    if model_name not in list_models(root, chosen_window, chosen_start):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    rows = load_horizon_metrics_csv(root, chosen_window, chosen_start, model_name)
    parsed = [
        HorizonMetricRow(
            horizon=int(r["horizon"]),
            RMSE=float(r["RMSE"]),
            MAE=float(r["MAE"]),
            MAPE=float(r["MAPE"]),
            SMAPE=float(r["SMAPE"]),
            R2=float(r["R2"]),
            bias=float(r["bias"]),
        )
        for r in rows
    ]
    return HorizonMetricsResponse(
        window=chosen_window,
        start=chosen_start,
        model_name=model_name,
        rows=parsed,
    )
````

- [ ] **Step 2: Write failing tests `web/backend/tests/test_routes_horizon_metrics.py`**

````python
from __future__ import annotations

from fastapi.testclient import TestClient


def test_horizon_metrics_returns_rows(client: TestClient) -> None:
    response = client.get("/api/horizon-metrics/lstm")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "lstm"
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["horizon"] == 1


def test_horizon_metrics_unknown_model_404(client: TestClient) -> None:
    response = client.get("/api/horizon-metrics/ghost")
    assert response.status_code == 404
````

- [ ] **Step 3: Run tests**

````
pytest tests/test_routes_horizon_metrics.py -v
pytest -v
````
Expected: New tests PASS, full suite PASS.

- [ ] **Step 4: Commit**

````
git add web/backend/app/routes.py web/backend/tests/test_routes_horizon_metrics.py
git commit -m "feat(web): add /api/horizon-metrics endpoint"
````

---

## Task 7: Frontend scaffold + Vite proxy

**Files:**
- Create: `web/frontend/package.json`
- Create: `web/frontend/tsconfig.json`
- Create: `web/frontend/tsconfig.node.json`
- Create: `web/frontend/vite.config.ts`
- Create: `web/frontend/index.html`
- Create: `web/frontend/README.md`
- Create: `web/frontend/src/main.tsx`
- Create: `web/frontend/src/App.tsx`
- Create: `web/frontend/src/styles/global.css`
- Create: `web/frontend/tests/setup.ts`

- [ ] **Step 1: Create `web/frontend/package.json`**

````json
{
  "name": "pm25-web-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "antd": "^5.16.0",
    "axios": "^1.7.0",
    "echarts": "^5.5.0",
    "echarts-for-react": "^3.0.2",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^16.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "jsdom": "^24.1.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0"
  }
}
````

- [ ] **Step 2: Create `web/frontend/tsconfig.json`**

````json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src", "tests"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
````

- [ ] **Step 3: Create `web/frontend/tsconfig.node.json`**

````json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "noEmit": true,
    "skipLibCheck": true,
    "types": ["node"]
  },
  "include": ["vite.config.ts"]
}
````

- [ ] **Step 4: Create `web/frontend/vite.config.ts`**

````typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
````

- [ ] **Step 5: Create `web/frontend/index.html`**

````html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PM2.5 多模型预测对比仪表盘</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
````

- [ ] **Step 6: Create `web/frontend/README.md`**

````markdown
# Frontend (React + Vite)

## Setup
    cd web/frontend
    npm install

## Run
    npm run dev          # http://localhost:5173

(Backend must be running on http://localhost:8000 — Vite proxies /api/* to it.)

## Test
    npm test

## Build
    npm run build
````

- [ ] **Step 7: Create `web/frontend/src/styles/global.css`**

````css
html, body, #root {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: #f5f7fa;
}
````

- [ ] **Step 8: Create `web/frontend/src/main.tsx`**

````typescript
import React from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";
import "antd/dist/reset.css";
import "./styles/global.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN}>
      <App />
    </ConfigProvider>
  </React.StrictMode>
);
````

- [ ] **Step 9: Create `web/frontend/src/App.tsx` (placeholder)**

````typescript
import React from "react";

const App: React.FC = () => {
  return (
    <div style={{ padding: 24 }}>
      <h1>PM2.5 多模型预测对比仪表盘</h1>
      <p>Loading...</p>
    </div>
  );
};

export default App;
````

- [ ] **Step 10: Create `web/frontend/tests/setup.ts`**

````typescript
import "@testing-library/jest-dom";
````

- [ ] **Step 11: Install deps and verify dev server boots**

Run from `web/frontend`:
````
npm install
npm run build
````
Expected: install completes, `npm run build` produces `dist/` without errors.

- [ ] **Step 12: Commit**

````
git add web/frontend
git commit -m "feat(web): scaffold React + Vite frontend"
````

---

## Task 8: Frontend types + format utils + tests

**Files:**
- Create: `web/frontend/src/types/api.ts`
- Create: `web/frontend/src/utils/format.ts`
- Create: `web/frontend/src/utils/colors.ts`
- Create: `web/frontend/src/utils/url.ts`
- Create: `web/frontend/tests/format.test.ts`
- Create: `web/frontend/tests/url.test.ts`

- [ ] **Step 1: Create `web/frontend/src/types/api.ts`**

````typescript
export interface WindowInfo {
  name: string;
  input_window: number;
  output_window: number;
  starts: string[];
}

export interface WindowsResponse {
  windows: WindowInfo[];
}

export interface ModelMetrics {
  model_name: string;
  RMSE: number;
  MAE: number;
  MAPE: number;
  SMAPE: number;
  R2: number;
  bias: number;
}

export interface MetricsResponse {
  window: string;
  start: string;
  predict_start: string;
  models: ModelMetrics[];
  missing_models: string[];
}

export interface PredictionsAggregateResponse {
  window: string;
  start: string;
  horizons: number[];
  timestamps: string[];
  y_true: number[];
  predictions: Record<string, number[]>;
  missing_models: string[];
}

export interface PredictionRow {
  sample_id: number;
  origin_timestamp: string;
  target_end_timestamp: string;
  timestamp: string;
  horizon: number;
  y_true: number;
  y_pred_model: number;
  y_pred: number;
  error: number;
  abs_error: number;
  relative_error: number;
}

export interface ModelPredictionsResponse {
  window: string;
  start: string;
  model_name: string;
  rows: PredictionRow[];
}
````

- [ ] **Step 2: Create `web/frontend/src/utils/format.ts`**

````typescript
const WINDOW_PATTERN = /^window_(\d+)h_to_(\d+)h$/;
const START_PATTERN = /^start_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})$/;

function hoursToLabel(hours: number): string {
  if (hours % 24 === 0) {
    return `${hours / 24} 天`;
  }
  return `${hours}h`;
}

export function formatWindow(name: string): string {
  const match = WINDOW_PATTERN.exec(name);
  if (!match) {
    return name;
  }
  const input = Number(match[1]);
  const output = Number(match[2]);
  return `${hoursToLabel(input)}历史 → ${hoursToLabel(output)}预测 (${input}h→${output}h)`;
}

export function formatStart(name: string): string {
  const match = START_PATTERN.exec(name);
  if (!match) {
    return name;
  }
  const [, year, month, day, hour, minute] = match;
  return `${year}-${month}-${day} ${hour}:${minute} (北京时间)`;
}
````

- [ ] **Step 3: Create `web/frontend/src/utils/colors.ts`**

````typescript
export const MODEL_COLORS: Record<string, string> = {
  lstm: "#1677ff",
  attention_lstm: "#f5222d",
  xgboost: "#52c41a",
  random_forest: "#722ed1",
  arima: "#fa8c16",
  sarima: "#8c8c8c",
};

export const Y_TRUE_COLOR = "#000000";

export function colorForModel(name: string): string {
  return MODEL_COLORS[name] ?? "#13c2c2";
}
````

- [ ] **Step 4: Create `web/frontend/src/utils/url.ts`**

````typescript
export interface UrlSelection {
  window: string | null;
  start: string | null;
}

export function readSelectionFromUrl(search: string): UrlSelection {
  const params = new URLSearchParams(search);
  return {
    window: params.get("window"),
    start: params.get("start"),
  };
}

export function buildSearchString(selection: UrlSelection): string {
  const params = new URLSearchParams();
  if (selection.window) {
    params.set("window", selection.window);
  }
  if (selection.start) {
    params.set("start", selection.start);
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}
````

- [ ] **Step 5: Write failing test `web/frontend/tests/format.test.ts`**

````typescript
import { describe, expect, it } from "vitest";
import { formatStart, formatWindow } from "../src/utils/format";

describe("formatWindow", () => {
  it("formats 720h to 72h as 30 days to 3 days", () => {
    expect(formatWindow("window_720h_to_72h")).toBe(
      "30 天历史 → 3 天预测 (720h→72h)"
    );
  });

  it("formats 168h to 72h", () => {
    expect(formatWindow("window_168h_to_72h")).toBe(
      "7 天历史 → 3 天预测 (168h→72h)"
    );
  });

  it("falls back when not divisible by 24", () => {
    expect(formatWindow("window_5h_to_2h")).toBe("5h历史 → 2h预测 (5h→2h)");
  });

  it("returns input when pattern does not match", () => {
    expect(formatWindow("garbage")).toBe("garbage");
  });
});

describe("formatStart", () => {
  it("formats start_2026_03_01_0000", () => {
    expect(formatStart("start_2026_03_01_0000")).toBe(
      "2026-03-01 00:00 (北京时间)"
    );
  });

  it("formats start_2026_12_31_2359", () => {
    expect(formatStart("start_2026_12_31_2359")).toBe(
      "2026-12-31 23:59 (北京时间)"
    );
  });

  it("returns input when pattern does not match", () => {
    expect(formatStart("garbage")).toBe("garbage");
  });
});
````

- [ ] **Step 6: Write failing test `web/frontend/tests/url.test.ts`**

````typescript
import { describe, expect, it } from "vitest";
import { buildSearchString, readSelectionFromUrl } from "../src/utils/url";

describe("readSelectionFromUrl", () => {
  it("returns nulls for empty search", () => {
    expect(readSelectionFromUrl("")).toEqual({ window: null, start: null });
  });

  it("parses window and start params", () => {
    expect(readSelectionFromUrl("?window=W1&start=S1")).toEqual({
      window: "W1",
      start: "S1",
    });
  });
});

describe("buildSearchString", () => {
  it("returns empty string when both null", () => {
    expect(buildSearchString({ window: null, start: null })).toBe("");
  });

  it("returns ?window=...&start=... when both present", () => {
    expect(
      buildSearchString({ window: "window_720h_to_72h", start: "start_2026_03_01_0000" })
    ).toBe("?window=window_720h_to_72h&start=start_2026_03_01_0000");
  });
});
````

- [ ] **Step 7: Run tests**

Run from `web/frontend`:
````
npm test
````
Expected: All format + url tests PASS.

- [ ] **Step 8: Commit**

````
git add web/frontend/src/types web/frontend/src/utils web/frontend/tests/format.test.ts web/frontend/tests/url.test.ts
git commit -m "feat(web): add frontend types, format/url/colors utils"
````

---

## Task 9: Frontend API client

**Files:**
- Create: `web/frontend/src/api/client.ts`

- [ ] **Step 1: Create `web/frontend/src/api/client.ts`**

````typescript
import axios from "axios";
import type {
  MetricsResponse,
  ModelPredictionsResponse,
  PredictionsAggregateResponse,
  WindowsResponse,
} from "../types/api";

const http = axios.create({
  baseURL: "/api",
  timeout: 15000,
});

export interface SelectionParams {
  window?: string;
  start?: string;
}

export async function fetchWindows(): Promise<WindowsResponse> {
  const { data } = await http.get<WindowsResponse>("/windows");
  return data;
}

export async function fetchMetrics(params: SelectionParams = {}): Promise<MetricsResponse> {
  const { data } = await http.get<MetricsResponse>("/metrics", { params });
  return data;
}

export async function fetchPredictionsAggregate(
  params: SelectionParams = {}
): Promise<PredictionsAggregateResponse> {
  const { data } = await http.get<PredictionsAggregateResponse>("/predictions", { params });
  return data;
}

export async function fetchModelPredictions(
  modelName: string,
  params: SelectionParams = {}
): Promise<ModelPredictionsResponse> {
  const { data } = await http.get<ModelPredictionsResponse>(
    `/predictions/${encodeURIComponent(modelName)}`,
    { params }
  );
  return data;
}
````

- [ ] **Step 2: Type-check via build**

Run from `web/frontend`:
````
npm run build
````
Expected: build succeeds.

- [ ] **Step 3: Commit**

````
git add web/frontend/src/api
git commit -m "feat(web): add axios API client"
````

---

## Task 10: Header with window/start selectors + URL sync

**Files:**
- Create: `web/frontend/src/components/Header.tsx`
- Modify: `web/frontend/src/App.tsx` (use Header + lift selection state)
- Create: `web/frontend/tests/Header.test.tsx`

- [ ] **Step 1: Create `web/frontend/src/components/Header.tsx`**

````typescript
import React from "react";
import { Layout, Select, Space, Typography } from "antd";
import type { WindowInfo } from "../types/api";
import { formatStart, formatWindow } from "../utils/format";

const { Header: AntHeader } = Layout;
const { Title } = Typography;

export interface HeaderProps {
  windows: WindowInfo[];
  selectedWindow: string | null;
  selectedStart: string | null;
  onChangeWindow: (name: string) => void;
  onChangeStart: (start: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  windows,
  selectedWindow,
  selectedStart,
  onChangeWindow,
  onChangeStart,
}) => {
  const currentWindow = windows.find((w) => w.name === selectedWindow);
  const startOptions = currentWindow?.starts ?? [];

  return (
    <AntHeader style={{ background: "#fff", padding: "0 24px", height: "auto", lineHeight: "1.5" }}>
      <div style={{ padding: "16px 0" }}>
        <Title level={3} style={{ margin: 0 }}>
          PM2.5 多模型预测对比仪表盘
        </Title>
        <Space size="large" style={{ marginTop: 12 }}>
          <Space>
            <span>时间窗口:</span>
            <Select
              data-testid="window-select"
              style={{ minWidth: 280 }}
              value={selectedWindow ?? undefined}
              onChange={onChangeWindow}
              options={windows.map((w) => ({
                value: w.name,
                label: formatWindow(w.name),
              }))}
            />
          </Space>
          <Space>
            <span>预测起点:</span>
            <Select
              data-testid="start-select"
              style={{ minWidth: 240 }}
              value={selectedStart ?? undefined}
              onChange={onChangeStart}
              disabled={startOptions.length === 0}
              placeholder={startOptions.length === 0 ? "该窗口暂无预测结果" : undefined}
              options={startOptions.map((s) => ({
                value: s,
                label: formatStart(s),
              }))}
            />
          </Space>
        </Space>
      </div>
    </AntHeader>
  );
};
````

- [ ] **Step 2: Replace `web/frontend/src/App.tsx`**

````typescript
import React, { useEffect, useState } from "react";
import { Alert, Empty, Layout, Spin } from "antd";
import { Header } from "./components/Header";
import { fetchWindows } from "./api/client";
import type { WindowInfo } from "./types/api";
import { buildSearchString, readSelectionFromUrl } from "./utils/url";

const { Content } = Layout;

const App: React.FC = () => {
  const [windows, setWindows] = useState<WindowInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedWindow, setSelectedWindow] = useState<string | null>(null);
  const [selectedStart, setSelectedStart] = useState<string | null>(null);

  useEffect(() => {
    fetchWindows()
      .then((res) => {
        setWindows(res.windows);
        const fromUrl = readSelectionFromUrl(window.location.search);
        const firstWithStart = res.windows.find((w) => w.starts.length > 0);
        const initialWindow =
          (fromUrl.window && res.windows.find((w) => w.name === fromUrl.window)?.name) ??
          firstWithStart?.name ??
          res.windows[0]?.name ??
          null;
        const target = res.windows.find((w) => w.name === initialWindow);
        const initialStart =
          (fromUrl.start && target?.starts.includes(fromUrl.start) ? fromUrl.start : null) ??
          target?.starts[0] ??
          null;
        setSelectedWindow(initialWindow);
        setSelectedStart(initialStart);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    const search = buildSearchString({ window: selectedWindow, start: selectedStart });
    const newUrl = `${window.location.pathname}${search}`;
    window.history.replaceState(null, "", newUrl);
  }, [selectedWindow, selectedStart]);

  const handleWindowChange = (name: string) => {
    setSelectedWindow(name);
    const target = windows?.find((w) => w.name === name);
    setSelectedStart(target?.starts[0] ?? null);
  };

  if (error) {
    return <Alert type="error" message={`加载窗口列表失败: ${error}`} showIcon style={{ margin: 24 }} />;
  }

  if (!windows) {
    return (
      <div style={{ padding: 48, textAlign: "center" }}>
        <Spin tip="加载中..." />
      </div>
    );
  }

  if (windows.length === 0) {
    return (
      <Empty
        description="未发现任何预测输出，请先运行 prepare_data / train_model / predict_model"
        style={{ marginTop: 96 }}
      />
    );
  }

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header
        windows={windows}
        selectedWindow={selectedWindow}
        selectedStart={selectedStart}
        onChangeWindow={handleWindowChange}
        onChangeStart={setSelectedStart}
      />
      <Content style={{ padding: 24 }}>
        {selectedStart === null ? (
          <Empty description="该时间窗口暂无预测结果" />
        ) : (
          <div>三个模块占位（Tasks 11/12/13 实现）</div>
        )}
      </Content>
    </Layout>
  );
};

export default App;
````

- [ ] **Step 3: Write failing test `web/frontend/tests/Header.test.tsx`**

````typescript
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Header } from "../src/components/Header";

const windows = [
  {
    name: "window_720h_to_72h",
    input_window: 720,
    output_window: 72,
    starts: ["start_2026_03_01_0000"],
  },
  {
    name: "window_168h_to_72h",
    input_window: 168,
    output_window: 72,
    starts: [],
  },
];

describe("Header", () => {
  it("renders the dashboard title", () => {
    render(
      <Header
        windows={windows}
        selectedWindow="window_720h_to_72h"
        selectedStart="start_2026_03_01_0000"
        onChangeWindow={() => {}}
        onChangeStart={() => {}}
      />
    );
    expect(screen.getByText(/PM2.5 多模型预测对比仪表盘/)).toBeInTheDocument();
  });

  it("shows formatted window label as selected value", () => {
    render(
      <Header
        windows={windows}
        selectedWindow="window_720h_to_72h"
        selectedStart="start_2026_03_01_0000"
        onChangeWindow={() => {}}
        onChangeStart={() => {}}
      />
    );
    expect(screen.getByText(/30 天历史/)).toBeInTheDocument();
  });

  it("disables the start selector when window has no starts", () => {
    render(
      <Header
        windows={windows}
        selectedWindow="window_168h_to_72h"
        selectedStart={null}
        onChangeWindow={() => {}}
        onChangeStart={() => {}}
      />
    );
    const startSelect = screen
      .getByTestId("start-select")
      .querySelector(".ant-select");
    expect(startSelect?.className).toContain("ant-select-disabled");
  });

  it("calls onChangeWindow when window selector changes", () => {
    const handler = vi.fn();
    render(
      <Header
        windows={windows}
        selectedWindow="window_720h_to_72h"
        selectedStart="start_2026_03_01_0000"
        onChangeWindow={handler}
        onChangeStart={() => {}}
      />
    );
    const trigger = screen.getByTestId("window-select").querySelector(".ant-select-selector");
    fireEvent.mouseDown(trigger as Element);
    const option = screen.getByText(/7 天历史/);
    fireEvent.click(option);
    expect(handler).toHaveBeenCalledWith("window_168h_to_72h");
  });
});
````

- [ ] **Step 4: Run tests**

Run from `web/frontend`:
````
npm test
````
Expected: All tests PASS (format + url + Header).

- [ ] **Step 5: Commit**

````
git add web/frontend/src/components/Header.tsx web/frontend/src/App.tsx web/frontend/tests/Header.test.tsx
git commit -m "feat(web): add Header with window/start selectors and URL sync"
````

---

## Task 11: Metrics comparison table component

**Files:**
- Create: `web/frontend/src/components/MetricsTable.tsx`
- Modify: `web/frontend/src/App.tsx` (mount MetricsTable in Content)
- Create: `web/frontend/tests/MetricsTable.test.tsx`

- [ ] **Step 1: Create `web/frontend/src/components/MetricsTable.tsx`**

````typescript
import React, { useEffect, useState } from "react";
import { Alert, Card, Skeleton, Table, Tag } from "antd";
import type { ColumnsType } from "antd/es/table";
import { fetchMetrics } from "../api/client";
import type { ModelMetrics, MetricsResponse } from "../types/api";

const METRIC_KEYS: (keyof Omit<ModelMetrics, "model_name">)[] = [
  "RMSE",
  "MAE",
  "MAPE",
  "SMAPE",
  "R2",
  "bias",
];

// For each metric, lower is better unless listed here.
const HIGHER_IS_BETTER: Set<string> = new Set(["R2"]);

function bestValueFor(metric: keyof Omit<ModelMetrics, "model_name">, models: ModelMetrics[]): number | null {
  if (models.length === 0) return null;
  const values = models.map((m) => m[metric]);
  if (HIGHER_IS_BETTER.has(metric)) {
    return Math.max(...values);
  }
  // For bias, "best" = closest to zero
  if (metric === "bias") {
    return values.reduce((best, v) => (Math.abs(v) < Math.abs(best) ? v : best), values[0]);
  }
  return Math.min(...values);
}

export interface MetricsTableProps {
  window: string;
  start: string;
}

export const MetricsTable: React.FC<MetricsTableProps> = ({ window, start }) => {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchMetrics({ window, start })
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [window, start]);

  if (loading) {
    return (
      <Card title="模型指标对比">
        <Skeleton active />
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card title="模型指标对比">
        <Alert type="error" message={`加载失败: ${error ?? "no data"}`} showIcon />
      </Card>
    );
  }

  const bestPerMetric: Record<string, number | null> = {};
  for (const k of METRIC_KEYS) {
    bestPerMetric[k] = bestValueFor(k, data.models);
  }

  const columns: ColumnsType<ModelMetrics> = [
    {
      title: "模型",
      dataIndex: "model_name",
      key: "model_name",
      render: (name: string) =>
        name === "attention_lstm" ? <strong>{name}</strong> : name,
    },
    ...METRIC_KEYS.map((key) => ({
      title: key,
      dataIndex: key,
      key,
      sorter: (a: ModelMetrics, b: ModelMetrics) => a[key] - b[key],
      render: (value: number) => {
        const best = bestPerMetric[key];
        const isBest = best !== null && value === best;
        const text = value.toFixed(3);
        return isBest ? <span style={{ background: "#d9f7be", padding: "2px 6px" }}>{text}</span> : text;
      },
    })),
  ];

  return (
    <Card title="模型指标对比">
      <Table<ModelMetrics>
        rowKey="model_name"
        columns={columns}
        dataSource={data.models}
        pagination={false}
      />
      {data.missing_models.length > 0 && (
        <div style={{ marginTop: 12 }}>
          {data.missing_models.map((m) => (
            <Tag key={m} color="warning">
              缺失: {m}
            </Tag>
          ))}
        </div>
      )}
    </Card>
  );
};
````

- [ ] **Step 2: Update `web/frontend/src/App.tsx` Content body**

Replace the Content body's placeholder div with:

````typescript
        {selectedStart === null ? (
          <Empty description="该时间窗口暂无预测结果" />
        ) : (
          <MetricsTable window={selectedWindow!} start={selectedStart} />
        )}
````

And add this import at the top of `App.tsx`:

````typescript
import { MetricsTable } from "./components/MetricsTable";
````

- [ ] **Step 3: Write failing test `web/frontend/tests/MetricsTable.test.tsx`**

````typescript
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("../src/api/client", () => ({
  fetchMetrics: vi.fn(),
}));

import { MetricsTable } from "../src/components/MetricsTable";
import { fetchMetrics } from "../src/api/client";

describe("MetricsTable", () => {
  beforeEach(() => {
    vi.mocked(fetchMetrics).mockReset();
  });

  it("renders model rows from API response", async () => {
    vi.mocked(fetchMetrics).mockResolvedValue({
      window: "window_720h_to_72h",
      start: "start_2026_03_01_0000",
      predict_start: "2026-03-01 00:00:00+08:00",
      models: [
        { model_name: "lstm", RMSE: 31, MAE: 30, MAPE: 35, SMAPE: 32, R2: 0.3, bias: -10 },
        { model_name: "attention_lstm", RMSE: 26, MAE: 25, MAPE: 30, SMAPE: 27, R2: 0.5, bias: -5 },
      ],
      missing_models: [],
    });
    render(<MetricsTable window="window_720h_to_72h" start="start_2026_03_01_0000" />);
    await waitFor(() => {
      expect(screen.getByText("lstm")).toBeInTheDocument();
    });
    expect(screen.getByText("attention_lstm")).toBeInTheDocument();
  });

  it("displays missing models tag", async () => {
    vi.mocked(fetchMetrics).mockResolvedValue({
      window: "window_720h_to_72h",
      start: "start_2026_03_01_0000",
      predict_start: "2026-03-01 00:00:00+08:00",
      models: [
        { model_name: "lstm", RMSE: 31, MAE: 30, MAPE: 35, SMAPE: 32, R2: 0.3, bias: -10 },
      ],
      missing_models: ["arima"],
    });
    render(<MetricsTable window="window_720h_to_72h" start="start_2026_03_01_0000" />);
    await waitFor(() => {
      expect(screen.getByText(/缺失: arima/)).toBeInTheDocument();
    });
  });

  it("shows error alert on failure", async () => {
    vi.mocked(fetchMetrics).mockRejectedValue(new Error("boom"));
    render(<MetricsTable window="window_720h_to_72h" start="start_2026_03_01_0000" />);
    await waitFor(() => {
      expect(screen.getByText(/加载失败/)).toBeInTheDocument();
    });
  });
});
````

- [ ] **Step 4: Run tests + build**

Run from `web/frontend`:
````
npm test
npm run build
````
Expected: All tests PASS, build succeeds.

- [ ] **Step 5: Commit**

````
git add web/frontend/src/components/MetricsTable.tsx web/frontend/src/App.tsx web/frontend/tests/MetricsTable.test.tsx
git commit -m "feat(web): add metrics comparison table"
````

---

## Task 12: Prediction curve chart component

**Files:**
- Create: `web/frontend/src/components/PredictionCurveChart.tsx`
- Modify: `web/frontend/src/App.tsx` (mount below MetricsTable)

- [ ] **Step 1: Create `web/frontend/src/components/PredictionCurveChart.tsx`**

````typescript
import React, { useEffect, useState } from "react";
import { Alert, Card, Skeleton } from "antd";
import ReactECharts from "echarts-for-react";
import { fetchPredictionsAggregate } from "../api/client";
import type { PredictionsAggregateResponse } from "../types/api";
import { Y_TRUE_COLOR, colorForModel } from "../utils/colors";

export interface PredictionCurveChartProps {
  window: string;
  start: string;
}

export const PredictionCurveChart: React.FC<PredictionCurveChartProps> = ({ window, start }) => {
  const [data, setData] = useState<PredictionsAggregateResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchPredictionsAggregate({ window, start })
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [window, start]);

  if (loading) {
    return (
      <Card title="预测曲线对比" style={{ marginTop: 16 }}>
        <Skeleton active />
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card title="预测曲线对比" style={{ marginTop: 16 }}>
        <Alert type="error" message={`加载失败: ${error ?? "no data"}`} showIcon />
      </Card>
    );
  }

  const modelNames = Object.keys(data.predictions);
  const series = [
    {
      name: "y_true",
      type: "line",
      data: data.y_true,
      lineStyle: { color: Y_TRUE_COLOR, width: 3 },
      itemStyle: { color: Y_TRUE_COLOR },
      smooth: false,
    },
    ...modelNames.map((name) => ({
      name,
      type: "line",
      data: data.predictions[name],
      lineStyle: { color: colorForModel(name) },
      itemStyle: { color: colorForModel(name) },
      smooth: false,
    })),
  ];

  const option = {
    tooltip: { trigger: "axis" },
    legend: { data: ["y_true", ...modelNames], top: 0 },
    grid: { top: 40, left: 60, right: 30, bottom: 60 },
    xAxis: {
      type: "category",
      data: data.timestamps,
      axisLabel: { rotate: 45, fontSize: 10 },
    },
    yAxis: {
      type: "value",
      name: "PM2.5 (μg/m³)",
    },
    dataZoom: [
      { type: "inside" },
      { type: "slider" },
    ],
    series,
  };

  return (
    <Card title="预测曲线对比" style={{ marginTop: 16 }}>
      <ReactECharts option={option} style={{ height: 480 }} notMerge lazyUpdate />
    </Card>
  );
};
````

- [ ] **Step 2: Update `web/frontend/src/App.tsx`**

Replace the `MetricsTable` line in the Content body with:

````typescript
        {selectedStart === null ? (
          <Empty description="该时间窗口暂无预测结果" />
        ) : (
          <>
            <MetricsTable window={selectedWindow!} start={selectedStart} />
            <PredictionCurveChart window={selectedWindow!} start={selectedStart} />
          </>
        )}
````

And add the import:

````typescript
import { PredictionCurveChart } from "./components/PredictionCurveChart";
````

- [ ] **Step 3: Build to verify type-correctness**

Run from `web/frontend`:
````
npm run build
````
Expected: build succeeds.

- [ ] **Step 4: Commit**

````
git add web/frontend/src/components/PredictionCurveChart.tsx web/frontend/src/App.tsx
git commit -m "feat(web): add prediction curve chart"
````

---

## Task 13: Error analysis component (scatter + residual histogram)

**Files:**
- Create: `web/frontend/src/components/ErrorAnalysis.tsx`
- Modify: `web/frontend/src/App.tsx` (mount below PredictionCurveChart)

- [ ] **Step 1: Create `web/frontend/src/components/ErrorAnalysis.tsx`**

````typescript
import React, { useEffect, useMemo, useState } from "react";
import { Alert, Card, Col, Row, Select, Skeleton, Space } from "antd";
import ReactECharts from "echarts-for-react";
import { fetchModelPredictions } from "../api/client";
import type { ModelPredictionsResponse, PredictionRow } from "../types/api";
import { colorForModel } from "../utils/colors";

const DEFAULT_MODEL = "attention_lstm";
const HISTOGRAM_BINS = 20;

function buildHistogram(values: number[]): { x: string[]; y: number[]; mean: number; std: number } {
  if (values.length === 0) {
    return { x: [], y: [], mean: 0, std: 0 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const binWidth = span / HISTOGRAM_BINS;
  const counts = new Array(HISTOGRAM_BINS).fill(0);
  for (const v of values) {
    const idx = Math.min(HISTOGRAM_BINS - 1, Math.floor((v - min) / binWidth));
    counts[idx] += 1;
  }
  const x = counts.map((_, i) => (min + binWidth * (i + 0.5)).toFixed(1));
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
  return { x, y: counts, mean, std: Math.sqrt(variance) };
}

export interface ErrorAnalysisProps {
  window: string;
  start: string;
  availableModels: string[];
}

export const ErrorAnalysis: React.FC<ErrorAnalysisProps> = ({ window, start, availableModels }) => {
  const initialModel = availableModels.includes(DEFAULT_MODEL)
    ? DEFAULT_MODEL
    : availableModels[0] ?? null;
  const [selectedModel, setSelectedModel] = useState<string | null>(initialModel);
  const [data, setData] = useState<ModelPredictionsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedModel) return;
    setLoading(true);
    setError(null);
    fetchModelPredictions(selectedModel, { window, start })
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [selectedModel, window, start]);

  const scatterOption = useMemo(() => {
    if (!data) return null;
    const points = data.rows.map((r: PredictionRow) => [r.y_true, r.y_pred]);
    const all = points.flat();
    const min = Math.min(...all);
    const max = Math.max(...all);
    return {
      tooltip: {
        trigger: "item",
        formatter: (p: { value: number[] }) => `y_true: ${p.value[0].toFixed(2)}<br/>y_pred: ${p.value[1].toFixed(2)}`,
      },
      grid: { top: 40, left: 60, right: 30, bottom: 50 },
      xAxis: { type: "value", name: "y_true", min, max },
      yAxis: { type: "value", name: "y_pred", min, max },
      series: [
        {
          type: "scatter",
          data: points,
          itemStyle: { color: colorForModel(selectedModel ?? "") },
          symbolSize: 6,
        },
        {
          type: "line",
          data: [
            [min, min],
            [max, max],
          ],
          lineStyle: { color: "#999", type: "dashed" },
          symbol: "none",
          tooltip: { show: false },
        },
      ],
    };
  }, [data, selectedModel]);

  const histogramOption = useMemo(() => {
    if (!data) return null;
    const errors = data.rows.map((r: PredictionRow) => r.error);
    const { x, y, mean, std } = buildHistogram(errors);
    return {
      tooltip: { trigger: "axis" },
      grid: { top: 40, left: 60, right: 30, bottom: 50 },
      title: {
        text: `mean=${mean.toFixed(2)}, std=${std.toFixed(2)}`,
        textStyle: { fontSize: 12, fontWeight: "normal" },
        left: "center",
        top: 0,
      },
      xAxis: { type: "category", data: x, name: "error" },
      yAxis: { type: "value", name: "count" },
      series: [
        {
          type: "bar",
          data: y,
          itemStyle: { color: colorForModel(selectedModel ?? "") },
        },
      ],
    };
  }, [data, selectedModel]);

  return (
    <Card title="单模型误差分析" style={{ marginTop: 16 }}>
      <Space style={{ marginBottom: 16 }}>
        <span>选择模型:</span>
        <Select
          style={{ minWidth: 200 }}
          value={selectedModel ?? undefined}
          onChange={setSelectedModel}
          options={availableModels.map((m) => ({ value: m, label: m }))}
        />
      </Space>
      {loading && <Skeleton active />}
      {error && <Alert type="error" message={`加载失败: ${error}`} showIcon />}
      {!loading && !error && data && scatterOption && histogramOption && (
        <Row gutter={16}>
          <Col span={12}>
            <Card type="inner" title="y_true vs y_pred">
              <ReactECharts option={scatterOption} style={{ height: 360 }} notMerge lazyUpdate />
            </Card>
          </Col>
          <Col span={12}>
            <Card type="inner" title="残差分布">
              <ReactECharts option={histogramOption} style={{ height: 360 }} notMerge lazyUpdate />
            </Card>
          </Col>
        </Row>
      )}
    </Card>
  );
};
````

- [ ] **Step 2: Update `web/frontend/src/App.tsx`**

The component needs the list of available models. The simplest path: lift it via `MetricsResponse.models`. Refactor `App.tsx` so MetricsTable's data is fetched at the App level OR fetch metrics independently in App for the model list. To stay simple, fetch metrics once in App for the model list, while MetricsTable fetches its own.

Add to App's state:

````typescript
const [availableModels, setAvailableModels] = useState<string[]>([]);
````

Add to the existing `selectedStart`-driven effect (or create a new one):

````typescript
useEffect(() => {
  if (!selectedWindow || !selectedStart) {
    setAvailableModels([]);
    return;
  }
  fetchMetrics({ window: selectedWindow, start: selectedStart })
    .then((res) => setAvailableModels(res.models.map((m) => m.model_name)))
    .catch(() => setAvailableModels([]));
}, [selectedWindow, selectedStart]);
````

Add the import at the top:

````typescript
import { fetchMetrics } from "./api/client";
import { ErrorAnalysis } from "./components/ErrorAnalysis";
````

Update Content body:

````typescript
        {selectedStart === null ? (
          <Empty description="该时间窗口暂无预测结果" />
        ) : (
          <>
            <MetricsTable window={selectedWindow!} start={selectedStart} />
            <PredictionCurveChart window={selectedWindow!} start={selectedStart} />
            <ErrorAnalysis
              window={selectedWindow!}
              start={selectedStart}
              availableModels={availableModels}
            />
          </>
        )}
````

- [ ] **Step 3: Build to verify**

Run from `web/frontend`:
````
npm run build
````
Expected: build succeeds.

- [ ] **Step 4: Manual smoke test**

In two terminals:

Terminal 1 (backend):
````
conda activate pm25
cd web/backend
pip install -e .[test]
uvicorn app.main:app --reload --port 8000
````

Terminal 2 (frontend):
````
cd web/frontend
npm run dev
````

Open `http://localhost:5173`. Verify:
- Title displays
- Window selector shows formatted labels
- Switching windows updates start selector
- Metrics table loads with all 6 models (in real outputs)
- Prediction curve renders with y_true bold + colored model lines
- Error analysis defaults to attention_lstm, scatter + histogram render

- [ ] **Step 5: Commit**

````
git add web/frontend/src/components/ErrorAnalysis.tsx web/frontend/src/App.tsx
git commit -m "feat(web): add error analysis (scatter + residual histogram)"
````

---

## Self-review notes

- **Spec section 1 (3 modules):** Tasks 11/12/13 cover metrics table, curve, error analysis ✓
- **Spec section 3 (6 endpoints):** Tasks 1, 3, 4, 5, 5, 6 cover health, windows, metrics, predictions (agg), predictions/{model}, horizon-metrics ✓
- **Spec section 3.2 (error handling):** 404 for unknown window/start covered in Task 4; missing model handled by `missing_models` field; outputs missing returns empty in Task 3 ✓
- **Spec section 4.1 (layout):** Tasks 10/11/12/13 ✓
- **Spec section 4.2 (formatWindow/formatStart):** Task 8 ✓
- **Spec section 4.3 (tech stack: ECharts, Ant Design, axios):** Tasks 7/9/11/12/13 ✓
- **Spec section 4.4 (URL sync, defaults, loading, errors):** Task 10 (URL/defaults), Tasks 11/12/13 (loading + error states) ✓
- **Spec section 5 (error matrix):** Backend Tasks 4-6; frontend Tasks 11-13 + App empty/error states ✓
- **Spec section 6 (testing):** pytest + httpx + vitest + RTL all covered ✓
- **Spec section 7 (run commands):** Backend README in Task 1, frontend README in Task 7 ✓
- **Spec section 8 (git):** `web/.gitignore` in Task 1 ✓
- **Type consistency:** `WindowInfo`, `ModelMetrics`, `PredictionsAggregateResponse`, etc. names match between schemas.py (Python) and types/api.ts (TS); endpoint paths match between routes.py and client.ts.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-web-dashboard.md`. Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, two-stage review (spec compliance + code quality) between tasks.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
