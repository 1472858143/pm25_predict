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
