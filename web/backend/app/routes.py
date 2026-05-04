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
