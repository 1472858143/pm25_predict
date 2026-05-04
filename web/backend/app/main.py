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
