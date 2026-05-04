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
