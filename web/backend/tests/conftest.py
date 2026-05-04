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
    from importlib import reload

    from app import main as main_module

    reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client
