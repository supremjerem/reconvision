"""Shared fixtures and the model-availability gate.

Tests marked `models` need roughly 360 MB of weights that are not in the
repository. Rather than a flag someone has to remember, they skip themselves
based on whether the weights are actually present, so the same command does the
right thing on a developer machine and in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
DETECTOR_PATH = MODELS_DIR / "yolo11s.onnx"
FACE_PACK_DIR = MODELS_DIR / "models" / "buffalo_l"


def models_available() -> bool:
    return DETECTOR_PATH.exists() and FACE_PACK_DIR.exists()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if models_available():
        return
    skip = pytest.mark.skip(reason="model weights absent; run `reconvision export-models`")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def detector_path() -> Path:
    return DETECTOR_PATH


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS_DIR


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    """Where fetched sample photographs live. Git-ignored, like the weights."""
    return MODELS_DIR.parent / "samples"
