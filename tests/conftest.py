"""Shared pytest fixtures and skip markers."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_final.json"


requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not available (models/xgb_final.json missing)",
)
