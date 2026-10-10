"""Tests for the drift detection module (small synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift import compute_shift


@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "amount": rng.normal(1000, 100, size=500),
        "count": rng.integers(0, 10, size=500),
    })
    cur_same = pd.DataFrame({
        "amount": rng.normal(1000, 100, size=500),
        "count": rng.integers(0, 10, size=500),
    })
    cur_shifted = pd.DataFrame({
        "amount": rng.normal(2000, 100, size=500),  # mean shifted 10 sigma
        "count": rng.integers(0, 10, size=500),
    })
    return ref, cur_same, cur_shifted


def test_no_drift_detected(small_data) -> None:
    ref, cur_same, _ = small_data
    result = compute_shift(ref, cur_same)
    # Z-score should be small for same distribution
    assert result["drift_ratio"] < 0.5


def test_drift_detected(small_data) -> None:
    ref, _, cur_shifted = small_data
    result = compute_shift(ref, cur_shifted)
    # Amount shifted by 10 sigma → drift on amount feature
    assert result["n_drifted"] >= 1
    assert "amount" in result["per_feature"]


def test_drift_metrics_shape(small_data) -> None:
    ref, cur_same, _ = small_data
    result = compute_shift(ref, cur_same)
    assert "total_features" in result
    assert "per_feature" in result
    assert result["total_features"] == 2
