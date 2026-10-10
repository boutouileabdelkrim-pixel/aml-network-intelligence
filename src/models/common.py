"""Shared utilities for ML training: loading, splitting, metrics.

Central place so all models use the same feature list, target, and split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "full.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "is_laundering"

# Columns to DROP before training (IDs, raw strings, timestamps)
DROP_COLS = [
    "timestamp",
    "from_account",
    "to_account",
    "from_bank",
    "to_bank",
    "payment_format",  # string categorical (encodings already present)
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as ML features."""
    return [c for c in df.columns if c not in DROP_COLS and c != TARGET]


def load_features(columns: list[str] | None = None) -> pd.DataFrame:
    """Load full.parquet. Optionally select a subset of columns (fast)."""
    if columns is not None and TARGET not in columns:
        columns = [*columns, TARGET]
    return pd.read_parquet(FEATURES_PATH, columns=columns, engine="pyarrow")


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute standard binary classification metrics for imbalanced data."""
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_at_0.5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Precision in the top-k highest-scored samples."""
    if k > len(y_prob):
        k = len(y_prob)
    idx = np.argsort(y_prob)[-k:]
    return float(y_true[idx].mean())


def recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Recall in the top-k highest-scored samples."""
    if k > len(y_prob):
        k = len(y_prob)
    idx = np.argsort(y_prob)[-k:]
    return float(y_true[idx].sum() / max(y_true.sum(), 1))


def save_metrics(metrics: dict, name: str) -> Path:
    """Save metrics dict to reports/metrics_<name>.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"metrics_{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
