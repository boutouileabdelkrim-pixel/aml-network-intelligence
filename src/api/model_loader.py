"""Load the XGBoost model + feature list once (singleton pattern)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "xgb_final.json"
FEATURES_PATH = MODELS_DIR / "xgb_final_features.pkl"
METRICS_PATH = REPORTS_DIR / "metrics_xgboost_final.json"

MODEL_NAME = "aml-xgb-final"
MODEL_VERSION = "v1"
DECISION_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7
CRITICAL_RISK_THRESHOLD = 0.9


@lru_cache(maxsize=1)
def load_model() -> xgb.XGBClassifier:
    """Load XGBoost model once and cache it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    return model


@lru_cache(maxsize=1)
def load_feature_names() -> list[str]:
    """Load the list of feature names used during training."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features list not found: {FEATURES_PATH}")
    features = joblib.load(FEATURES_PATH)
    logger.info(f"Loaded {len(features)} feature names")
    return features


@lru_cache(maxsize=1)
def load_training_metrics() -> dict:
    """Load the metrics from the last training run."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def risk_level(prob: float) -> str:
    """Map probability to a risk tier."""
    if prob >= CRITICAL_RISK_THRESHOLD:
        return "CRITICAL"
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    if prob >= DECISION_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def decision(prob: float) -> str:
    """Map probability to an operational decision."""
    if prob >= HIGH_RISK_THRESHOLD:
        return "FLAG"
    if prob >= DECISION_THRESHOLD:
        return "REVIEW"
    return "ALLOW"


def features_to_vector(features: dict) -> np.ndarray:
    """Convert a dict of feature name -> value into an ordered numpy row.

    Missing features default to 0.0 — XGBoost handles sparse inputs.
    """
    feat_names = load_feature_names()
    row = [float(features.get(name, 0.0)) for name in feat_names]
    return np.array([row], dtype="float32")


def top_contributing_features(features: dict, k: int = 5) -> list[str]:
    """Return the top-k features (by abs value) present in the input.

    This is a lightweight heuristic for API response — a real production
    system would compute per-prediction SHAP values.
    """
    feat_names = load_feature_names()
    present = [(name, abs(float(features.get(name, 0.0)))) for name in feat_names]
    present.sort(key=lambda x: x[1], reverse=True)
    return [name for name, val in present[:k] if val > 0][:k]
