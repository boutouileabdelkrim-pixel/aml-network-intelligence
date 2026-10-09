"""Unsupervised anomaly detection with Isolation Forest.

Provides a complementary signal to XGBoost:
- XGBoost learns from labels (supervised)
- IsolationForest detects outliers regardless of labels (unsupervised)
- Their disagreement often flags ambiguous cases worth human review
"""

from __future__ import annotations

import sys
import time

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.models.common import (
    MODELS_DIR,
    TARGET,
    evaluate,
    get_feature_columns,
    load_features,
    save_metrics,
)
from src.models.split import load_split

CONTAMINATION = 0.005  # expected fraction of anomalies (0.5% ≈ 5x fraud rate)
N_ESTIMATORS = 100
RANDOM_STATE = 42
SAMPLE_SIZE = "auto"


def train_anomaly() -> dict:
    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    idx_train, idx_val, idx_test = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")

    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test], y[idx_test]

    logger.info(f"Training IsolationForest (contamination={CONTAMINATION}) ...")
    t1 = time.time()
    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(X_tr)
    logger.info(f"Training done in {time.time()-t1:.1f}s")

    logger.info("Scoring test set ...")
    # score_samples: lower = more anomalous. Convert to a "fraud probability"-like score.
    raw_scores = clf.score_samples(X_te)
    # Normalize to [0, 1] where 1 = most anomalous
    min_s, max_s = raw_scores.min(), raw_scores.max()
    p_te = (max_s - raw_scores) / (max_s - min_s + 1e-9)

    m_te = evaluate(y_te, p_te)
    logger.info("=== TEST (IsolationForest) ===")
    for k, v in m_te.items():
        logger.info(f"  {k:20s}: {v}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "iforest_model.pkl")
    joblib.dump(feat_cols, MODELS_DIR / "iforest_features.pkl")
    save_metrics({"test": m_te}, "isolation_forest")
    logger.success("Saved iforest_model.pkl + metrics")

    return {"test": m_te}


def main() -> None:
    train_anomaly()


if __name__ == "__main__":
    sys.exit(main())
