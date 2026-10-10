"""Baseline: Logistic Regression on standardized features.

Purpose: get a floor score to compare XGBoost/LightGBM against.
Runs fast (a few minutes) even on 5M x 259 features.
"""

from __future__ import annotations

import sys
import time

import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.common import (
    MODELS_DIR,
    TARGET,
    evaluate,
    get_feature_columns,
    load_features,
    save_metrics,
)
from src.models.split import load_split


def train_baseline() -> dict:
    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    logger.info(f"Using {len(feat_cols)} features")

    idx_train, idx_val, idx_test = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")

    # Split
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_va, y_va = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]
    logger.info(f"Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")

    # Standardize
    logger.info("Standardizing ...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # Train (with class_weight='balanced' for imbalance)
    logger.info("Training Logistic Regression ...")
    t1 = time.time()
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(X_tr, y_tr)
    logger.info(f"Training done in {time.time()-t1:.1f}s")

    # Predict
    logger.info("Predicting on val + test ...")
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    # Evaluate
    m_va = evaluate(y_va, p_va)
    m_te = evaluate(y_te, p_te)
    logger.info("=== VALIDATION ===")
    for k, v in m_va.items():
        logger.info(f"  {k:20s}: {v}")
    logger.info("=== TEST ===")
    for k, v in m_te.items():
        logger.info(f"  {k:20s}: {v}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "baseline_logreg.pkl")
    joblib.dump(scaler, MODELS_DIR / "baseline_scaler.pkl")
    save_metrics({"val": m_va, "test": m_te}, "baseline_logreg")
    logger.success("Saved baseline_logreg.pkl + metrics")

    return {"val": m_va, "test": m_te}


def main() -> None:
    train_baseline()


if __name__ == "__main__":
    sys.exit(main())
