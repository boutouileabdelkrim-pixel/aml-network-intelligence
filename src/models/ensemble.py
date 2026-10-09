"""Ensemble: weighted average of XGBoost and IsolationForest scores.

Simple but effective: XGBoost captures labeled fraud, IF catches structural
anomalies. Weight 0.8/0.2 empirically favors the supervised model.
"""

from __future__ import annotations

import sys
import time

import joblib
import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import IsolationForest

from src.models.common import (
    MODELS_DIR,
    TARGET,
    evaluate,
    get_feature_columns,
    load_features,
    save_metrics,
)
from src.models.split import load_split

XGB_WEIGHT = 0.8
IF_WEIGHT = 0.2


def run_ensemble() -> dict:
    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    idx_train, idx_val, idx_test = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")

    X_te, y_te = X[idx_test], y[idx_test]

    logger.info("Loading trained models ...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODELS_DIR / "xgb_final.json")
    iforest = joblib.load(MODELS_DIR / "iforest_model.pkl")

    logger.info("Scoring with XGBoost ...")
    p_xgb = xgb_model.predict_proba(X_te)[:, 1]

    logger.info("Scoring with IsolationForest ...")
    raw = iforest.score_samples(X_te)
    p_if = (raw.max() - raw) / (raw.max() - raw.min() + 1e-9)

    logger.info(f"Combining with weights XGB={XGB_WEIGHT}, IF={IF_WEIGHT} ...")
    p_ens = XGB_WEIGHT * p_xgb + IF_WEIGHT * p_if

    # Evaluate all three
    m_xgb = evaluate(y_te, p_xgb)
    m_if = evaluate(y_te, p_if)
    m_ens = evaluate(y_te, p_ens)

    logger.info("=== XGBOOST ===")
    logger.info(f"  PR-AUC: {m_xgb['pr_auc']:.5f}  F1@0.5: {m_xgb['f1_at_0.5']:.5f}")
    logger.info("=== ISOLATION FOREST ===")
    logger.info(f"  PR-AUC: {m_if['pr_auc']:.5f}  F1@0.5: {m_if['f1_at_0.5']:.5f}")
    logger.info("=== ENSEMBLE ===")
    logger.info(f"  PR-AUC: {m_ens['pr_auc']:.5f}  F1@0.5: {m_ens['f1_at_0.5']:.5f}")

    save_metrics(
        {"xgb": m_xgb, "iforest": m_if, "ensemble": m_ens},
        "ensemble",
    )
    logger.success("Saved ensemble metrics")

    return {"xgb": m_xgb, "iforest": m_if, "ensemble": m_ens}


def main() -> None:
    run_ensemble()


if __name__ == "__main__":
    sys.exit(main())

