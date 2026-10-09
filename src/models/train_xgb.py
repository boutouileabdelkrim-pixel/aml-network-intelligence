"""XGBoost training with GPU support and class imbalance handling.

- scale_pos_weight = neg/pos ratio (handles 0.14% positive rate)
- early_stopping on validation PR-AUC
- GPU if available (Kaggle T4)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from loguru import logger

from src.models.common import (
    MODELS_DIR,
    TARGET,
    evaluate,
    get_feature_columns,
    load_features,
    save_metrics,
)
from src.models.split import load_split

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",       # CPU fast; GPU: "gpu_hist" if available
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def train_xgb() -> dict:
    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    logger.info(f"Using {len(feat_cols)} features")

    idx_train, idx_val, idx_test = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")

    X_tr, y_tr = X[idx_train], y[idx_train]
    X_va, y_va = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]

    # Handle imbalance
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)
    logger.info(f"scale_pos_weight = {spw:.1f} (neg={neg}, pos={pos})")

    # Detect GPU
    try:
        import torch

        if torch.cuda.is_available():
            PARAMS["tree_method"] = "hist"
            PARAMS["device"] = "cuda"
            logger.info("GPU detected — using device=cuda")
        else:
            logger.info("No GPU — using CPU")
    except Exception:
        logger.info("No torch — using CPU")

    logger.info(f"XGBoost params: {PARAMS}")

    logger.info("Training XGBoost ...")
    t1 = time.time()
    model = xgb.XGBClassifier(
        **PARAMS,
        scale_pos_weight=spw,
        early_stopping_rounds=30,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    logger.info(f"Training done in {time.time()-t1:.1f}s — best_iter={model.best_iteration}")

    # Predict
    logger.info("Predicting ...")
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

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
    model.save_model(MODELS_DIR / "xgb_model.json")
    joblib.dump(feat_cols, MODELS_DIR / "xgb_features.pkl")
    save_metrics({"val": m_va, "test": m_te}, "xgboost")
    logger.success("Saved xgb_model.json + metrics")

    # Feature importance
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    logger.info("Top 20 features:")
    for i in top_idx:
        logger.info(f"  {feat_cols[i]:40s} {imp[i]:.5f}")

    return {"val": m_va, "test": m_te}


def main() -> None:
    train_xgb()


if __name__ == "__main__":
    sys.exit(main())
