"""LightGBM training (GPU when available).

Same setup as XGBoost: scale_pos_weight, early stopping on PR-AUC.
"""

from __future__ import annotations

import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
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
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def train_lgbm() -> dict:
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

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)
    logger.info(f"scale_pos_weight = {spw:.1f}")

    # Detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            PARAMS["device"] = "gpu"
            PARAMS["gpu_platform_id"] = 0
            PARAMS["gpu_device_id"] = 0
            logger.info("GPU detected — using device=gpu")
        else:
            PARAMS["device"] = "cpu"
            logger.info("No GPU — using CPU")
    except Exception:
        PARAMS["device"] = "cpu"
        logger.info("torch missing — using CPU")

    logger.info(f"LightGBM params: {PARAMS}")

    logger.info("Training LightGBM ...")
    t1 = time.time()
    model = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=spw)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    logger.info(f"Training done in {time.time()-t1:.1f}s — best_iter={model.best_iteration_}")

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
    model.booster_.save_model(str(MODELS_DIR / "lgbm_model.txt"))
    joblib.dump(feat_cols, MODELS_DIR / "lgbm_features.pkl")
    save_metrics({"val": m_va, "test": m_te}, "lightgbm")
    logger.success("Saved lgbm_model.txt + metrics")

    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    logger.info("Top 20 features:")
    for i in top_idx:
        logger.info(f"  {feat_cols[i]:40s} {imp[i]:.5f}")

    return {"val": m_va, "test": m_te}


def main() -> None:
    train_lgbm()


if __name__ == "__main__":
    sys.exit(main())
