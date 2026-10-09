"""Train final XGBoost with Optuna-tuned hyperparameters.

Uses models/best_xgb_params.json (from Phase 6 Optuna tuning).
Produces the model that will be used for SHAP (Phase 7) and API (Phase 9).
"""

from __future__ import annotations

import json
import sys
import time

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


def load_best_params() -> dict:
    path = MODELS_DIR / "best_xgb_params.json"
    if not path.exists():
        logger.error(f"Missing {path}. Run src.models.tune_optuna first.")
        raise SystemExit(1)
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded best params from {path}")
    logger.info(f"  Best val PR-AUC: {data['best_value_pr_auc']:.5f}")
    return data["best_params"]


def train_final() -> dict:
    params = load_best_params()

    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    idx_train, idx_val, idx_test = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")

    X_tr, y_tr = X[idx_train], y[idx_train]
    X_va, y_va = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)
    logger.info(f"scale_pos_weight = {spw:.1f}")

    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": 1500,
        "scale_pos_weight": spw,
        **params,
    }

    try:
        import torch

        if torch.cuda.is_available():
            full_params["device"] = "cuda"
            logger.info("GPU detected")
    except Exception:
        pass

    logger.info("Training final XGBoost ...")
    t1 = time.time()
    model = xgb.XGBClassifier(**full_params, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    logger.info(f"Done in {time.time()-t1:.1f}s — best_iter={model.best_iteration}")

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
    model.save_model(MODELS_DIR / "xgb_final.json")
    joblib.dump(feat_cols, MODELS_DIR / "xgb_final_features.pkl")
    save_metrics({"val": m_va, "test": m_te}, "xgboost_final")
    logger.success("Saved xgb_final.json + metrics")

    return {"val": m_va, "test": m_te}


def main() -> None:
    train_final()


if __name__ == "__main__":
    sys.exit(main())
