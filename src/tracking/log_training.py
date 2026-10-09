"""Train XGBoost and log everything to MLflow.

This is the "reproducible run" script that would be used in production:
- Trains with the best params
- Logs hyperparameters, metrics, model
- Registers the model in the Model Registry

Run on Kaggle (needs GPU + full.parquet).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.common import (
    MODELS_DIR,
    TARGET,
    get_feature_columns,
    load_features,
)
from src.models.split import load_split
from src.tracking.mlflow_setup import EXPERIMENT_NAME, MLRUNS_DIR, setup

MODEL_NAME = "aml-xgb-final"


def _metrics(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run() -> dict:
    # 1. Setup MLflow
    setup()
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load best params
    with open(MODELS_DIR / "best_xgb_params.json") as f:
        best = json.load(f)
    params = best["best_params"]
    logger.info(f"Best params loaded: {params}")

    # 3. Load features
    logger.info("Loading features ...")
    df = load_features()
    feat_cols = get_feature_columns(df)
    idx_train, idx_val, idx_test = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_va, y_va = X[idx_val], y[idx_val]
    X_te, y_te = X[idx_test], y[idx_test]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)

    # 4. Start MLflow run
    with mlflow.start_run(run_name="xgb-tuned-optuna") as run_ctx:
        logger.info(f"MLflow run_id = {run_ctx.info.run_id}")

        # Log params
        full_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_estimators": 1500,
            "scale_pos_weight": spw,
            "random_state": 42,
            **params,
        }
        mlflow.log_params(full_params)
        mlflow.log_param("n_train_rows", len(X_tr))
        mlflow.log_param("n_features", len(feat_cols))

        # Train
        logger.info("Training XGBoost ...")
        t0 = time.time()
        model = xgb.XGBClassifier(**full_params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("best_iteration", model.best_iteration)
        logger.info(f"Trained in {train_time:.1f}s — best_iter={model.best_iteration}")

        # Evaluate
        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]

        m_val = _metrics(y_va, p_va)
        m_test = _metrics(y_te, p_te)

        for k, v in m_val.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in m_test.items():
            mlflow.log_metric(f"test_{k}", v)

        logger.info(f"VAL  — PR-AUC: {m_val['pr_auc']:.5f}  F1: {m_val['f1']:.5f}")
        logger.info(f"TEST — PR-AUC: {m_test['pr_auc']:.5f}  F1: {m_test['f1']:.5f}")

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Log feature list
        mlflow.log_dict({"features": feat_cols}, "features.json")

        logger.success(f"Logged run {run_ctx.info.run_id}")

    return {"run_id": run_ctx.info.run_id, "val": m_val, "test": m_test}


def main() -> None:
    run()


if __name__ == "__main__":
    sys.exit(main())
