"""Import an existing training run into MLflow.

Use case: the model was trained on Kaggle (external environment) and
we want to register its params + metrics + model artifact into the local
MLflow tracking server for reproducibility and visualization.

This is the standard MLOps pattern: train remote, register locally.
"""

from __future__ import annotations

import json
import sys

import mlflow
import mlflow.xgboost
import xgboost as xgb
from loguru import logger

from src.models.common import MODELS_DIR, REPORTS_DIR
from src.tracking.mlflow_setup import EXPERIMENT_NAME, setup

METRICS_XGB = REPORTS_DIR / "metrics_xgboost_final.json"
MODEL_PATH = MODELS_DIR / "xgb_final.json"
PARAMS_PATH = MODELS_DIR / "best_xgb_params.json"


def import_run() -> str:
    setup()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with open(METRICS_XGB) as f:
        metrics = json.load(f)
    with open(PARAMS_PATH) as f:
        params_data = json.load(f)

    run_name = "xgb-tuned-optuna-kaggle"
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run_id: {run.info.run_id}")

        # Params
        full_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_estimators": 1500,
            "scale_pos_weight": 1243.7,
            "random_state": 42,
            "device": "cuda",
            **params_data["best_params"],
        }
        mlflow.log_params(full_params)
        mlflow.log_param("n_train_rows", 3_554_841)
        mlflow.log_param("n_features", 252)
        mlflow.log_param("source", "kaggle")

        # Metrics
        for k, v in metrics["val"].items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in metrics["test"].items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("best_iteration", metrics.get("best_iter", 1475))
        mlflow.log_metric("train_time_seconds", metrics.get("train_time_s", 279.2))

        # Model (load from file — real model)
        logger.info(f"Loading XGBoost model from {MODEL_PATH}")
        booster = xgb.XGBClassifier()
        booster.load_model(MODEL_PATH)
        mlflow.xgboost.log_model(
            booster,
            name="model",
            registered_model_name="aml-xgb-final",
        )

        logger.success(f"Imported run {run.info.run_id}")
        print(f"RUN_ID={run.info.run_id}")

    return run.info.run_id


def main() -> None:
    import_run()


if __name__ == "__main__":
    sys.exit(main())
