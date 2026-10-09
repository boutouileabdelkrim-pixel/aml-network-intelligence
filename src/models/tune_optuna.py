"""Optuna hyperparameter tuning for XGBoost.

Runs N_TRIALS studies, each training on train and scoring PR-AUC on val.
Saves the best params to models/best_xgb_params.json.
"""

from __future__ import annotations

import json
import sys
import time

import optuna
import xgboost as xgb
from loguru import logger
from sklearn.metrics import average_precision_score

from src.models.common import MODELS_DIR, TARGET, get_feature_columns, load_features
from src.models.split import load_split

N_TRIALS = 12
SEED = 42
EARLY_STOP = 30


def build_objective(X_tr, y_tr, X_va, y_va, spw):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": -1,
            "scale_pos_weight": spw,
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": 1000,
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        try:
            import torch

            if torch.cuda.is_available():
                params["device"] = "cuda"
        except Exception:
            pass

        model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOP)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_va = model.predict_proba(X_va)[:, 1]
        pr_auc = average_precision_score(y_va, p_va)
        return pr_auc

    return objective


def tune() -> dict:
    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    idx_train, idx_val, _ = load_split()

    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_va, y_va = X[idx_val], y[idx_val]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / max(pos, 1)
    logger.info(f"scale_pos_weight = {spw:.1f}")

    logger.info(f"Starting Optuna study — {N_TRIALS} trials ...")
    t1 = time.time()

    # Afficher chaque trial dans les logs
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    try:
        study.optimize(
            build_objective(X_tr, y_tr, X_va, y_va, spw),
            n_trials=N_TRIALS,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving partial results ...")

    logger.info(f"Study done in {time.time()-t1:.1f}s")
    logger.info(f"Best PR-AUC (val): {study.best_value:.5f}")
    logger.info(f"Best params: {study.best_params}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "best_value_pr_auc": study.best_value,
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
    }
    with open(MODELS_DIR / "best_xgb_params.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.success(f"Saved {MODELS_DIR / 'best_xgb_params.json'}")

    # Top 5 trials
    trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -1,
        reverse=True,
    )[:5]
    logger.info("Top 5 trials:")
    for i, t in enumerate(trials, 1):
        logger.info(f"  #{i} — PR-AUC={t.value:.5f} — params={t.params}")

    return out


def main() -> None:
    tune()


if __name__ == "__main__":
    sys.exit(main())
