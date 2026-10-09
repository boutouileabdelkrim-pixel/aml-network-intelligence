"""SHAP explainability for the final XGBoost model.

Produces:
- Global feature importance (bar + summary beeswarm)
- Local explanations for top 5 highest-scored transactions (waterfall)
- Dependence plots for top 3 features
- JSON with top-30 feature importances

Sampling: uses 10,000 test rows for global (fast, representative) and 5
top-scored rows for local (interpretable).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger

from src.models.common import (
    MODELS_DIR,
    REPORTS_DIR,
    TARGET,
    get_feature_columns,
    load_features,
)
from src.models.split import load_split

# Sampling
N_GLOBAL = 10_000
N_LOCAL = 5
TOP_K_FEATURES = 30


def _load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / "xgb_final.json")
    feat_cols = joblib.load(MODELS_DIR / "xgb_final_features.pkl")
    return model, feat_cols


def run_shap() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features ...")
    t0 = time.time()
    df = load_features()
    logger.info(f"Loaded in {time.time()-t0:.1f}s — shape: {df.shape}")

    model, feat_cols = _load_model()
    logger.info(f"Model loaded — {len(feat_cols)} features")

    idx_train, idx_val, idx_test = load_split()
    X = df[feat_cols].values.astype("float32")
    y = df[TARGET].values.astype("int8")

    X_te, y_te = X[idx_test], y[idx_test]

    # Global sample
    logger.info(f"Sampling {N_GLOBAL:,} test rows for global SHAP ...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_te), size=N_GLOBAL, replace=False)
    X_sample = X_te[sample_idx]
    y_sample = y_te[sample_idx]
    logger.info(f"  positive rate in sample: {y_sample.mean()*100:.4f}%")

    # Build explainer
    logger.info("Building TreeExplainer ...")
    t1 = time.time()
    explainer = shap.TreeExplainer(model)
    logger.info(f"Explainer built in {time.time()-t1:.1f}s")

    # Global SHAP values
    logger.info(f"Computing SHAP values on {N_GLOBAL:,} rows ...")
    t2 = time.time()
    shap_values = explainer.shap_values(X_sample)
    logger.info(f"SHAP computed in {time.time()-t2:.1f}s")

    # ---- 1. Global bar plot ----
    logger.info("Plotting global bar importance ...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feat_cols,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importance_bar.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ---- 2. Global summary beeswarm ----
    logger.info("Plotting summary beeswarm ...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feat_cols,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ---- 3. Feature importance values ----
    logger.info("Computing mean |SHAP| ...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    top_features = [
        {"feature": feat_cols[i], "mean_abs_shap": float(mean_abs_shap[i])}
        for i in order[:TOP_K_FEATURES]
    ]

    # ---- 4. Local explanations on top-scored test rows ----
    logger.info(f"Computing local explanations for top {N_LOCAL} scored rows ...")
    p_te = model.predict_proba(X_te)[:, 1]
    top_idx = np.argsort(p_te)[-N_LOCAL:][::-1]
    X_top = X_te[top_idx]
    y_top = y_te[top_idx]

    shap_top = explainer.shap_values(X_top)

    for i, (idx, y_true) in enumerate(zip(top_idx, y_top)):
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_top[i],
                base_values=explainer.expected_value,
                data=X_top[i],
                feature_names=feat_cols,
            ),
            max_display=15,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            REPORTS_DIR / f"shap_waterfall_top{i+1}.png", dpi=120, bbox_inches="tight"
        )
        plt.close()

    # ---- 5. Dependence plots for top 3 features ----
    logger.info("Plotting dependence plots for top 3 features ...")
    for rank in range(3):
        feat_name = feat_cols[order[rank]]
        plt.figure()
        shap.dependence_plot(
            feat_name,
            shap_values,
            X_sample,
            feature_names=feat_cols,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            REPORTS_DIR / f"shap_dependence_{rank+1}_{feat_name}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()

    # ---- 6. Save metrics ----
    out = {
        "n_global_sample": N_GLOBAL,
        "sample_positive_rate": float(y_sample.mean()),
        "top_30_features": top_features,
        "top_5_scored_transactions": [
            {
                "rank": i + 1,
                "predicted_prob": float(p_te[idx]),
                "true_label": int(y_top[i]),
            }
            for i, idx in enumerate(top_idx)
        ],
    }
    with open(REPORTS_DIR / "shap_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.success(f"Saved shap_metrics.json + plots to {REPORTS_DIR}")

    logger.info("=== TOP 20 FEATURES (mean |SHAP|) ===")
    for i, feat in enumerate(top_features[:20], 1):
        logger.info(f"  #{i:2d} — {feat['feature']:40s} {feat['mean_abs_shap']:.6f}")

    return out


def main() -> None:
    run_shap()


if __name__ == "__main__":
    sys.exit(main())
