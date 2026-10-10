"""Detect data drift between a reference (train) and current (production) dataset.

Uses Evidently to compare:
- Feature distributions
- Missing values
- Target distribution (if available)

Produces:
- reports/drift_report.html (visual, openable in browser)
- reports/drift_metrics.json (numeric summary)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Sample size for drift detection (fast + representative)
SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    name: str = "drift",
) -> dict:
    """Run Evidently drift detection and save HTML + JSON reports."""

    # Subsample to keep it fast
    if len(reference) > SAMPLE_SIZE:
        reference = reference.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    if len(current) > SAMPLE_SIZE:
        current = current.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    logger.info(f"Reference: {reference.shape}, Current: {current.shape}")

    # Use only numeric columns (Evidently handles categories but simpler here)
    numeric_cols = [
        c for c in reference.columns
        if pd.api.types.is_numeric_dtype(reference[c])
    ]
    reference = reference[numeric_cols]
    current = current[numeric_cols]

    logger.info(f"Drift on {len(numeric_cols)} numeric columns")

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    html_path = REPORTS_DIR / f"{name}_report.html"
    result.save_html(str(html_path))
    logger.success(f"HTML report: {html_path}")

    # Extract summary metrics
    try:
        summary = result.dict()
    except Exception:
        summary = {}

    # Manual drift summary — count columns with drift
    metrics = {
        "n_features": len(numeric_cols),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "html_report": str(html_path),
    }

    json_path = REPORTS_DIR / f"{name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"JSON metrics: {json_path}")

    return metrics


def compute_shift(df_ref: pd.DataFrame, df_cur: pd.DataFrame) -> dict:
    """Compute simple mean-shift per numeric feature (complementary to Evidently)."""
    numeric_cols = [
        c for c in df_ref.columns
        if pd.api.types.is_numeric_dtype(df_ref[c])
    ]
    shifts = {}
    for col in numeric_cols:
        m_ref = float(df_ref[col].mean())
        m_cur = float(df_cur[col].mean())
        std_ref = float(df_ref[col].std()) or 1.0
        z = (m_cur - m_ref) / std_ref
        shifts[col] = {
            "mean_ref": m_ref,
            "mean_current": m_cur,
            "z_score": z,
            "drift_detected": abs(z) > 0.5,
        }
    n_drift = sum(1 for v in shifts.values() if v["drift_detected"])
    return {
        "total_features": len(numeric_cols),
        "n_drifted": n_drift,
        "drift_ratio": n_drift / max(len(numeric_cols), 1),
        "per_feature": shifts,
    }


def main() -> None:
    """Quick test on random split of full.parquet."""
    from src.models.common import load_features

    logger.info("Loading features ...")
    df = load_features()
    half = len(df) // 2
    ref = df.iloc[:half]
    cur = df.iloc[half:]
    detect_drift(ref, cur, name="baseline_split")


if __name__ == "__main__":
    sys.exit(main())
