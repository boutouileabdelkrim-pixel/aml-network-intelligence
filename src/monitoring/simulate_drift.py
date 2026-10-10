"""Simulate 3 drift scenarios and detect them with Evidently.

Scenarios:
1. Baseline  — same distribution as reference (expect: no drift)
2. Amount shift — amount_paid x1.5, amount_received x1.5 (expect: drift)
3. Velocity spike — cnt_1h x3, velocity_tx_per_day x5 (expect: drift)

Each scenario produces a report saved to reports/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from src.models.common import load_features
from src.monitoring.drift import detect_drift

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def baseline(ref: pd.DataFrame) -> pd.DataFrame:
    """Same distribution — no drift expected."""
    return ref.sample(10_000, random_state=1).copy()


def amount_shift(ref: pd.DataFrame) -> pd.DataFrame:
    """Amounts x1.5 — moderate drift on amount features."""
    cur = ref.sample(10_000, random_state=2).copy()
    for col in ["amount_paid", "amount_received"]:
        if col in cur.columns:
            cur[col] = cur[col] * 1.5
    return cur


def velocity_spike(ref: pd.DataFrame) -> pd.DataFrame:
    """Velocity x3-5 — strong drift on temporal features."""
    cur = ref.sample(10_000, random_state=3).copy()
    for col in ["acc_from_cnt_1h", "acc_from_cnt_1d", "acc_from_velocity_tx_per_day"]:
        if col in cur.columns:
            cur[col] = cur[col] * 3.0
    return cur


def main() -> None:
    logger.info("Loading features (10k sample) ...")
    df = load_features()
    ref = df.sample(20_000, random_state=42).copy()

    scenarios = {
        "scenario_1_baseline": baseline(ref),
        "scenario_2_amount_shift": amount_shift(ref),
        "scenario_3_velocity_spike": velocity_spike(ref),
    }

    summary = {}
    for name, current in scenarios.items():
        logger.info(f"\n=== {name} ===")
        metrics = detect_drift(ref, current, name=name)
        summary[name] = metrics
        logger.info(f"  → {metrics}")

    with open(REPORTS_DIR / "drift_scenarios_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.success("All scenarios done. See reports/drift_scenarios_summary.json")


if __name__ == "__main__":
    sys.exit(main())
