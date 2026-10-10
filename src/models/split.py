"""Create a temporal train/val/test split.

Rationale: for AML, we must avoid *temporal leakage* — training on future
data and testing on past data is unrealistic. So we sort by timestamp and
take the first 70% as train, next 15% as val, last 15% as test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.common import TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = PROJECT_ROOT / "data" / "features" / "split_indices.npz"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remaining 15%


def make_split() -> Path:
    logger.info("Loading full.parquet (timestamp + target only) ...")
    df = pd.read_parquet(
        PROJECT_ROOT / "data" / "features" / "full.parquet",
        columns=["timestamp", TARGET],
        engine="pyarrow",
    )
    logger.info(f"Loaded {len(df):,} rows")

    # Sort by timestamp
    logger.info("Sorting by timestamp ...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)

    target = df[TARGET].values

    logger.info("Split summary:")
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        sub = target[idx]
        logger.info(
            f"  {name:5s} — {len(idx):>9,} rows — "
            f"pos={sub.sum():>6,} ({sub.mean()*100:.4f}%)"
        )

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SPLIT_PATH,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        timestamp_min=str(df["timestamp"].iloc[0]),
        timestamp_max=str(df["timestamp"].iloc[-1]),
    )
    logger.success(f"Wrote {SPLIT_PATH}")

    # Also save a human-readable summary
    summary_path = SPLIT_PATH.with_suffix(".json")
    import json

    summary = {
        "n_total": int(n),
        "train": {
            "n": len(idx_train),
            "n_pos": int(target[idx_train].sum()),
            "rate": float(target[idx_train].mean()),
        },
        "val": {
            "n": len(idx_val),
            "n_pos": int(target[idx_val].sum()),
            "rate": float(target[idx_val].mean()),
        },
        "test": {
            "n": len(idx_test),
            "n_pos": int(target[idx_test].sum()),
            "rate": float(target[idx_test].mean()),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {summary_path}")

    return SPLIT_PATH


def load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the saved split indices."""
    data = np.load(SPLIT_PATH)
    return data["idx_train"], data["idx_val"], data["idx_test"]


def main() -> None:
    make_split()


if __name__ == "__main__":
    sys.exit(main())

