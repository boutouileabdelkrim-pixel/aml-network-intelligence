"""Orchestrator: build the full tabular feature table and save to parquet.

MEMORY-OPTIMIZED: converts to float32 early, drops raw string columns
before the heavy stages, and calls gc.collect() between stages.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.tabular_basic import build_basic_features
from src.features.tabular_temporal import build_temporal_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_clean.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "tabular.parquet"


def _reduce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to memory-efficient dtypes."""
    for col in df.columns:
        dt = df[col].dtype
        if dt == "float64":
            df[col] = df[col].astype("float32")
        elif dt == "int64":
            df[col] = df[col].astype("int32")
    return df


def build_features() -> Path:
    if not CLEAN_PATH.exists():
        logger.error(f"Missing {CLEAN_PATH}. Run clean first (Phase 3).")
        raise SystemExit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {CLEAN_PATH} ...")
    df = pd.read_parquet(CLEAN_PATH, engine="pyarrow")
    logger.info(f"Loaded shape: {df.shape} — mem: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Early dtype reduction
    df = _reduce_dtypes(df)
    logger.info(f"After reduce — mem: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Drop raw strings we don't need (keep account IDs for joins)

    # ---- Basic features ----
    logger.info("Building basic features ...")
    t0 = time.time()
    df = build_basic_features(df)
    gc.collect()
    logger.info(
        f"Basic features done in {time.time() - t0:.1f}s — shape: {df.shape} — "
        f"mem: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB"
    )

    # ---- Temporal features ----
    logger.info("Building temporal features ...")
    t1 = time.time()
    df = build_temporal_features(df)
    gc.collect()
    logger.info(
        f"Temporal features done in {time.time() - t1:.1f}s — shape: {df.shape} — "
        f"mem: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB"
    )


    # Drop raw strings AFTER features are built
    df = df.drop(columns=["receiving_currency", "payment_currency"], errors="ignore")
    gc.collect()


    # Final cast + NaN fill
    df["is_laundering"] = df["is_laundering"].astype("int8")
    df = df.fillna(0.0)
    gc.collect()

    logger.info(f"Final shape: {df.shape} — mem: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    logger.info(f"Columns ({len(df.columns)}): {list(df.columns)}")

    logger.info(f"Writing {OUT_PATH} ...")
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.success(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")
    return OUT_PATH


def main() -> None:
    build_features()


if __name__ == "__main__":
    sys.exit(main())
