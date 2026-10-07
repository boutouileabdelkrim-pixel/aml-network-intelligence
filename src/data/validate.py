"""Validate the cleaned AML dataset schema and data quality.

Fails loudly if any invariant is broken (missing columns, NaNs in critical
fields, wrong target values, unsorted timestamps, negative amounts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_clean.parquet"

REQUIRED_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

CRITICAL_NOT_NULL = ["timestamp", "from_account", "to_account", "is_laundering"]


def validate() -> None:
    if not CLEAN_PATH.exists():
        logger.error(f"Missing {CLEAN_PATH}. Run clean first.")
        raise SystemExit(1)

    df = pd.read_parquet(CLEAN_PATH, engine="pyarrow")
    errors: list[str] = []

    # 1. Columns present
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # 2. No NaN in critical fields
    for col in CRITICAL_NOT_NULL:
        if col in df.columns and df[col].isna().any():
            errors.append(f"NaN found in critical column: {col}")

    # 3. Target is binary
    if "is_laundering" in df.columns:
        invalid = ~df["is_laundering"].isin([0, 1])
        if invalid.any():
            errors.append(f"Invalid target values: {df.loc[invalid, 'is_laundering'].unique().tolist()}")

    # 4. Timestamps monotonic
    if "timestamp" in df.columns and not df["timestamp"].is_monotonic_increasing:
        errors.append("Timestamps are not sorted ascending")

    # 5. Positive amounts
    for col in ["amount_received", "amount_paid"]:
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Negative values in {col}")

    if errors:
        logger.error("Validation FAILED:")
        for e in errors:
            logger.error(f"  - {e}")
        raise SystemExit(1)

    logger.success(f"Validation PASSED — {len(df):,} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")


def main() -> None:
    validate()


if __name__ == "__main__":
    sys.exit(main())
