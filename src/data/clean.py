"""Clean the raw IBM AML dataset into a typed, analysis-ready parquet file.

Handles:
- Duplicated "Account" column (raw CSV has it twice)
- Bank IDs as strings (preserve leading zeros: "010" != 10)
- Timestamp parsing
- Drop rows with missing critical fields
- Sort by timestamp
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_clean.parquet"

# The raw CSV has a duplicated "Account" column name.
# pandas auto-renames the second one to "Account.1".
RENAME_MAP = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

CRITICAL_COLUMNS = ["timestamp", "from_account", "to_account", "amount_paid"]


def load_raw() -> pd.DataFrame:
    logger.info(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(
        RAW_PATH,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    logger.info(f"Loaded shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(RENAME_MAP) - set(df.columns)
    if missing:
        logger.error(f"Missing expected columns in raw file: {missing}")
        raise SystemExit(1)
    return df.rename(columns=RENAME_MAP)


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["amount_received", "amount_paid"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").astype("Int8")
    return df


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLUMNS)
    df = df[df["is_laundering"].notna()]
    df["is_laundering"] = df["is_laundering"].astype("int8")
    after = len(df)
    logger.info(f"Dropped {before - after} rows with critical NaN ({after} remaining)")
    return df


def clean() -> Path:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = rename_columns(df)
    df = cast_types(df)
    df = drop_bad_rows(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Writing {OUT_PATH} ...")
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    logger.success(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    return OUT_PATH


def main() -> None:
    clean()


if __name__ == "__main__":
    sys.exit(main())
