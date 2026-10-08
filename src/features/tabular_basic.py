"""Basic tabular features per account and per transaction.

MEMORY-OPTIMIZED: uses .map() instead of .merge() to avoid creating
5M-row intermediate DataFrames. Uses float32 / int32 throughout.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from loguru import logger

QUANTILES = [0.25, 0.5, 0.75, 0.95, 0.99]
ROUND_AMOUNTS = [100.0, 1000.0, 10000.0]
STRUCTURING_THRESHOLDS = [5000.0, 9500.0, 10000.0, 50000.0]


def account_aggregates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Compute per-account aggregate stats. Result indexed by account."""
    account_col = f"{side}_account"
    amount_col = "amount_paid" if side == "from" else "amount_received"
    prefix = f"acc_{side}_"

    grp = df.groupby(account_col)[amount_col]

    stats = pd.DataFrame(
        {
            f"{prefix}n_tx": grp.count().astype("int32"),
            f"{prefix}sum": grp.sum().astype("float32"),
            f"{prefix}mean": grp.mean().astype("float32"),
            f"{prefix}std": grp.std().fillna(0.0).astype("float32"),
            f"{prefix}min": grp.min().astype("float32"),
            f"{prefix}max": grp.max().astype("float32"),
        }
    )

    q = grp.quantile(QUANTILES).unstack()
    q.columns = [f"{prefix}q{int(c * 100)}" for c in q.columns]
    for c in q.columns:
        q[c] = q[c].astype("float32")
    stats = stats.join(q)

    stats[f"{prefix}range"] = (stats[f"{prefix}max"] - stats[f"{prefix}min"]).astype("float32")
    stats[f"{prefix}cv"] = np.where(
        stats[f"{prefix}mean"] > 0,
        stats[f"{prefix}std"] / stats[f"{prefix}mean"],
        0.0,
    ).astype("float32")

    return stats


def _attach_via_map(df: pd.DataFrame, stats: pd.DataFrame, side: str) -> None:
    """Attach account stats to df using .map() — memory efficient."""
    account_col = f"{side}_account"
    amount_col = "amount_paid" if side == "from" else "amount_received"
    prefix = f"acc_{side}_"

    # Map each stat column (each map creates a 5M Series but is freed immediately)
    for col in stats.columns:
        df[col] = df[account_col].map(stats[col]).astype("float32")

    # Derived ratios
    mean_col = f"{prefix}mean"
    std_col = f"{prefix}std"
    q95_col = f"{prefix}q95"

    df[f"tx_{side}_amount_ratio"] = np.where(
        df[mean_col] > 0, df[amount_col] / df[mean_col], 0.0
    ).astype("float32")
    df[f"tx_{side}_amount_ratio_q95"] = np.where(
        df[q95_col] > 0, df[amount_col] / df[q95_col], 0.0
    ).astype("float32")
    df[f"tx_{side}_amount_zscore"] = np.where(
        df[std_col] > 0, (df[amount_col] - df[mean_col]) / df[std_col], 0.0
    ).astype("float32")


def categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-currency flag + format/currency frequency encodings."""
    df["is_cross_currency"] = (
        df["receiving_currency"].astype(str) != df["payment_currency"].astype(str)
    ).astype("int8")

    fmt_counts = df["payment_format"].value_counts(normalize=True)
    df["payment_format_freq"] = df["payment_format"].map(fmt_counts).astype("float32")

    cur_counts = df["receiving_currency"].value_counts(normalize=True)
    df["receiving_currency_freq"] = df["receiving_currency"].map(cur_counts).astype("float32")

    for fmt in ["ACH", "Wire", "Cheque", "Credit Card", "Reinvestment"]:
        df[f"fmt_{fmt.lower().replace(' ', '_')}"] = (
            df["payment_format"].astype(str) == fmt
        ).astype("int8")

    return df


def structuring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flags for structuring patterns."""
    amt = df["amount_paid"].astype("float32")

    for r in ROUND_AMOUNTS:
        df[f"is_round_{int(r)}"] = ((amt % r) == 0).astype("int8")

    for thr in STRUCTURING_THRESHOLDS:
        df[f"near_thr_{int(thr)}"] = ((amt >= thr * 0.95) & (amt < thr)).astype("int8")

    df["amount_bucket"] = pd.cut(
        amt, bins=[-1, 100, 1_000, 10_000, 100_000, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype("int8")

    return df


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full basic feature pipeline with memory cleanup between steps."""
    logger.info("  [basic] from-account aggregates ...")
    from_stats = account_aggregates(df, side="from")
    _attach_via_map(df, from_stats, side="from")
    del from_stats
    gc.collect()
    logger.info(f"  [basic] from-side done — shape: {df.shape}")

    logger.info("  [basic] to-account aggregates ...")
    to_stats = account_aggregates(df, side="to")
    _attach_via_map(df, to_stats, side="to")
    del to_stats
    gc.collect()
    logger.info(f"  [basic] to-side done — shape: {df.shape}")

    df["tx_net_amount"] = (df["amount_paid"] - df["amount_received"]).astype("float32")
    df["tx_amount_diff"] = (df["amount_paid"] - df["amount_received"]).abs().astype("float32")

    df = categorical_features(df)
    gc.collect()
    df = structuring_features(df)
    gc.collect()

    return df
