"""Temporal features per transaction (vectorized, fast).

Computes:
- Recency: time since account's first/last transaction
- Time since previous transaction (per account)
- Rolling counts and sums in windows: 1h, 1D, 7D, 30D
- Velocity: transactions and amount per day

NOTE: rolling windows are computed with numpy.searchsorted for O(n log n)
speed instead of pandas groupby().rolling() which is O(n^2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

WINDOWS = ["1h", "1D", "7D", "30D"]


def _windows_tag(window: str) -> str:
    """Convert pandas freq string to a safe column tag."""
    return window.replace(" ", "").lower()


def recency_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Recency features per {side}_account."""
    account_col = f"{side}_account"
    prefix = f"acc_{side}_"

    grp = df.groupby(account_col)["timestamp"]
    first_ts = grp.transform("min")
    last_ts = grp.transform("max")

    df[f"{prefix}ts_since_first_s"] = (df["timestamp"] - first_ts).dt.total_seconds()
    df[f"{prefix}ts_since_last_s"] = (last_ts - df["timestamp"]).dt.total_seconds()
    df[f"{prefix}days_since_first"] = df[f"{prefix}ts_since_first_s"] / 86400.0

    return df


def inter_transaction_delta(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Time since previous transaction of the same {side}_account."""
    account_col = f"{side}_account"
    prefix = f"acc_{side}_"

    df = df.sort_values([account_col, "timestamp"]).reset_index(drop=True)
    df[f"{prefix}ts_since_prev_s"] = (
        df.groupby(account_col)["timestamp"].diff().dt.total_seconds()
    )
    df[f"{prefix}ts_since_prev_s"] = df[f"{prefix}ts_since_prev_s"].fillna(-1.0)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _rolling_per_group(
    ts_ns: np.ndarray,
    amounts: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    window_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling count & sum per group using numpy.searchsorted.

    Args:
        ts_ns: int64 array of timestamps in nanoseconds (sorted by group, then time)
        amounts: float64 array of amounts
        group_starts: int array of group start indices
        group_ends: int array of group end indices (exclusive)
        window_ns: window size in nanoseconds

    Returns:
        (counts, sums) arrays of same length as ts_ns
    """
    n = len(ts_ns)
    counts = np.zeros(n, dtype="int32")
    sums = np.zeros(n, dtype="float64")

    for start, end in zip(group_starts, group_ends):
        ts_g = ts_ns[start:end]
        amt_g = amounts[start:end]
        m = len(ts_g)

        # Cumulative sum with leading 0
        cumsum = np.empty(m + 1, dtype="float64")
        cumsum[0] = 0.0
        np.cumsum(amt_g, out=cumsum[1:])

        # For each row, find first index in window [ts - window, ts]
        lower_ts = ts_g - window_ns
        lower_idx = np.searchsorted(ts_g, lower_ts, side="left")

        idx = np.arange(m, dtype="int64")
        counts[start:end] = (idx - lower_idx + 1).astype("int32")
        sums[start:end] = cumsum[idx + 1] - cumsum[lower_idx]

    return counts, sums


def rolling_window_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Fast rolling count + sum per {side}_account over multiple windows."""
    account_col = f"{side}_account"
    amount_col = "amount_paid" if side == "from" else "amount_received"
    prefix = f"acc_{side}_"

    # Sort by (account, timestamp) — required for searchsorted trick
    work = df[[account_col, "timestamp", amount_col]].copy()
    work = work.sort_values([account_col, "timestamp"]).reset_index(drop=True)

    # Convert to numpy for speed
    ts_ns = work["timestamp"].values.astype("datetime64[ns]").astype("int64")
    amounts = work[amount_col].values.astype("float64")
    accounts = work[account_col].values

    # Group boundaries (accounts are sorted, so changes mark new groups)
    change_idx = np.where(accounts[1:] != accounts[:-1])[0] + 1
    group_starts = np.concatenate([[0], change_idx])
    group_ends = np.concatenate([change_idx, [len(accounts)]])

    logger.info(
        f"    rolling windows ({side}) — {len(group_starts):,} unique accounts"
    )

    for w in WINDOWS:
        tag = _windows_tag(w)
        window_ns = int(pd.Timedelta(w).value)

        cnt, sm = _rolling_per_group(
            ts_ns, amounts, group_starts, group_ends, window_ns
        )
        work[f"{prefix}cnt_{tag}"] = cnt
        work[f"{prefix}sum_{tag}"] = sm

    # Merge back — same sort order as work
    df = df.sort_values([account_col, "timestamp"]).reset_index(drop=True)
    new_cols = [
        c for c in work.columns if c not in (account_col, "timestamp", amount_col)
    ]
    for col in new_cols:
        df[col] = work[col].values

    # Restore original sort (by timestamp)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def velocity_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Velocity: transactions per day and amount per day (per account)."""
    prefix = f"acc_{side}_"

    span_s = df[f"{prefix}ts_since_first_s"]
    span_days = (span_s / 86400.0).clip(lower=1.0 / 24.0)

    df[f"{prefix}velocity_tx_per_day"] = df[f"{prefix}n_tx"] / span_days
    df[f"{prefix}velocity_amount_per_day"] = df[f"{prefix}sum"] / span_days

    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full temporal pipeline."""
    for side in ["from", "to"]:
        logger.info(f"  temporal features — side={side}")
        df = recency_features(df, side)
        logger.info("    recency done")
        df = inter_transaction_delta(df, side)
        logger.info("    inter-transaction delta done")
        df = rolling_window_features(df, side)
        logger.info("    rolling windows done")
        df = velocity_features(df, side)
        logger.info("    velocity done")
    return df
