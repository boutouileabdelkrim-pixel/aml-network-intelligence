"""Unit tests for tabular feature builders.

Uses small synthetic data — fast and deterministic.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.features.tabular_basic import (
    account_aggregates,
    categorical_features,
    structuring_features,
)
from src.features.tabular_temporal import (
    inter_transaction_delta,
    recency_features,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """3 accounts, 6 transactions over 2 days."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 12:00",
                    "2024-01-02 08:00",
                    "2024-01-02 09:30",
                    "2024-01-02 15:00",
                    "2024-01-02 18:00",
                ]
            ),
            "from_bank": ["B1", "B1", "B2", "B2", "B3", "B3"],
            "from_account": ["A1", "A1", "A2", "A2", "A1", "A3"],
            "to_bank": ["B2", "B3", "B1", "B3", "B2", "B1"],
            "to_account": ["A2", "A3", "A1", "A3", "A2", "A1"],
            "amount_received": [100.0, 200.0, 100.0, 300.0, 250.0, 50.0],
            "receiving_currency": ["USD", "USD", "EUR", "USD", "EUR", "USD"],
            "amount_paid": [100.0, 200.0, 100.0, 300.0, 250.0, 50.0],
            "payment_currency": ["USD", "USD", "USD", "USD", "EUR", "USD"],
            "payment_format": ["ACH", "Wire", "ACH", "Wire", "ACH", "Cheque"],
            "is_laundering": [0, 0, 0, 1, 0, 0],
        }
    )


def test_account_aggregates_from(small_df: pd.DataFrame) -> None:
    stats = account_aggregates(small_df, side="from")
    assert "acc_from_n_tx" in stats.columns
    assert stats.loc["A1", "acc_from_n_tx"] == 3
    assert stats.loc["A2", "acc_from_n_tx"] == 2
    assert stats.loc["A3", "acc_from_n_tx"] == 1
    assert stats.loc["A1", "acc_from_sum"] == pytest.approx(550.0)
    assert stats.loc["A1", "acc_from_mean"] == pytest.approx(550.0 / 3)


def test_account_aggregates_to(small_df: pd.DataFrame) -> None:
    stats = account_aggregates(small_df, side="to")
    assert stats.loc["A1", "acc_to_n_tx"] == 2
    assert stats.loc["A3", "acc_to_n_tx"] == 2


def test_categorical_features(small_df: pd.DataFrame) -> None:
    out = categorical_features(small_df.copy())
    assert "is_cross_currency" in out.columns
    assert out.loc[2, "is_cross_currency"] == 1
    assert out.loc[0, "is_cross_currency"] == 0


def test_structuring_features(small_df: pd.DataFrame) -> None:
    out = structuring_features(small_df.copy())
    assert "is_round_100" in out.columns
    assert out.loc[0, "is_round_100"] == 1
    assert out.loc[4, "is_round_100"] == 0


def test_recency_features(small_df: pd.DataFrame) -> None:
    out = recency_features(small_df.copy(), side="from")
    assert "acc_from_ts_since_first_s" in out.columns
    assert "acc_from_ts_since_last_s" in out.columns
    a1_rows = out[out["from_account"] == "A1"]
    assert a1_rows["acc_from_ts_since_first_s"].min() == 0


def test_inter_transaction_delta(small_df: pd.DataFrame) -> None:
    out = inter_transaction_delta(small_df.copy(), side="from")
    assert "acc_from_ts_since_prev_s" in out.columns
    a1_rows = out[out["from_account"] == "A1"].sort_values("timestamp")
    assert a1_rows.iloc[0]["acc_from_ts_since_prev_s"] == -1.0
    assert a1_rows.iloc[1]["acc_from_ts_since_prev_s"] == pytest.approx(7200.0)


def test_no_nan_in_critical_outputs(small_df: pd.DataFrame) -> None:
    out = structuring_features(categorical_features(small_df.copy()))
    for col in ["is_cross_currency", "is_round_100", "amount_bucket"]:
        assert out[col].notna().all(), f"NaN found in {col}"
