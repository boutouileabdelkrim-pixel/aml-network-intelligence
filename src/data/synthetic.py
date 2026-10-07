"""Generate a small synthetic AML dataset for testing.

Useful as a fallback if the Kaggle download fails, or for fast
local testing without loading 5M rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_aml.csv"

RNG = np.random.default_rng(42)

CURRENCIES = ["US Dollar", "Euro", "Yuan", "Bitcoin"]
FORMATS = ["ACH", "Wire", "Cheque", "Credit Card", "Reinvestment"]


def generate(n_rows: int = 50_000, laundering_ratio: float = 0.01) -> pd.DataFrame:
    """Generate a synthetic AML transaction dataset."""
    n_launder = int(n_rows * laundering_ratio)
    n_normal = n_rows - n_launder

    banks = RNG.integers(1, 20, size=n_rows)
    accounts = [f"ACC{RNG.integers(10_000_000, 99_999_999)}" for _ in range(n_rows)]
    amounts = np.round(RNG.lognormal(mean=7, sigma=1.5, size=n_rows), 2)

    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="min"),
            "From Bank": banks,
            "Account": accounts,
            "To Bank": RNG.integers(1, 20, size=n_rows),
            "Account.1": [f"ACC{RNG.integers(10_000_000, 99_999_999)}" for _ in range(n_rows)],
            "Amount Received": amounts,
            "Receiving Currency": RNG.choice(CURRENCIES, size=n_rows),
            "Amount Paid": amounts,
            "Payment Currency": RNG.choice(CURRENCIES, size=n_rows),
            "Payment Format": RNG.choice(FORMATS, size=n_rows),
            "Is Laundering": np.array([1] * n_launder + [0] * n_normal),
        }
    )

    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
