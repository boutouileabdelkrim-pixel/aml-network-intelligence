# Data — IBM AML HI-Small

## Source

- **Provider**: IBM Research
- **Dataset**: Transactions for Anti-Money Laundering (AML)
- **Variant**: HI-Small
- **Kaggle slug**: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
- **License**: Community Data License Agreement - Sharing (CDLA-S) v1.0

## Files currently in data/raw/

| File | Size | Description |
|------|------|-------------|
| `HI-Small_Trans.csv` | 454 MB | 5,078,345 transactions with label `Is Laundering` |
| `HI-Small_accounts.csv` | 33 MB | Account metadata (bank, entity) |
| `HI-Small_Patterns.txt` | 317 KB | Ground-truth AML patterns (FAN-IN, FAN-OUT, CYCLE...) |

The full Kaggle archive (~8 GB) contains 7 variants (HI-Small/Medium/Large, LI-Small/Medium/Large).
We only keep HI-Small for this project to save disk and speed up iteration.

The original archive is kept outside the repo at `~/aml-zip-backup/` for reproducibility.

## Schema — HI-Small_Trans.csv

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | datetime | Transaction timestamp |
| `From Bank` | int | Source bank ID |
| `Account` | string | Source account ID |
| `To Bank` | int | Destination bank ID |
| `Account.1` | string | Destination account ID |
| `Amount Received` | float | Amount credited |
| `Receiving Currency` | string | Currency of received amount |
| `Amount Paid` | float | Amount debited |
| `Payment Currency` | string | Currency of paid amount |
| `Payment Format` | string | ACH, Wire, Cheque, Credit Card, Reinvestment |
| `Is Laundering` | int (0/1) | Label: 1 = laundering |

## Class imbalance

- Total rows: **5,078,345**
- Laundering rate: **~0.1 %** (extremely imbalanced)
- Metric choice: **PR-AUC** preferred over ROC-AUC
- Strategy: `scale_pos_weight` for XGBoost / LightGBM, stratified splits

## How to download

The full dataset is fetched with:

    uv run python -m src.data.download

Note: the Kaggle archive is ~8 GB and contains all variants. To extract only HI-Small
after download (much faster than Python's zipfile):

    cd data/raw
    unzip -o -j ~/aml-zip-backup/ibm-transactions-for-anti-money-laundering-aml.zip "HI-Small*" -d .

## Fallback — Synthetic dataset

If Kaggle is unavailable, a synthetic dataset can be generated:

    uv run python -m src.data.synthetic

It writes `data/raw/synthetic_aml.csv` with 50k rows and a 1 % laundering ratio.

## References

- Altman, E., et al. (2023). *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*. NeurIPS.
- Kaggle page: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
