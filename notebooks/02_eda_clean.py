# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3.11 (aml-network-intelligence)
#     language: python
#     name: aml-net
# ---

# %% [markdown]
# # 02 — EDA Clean: IBM AML HI-Small (post-cleaning)
#
# Validates the cleaning pipeline and produces the summary visual report
# used in the README.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_clean.parquet"
REPORTS = PROJECT_ROOT / "reports"

print("Clean file:", CLEAN_PATH.exists(), f"({CLEAN_PATH.stat().st_size / 1e6:.1f} MB)")

# %%
df = pd.read_parquet(CLEAN_PATH, engine="pyarrow")
print("Shape:", df.shape)
df.head()

# %% [markdown]
# ## 1. Summary statistics

# %%
print("Timestamp range :", df["timestamp"].min(), "→", df["timestamp"].max())
print("Target rate     :", f"{(df['is_laundering'] == 1).mean() * 100:.4f}%")
print("Amount Paid p50 :", f"{df['amount_paid'].median():,.2f}")
print("Amount Paid p99 :", f"{df['amount_paid'].quantile(0.99):,.2f}")

# %% [markdown]
# ## 2. Amount distribution — clean

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(df["amount_paid"].clip(lower=0, upper=df["amount_paid"].quantile(0.99)),
             bins=100, color="steelblue", edgecolor="white")
axes[0].set_title("amount_paid (0 → p99)")
axes[0].set_xlabel("amount")
axes[0].set_ylabel("count")

axes[1].hist(df["amount_paid"].clip(lower=1).pipe(lambda s: s[s > 0]).apply("log1p"),
             bins=100, color="crimson", edgecolor="white")
axes[1].set_title("log1p(amount_paid)")
axes[1].set_xlabel("log1p(amount)")
plt.tight_layout()
plt.savefig(REPORTS / "amount_distribution_clean.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Payment formats & currencies

# %%
print("Payment Format:")
print(df["payment_format"].value_counts())

print("\nPayment Currency:")
print(df["payment_currency"].value_counts())

# %% [markdown]
# ## 4. Aggregate-level features (preview for Phase 4)
#
# These are the kind of aggregates we will use for the model.

# %%
account_stats = df.groupby("from_account").agg(
    n_out=("amount_paid", "count"),
    total_out=("amount_paid", "sum"),
    mean_out=("amount_paid", "mean"),
).sort_values("n_out", ascending=False)

print("Top 5 most active source accounts:")
account_stats.head()

# %%
print("\nDistribution of transaction count per account:")
print(account_stats["n_out"].describe())
