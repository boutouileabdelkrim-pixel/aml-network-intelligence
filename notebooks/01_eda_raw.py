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
# # 01 — EDA Raw: IBM AML HI-Small
#
# **Author**: boutouileabdelkrim-pixel
# **Date**: 2026-10-07
# **Goal**: Understand structure, types, missing values, class imbalance,
# temporal patterns and duplicates before any cleaning.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve project root robustly (works both from project root and from notebooks/)
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW = PROJECT_ROOT / "data" / "raw"
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TRANS_PATH = RAW / "HI-Small_Trans.csv"
ACC_PATH = RAW / "HI-Small_accounts.csv"

plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["figure.dpi"] = 100

print("Project root:", PROJECT_ROOT)
print("Trans file  :", TRANS_PATH.exists(), f"({TRANS_PATH.stat().st_size / 1e6:.1f} MB)")
print("Acc file    :", ACC_PATH.exists(), f"({ACC_PATH.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 1. Load data
#
# We load with the pyarrow backend for speed and clean dtype detection.
# Note: the raw CSV has a **duplicated** `Account` column, pandas will
# auto-rename the second one to `Account.1`.

# %%
df = pd.read_csv(
    TRANS_PATH,
    dtype_backend="pyarrow",
    parse_dates=["Timestamp"],
)
print("Shape:", df.shape)
print("\nColumns:")
for c in df.columns:
    print("  -", c)

# %%
df.head()

# %%
df.dtypes

# %% [markdown]
# ## 2. Missing values

# %%
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(4)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
missing_df

# %% [markdown]
# ## 3. Target distribution (class imbalance)

# %%
target_counts = df["Is Laundering"].value_counts().sort_index()
target_pct = df["Is Laundering"].value_counts(normalize=True).sort_index() * 100

print("Counts:")
print(target_counts)
print("\nPercentage:")
print(target_pct.round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

target_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "crimson"])
axes[0].set_title("Class counts (log scale)")
axes[0].set_yscale("log")
axes[0].set_xticklabels(["Normal (0)", "Laundering (1)"], rotation=0)
axes[0].set_ylabel("count (log)")

axes[1].pie(
    target_counts,
    labels=["Normal", "Laundering"],
    autopct="%1.3f%%",
    colors=["steelblue", "crimson"],
    startangle=90,
)
axes[1].set_title("Class imbalance (linear)")
plt.tight_layout()
plt.savefig(REPORTS / "target_distribution.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Amount distributions

# %%
print("=== Amount Paid ===")
print(df["Amount Paid"].describe().astype(float))

print("\n=== Amount Received ===")
print(df["Amount Received"].describe().astype(float))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log-scale histogram
axes[0].hist(
    np.log1p(df["Amount Paid"].astype(float).dropna()),
    bins=100,
    color="steelblue",
    edgecolor="white",
)
axes[0].set_title("log1p(Amount Paid) — distribution")
axes[0].set_xlabel("log1p(amount)")
axes[0].set_ylabel("count")

# Boxplot by class (sample for speed)
df_sample = df.sample(n=min(200_000, len(df)), random_state=42)
amounts_by_target = [
    np.log1p(df_sample[df_sample["Is Laundering"] == 0]["Amount Paid"].astype(float).dropna()),
    np.log1p(df_sample[df_sample["Is Laundering"] == 1]["Amount Paid"].astype(float).dropna()),
]
axes[1].boxplot(amounts_by_target, tick_labels=["Normal", "Laundering"])
axes[1].set_title("log1p(Amount Paid) by class")
axes[1].set_ylabel("log1p(amount)")
plt.tight_layout()
plt.savefig(REPORTS / "amount_distribution.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Categorical distributions

# %%
for col in ["Payment Format", "Receiving Currency", "Payment Currency"]:
    print(f"\n=== {col} ===")
    print(df[col].value_counts().head(10))

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ["Payment Format", "Receiving Currency", "Payment Currency"]):
    df[col].value_counts().head(8).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(col)
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(REPORTS / "categoricals.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Temporal analysis

# %%
df["date"] = df["Timestamp"].dt.date
df["hour"] = df["Timestamp"].dt.hour

print("Date range:", df["Timestamp"].min(), "→", df["Timestamp"].max())
print("Duration  :", (df["Timestamp"].max() - df["Timestamp"].min()).days, "days")

daily = df.groupby("date").size()
hourly = df.groupby("hour").size()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
daily.plot(ax=axes[0], color="steelblue")
axes[0].set_title("Transactions per day")
axes[0].set_xlabel("date")
axes[0].set_ylabel("count")

hourly.plot(kind="bar", ax=axes[1], color="steelblue")
axes[1].set_title("Transactions per hour of day")
axes[1].set_xlabel("hour")
axes[1].set_ylabel("count")
plt.tight_layout()
plt.savefig(REPORTS / "temporal.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Duplicates & sanity checks

# %%
print("Duplicated full rows :", df.duplicated().sum())

from_acc = df["Account"].astype(str)
to_acc = df["Account.1"].astype(str)
self_loops = (from_acc == to_acc).sum()
print("Self-loop transactions (from == to) :", self_loops)

print("\nUnique source accounts      :", df["Account"].nunique())
print("Unique destination accounts :", df["Account.1"].nunique())
print("Unique banks (from)         :", df["From Bank"].nunique())
print("Unique banks (to)           :", df["To Bank"].nunique())
print("Unique payment formats      :", df["Payment Format"].nunique())
print("Unique currencies           :", df["Payment Currency"].nunique())

# %% [markdown]
# ## 8. Key insights (auto-generated)

# %%
insights = {
    "total_transactions": int(len(df)),
    "laundering_count": int((df["Is Laundering"] == 1).sum()),
    "laundering_rate_pct": float((df["Is Laundering"] == 1).mean() * 100),
    "date_range_days": int((df["Timestamp"].max() - df["Timestamp"].min()).days),
    "unique_source_accounts": int(df["Account"].nunique()),
    "unique_dest_accounts": int(df["Account.1"].nunique()),
    "unique_banks": int(max(df["From Bank"].nunique(), df["To Bank"].nunique())),
    "self_loop_count": int(self_loops),
    "duplicate_rows": int(df.duplicated().sum()),
}
for k, v in insights.items():
    print(f"{k:30s}: {v}")
