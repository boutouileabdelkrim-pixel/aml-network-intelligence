# 🏦 AML Network Intelligence

> Anti-Money Laundering detection using **graph intelligence** and **machine learning**.

[![CI](https://github.com/boutouileabdelkrim-pixel/aml-network-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/boutouileabdelkrim-pixel/aml-network-intelligence/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)](https://mlflow.org/)

---

## 🎯 Problem Statement

Suspicious financial activity is not always visible in a single transaction — the real signal often lies in the **network of accounts**.

    Account A
       |
       v
    Account B ---> Account D
       |              |
       v              v
    Account C ---------+

This project detects **abnormal structures and behaviors** in financial networks using a hybrid approach: **tabular ML** + **graph-based feature engineering**.

---

## 🏗️ Architecture

    Raw transactions
          |
          v
    Data cleaning & EDA
          |
          v
    Feature Engineering ---> Tabular features ---> XGBoost / LightGBM
          |
          +---> Graph features ---> Node2Vec / Communities ---> Anomaly detection
          |
          v
    Explainability (SHAP)
          |
          v
    MLflow tracking ---> Docker ---> FastAPI (batch + realtime) ---> Monitoring

---

## 🚀 Quickstart

```bash
git clone git@github.com:boutouileabdelkrim-pixel/aml-network-intelligence.git
cd aml-network-intelligence
make setup
make data
make train
make api
```

---

## 📁 Project Structure

    aml-network-intelligence/
    |-- data/               # Raw / interim / processed / features
    |-- notebooks/          # EDA, graph analysis, SHAP
    |-- src/                # Python source code
    |   |-- data/           # Cleaning, validation
    |   |-- features/       # Tabular + graph features
    |   |-- models/         # Training, tuning
    |   |-- api/            # FastAPI serving
    |   |-- monitoring/     # Drift detection
    |   +-- tracking/       # MLflow
    |-- tests/              # Unit tests
    |-- docker/             # Dockerfile + compose
    |-- docs/               # Documentation
    +-- reports/            # Generated artifacts

---


## 🔍 Exploratory Data Analysis

The raw dataset contains **5,078,345 transactions** over **11 days**
(2022-09-01 → 2022-09-11) with an extreme class imbalance:
only **~0.1 %** are labeled as laundering.

| Metric | Value |
|--------|-------|
| Total transactions | 5,078,345 |
| Laundering transactions | ~6,900 |
| Laundering rate | ~0.14 % |
| Unique source accounts | ~500 k |
| Unique banks | ~30 k |
| Payment formats | 5 (ACH, Wire, Cheque, Credit Card, Reinvestment) |
| Currencies | 5+ (US Dollar, Euro, Yuan, ...) |

### Visual insights

| Target distribution | Amount distribution |
|---|---|
| ![target](reports/target_distribution.png) | ![amount](reports/amount_distribution.png) |

| Categoricals | Temporal |
|---|---|
| ![cats](reports/categoricals.png) | ![temporal](reports/temporal.png) |

See notebooks for full analysis:
- [`notebooks/01_eda_raw.ipynb`](notebooks/01_eda_raw.ipynb) — raw EDA
- [`notebooks/02_eda_clean.ipynb`](notebooks/02_eda_clean.ipynb) — post-cleaning










## 📊 Results

_To be filled after training (Phase 6)._

| Model | PR-AUC | ROC-AUC | F1 |
|-------|--------|---------|-----|
| XGBoost | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |

---

## 🛠️ Tech Stack

- **Data**: pandas, numpy, scipy, pyarrow
- **ML**: scikit-learn, XGBoost, LightGBM, Optuna, SHAP
- **Graph**: NetworkX, igraph, PyTorch Geometric
- **API**: FastAPI, Uvicorn
- **Ops**: MLflow, Docker, Evidently

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

**boutouileabdelkrim-pixel**
- GitHub: [@boutouileabdelkrim-pixel](https://github.com/boutouileabdelkrim-pixel)
