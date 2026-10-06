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
