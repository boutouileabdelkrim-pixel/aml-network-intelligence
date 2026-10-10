"""API tests using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from tests.conftest import requires_model


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@requires_model
def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "version" in data


@requires_model
def test_model_info(client: TestClient) -> None:
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "aml-xgb-final"
    assert data["n_features"] > 0
    assert 0 < data["threshold"] < 1


@requires_model
def test_predict_simple(client: TestClient) -> None:
    payload = {
        "from_account": "ACC001",
        "to_account": "ACC002",
        "amount_paid": 5000.0,
        "amount_received": 5000.0,
        "acc_from_cnt_1h": 50.0,
        "acc_from_n_tx": 200.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["is_laundering_prob"] <= 1
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert data["decision"] in ("ALLOW", "REVIEW", "FLAG")
    assert "latency_ms" in data


@requires_model
def test_predict_minimal(client: TestClient) -> None:
    """Only account IDs — everything else defaults to 0."""
    r = client.post("/predict", json={"from_account": "A", "to_account": "B"})
    assert r.status_code == 200
    assert 0 <= r.json()["is_laundering_prob"] <= 1


@requires_model
def test_predict_batch(client: TestClient) -> None:
    batch = [
        {"from_account": "A1", "to_account": "B1", "acc_from_cnt_1h": 5.0},
        {"from_account": "A2", "to_account": "B2", "acc_from_cnt_1h": 500.0},
        {"from_account": "A3", "to_account": "B3", "acc_from_cnt_1h": 1.0},
    ]
    r = client.post("/predict/batch", json=batch)
    assert r.status_code == 200
    data = r.json()
    assert data["n_transactions"] == 3
    assert len(data["predictions"]) == 3


@requires_model
def test_batch_empty(client: TestClient) -> None:
    r = client.post("/predict/batch", json=[])
    assert r.status_code == 400
