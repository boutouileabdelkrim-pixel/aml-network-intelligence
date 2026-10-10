"""FastAPI application for AML prediction."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.model_loader import (
    DECISION_THRESHOLD,
    MODEL_NAME,
    MODEL_VERSION,
    decision,
    features_to_vector,
    load_feature_names,
    load_model,
    load_training_metrics,
    risk_level,
    top_contributing_features,
)
from src.api.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionFeatures,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup."""
    logger.info("Starting AML API — loading model ...")
    load_model()
    load_feature_names()
    logger.success("Model ready. API is live.")
    yield
    logger.info("Shutting down AML API.")


app = FastAPI(
    title="AML Network Intelligence API",
    description="Anti-Money Laundering detection using graph intelligence + XGBoost.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# =========================================================
# Health
# =========================================================
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True, version=MODEL_VERSION)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="degraded", model_loaded=False, version=MODEL_VERSION)


# =========================================================
# Model info
# =========================================================
@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    metrics = load_training_metrics()
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        n_features=len(load_feature_names()),
        training_metrics=metrics.get("test", {}),
        threshold=DECISION_THRESHOLD,
    )


# =========================================================
# Predict one
# =========================================================
@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(tx: TransactionFeatures) -> PredictionResponse:
    t0 = time.perf_counter()
    try:
        model = load_model()
        features = tx.model_dump()
        X = features_to_vector(features)
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        is_laundering_prob=round(prob, 6),
        risk_level=risk_level(prob),
        decision=decision(prob),
        threshold=DECISION_THRESHOLD,
        top_features=top_contributing_features(features, k=5),
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 2),
    )


# =========================================================
# Predict batch
# =========================================================
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predict"])
def predict_batch(transactions: list[TransactionFeatures]) -> BatchPredictionResponse:
    if not transactions:
        raise HTTPException(status_code=400, detail="Empty batch")

    model = load_model()
    rows = [features_to_vector(tx.model_dump())[0] for tx in transactions]
    X = np.array(rows, dtype="float32")
    probs = model.predict_proba(X)[:, 1]

    items: list[BatchPredictionItem] = []
    n_flagged = 0
    n_reviewed = 0
    for tx, p in zip(transactions, probs):
        p = float(p)
        d = decision(p)
        if d == "FLAG":
            n_flagged += 1
        elif d == "REVIEW":
            n_reviewed += 1
        items.append(
            BatchPredictionItem(
                from_account=tx.from_account,
                to_account=tx.to_account,
                is_laundering_prob=round(p, 6),
                risk_level=risk_level(p),
                decision=d,
            )
        )

    return BatchPredictionResponse(
        n_transactions=len(transactions),
        n_flagged=n_flagged,
        n_reviewed=n_reviewed,
        predictions=items,
    )
