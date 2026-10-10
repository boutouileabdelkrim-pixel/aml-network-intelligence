"""Pydantic schemas for the AML prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
Decision = Literal["ALLOW", "REVIEW", "FLAG"]


class TransactionFeatures(BaseModel):
    """Input: pre-computed features for one transaction.

    In production, these come from a feature store. Missing features
    default to 0 (XGBoost handles them naturally).
    """

    model_config = {"extra": "allow"}  # accept unknown feature names

    from_account: str = Field(..., description="Source account ID")
    to_account: str = Field(..., description="Destination account ID")

    # Common pre-computed features (optional — default to 0 if missing)
    amount_paid: float = Field(0.0, ge=0)
    amount_received: float = Field(0.0, ge=0)
    acc_from_cnt_1h: float = Field(0.0, ge=0)
    acc_from_cnt_1d: float = Field(0.0, ge=0)
    acc_from_n_tx: float = Field(0.0, ge=0)


class PredictionResponse(BaseModel):
    """Output: risk prediction for one transaction."""

    is_laundering_prob: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    decision: Decision
    threshold: float
    top_features: list[str]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    n_features: int
    training_metrics: dict
    threshold: float


class BatchPredictionItem(BaseModel):
    from_account: str
    to_account: str
    is_laundering_prob: float
    risk_level: RiskLevel
    decision: Decision


class BatchPredictionResponse(BaseModel):
    n_transactions: int
    n_flagged: int
    n_reviewed: int
    predictions: list[BatchPredictionItem]

