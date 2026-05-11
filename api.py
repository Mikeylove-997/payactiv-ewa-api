"""
EWA Demand Forecasting API
FastAPI service for batch EWA withdrawal frequency prediction with SHAP explanations.

Usage:
    pip install fastapi uvicorn xgboost shap pandas numpy
    uvicorn api:app --reload --port 8000

Docs auto-generated at: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH    = BASE_DIR / "ewa_risk_model.json"
FEATURES_PATH = BASE_DIR / "ewa_model_features.pkl"
METADATA_PATH = BASE_DIR / "ewa_model_metadata.pkl"

# ── Global model state (loaded once at startup) ────────────────────────────────
class ModelState:
    model: XGBRegressor = None
    explainer: shap.TreeExplainer = None
    feature_cols: list[str] = None
    calibration_multiplier: float = 1.0
    metadata: dict = {}

state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; release on shutdown."""
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    state.model = model

    with open(FEATURES_PATH, "rb") as f:
        state.feature_cols = pickle.load(f)

    with open(METADATA_PATH, "rb") as f:
        state.metadata = pickle.load(f)

    state.calibration_multiplier = state.metadata.get("calibration_multiplier", 1.0)

    # TreeExplainer is fast for XGBoost — build it once, reuse per request
    state.explainer = shap.TreeExplainer(state.model)

    print(f"[startup] Model loaded — {len(state.feature_cols)} features, "
          f"calibration={state.calibration_multiplier:.4f}")
    yield
    print("[shutdown] Model released")


app = FastAPI(
    title="EWA Demand Forecasting API",
    description="Predicts EWA withdrawal frequency and demand tier for a batch of users.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class UserFeatures(BaseModel):
    """
    One user's pre-computed Window-1 features.
    All 34 fields are required. Pass 0 for missing/unknown values.
    """
    # Spending volume
    total_spend: float = Field(..., description="Total spending in Window 1")
    txn_count: float = Field(..., description="Total transaction count")

    # 30-day rolling spending stats
    spend_mean_30d: float
    spend_volatility_30d: float
    spend_cv_30d: float = Field(..., description="Coefficient of variation (volatility / mean)")
    spend_txn_count_30d: float
    max_daily_spend_30d: float

    # Lag-1 rolling stats (prior 30-day window)
    spend_mean_30d_lag1: float
    spend_volatility_30d_lag1: float
    spend_cv_30d_lag1: float

    # Spending composition
    essential_ratio_30d: float = Field(..., ge=0, le=1, description="Share of essential spending")
    non_essential_spend_30d: float
    non_essential_spend_vol_30d: float
    disc_active_days_30d: float = Field(..., description="Days with discretionary spending")
    total_spend_30d: float

    # Merchant & source mix
    top_merchant_spend_ratio: float = Field(..., ge=0, le=1)
    bank_spend_30d: float
    card_spend_share_30d: float = Field(..., ge=0, le=1)
    card_to_bank_spend_ratio_30d: float

    # Trend & behavioral signals
    spend_trend_w2_vs_w1: float = Field(..., description="% spend change early→late Window 1")
    spend_volatility_trend_7d: float
    frequent_small_txn_ratio: float = Field(..., ge=0, le=1)
    merchant_churn_rate: float = Field(..., ge=0, le=1)

    # Financial distress index
    w1_distress_ratio: float = Field(..., ge=0, le=1,
        description="Share of Window-1 days where spend > income")

    # EWA history in Window 1
    ewa_usage_count: float
    ewa_total_amount: float
    ewa_avg_amount: float
    ewa_max_amount: float
    ewa_avg_income_rate: float
    ewa_min_income_rate: float

    # Contract type (one-hot encoded — set exactly one to 1, rest to 0)
    ewa_contract_FullTimeHourly: float = Field(..., ge=0, le=1)
    ewa_contract_FullTimeSalaried: float = Field(..., ge=0, le=1)
    ewa_contract_PartTimeHourly: float = Field(..., ge=0, le=1)
    ewa_contract_PartTimeSalaried: float = Field(..., ge=0, le=1)


class PredictRequest(BaseModel):
    users: list[dict] = Field(
        ...,
        description='List of {"user_id": str, "features": {UserFeatures fields}}',
        min_length=1,
        max_length=5000,
    )
    include_shap: bool = Field(True, description="Include per-feature SHAP values in response")
    top_n_drivers: int = Field(5, ge=1, le=34, description="Number of top SHAP drivers to return")


class UserPrediction(BaseModel):
    user_id: str
    predicted_score: float = Field(..., description="Raw log-scale score from model")
    predicted_ewa_count: float = Field(..., description="Back-transformed & calibrated EWA count")
    demand_tier: str = Field(..., description="Low | Medium | High")
    shap_values: Optional[dict[str, float]] = Field(None, description="SHAP value per feature")
    top_drivers: Optional[list[dict]] = Field(None, description="Top N features by |SHAP|")


class PredictResponse(BaseModel):
    predictions: list[UserPrediction]
    model_version: str
    calibration_multiplier: float
    tier_thresholds: dict = Field(...,
        description="Score thresholds used for Low/Medium/High tiers (percentile-based on this batch)")
    total_users: int


# ── Helper ─────────────────────────────────────────────────────────────────────

def assign_tier(score: float, low_thresh: float, high_thresh: float) -> str:
    if score <= low_thresh:
        return "Low"
    if score <= high_thresh:
        return "Medium"
    return "High"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "feature_count": len(state.feature_cols) if state.feature_cols else 0,
    }


@app.get("/model-info")
def model_info():
    return state.metadata


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # ── Build feature matrix ───────────────────────────────────────────────────
    user_ids = []
    rows = []

    for entry in req.users:
        if "user_id" not in entry or "features" not in entry:
            raise HTTPException(
                status_code=422,
                detail='Each item must have "user_id" and "features" keys.',
            )
        user_ids.append(str(entry["user_id"]))

        feat = entry["features"]
        # Validate using Pydantic model (raises 422 automatically on bad values)
        validated = UserFeatures(**feat)
        row = [getattr(validated, col) for col in state.feature_cols]
        rows.append(row)

    X = pd.DataFrame(rows, columns=state.feature_cols)

    # ── Predict ────────────────────────────────────────────────────────────────
    log_scores = state.model.predict(X)   # log(count + 1)
    calibrated_counts = np.maximum(np.expm1(log_scores) * state.calibration_multiplier, 0)

    # ── Tier thresholds (percentile-based on this batch) ──────────────────────
    # For production: replace with fixed thresholds saved from training set.
    low_thresh  = float(np.percentile(log_scores, 30))
    high_thresh = float(np.percentile(log_scores, 70))

    # ── SHAP (optional) ────────────────────────────────────────────────────────
    shap_matrix = None
    if req.include_shap:
        shap_matrix = state.explainer.shap_values(X)   # shape: (n_users, n_features)

    # ── Build response ─────────────────────────────────────────────────────────
    predictions = []
    for i, uid in enumerate(user_ids):
        shap_vals = None
        top_drivers = None

        if shap_matrix is not None:
            raw = shap_matrix[i]
            shap_vals = {col: round(float(raw[j]), 5) for j, col in enumerate(state.feature_cols)}

            sorted_idx = np.argsort(np.abs(raw))[::-1][: req.top_n_drivers]
            top_drivers = [
                {
                    "feature": state.feature_cols[j],
                    "shap": round(float(raw[j]), 5),
                    "value": round(float(X.iloc[i, j]), 4),
                    "direction": "increases EWA demand" if raw[j] > 0 else "decreases EWA demand",
                }
                for j in sorted_idx
            ]

        predictions.append(
            UserPrediction(
                user_id=uid,
                predicted_score=round(float(log_scores[i]), 4),
                predicted_ewa_count=round(float(calibrated_counts[i]), 2),
                demand_tier=assign_tier(log_scores[i], low_thresh, high_thresh),
                shap_values=shap_vals,
                top_drivers=top_drivers,
            )
        )

    return PredictResponse(
        predictions=predictions,
        model_version=state.metadata.get("model_type", "XGBRegressor"),
        calibration_multiplier=state.calibration_multiplier,
        tier_thresholds={"low_max": round(low_thresh, 4), "high_min": round(high_thresh, 4)},
        total_users=len(predictions),
    )
