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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# EWA model paths
MODEL_PATH      = BASE_DIR / "ewa_risk_model.json"
FEATURES_PATH   = BASE_DIR / "ewa_model_features.pkl"
METADATA_PATH   = BASE_DIR / "ewa_model_metadata.pkl"
USER_DATA_PATH  = BASE_DIR / "user_features_api.csv"

# Financial Distress model paths
FD_MODEL_PATH    = BASE_DIR / "fd_risk_model.pkl"
FD_FEATURES_PATH = BASE_DIR / "fd_model_features.pkl"
FD_METADATA_PATH = BASE_DIR / "fd_model_metadata.pkl"
FD_USER_DATA_PATH = BASE_DIR / "fd_user_features_api.csv"

# ── Global model state (loaded once at startup) ────────────────────────────────
class ModelState:
    # EWA model
    model: XGBRegressor = None
    explainer: shap.TreeExplainer = None
    feature_cols: list[str] = None
    calibration_multiplier: float = 1.0
    metadata: dict = {}
    user_data: pd.DataFrame = None

    # Financial Distress model
    fd_model: RandomForestClassifier = None
    fd_feature_cols: list[str] = None
    fd_metadata: dict = {}
    fd_user_data: pd.DataFrame = None

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

    # Load user features lookup table if available
    if USER_DATA_PATH.exists():
        state.user_data = pd.read_csv(USER_DATA_PATH).set_index("user_id")
        print(f"[startup] User data loaded — {len(state.user_data):,} users")
    else:
        print("[startup] No user_features_api.csv found — /user_risk, /top_features, /high_risk_users unavailable")

    print(f"[startup] EWA model loaded — {len(state.feature_cols)} features, "
          f"calibration={state.calibration_multiplier:.4f}")

    # Load Financial Distress model
    with open(FD_MODEL_PATH, "rb") as f:
        state.fd_model = pickle.load(f)
    with open(FD_FEATURES_PATH, "rb") as f:
        state.fd_feature_cols = pickle.load(f)
    with open(FD_METADATA_PATH, "rb") as f:
        state.fd_metadata = pickle.load(f)
    if FD_USER_DATA_PATH.exists():
        state.fd_user_data = pd.read_csv(FD_USER_DATA_PATH).set_index("user_id")
        print(f"[startup] FD model loaded — {len(state.fd_feature_cols)} features, {len(state.fd_user_data):,} users")

    yield
    print("[shutdown] Models released")


app = FastAPI(
    title="EWA Demand Forecasting API",
    description="Predicts EWA withdrawal frequency and demand tier for a batch of users.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/")
def root():
    return {
        "name": "Payactiv EWA Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


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
        try:
            validated = UserFeatures(**feat)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid features for user '{entry['user_id']}': {str(e)}"
            )
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


def _check_user_data():
    if state.user_data is None:
        raise HTTPException(status_code=503, detail="User data not loaded. Add user_features_api.csv to the API folder.")


def _predict_for_user(user_id: str):
    if user_id not in state.user_data.index:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    row = state.user_data.loc[user_id][state.feature_cols]
    X = pd.DataFrame([row.values], columns=state.feature_cols)
    log_score = float(state.model.predict(X)[0])
    count = float(max(np.expm1(log_score) * state.calibration_multiplier, 0))
    shap_vals = state.explainer.shap_values(X)[0]
    return X, log_score, count, shap_vals


@app.get("/user_risk/{user_id}")
def get_user_risk(user_id: str):
    """Get predicted EWA withdrawal count and demand tier for a single user."""
    _check_user_data()
    X, log_score, count, shap_vals = _predict_for_user(user_id)

    all_scores = state.model.predict(pd.DataFrame(state.user_data[state.feature_cols]))
    low_thresh  = float(np.percentile(all_scores, 30))
    high_thresh = float(np.percentile(all_scores, 70))

    return {
        "user_id": user_id,
        "predicted_ewa_count": round(count, 2),
        "demand_tier": assign_tier(log_score, low_thresh, high_thresh),
        "predicted_score": round(log_score, 4),
    }


@app.get("/top_features/{user_id}")
def get_top_features(user_id: str, top_n: int = 5):
    """Get top SHAP drivers explaining the EWA prediction for a single user."""
    _check_user_data()
    X, log_score, count, shap_vals = _predict_for_user(user_id)

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    drivers = [
        {
            "rank": i + 1,
            "feature": state.feature_cols[j],
            "shap": round(float(shap_vals[j]), 5),
            "value": round(float(X.iloc[0, j]), 4),
            "direction": "increases EWA demand" if shap_vals[j] > 0 else "decreases EWA demand",
        }
        for i, j in enumerate(sorted_idx)
    ]

    return {
        "user_id": user_id,
        "predicted_ewa_count": round(count, 2),
        "top_drivers": drivers,
    }


@app.get("/high_risk_users")
def get_high_risk_users(threshold: float = 0.7):
    """Get all users whose predicted score is in the top percentile (default: top 30%)."""
    _check_user_data()

    X_all = pd.DataFrame(state.user_data[state.feature_cols])
    all_scores = state.model.predict(X_all)
    counts = np.maximum(np.expm1(all_scores) * state.calibration_multiplier, 0)

    cutoff = float(np.percentile(all_scores, threshold * 100))
    high_risk = [
        {
            "user_id": uid,
            "predicted_ewa_count": round(float(counts[i]), 2),
            "predicted_score": round(float(all_scores[i]), 4),
        }
        for i, uid in enumerate(state.user_data.index)
        if all_scores[i] >= cutoff
    ]
    high_risk.sort(key=lambda x: x["predicted_score"], reverse=True)

    return {
        "threshold_percentile": f"top {round((1 - threshold) * 100)}%",
        "total_high_risk_users": len(high_risk),
        "users": high_risk,
    }


# ── Financial Distress Endpoints ───────────────────────────────────────────────

def _check_fd_user_data():
    if state.fd_user_data is None:
        raise HTTPException(status_code=503, detail="FD user data not loaded.")


def _predict_distress_for_user(user_id: str):
    if user_id not in state.fd_user_data.index:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in distress model.")
    row = state.fd_user_data.loc[user_id][state.fd_feature_cols]
    X = pd.DataFrame([row.values], columns=state.fd_feature_cols)
    prob = float(state.fd_model.predict_proba(X)[0][1])
    label = "Yes" if prob >= 0.5 else "No"
    return X, prob, label


@app.get("/distress_risk/{user_id}")
def get_distress_risk(user_id: str):
    """Get financial distress probability and label for a single user."""
    _check_fd_user_data()
    X, prob, label = _predict_distress_for_user(user_id)
    return {
        "user_id": user_id,
        "distress_probability": round(prob, 4),
        "financially_distressed": label,
        "risk_level": "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low",
    }


@app.get("/distress_features/{user_id}")
def get_distress_features(user_id: str, top_n: int = 5):
    """Get top features driving financial distress prediction for a single user."""
    _check_fd_user_data()
    X, prob, label = _predict_distress_for_user(user_id)

    importances = state.fd_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    drivers = [
        {
            "rank": i + 1,
            "feature": state.fd_feature_cols[j],
            "importance": round(float(importances[j]), 5),
            "value": round(float(X.iloc[0, j]), 4),
        }
        for i, j in enumerate(sorted_idx)
    ]
    return {
        "user_id": user_id,
        "distress_probability": round(prob, 4),
        "financially_distressed": label,
        "top_drivers": drivers,
    }


@app.get("/high_distress_users")
def get_high_distress_users(min_probability: float = 0.5):
    """Get all users predicted to be financially distressed above a probability threshold."""
    _check_fd_user_data()

    X_all = pd.DataFrame(state.fd_user_data[state.fd_feature_cols])
    probs = state.fd_model.predict_proba(X_all)[:, 1]

    distressed = [
        {
            "user_id": uid,
            "distress_probability": round(float(probs[i]), 4),
            "risk_level": "High" if probs[i] >= 0.6 else "Medium",
        }
        for i, uid in enumerate(state.fd_user_data.index)
        if probs[i] >= min_probability
    ]
    distressed.sort(key=lambda x: x["distress_probability"], reverse=True)

    return {
        "min_probability": min_probability,
        "total_distressed_users": len(distressed),
        "users": distressed,
    }
