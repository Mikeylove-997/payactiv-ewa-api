# Payactiv Risk Prediction API — Technical Documentation

**Live API:** https://web-production-00c9c.up.railway.app  
**Interactive Docs:** https://web-production-00c9c.up.railway.app/docs  
**GitHub:** https://github.com/Mikeylove-997/payactiv-ewa-api  

---

## Overview

A FastAPI-based REST API serving two ML models:
- **EWA Demand Model** (XGBoost) — predicts EWA withdrawal frequency per user
- **Financial Distress Model** (Random Forest) — predicts financial distress probability per user

The API loads both models once at startup and serves predictions instantly for any user ID.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| EWA model | XGBoost Regressor |
| Distress model | Random Forest Classifier |
| Explainability | SHAP (TreeExplainer) |
| Data processing | Pandas, NumPy |
| Validation | Pydantic |
| Deployment | Railway |

---

## How the Two Models Are Integrated

Both models are loaded at API startup and run independently. The API file (`api.py`) maintains a single `ModelState` object that holds both models in memory:

```python
class ModelState:
    # EWA model
    model: XGBRegressor          # XGBoost regressor
    explainer: TreeExplainer     # SHAP explainer
    feature_cols: list[str]      # 34 feature names
    calibration_multiplier: float
    user_data: pd.DataFrame      # 11,988 users lookup

    # Financial Distress model
    fd_model: RandomForestClassifier
    fd_feature_cols: list[str]   # 23 feature names
    fd_user_data: pd.DataFrame   # 5,359 users lookup
```

When the server starts, both models load from their saved artifact files. Requests to EWA endpoints use `state.model`; requests to Financial Distress endpoints use `state.fd_model`. They never interfere with each other.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Mikeylove-997/payactiv-ewa-api
cd payactiv-ewa-api

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn api:app --reload --port 8000

# Open interactive docs
open http://localhost:8000/docs
```

---

## Endpoints

### System Endpoints

---

#### `GET /`
Welcome message. Confirms the API is reachable.

**Response:**
```json
{
  "name": "Payactiv EWA Demand Forecasting API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

#### `GET /health`
Returns whether the API is running and both models are loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_count": 34
}
```

---

#### `GET /model-info`
Returns EWA model metadata including feature list, metrics, and time windows.

**Response:**
```json
{
  "model_type": "XGBRegressor",
  "target": "log(ewa_w2_count + 1)",
  "calibration_multiplier": 1.3938,
  "feature_count": 34,
  "features": ["total_spend", "txn_count", "..."],
  "window1_end": "2026-01-31",
  "window2_start": "2026-02-01",
  "window2_end": "2026-04-07",
  "metrics": {
    "mae_log": 0.5562,
    "mae_count": 3.68,
    "r2": 0.6143,
    "spearman": 0.7548
  }
}
```

---

### EWA Demand Endpoints

---

#### `POST /predict`
Batch prediction for multiple users. Accepts pre-computed features and returns predictions with SHAP explanations.

**Request:**
```json
{
  "users": [
    {
      "user_id": "USER_001",
      "features": {
        "total_spend": 4200.0,
        "txn_count": 87,
        "spend_mean_30d": 140.0,
        "spend_volatility_30d": 52.3,
        "spend_cv_30d": 0.37,
        "spend_txn_count_30d": 62.0,
        "max_daily_spend_30d": 320.0,
        "spend_mean_30d_lag1": 130.0,
        "spend_volatility_30d_lag1": 48.1,
        "spend_cv_30d_lag1": 0.37,
        "essential_ratio_30d": 0.61,
        "non_essential_spend_30d": 850.0,
        "non_essential_spend_vol_30d": 40.2,
        "disc_active_days_30d": 18.0,
        "total_spend_30d": 2100.0,
        "top_merchant_spend_ratio": 0.22,
        "bank_spend_30d": 1500.0,
        "card_spend_share_30d": 0.43,
        "card_to_bank_spend_ratio_30d": 0.75,
        "spend_trend_w2_vs_w1": 0.12,
        "spend_volatility_trend_7d": 0.05,
        "frequent_small_txn_ratio": 0.30,
        "merchant_churn_rate": 0.45,
        "w1_distress_ratio": 0.40,
        "ewa_usage_count": 3.0,
        "ewa_total_amount": 600.0,
        "ewa_avg_amount": 200.0,
        "ewa_max_amount": 250.0,
        "ewa_avg_income_rate": 0.18,
        "ewa_min_income_rate": 0.10,
        "ewa_contract_FullTimeHourly": 1,
        "ewa_contract_FullTimeSalaried": 0,
        "ewa_contract_PartTimeHourly": 0,
        "ewa_contract_PartTimeSalaried": 0
      }
    }
  ],
  "include_shap": true,
  "top_n_drivers": 5
}
```

**Response:**
```json
{
  "predictions": [
    {
      "user_id": "USER_001",
      "predicted_score": 0.0715,
      "predicted_ewa_count": 0.1,
      "demand_tier": "Low",
      "shap_values": { "ewa_usage_count": -0.43, "...": "..." },
      "top_drivers": [
        {
          "feature": "ewa_usage_count",
          "shap": -0.43,
          "value": 3.0,
          "direction": "decreases EWA demand"
        }
      ]
    }
  ],
  "model_version": "XGBRegressor",
  "calibration_multiplier": 1.3938,
  "tier_thresholds": { "low_max": 0.07, "high_min": 0.07 },
  "total_users": 1
}
```

---

#### `GET /user_risk/{user_id}`
Returns predicted EWA withdrawal count and demand tier for a single user by ID.

**Example:** `GET /user_risk/Mymo1002027`

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "predicted_ewa_count": 20.48,
  "demand_tier": "High",
  "predicted_score": 2.7531
}
```

---

#### `GET /top_features/{user_id}?top_n=5`
Returns top SHAP drivers explaining EWA demand for a single user.

**Example:** `GET /top_features/Mymo1002027?top_n=5`

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "predicted_ewa_count": 20.48,
  "top_drivers": [
    {
      "rank": 1,
      "feature": "ewa_usage_count",
      "shap": 0.891,
      "value": 62.0,
      "direction": "increases EWA demand"
    }
  ]
}
```

---

#### `GET /high_risk_users?threshold=0.7`
Returns all users in the top risk percentile. Default threshold = 0.7 (top 30%).

**Example:** `GET /high_risk_users?threshold=0.9` (top 10%)

**Response:**
```json
{
  "threshold_percentile": "top 30%",
  "total_high_risk_users": 3597,
  "users": [
    {
      "user_id": "Mymo1002027",
      "predicted_ewa_count": 20.48,
      "predicted_score": 2.7531
    }
  ]
}
```

---

### Financial Distress Endpoints

---

#### `GET /distress_risk/{user_id}`
Returns financial distress probability and label for a single user.

**Example:** `GET /distress_risk/Mymo1002027`

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "distress_probability": 0.172,
  "financially_distressed": "No",
  "risk_level": "Low"
}
```

---

#### `GET /distress_features/{user_id}?top_n=5`
Returns top feature importances driving the distress prediction for a single user.

**Example:** `GET /distress_features/Mymo1002027?top_n=5`

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "distress_probability": 0.172,
  "financially_distressed": "No",
  "top_drivers": [
    {
      "rank": 1,
      "feature": "card_to_bank_spend_ratio_30d",
      "importance": 0.089,
      "value": 0.5
    }
  ]
}
```

---

#### `GET /high_distress_users?min_probability=0.5`
Returns all users predicted to be financially distressed above the probability threshold.

**Response:**
```json
{
  "min_probability": 0.5,
  "total_distressed_users": 1316,
  "users": [
    {
      "user_id": "Mymo1003871",
      "distress_probability": 0.94,
      "risk_level": "High"
    }
  ]
}
```

---

#### `POST /distress/predict`
Adapter endpoint for the Gradio dashboard. Accepts spending features and returns distress prediction. If the user ID is found in the database, uses the real model. Otherwise estimates from the provided features.

**Request:**
```json
{
  "user_id": "Mymo1002027",
  "total_spend": 4200,
  "txn_count": 87,
  "total_spend_30d": 2100,
  "max_daily_spend_30d": 320,
  "txn_count_30d": 62,
  "ewa_usage_count": 62,
  "ewa_total_amount": 3626,
  "ewa_avg_amount": 58,
  "ewa_max_amount": 250
}
```

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "risk_score": 0.172,
  "risk_level": "Low risk",
  "model_type": "RandomForestClassifier",
  "timestamp": "2026-05-18T19:14:45"
}
```

---

## Error Handling

| Error | HTTP Code | Meaning |
|---|---|---|
| User not found | 404 | User ID does not exist in the database |
| Invalid feature value | 422 | Feature value out of valid range |
| Empty batch | 422 | At least 1 user required in `/predict` |
| Model not loaded | 503 | User data CSV not found at startup |

**Example 404 response:**
```json
{
  "detail": "User 'FAKEID' not found."
}
```

---

## Project File Structure

```
EWA API/
├── api.py                       # FastAPI app — all 10 endpoints
├── mcp_server.py                # Claude Desktop MCP integration
├── ewa_risk_model.json          # Trained XGBoost model (EWA)
├── ewa_model_features.pkl       # EWA feature list (34 features)
├── ewa_model_metadata.pkl       # EWA calibration + metrics
├── user_features_api.csv        # EWA user features (11,988 users)
├── fd_risk_model.pkl            # Trained Random Forest (distress)
├── fd_model_features.pkl        # Distress feature list (23 features)
├── fd_model_metadata.pkl        # Distress metrics
├── fd_user_features_api.csv     # Distress user features (5,359 users)
├── requirements.txt             # Package dependencies
├── Procfile                     # Railway deployment config
├── README.md                    # Project overview
├── API_DOCUMENTATION.md         # This file
├── PROJECT_SUMMARY.md           # Full project summary
└── MCP_SETUP.md                 # MCP client setup guide
```
