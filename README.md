# Payactiv EWA Demand Forecasting API

A FastAPI backend that predicts EWA (Earned Wage Access) withdrawal demand for users using a trained XGBoost model.

**Live API:** `https://web-production-00c9c.up.railway.app`  
**Interactive Docs:** `https://web-production-00c9c.up.railway.app/docs`

---

## What It Does

Given a user's pre-computed spending and EWA history features, the API predicts:
- How many EWA withdrawals they are expected to make in the next period
- Which demand tier they fall into (Low / Medium / High)
- Which features are driving that prediction (SHAP explanation)

---

## Model

| Detail | Value |
|---|---|
| Model type | XGBoost Regressor |
| Target variable | log(EWA withdrawal count + 1) |
| Features | 34 pre-computed user features |
| Calibration multiplier | 1.3938 |
| R² | 0.6143 |
| Spearman correlation | 0.7548 |
| Training window | Nov 6, 2025 – Jan 31, 2026 |
| Prediction window | Feb 1, 2026 – Apr 7, 2026 |

### Top 5 SHAP Drivers (Global Feature Importance)

| Rank | Feature | Plain Meaning |
|---|---|---|
| 1 | `ewa_usage_count` | How many times user used EWA before |
| 2 | `ewa_total_amount` | Total dollars withdrawn via EWA |
| 3 | `ewa_avg_amount` | Average amount per EWA withdrawal |
| 4 | `spend_trend_w2_vs_w1` | Whether spending is trending up or down |
| 5 | `disc_active_days_30d` | Days with discretionary spending in last 30 days |

---

## Endpoints

### 1. Health Check
```
GET /health
```
Returns whether the API is running and the model is loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "feature_count": 34
}
```

---

### 2. Model Info
```
GET /model-info
```
Returns model metadata including features, metrics, and time windows.

---

### 3. Batch Predict
```
POST /predict
```
Accepts a batch of users with their pre-computed features and returns predictions.

**Request Body:**
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

### 4. User Risk Lookup ⚠️ Requires user_features_api.csv
```
GET /user_risk/{user_id}
```
Returns predicted EWA count and demand tier for a single user by ID.

**Example:** `GET /user_risk/Mymo1002027`

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "predicted_ewa_count": 3.2,
  "demand_tier": "High",
  "predicted_score": 1.4821
}
```

---

### 5. Top Features Lookup ⚠️ Requires user_features_api.csv
```
GET /top_features/{user_id}?top_n=5
```
Returns top SHAP drivers explaining the prediction for a single user.

**Example:** `GET /top_features/Mymo1002027?top_n=5`

**Response:**
```json
{
  "user_id": "Mymo1002027",
  "predicted_ewa_count": 3.2,
  "top_drivers": [
    {
      "rank": 1,
      "feature": "ewa_usage_count",
      "shap": 0.52,
      "value": 8.0,
      "direction": "increases EWA demand"
    }
  ]
}
```

---

### 6. High Risk Users ⚠️ Requires user_features_api.csv
```
GET /high_risk_users?threshold=0.7
```
Returns all users in the top risk percentile. Default threshold = 0.7 (top 30%).

**Response:**
```json
{
  "threshold_percentile": "top 30%",
  "total_high_risk_users": 3596,
  "users": [
    {
      "user_id": "Mymo1002027",
      "predicted_ewa_count": 3.2,
      "predicted_score": 1.4821
    }
  ]
}
```

> **Note:** Endpoints 4, 5, and 6 require `user_features_api.csv` to be present in the API folder. This file contains pre-computed user features and is pending data privacy approval before being added to the repository.

---

## Demand Tiers

| Tier | Definition | Business Action |
|---|---|---|
| **High** | Top 30% predicted score | Reserve higher liquidity |
| **Medium** | Middle 40% predicted score | Standard provisioning |
| **Low** | Bottom 30% predicted score | Minimal provisioning |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| ML model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Data processing | Pandas, NumPy |
| Validation | Pydantic |
| Deployment | Railway |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn api:app --reload --port 8000

# Open docs
open http://localhost:8000/docs
```

---

## Project Structure

```
API/
├── api.py                      # FastAPI app — all 6 endpoints
├── ewa_risk_model.json         # Trained XGBoost model
├── ewa_model_features.pkl      # Feature column list (34 features)
├── ewa_model_metadata.pkl      # Calibration multiplier + metrics
├── user_features_api.csv       # Pre-computed user features (pending approval)
├── requirements.txt            # Package dependencies
├── Procfile                    # Railway deployment config
├── dashboard.html              # Frontend dashboard (CSV upload)
└── .gitignore                  # Excludes raw data files
```

---

## What's Next

- [ ] Data privacy approval for `user_features_api.csv`
- [ ] Week 4: Error handling + edge case testing
- [ ] MCP wrapping for Claude integration
- [ ] Frontend connection to co-worker's Gradio app
