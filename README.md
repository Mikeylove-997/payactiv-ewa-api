# Payactiv Risk Prediction API

A FastAPI backend serving two ML models for EWA demand forecasting and financial distress prediction.

**Live API:** `https://web-production-00c9c.up.railway.app`  
**Interactive Docs:** `https://web-production-00c9c.up.railway.app/docs`

---

## What It Does

Two models working together to give Payactiv a complete picture of each user:

| Model | Question it answers | Business use |
|---|---|---|
| **EWA Demand** | How many withdrawals will this user make? | Liquidity planning |
| **Financial Distress** | Is this user financially struggling? | User support & outreach |

---

## Models

### Model 1 — EWA Demand Forecasting

| Detail | Value |
|---|---|
| Model type | XGBoost Regressor |
| Target variable | log(EWA withdrawal count + 1) |
| Features | 34 pre-computed user features |
| Calibration multiplier | 1.3938 |
| R² | 0.6143 |
| Spearman correlation | 0.7548 |
| Users | 11,988 |
| Training window | Nov 6, 2025 – Jan 31, 2026 |
| Prediction window | Feb 1, 2026 – Apr 7, 2026 |

**Top 5 SHAP Drivers:**

| Rank | Feature | Meaning |
|---|---|---|
| 1 | `ewa_usage_count` | How many times user used EWA before |
| 2 | `ewa_total_amount` | Total dollars withdrawn via EWA |
| 3 | `ewa_avg_amount` | Average amount per EWA withdrawal |
| 4 | `spend_trend_w2_vs_w1` | Whether spending is trending up or down |
| 5 | `disc_active_days_30d` | Days with discretionary spending in last 30 days |

---

### Model 2 — Financial Distress Prediction

| Detail | Value |
|---|---|
| Model type | Random Forest Classifier |
| Target variable | Financially distressed (Yes/No) |
| Features | 23 pre-computed user features |
| ROC-AUC | 0.8705 |
| F1 Score | 0.683 |
| Users | 5,359 |
| Distress threshold | Spend > income on ≥30% of Window 2 days |
| Training window | Nov 6, 2025 – Jan 31, 2026 |
| Prediction window | Feb 1, 2026 – Mar 24, 2026 |

**Top 5 Feature Importances:**

| Rank | Feature | Meaning |
|---|---|---|
| 1 | `card_to_bank_spend_ratio_30d` | Heavy card vs bank usage |
| 2 | `total_spend` | Overall spending level |
| 3 | `disc_active_days_30d` | Days with discretionary spending |
| 4 | `ewa_total_amount` | Total EWA withdrawn |
| 5 | `ewa_avg_amount` | Average EWA withdrawal amount |

---

## Endpoints

### System

#### `GET /`
Welcome message and links to docs.

#### `GET /health`
Returns whether the API is running and both models are loaded.

```json
{ "status": "ok", "model_loaded": true, "feature_count": 34 }
```

#### `GET /model-info`
Returns EWA model metadata including features, metrics, and time windows.

---

### EWA Demand Model

#### `POST /predict`
Batch prediction for multiple users. Send pre-computed features, get predictions + SHAP explanations back.

#### `GET /user_risk/{user_id}`
Predicted EWA withdrawal count and demand tier for one user.

**Example:** `GET /user_risk/Mymo1002027`
```json
{
  "user_id": "Mymo1002027",
  "predicted_ewa_count": 20.48,
  "demand_tier": "High",
  "predicted_score": 2.7531
}
```

#### `GET /top_features/{user_id}?top_n=5`
Top SHAP drivers explaining why a user has high or low EWA demand.

```json
{
  "user_id": "Mymo1002027",
  "predicted_ewa_count": 20.48,
  "top_drivers": [
    { "rank": 1, "feature": "ewa_usage_count", "shap": 0.891, "value": 62.0, "direction": "increases EWA demand" }
  ]
}
```

#### `GET /high_risk_users?threshold=0.7`
All users in the top risk percentile. Default = top 30%.

```json
{
  "threshold_percentile": "top 30%",
  "total_high_risk_users": 3597,
  "users": [{ "user_id": "Mymo1002027", "predicted_ewa_count": 20.48, "predicted_score": 2.7531 }]
}
```

---

### Financial Distress Model

#### `GET /distress_risk/{user_id}`
Financial distress probability and label for one user.

**Example:** `GET /distress_risk/Mymo1002027`
```json
{
  "user_id": "Mymo1002027",
  "distress_probability": 0.172,
  "financially_distressed": "No",
  "risk_level": "Low"
}
```

#### `GET /distress_features/{user_id}?top_n=5`
Top features driving the financial distress prediction for one user.

```json
{
  "user_id": "Mymo1002027",
  "distress_probability": 0.172,
  "financially_distressed": "No",
  "top_drivers": [
    { "rank": 1, "feature": "card_to_bank_spend_ratio_30d", "importance": 0.089, "value": 0.5 }
  ]
}
```

#### `GET /high_distress_users?min_probability=0.5`
All users predicted to be financially distressed above the probability threshold.

```json
{
  "min_probability": 0.5,
  "total_distressed_users": 1316,
  "users": [{ "user_id": "Mymo1003871", "distress_probability": 0.94, "risk_level": "High" }]
}
```

---

## Combined View (Both Models)

The real power comes from combining both predictions:

| EWA Demand | Distress | Meaning | Action |
|---|---|---|---|
| High | High | Heavy user + struggling | Urgent — liquidity + outreach |
| High | Low | Heavy user, financially stable | Reserve liquidity only |
| Low | High | Struggling but not using EWA | Wellness outreach |
| Low | Low | Stable user | No action needed |

---

## Claude Integration (MCP)

This API is connected to Claude Desktop via MCP. Claude can call all 6 user-facing tools directly:

```
get_user_risk(user_id)
get_top_features(user_id)
get_high_risk_users(threshold)
get_distress_risk(user_id)
get_distress_features(user_id)
get_high_distress_users(min_probability)
```

**Example conversation:**
> "Is user Mymo1002027 financially distressed?"  
> Claude → calls `/distress_risk/Mymo1002027` → returns answer in plain English

---

## Tech Stack

| Layer | Tool |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| EWA model | XGBoost |
| Distress model | Random Forest |
| Explainability | SHAP |
| Data processing | Pandas, NumPy |
| Validation | Pydantic |
| Deployment | Railway |
| AI integration | MCP (Claude Desktop) |

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
open http://localhost:8000/docs
```

---

## Project Structure

```
EWA API/
├── api.py                       # FastAPI app — 9 endpoints
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
└── .gitignore                   # Excludes raw data files
```
