# Payactiv Risk Prediction System — Project Summary

**Author:** Mike Zhao (Mikeylove-997)  
**Institution:** Santa Clara University  
**Sponsor:** Payactiv  
**Live API:** https://web-production-00c9c.up.railway.app  
**GitHub:** https://github.com/Mikeylove-997/payactiv-ewa-api  
**Frontend:** https://huggingface.co/spaces/JoyZhan/EWAANDFD  

---

## 1. Project Overview

This project delivers a complete end-to-end machine learning system for Payactiv, transforming raw financial transaction data into actionable predictions. The system answers two core business questions:

1. **How many EWA withdrawals will a user make next period?** (EWA Demand Forecasting)
2. **Is a user at risk of financial distress?** (Financial Distress Prediction)

Both predictions are accessible through a live REST API, a business-friendly dashboard, and a Claude AI integration — making the models usable by technical and non-technical stakeholders alike.

---

## 2. Data Sources

Three datasets were used, spanning November 6, 2025 – April 8, 2026:

| Dataset | Description | Size |
|---|---|---|
| Bank Accounts | Bank transaction records from Plaid | ~728 MB |
| Card Transactions | Payactiv credit card spending data | ~63 MB |
| EWA Records | EWA withdrawal history per user | ~22 MB |

All raw data remains on the local machine and never enters the API or GitHub repository. Only aggregated, anonymized feature statistics are used in the deployed system.

---

## 3. Machine Learning Models

### Model 1 — EWA Demand Forecasting

**Objective:** Predict how many EWA withdrawals a user will make in the next period (Window 2: Feb 1 – Apr 7, 2026), based on their spending behavior in Window 1 (Nov 6, 2025 – Jan 31, 2026).

**Model Type:** XGBoost Regressor  
**Target Variable:** log(EWA withdrawal count + 1)  
**Features:** 34 pre-computed user-level features  
**Users:** 11,988

**Performance Metrics:**

| Metric | Value |
|---|---|
| R² | 0.6143 |
| Spearman Correlation | 0.7548 |
| MAE (log scale) | 0.5562 |
| MAE (raw count) | 3.68 withdrawals |

**Top 5 SHAP Drivers (Global Feature Importance):**

| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | `ewa_usage_count` | Number of EWA withdrawals in Window 1 |
| 2 | `ewa_total_amount` | Total dollars withdrawn via EWA |
| 3 | `ewa_avg_amount` | Average amount per EWA withdrawal |
| 4 | `spend_trend_w2_vs_w1` | Whether spending is trending up or down |
| 5 | `disc_active_days_30d` | Days with discretionary spending in last 30 days |

**Demand Tiers:**

| Tier | Definition | Business Action |
|---|---|---|
| High | Top 30% predicted score | Reserve higher liquidity |
| Medium | Middle 40% predicted score | Standard provisioning |
| Low | Bottom 30% predicted score | Minimal provisioning needed |

---

### Model 2 — Financial Distress Prediction

**Objective:** Predict whether a user will be financially distressed in Window 2 (spend > income on ≥30% of days), based on Window 1 features.

**Model Type:** Random Forest Classifier (best among Logistic Regression, Random Forest, XGBoost)  
**Target Variable:** Binary — financially distressed (Yes/No)  
**Features:** 23 pre-computed user-level features  
**Users:** 5,359  
**Distress Threshold:** User is distressed if spend > income on ≥30% of Window 2 days

**Model Comparison:**

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8630 | 0.6014 | 0.7674 | 0.6743 |
| **Random Forest** | **0.8705** | **0.6419** | **0.7297** | **0.6830** |
| XGBoost | 0.8575 | 0.5951 | 0.7093 | 0.6472 |

**Top 5 Feature Importances:**

| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | `card_to_bank_spend_ratio_30d` | Heavy card vs bank usage |
| 2 | `total_spend` | Overall spending level |
| 3 | `disc_active_days_30d` | Days with discretionary spending |
| 4 | `ewa_total_amount` | Total EWA amount withdrawn |
| 5 | `ewa_avg_amount` | Average EWA withdrawal amount |

---

## 4. Combined Prediction Value

The real power comes from combining both models for each user:

| EWA Demand | Financial Distress | Meaning | Recommended Action |
|---|---|---|---|
| High | High | Heavy user + struggling financially | Urgent — reserve liquidity + outreach |
| High | Low | Heavy user, financially stable | Reserve liquidity only |
| Low | High | Struggling but not using EWA | Financial wellness outreach |
| Low | Low | Stable, low demand user | No immediate action needed |

---

## 5. System Architecture

```
Raw Data (Bank + Card + EWA)
        ↓
Feature Engineering (Jupyter Notebook)
        ↓
Trained ML Models (XGBoost + Random Forest)
        ↓
REST API (FastAPI + Railway)
        ↓
┌──────────────────────────────────────┐
│  Gradio Dashboard  │  Claude + MCP  │
│  (Hugging Face)    │  (Desktop App)  │
└──────────────────────────────────────┘
```

---

## 6. How the Two Models Were Integrated into One API

Initially, only the EWA Demand model was in the API. The Financial Distress model was added in a second phase without breaking any existing functionality.

**Changes made to integrate the second model:**

**Step 1 — Save Financial Distress artifacts from notebook**
A new cell was added at the end of `Final Financial Distress Model.ipynb` to export:
- `fd_risk_model.pkl` — trained Random Forest model
- `fd_model_features.pkl` — list of 23 feature names
- `fd_model_metadata.pkl` — metrics and config
- `fd_user_features_api.csv` — pre-computed features for 5,359 users

**Step 2 — Copy artifacts into the API folder**
All 4 files were copied into the `EWA API` folder alongside the existing EWA model files.

**Step 3 — Update `api.py`**
Three additions were made to the existing API file:
- Added new file paths for the FD model artifacts
- Added FD model loading in the startup function (runs once at launch)
- Added 3 new endpoints + 1 Gradio adapter endpoint

**Step 4 — Update `.gitignore`**
Added `!fd_user_features_api.csv` exception to allow the FD user features file on GitHub.

**Step 5 — Push to GitHub**
Railway automatically detected the changes and redeployed with both models running.

**No existing endpoints were changed.** The EWA model endpoints continue to work exactly as before. The Financial Distress endpoints are purely additive.

---

## 7. What Changed Between Versions

### Version 1 (EWA Only) — 6 endpoints
```
GET /
GET /health
GET /model-info
POST /predict
GET /user_risk/{user_id}
GET /top_features/{user_id}
GET /high_risk_users
```

### Version 2 (EWA + Financial Distress) — 10 endpoints
```
GET /                           ← unchanged
GET /health                     ← unchanged
GET /model-info                 ← unchanged
POST /predict                   ← unchanged
GET /user_risk/{user_id}        ← unchanged
GET /top_features/{user_id}     ← unchanged
GET /high_risk_users            ← unchanged
GET /distress_risk/{user_id}    ← NEW
GET /distress_features/{user_id}← NEW
GET /high_distress_users        ← NEW
POST /distress/predict          ← NEW (Gradio adapter)
```

---

## 8. Frontend Dashboard

The Gradio dashboard was built by a team member (Joy Zhan) and connected to this API.

**Original dashboard issues:**
- Used a hardcoded formula instead of the real ML model
- Had 10+ manual input fields
- Only showed Financial Distress, not EWA demand

**After integration:**
- Connected to the Railway API via `API_BASE_URL` environment variable
- Simplified to a single User ID input
- Shows both EWA demand and Financial Distress results together
- Displays top SHAP drivers
- Provides a recommended action based on combined results

---

## 9. Deliverables Completed

| Deliverable | Status |
|---|---|
| Unified data pipeline (notebooks) | ✓ Complete |
| EWA Demand model (XGBoost) | ✓ Complete |
| Financial Distress model (Random Forest) | ✓ Complete |
| REST API (10 endpoints, deployed) | ✓ Complete |
| MCP server (Claude Desktop integration) | ✓ Complete |
| Gradio frontend dashboard | ✓ Complete |
| GitHub repository with documentation | ✓ Complete |
| Final presentation | → Upcoming |

---

## 10. Technologies Used

| Layer | Technology |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Machine learning | XGBoost, Scikit-learn (Random Forest) |
| Explainability | SHAP (TreeExplainer) |
| API framework | FastAPI |
| API server | Uvicorn |
| Cloud deployment | Railway |
| Version control | GitHub |
| Frontend | Gradio (Hugging Face Spaces) |
| AI integration | MCP (Model Context Protocol) |
| AI assistant | Claude Desktop |
