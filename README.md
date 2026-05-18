# Payactiv Risk Prediction System

**Author:** Mike Zhao  
**Institution:** Santa Clara University — Practicum Project  
**Sponsor:** Payactiv  
**Live API:** https://web-production-00c9c.up.railway.app  
**Interactive Docs:** https://web-production-00c9c.up.railway.app/docs  
**GitHub:** https://github.com/Mikeylove-997/payactiv-ewa-api  
**Frontend Dashboard:** https://huggingface.co/spaces/JoyZhan/EWAANDFD  

---

## Project Overview

This project delivers a complete end-to-end machine learning system for Payactiv, transforming raw financial transaction data into actionable predictions accessible through a live API, a business dashboard, and a Claude AI integration.

**Two core questions answered:**
1. How many EWA withdrawals will a user make next period?
2. Is this user at risk of financial distress?

---

## Data Sources

Three datasets spanning Nov 6, 2025 – Apr 8, 2026:

| Dataset | Description |
|---|---|
| Bank Accounts | Bank transaction records from Plaid (~728 MB) |
| Card Transactions | Payactiv credit card spending data (~63 MB) |
| EWA Records | EWA withdrawal history per user (~22 MB) |

Raw data stays on the local machine only. Only aggregated feature statistics are deployed.

---

## Machine Learning Models

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

| Detail | Value |
|---|---|
| Model type | Random Forest Classifier (best of 3 models tested) |
| Target variable | Financially distressed — Yes/No |
| Features | 23 pre-computed user features |
| ROC-AUC | 0.8705 |
| F1 Score | 0.683 |
| Users | 5,359 |
| Distress threshold | Spend > income on ≥30% of Window 2 days |
| Training window | Nov 6, 2025 – Jan 31, 2026 |
| Prediction window | Feb 1, 2026 – Mar 24, 2026 |

**Model Comparison:**

| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | 0.8630 | 0.674 |
| **Random Forest** | **0.8705** | **0.683** |
| XGBoost | 0.8575 | 0.647 |

**Top 5 Feature Importances:**

| Rank | Feature | Meaning |
|---|---|---|
| 1 | `card_to_bank_spend_ratio_30d` | Heavy card vs bank usage |
| 2 | `total_spend` | Overall spending level |
| 3 | `disc_active_days_30d` | Days with discretionary spending |
| 4 | `ewa_total_amount` | Total EWA withdrawn |
| 5 | `ewa_avg_amount` | Average EWA withdrawal amount |

---

## Combined Prediction Value

| EWA Demand | Distress | Meaning | Recommended Action |
|---|---|---|---|
| High | High | Heavy user + struggling | Urgent — reserve liquidity + outreach |
| High | Low | Heavy user, financially stable | Reserve liquidity only |
| Low | High | Struggling but not using EWA | Financial wellness outreach |
| Low | Low | Stable, low demand user | No immediate action needed |

---

## How Both Models Were Integrated Into One API

Initially only the EWA model existed. The Financial Distress model was added in a second phase without changing any existing endpoints.

**Integration steps:**

1. Added save cell to `Final Financial Distress Model.ipynb` → exported 4 artifact files
2. Copied artifacts into the API folder alongside EWA model files
3. Added FD model loading to the API startup function
4. Added 3 new endpoints + 1 Gradio adapter endpoint to `api.py`
5. Updated `.gitignore` to allow `fd_user_features_api.csv`
6. Pushed to GitHub → Railway redeployed automatically

No existing endpoints were modified. The Financial Distress endpoints are purely additive.

---

## API Endpoints (10 total)

| Endpoint | Model | Purpose |
|---|---|---|
| `GET /` | — | Welcome message |
| `GET /health` | — | System status |
| `GET /model-info` | — | EWA model metadata |
| `POST /predict` | EWA | Batch predictions + SHAP |
| `GET /user_risk/{user_id}` | EWA | Single user EWA risk |
| `GET /top_features/{user_id}` | EWA | Top SHAP drivers |
| `GET /high_risk_users` | EWA | All high demand users |
| `GET /distress_risk/{user_id}` | Distress | Single user distress |
| `GET /distress_features/{user_id}` | Distress | Distress drivers |
| `GET /high_distress_users` | Distress | All distressed users |

Full endpoint documentation with request/response examples: see `API_DOCUMENTATION.md`

---

## Who Uses What

| Audience | Tool | How |
|---|---|---|
| Business managers | Gradio dashboard | Enter User ID → instant results |
| Internal analysts | Claude Desktop + MCP | Plain English questions |
| Developers | Swagger UI `/docs` | Technical testing |
| Payactiv's system | Direct API calls | Automated monthly runs |

---

## Claude AI Integration (MCP)

Six tools registered in Claude Desktop via Model Context Protocol:

```
get_user_risk(user_id)          → EWA demand prediction
get_top_features(user_id)       → SHAP explanation
get_high_risk_users(threshold)  → Monthly planning list
get_distress_risk(user_id)      → Financial distress check
get_distress_features(user_id)  → Distress explanation
get_high_distress_users()       → All vulnerable users
```

Full MCP documentation and setup guide: see `MCP_DOCUMENTATION.md`

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
| Frontend | Gradio (Hugging Face Spaces) |
| AI integration | MCP (Claude Desktop) |

---

## Running Locally

```bash
git clone https://github.com/Mikeylove-997/payactiv-ewa-api
cd payactiv-ewa-api
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
open http://localhost:8000/docs
```

---

## Project Structure

```
EWA API/
├── api.py                       # FastAPI app — 10 endpoints
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
├── README.md                    # This file
├── API_DOCUMENTATION.md         # Full endpoint reference
├── MCP_DOCUMENTATION.md         # MCP setup + tools guide
└── .gitignore                   # Excludes raw data files
```

---

## Deliverables

| Deliverable | Status |
|---|---|
| Unified data pipeline | ✓ Complete |
| EWA Demand model (XGBoost) | ✓ Complete |
| Financial Distress model (Random Forest) | ✓ Complete |
| REST API — 10 endpoints deployed | ✓ Complete |
| MCP server — Claude Desktop integration | ✓ Complete |
| Gradio frontend dashboard | ✓ Complete |
| GitHub repository with documentation | ✓ Complete |
