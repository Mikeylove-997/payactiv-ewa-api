# Payactiv Risk Prediction System — Integration Report

**Author:** Mike Zhao  
**Institution:** Santa Clara University — MS Business Analytics Practicum  
**Sponsor:** Payactiv  
**Date:** May 2026  
**Live API:** https://web-production-00c9c.up.railway.app  
**GitHub:** https://github.com/Mikeylove-997/payactiv-ewa-api  
**Frontend:** https://huggingface.co/spaces/JoyZhan/EWAANDFD  

---

## 1. Executive Summary

This report documents the complete integration pipeline built for the Payactiv practicum project — from trained machine learning models in Jupyter notebooks to a live, AI-accessible prediction system.

The system delivers two core predictions for each Payactiv user:
1. **EWA Demand** — how many EWA withdrawals a user will make next period
2. **Financial Distress** — whether a user will be financially distressed next period

These predictions are now accessible through three interfaces:
- A **REST API** deployed on Railway (for developers and automated systems)
- A **Gradio dashboard** on Hugging Face (for business users)
- A **Claude AI integration** via MCP (for analysts using natural language)

---

## 2. Starting Point — The Notebooks

Before the API was built, both models existed only as Jupyter notebooks on a local machine. This meant:

- Only the notebook author could run predictions
- Running predictions required loading ~800MB of raw CSV files
- Each run took 30–60 minutes to complete
- Results were trapped inside the notebook
- No one else could access the predictions without running the full pipeline

**The goal of this integration:** make both models accessible to anyone, instantly, without touching the notebooks or raw data.

---

## 3. Phase 1 — EWA Demand API

### 3.1 Saving Model Artifacts

The first step was extracting the trained EWA model from the notebook into reusable files. A save cell was added to `EWA Demand Forcasting Model.ipynb`:

```python
# Save XGBoost model
xgb_reg.save_model('ewa_risk_model.json')

# Save feature list
pickle.dump(feature_cols, open('ewa_model_features.pkl', 'wb'))

# Save metadata
pickle.dump(metadata, open('ewa_model_metadata.pkl', 'wb'))

# Export user features for API lookup
model_df[['user_id'] + feature_cols].to_csv('user_features_api.csv', index=False)
```

This produced 4 artifact files:

| File | Contents | Size |
|---|---|---|
| `ewa_risk_model.json` | Trained XGBoost model weights | 577 KB |
| `ewa_model_features.pkl` | List of 34 feature names | 787 bytes |
| `ewa_model_metadata.pkl` | Calibration multiplier + metrics | 1.2 KB |
| `user_features_api.csv` | Pre-computed features for 11,988 users | 3.6 MB |

**Key design decision:** The raw transaction CSVs (728MB+ total) are never included. Only the trained model and aggregated feature statistics are deployed.

### 3.2 Building the FastAPI Application

`api.py` was built using FastAPI with the following design principles:

- **Single model state object** — both models load once at startup and stay in memory
- **Pydantic validation** — all inputs are validated before reaching the model
- **SHAP integration** — TreeExplainer runs at startup for fast per-request SHAP values
- **Calibration multiplier** — raw log-scale predictions are back-transformed and calibrated
- **Negative count fix** — `np.maximum(..., 0)` ensures predictions are always ≥ 0

**Initial endpoints (6):**

```
GET  /              Welcome message
GET  /health        System status check
GET  /model-info    EWA model metadata
POST /predict       Batch predictions with SHAP
GET  /user_risk/{user_id}     Single user EWA risk lookup
GET  /top_features/{user_id}  Top SHAP drivers for one user
GET  /high_risk_users         All high demand users ranked
```

### 3.3 Deployment to Railway

The API was deployed to Railway (a cloud hosting platform) using:

- `Procfile` — tells Railway how to start the server:
  ```
  web: uvicorn api:app --host 0.0.0.0 --port $PORT
  ```
- `requirements.txt` — all package dependencies
- GitHub integration — Railway redeploys automatically on every `git push`

**Live URL:** `https://web-production-00c9c.up.railway.app`

### 3.4 Testing and Bug Fixes

After deployment, the following issues were identified and fixed:

| Issue | Fix |
|---|---|
| Predicted EWA count could be negative | Wrapped output in `np.maximum(..., 0)` |
| Invalid feature values returned 500 error | Added try/except with clear 422 error message |
| Root URL returned "Not Found" | Added `GET /` welcome endpoint |
| CORS blocked dashboard requests | Added `CORSMiddleware` to FastAPI app |

---

## 4. Phase 2 — Financial Distress Model Integration

### 4.1 Model Selection

The Financial Distress notebook (`Final Financial Distress Model.ipynb`) trained and compared three classifiers:

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8630 | 0.6014 | 0.7674 | 0.6743 |
| **Random Forest** | **0.8705** | **0.6419** | **0.7297** | **0.6830** |
| XGBoost Classifier | 0.8575 | 0.5951 | 0.7093 | 0.6472 |

**Random Forest was selected** as the best performer (highest ROC-AUC and F1).

### 4.2 Saving Financial Distress Artifacts

A save cell was added to the Financial Distress notebook:

```python
# Save Random Forest model
pickle.dump(rf, open('fd_risk_model.pkl', 'wb'))

# Save feature list (23 features)
pickle.dump(feature_cols, open('fd_model_features.pkl', 'wb'))

# Save metadata
pickle.dump(fd_metadata, open('fd_model_metadata.pkl', 'wb'))

# Export user features (5,359 users)
model_df[['user_id'] + feature_cols].to_csv('fd_user_features_api.csv', index=False)
```

**Note:** The Financial Distress model uses 23 features (not 34 like EWA) and covers 5,359 users (not 11,988) because it requires data in both Window 1 and Window 2.

### 4.3 Adding Endpoints to Existing API

**No existing endpoints were changed.** Three new endpoints and one Gradio adapter were added to `api.py`:

```
GET  /distress_risk/{user_id}      Financial distress probability + label
GET  /distress_features/{user_id}  Top feature importances for distress
GET  /high_distress_users          All users above distress threshold
POST /distress/predict             Adapter endpoint for Gradio dashboard
```

The `ModelState` class was extended to hold both models:

```python
class ModelState:
    # EWA model (existing)
    model: XGBRegressor
    explainer: shap.TreeExplainer
    feature_cols: list[str]
    calibration_multiplier: float
    user_data: pd.DataFrame        # 11,988 users

    # Financial Distress model (new)
    fd_model: RandomForestClassifier
    fd_feature_cols: list[str]
    fd_user_data: pd.DataFrame     # 5,359 users
```

Both models load at startup from their artifact files and stay in memory for fast inference.

### 4.4 Final API — 10 Endpoints

```
GET  /                              Welcome
GET  /health                        System status
GET  /model-info                    EWA model metadata
POST /predict                       Batch EWA predictions + SHAP
GET  /user_risk/{user_id}           EWA risk for one user
GET  /top_features/{user_id}        EWA SHAP drivers for one user
GET  /high_risk_users               All high EWA demand users
GET  /distress_risk/{user_id}       Distress probability for one user
GET  /distress_features/{user_id}   Distress drivers for one user
GET  /high_distress_users           All financially distressed users
POST /distress/predict              Gradio dashboard adapter
```

---

## 5. Phase 3 — MCP Server

### 5.1 What MCP Does

MCP (Model Context Protocol) is a standard created by Anthropic that allows Claude AI to call external tools automatically during a conversation. It bridges plain English questions to real API calls.

```
User asks:  "Who are the high risk users this month?"
Claude:     automatically calls /high_risk_users
Claude:     returns answer in plain English — no manual API call needed
```

### 5.2 How the MCP Server Was Built

`mcp_server.py` was created using the `FastMCP` library:

```python
from mcp.server.fastmcp import FastMCP
import httpx

API_BASE = "https://web-production-00c9c.up.railway.app"
mcp = FastMCP("Payactiv EWA Demand API")

@mcp.tool()
def get_user_risk(user_id: str) -> str:
    """
    Get the predicted EWA withdrawal count and demand tier for a specific user.
    Use this when asked about a specific user's EWA risk or demand level.
    """
    response = httpx.get(f"{API_BASE}/user_risk/{user_id}", timeout=30)
    data = response.json()
    return f"User {data['user_id']}: {data['predicted_ewa_count']} withdrawals, {data['demand_tier']} demand"

if __name__ == "__main__":
    mcp.run()
```

**Key design points:**
- `@mcp.tool()` registers each function as a tool Claude can call
- The docstring tells Claude **when** to use the tool — this is critical for correct tool selection
- `httpx` makes HTTP calls to the Railway API
- The MCP server contains no ML logic — all intelligence stays in the backend
- The server runs locally on the client's machine and calls the cloud API

### 5.3 Six MCP Tools Registered

| Tool | API Endpoint Called | Use Case |
|---|---|---|
| `get_user_risk(user_id)` | `GET /user_risk/{user_id}` | Check one user's EWA demand |
| `get_top_features(user_id)` | `GET /top_features/{user_id}` | Explain why a user is high risk |
| `get_high_risk_users(threshold)` | `GET /high_risk_users` | Monthly planning list |
| `get_distress_risk(user_id)` | `GET /distress_risk/{user_id}` | Check one user's distress |
| `get_distress_features(user_id)` | `GET /distress_features/{user_id}` | Explain distress drivers |
| `get_high_distress_users()` | `GET /high_distress_users` | All vulnerable users |

### 5.4 Registering with Claude Desktop

The MCP server was registered in Claude Desktop's configuration file:

```json
{
  "mcpServers": {
    "payactiv-ewa-api": {
      "command": "python3",
      "args": ["/Users/linyuzhao/Desktop/EWA API/mcp_server.py"]
    }
  }
}
```

After restarting Claude Desktop, all 6 tools become available automatically in every conversation.

### 5.5 Full MCP Pipeline

```
1. User types question in Claude Desktop (plain English)
2. Claude identifies the appropriate MCP tool
3. Claude calls mcp_server.py with the required parameters
4. mcp_server.py sends HTTP request to Railway API
5. Railway API loads the pre-trained model from memory
6. Model runs prediction on user features from CSV lookup
7. Prediction returned to mcp_server.py
8. mcp_server.py formats result as plain text
9. Claude presents the answer to the user in natural language
```

---

## 6. Phase 4 — Frontend Dashboard

### 6.1 Original Dashboard Issues

The Gradio dashboard was originally built by Joy Zhan on Hugging Face Spaces. The original version had several issues:

- Used a hardcoded formula instead of the real ML model:
  ```python
  risk_score = (total_spend / 10000) * 0.7 + (txn_count / 100) * 0.3
  ```
- Required 10+ manual input fields per user
- Showed only Financial Distress, not EWA demand
- Called `/distress/predict` which didn't exist in our API

### 6.2 Integration Fix

Three changes were made to connect the dashboard to the real backend:

**Change 1:** Set `API_BASE_URL` environment variable in Hugging Face Space settings to point to the Railway API.

**Change 2:** Added `POST /distress/predict` adapter endpoint to the API to match what the dashboard calls. If the user ID exists in the database, it uses the real Random Forest model. Otherwise it falls back to an estimation.

**Change 3:** Rewrote `gradio_app.py` to simplify the interface:
- Replaced 10+ input fields with a single User ID field
- Added both EWA demand and Financial Distress results side by side
- Added top SHAP drivers display
- Added recommended action based on combined results

### 6.3 Final Dashboard Flow

```
User enters User ID → clicks "Get Prediction"
        ↓
Dashboard calls 3 Railway API endpoints simultaneously:
  GET /user_risk/{user_id}
  GET /distress_risk/{user_id}
  GET /top_features/{user_id}
        ↓
Dashboard displays:
  - EWA demand tier + predicted count
  - Financial distress probability + risk level
  - Top 5 SHAP drivers
  - Recommended action
```

---

## 7. System Architecture — Complete Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER (local only)                    │
│  Bank CSV (~728MB) + Card CSV (~63MB) + EWA CSV (~22MB)     │
│                          ↓                                   │
│              Jupyter Notebooks (feature engineering)         │
│                          ↓                                   │
│         Model artifacts + user feature CSVs saved           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    API LAYER (Railway cloud)                  │
│                                                             │
│   FastAPI (api.py) — 10 endpoints                          │
│   ├── XGBoost model (EWA) — 11,988 users                   │
│   ├── Random Forest model (Distress) — 5,359 users         │
│   └── SHAP explainer                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Gradio     │  │    Claude    │  │  Swagger UI  │
│  Dashboard   │  │  Desktop +   │  │    /docs     │
│ (Hugging     │  │    MCP       │  │ (Developers) │
│   Face)      │  │ (Analysts)   │  │              │
│ (Managers)   │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 8. Key Technical Decisions

| Decision | Reason |
|---|---|
| FastAPI over Flask | Auto-generated Swagger UI, Pydantic validation, async support |
| Railway over AWS | Simpler setup, auto-deploy from GitHub, free tier sufficient |
| Single API for both models | Simpler architecture, shared startup, one URL to share |
| Percentile-based demand tiers | Consistent with notebook methodology |
| Calibration multiplier for EWA | Corrects systematic under-prediction from log transform |
| Random Forest for distress | Best ROC-AUC (0.87) among 3 models tested |
| MCP over custom chatbot | Leverages existing Claude infrastructure, no extra UI needed |

---

## 9. Limitations and Future Work

| Limitation | Future Improvement |
|---|---|
| Tier thresholds are batch-relative | Save fixed thresholds from training set |
| User IDs exposed in CSV files | Anonymize before making repo fully public |
| MCP requires local setup per client | Host MCP server on cloud for easier access |
| Financial Distress covers 5,359 users vs 11,988 for EWA | Align user coverage across both models |
| No API authentication | Add API key authentication for production |

---

## 10. Contributors

| Name | Role | Contribution |
|---|---|---|
| **Mike Zhao** | Backend Engineer | ML models, REST API, Railway deployment, MCP server, documentation |
| **Joy Zhan** | Frontend Engineer | Gradio dashboard design and Hugging Face deployment |

**Academic Supervisor:** Professor Hou, Santa Clara University  
**Industry Sponsor:** Payactiv
