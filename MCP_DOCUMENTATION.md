# MCP Server Documentation — Payactiv Risk Prediction

**GitHub:** https://github.com/Mikeylove-997/payactiv-ewa-api  
**MCP Server File:** `mcp_server.py`  
**Live API:** https://web-production-00c9c.up.railway.app  

---

## What is MCP?

MCP (Model Context Protocol) is a standard created by Anthropic that allows Claude AI to call external tools and APIs automatically during a conversation. It acts as a bridge between Claude and your backend systems.

**Without MCP:**
```
User → types question → Claude answers from its own knowledge only
```

**With MCP:**
```
User → asks "Who are the high risk users?"
Claude → automatically calls your Railway API
Claude → returns real predictions in plain English
```

MCP is not AI itself — it is a connector that makes two systems compatible:

```
Claude (AI)  ←→  MCP Server  ←→  Your Railway API  ←→  ML Models
```

---

## How MCP Works in This Project

### The Pipeline

```
1. User asks a question in Claude Desktop (plain English)
        ↓
2. Claude identifies which MCP tool to use
        ↓
3. mcp_server.py receives the tool call
        ↓
4. mcp_server.py makes an HTTP request to Railway API
        ↓
5. Railway API runs the XGBoost / Random Forest model
        ↓
6. Prediction returned to mcp_server.py
        ↓
7. mcp_server.py formats the result as plain text
        ↓
8. Claude presents the answer to the user
```

### What Makes This Work

The MCP server (`mcp_server.py`) is a lightweight Python file that:
- Defines tools using the `@mcp.tool()` decorator
- Each tool maps to one endpoint on the Railway API
- Uses `httpx` to make HTTP requests to the live API
- Returns human-readable text back to Claude

The key design decision: the MCP server contains **no ML logic** — it only calls the Railway API. All intelligence stays in the backend.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT MACHINE                        │
│                                                         │
│   Claude Desktop App                                    │
│         ↕  (MCP Protocol)                              │
│   mcp_server.py  ──────────────────────────────────→  │
│                         HTTP requests                   │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                    RAILWAY (CLOUD)                       │
│                                                         │
│   FastAPI (api.py)                                      │
│         ↕                                              │
│   XGBoost Model (EWA)                                  │
│   Random Forest Model (Financial Distress)              │
│   user_features_api.csv (11,988 users)                 │
│   fd_user_features_api.csv (5,359 users)               │
└─────────────────────────────────────────────────────────┘
```

---

## MCP Tools Available

The MCP server exposes 6 tools that Claude can call automatically:

### EWA Demand Tools

#### `get_user_risk(user_id)`
Calls `GET /user_risk/{user_id}` on the Railway API.

**When Claude uses it:** When asked about a specific user's EWA demand or withdrawal prediction.

**Example question:** "What's the EWA risk for user Mymo1002027?"

**Claude's response:**
```
User Mymo1002027:
  Predicted EWA withdrawals: 20.48
  Demand tier: High
  Risk score: 2.7531
```

---

#### `get_top_features(user_id, top_n=5)`
Calls `GET /top_features/{user_id}` on the Railway API.

**When Claude uses it:** When asked why a user is high risk or what's driving their EWA demand.

**Example question:** "Why is user Mymo1002027 flagged as high risk?"

**Claude's response:**
```
Top 5 drivers for user Mymo1002027 (predicted 20.48 withdrawals):
  1. ewa_usage_count = 62.0 → increases EWA demand (SHAP: 0.891)
  2. ewa_total_amount = 3626.0 → increases EWA demand (SHAP: 0.527)
  3. ewa_avg_amount = 58.48 → increases EWA demand (SHAP: 0.104)
  4. spend_trend_w2_vs_w1 = 139.19 → increases EWA demand (SHAP: 0.067)
  5. merchant_churn_rate = 0.5 → decreases EWA demand (SHAP: -0.059)
```

---

#### `get_high_risk_users(threshold=0.7, limit=10)`
Calls `GET /high_risk_users` on the Railway API.

**When Claude uses it:** When asked about monthly planning, which users need attention, or who to prioritize.

**Example question:** "Who are the top 10 high risk users this month?"

**Claude's response:**
```
High risk users (top 30%):
Total: 3,597 users

Top 10:
  Mymo1002027 → 20.48 withdrawals (score: 2.7531)
  Mymo1003871 → 18.32 withdrawals (score: 2.6198)
  ...
```

---

### Financial Distress Tools

#### `get_distress_risk(user_id)`
Calls `GET /distress_risk/{user_id}` on the Railway API.

**When Claude uses it:** When asked if a user is financially distressed or struggling financially.

**Example question:** "Is user Mymo1002027 financially distressed?"

**Claude's response:**
```
User Mymo1002027 Financial Distress:
  Distress probability: 17.2%
  Financially distressed: No
  Risk level: Low
```

---

#### `get_distress_features(user_id, top_n=5)`
Calls `GET /distress_features/{user_id}` on the Railway API.

**When Claude uses it:** When asked what's causing a user's financial distress.

**Example question:** "What's driving user Mymo1003871's financial distress?"

**Claude's response:**
```
Top 5 distress drivers for user Mymo1003871 (probability: 94.0%):
  1. card_to_bank_spend_ratio_30d = 0.89 (importance: 0.089)
  2. total_spend = 12400.0 (importance: 0.071)
  3. disc_active_days_30d = 28.0 (importance: 0.065)
  4. ewa_total_amount = 4200.0 (importance: 0.058)
  5. ewa_avg_amount = 350.0 (importance: 0.051)
```

---

#### `get_high_distress_users(min_probability=0.5, limit=10)`
Calls `GET /high_distress_users` on the Railway API.

**When Claude uses it:** When asked about financially vulnerable users or who needs financial support.

**Example question:** "Who are the most financially distressed users?"

**Claude's response:**
```
Financially distressed users (probability ≥ 50%):
Total: 1,316 users

Top 10:
  Mymo1003871 → 94.0% probability (High risk)
  Mymo1007906 → 91.2% probability (High risk)
  ...
```

---

## How MCP Was Built

### Step 1 — Install the MCP package
```bash
pip install mcp httpx
```

### Step 2 — Create `mcp_server.py`

The server is structured around the `FastMCP` class:

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

Key points:
- `@mcp.tool()` registers the function as a tool Claude can call
- The docstring tells Claude **when** to use the tool — this is critical
- `httpx.get()` makes the actual HTTP call to Railway
- The function returns plain text that Claude reads and summarizes

### Step 3 — Register in Claude Desktop config

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

Config file location:
- Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

### Step 4 — Restart Claude Desktop

Claude loads the MCP server on startup. The 6 tools become available automatically in every conversation.

---

## What MCP Is and Isn't

| MCP is | MCP is not |
|---|---|
| A connector between Claude and your API | An AI model |
| A local Python file on the client's machine | A cloud service |
| A tool registry Claude can call | A replacement for the API |
| Free to use | Dependent on Claude subscription |

---

## Who Should Use MCP vs Other Interfaces

| Audience | Best Interface | Reason |
|---|---|---|
| Business managers | Gradio dashboard | No setup, visual, easy |
| Internal analysts | Claude Desktop + MCP | Plain English queries, powerful |
| Developers | Swagger UI `/docs` | Technical testing |
| Automated systems | Direct API calls | No human involvement |

---

## Sample MCP Conversation

**User:** "Give me a full risk profile for user Mymo1002027"

**Claude (calls 3 tools automatically):**
```
Here's the full risk profile for Mymo1002027:

EWA Demand:
  Predicted withdrawals: 20.48
  Demand tier: High

Financial Distress:
  Distress probability: 17.2%
  Risk level: Low

Interpretation:
This user is a heavy EWA user (High demand) but is not financially 
distressed. They frequently use EWA as a financial tool but are 
managing their finances well. Recommended action: reserve higher 
liquidity for this user but no outreach needed.

Top drivers for EWA demand:
  1. ewa_usage_count = 62 → strong predictor of future demand
  2. ewa_total_amount = $3,626 → high historical withdrawal amount
  3. spend_trend_w2_vs_w1 = rising → spending is increasing
```

This entire response required zero manual API calls — Claude handled everything automatically.
