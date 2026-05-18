# MCP Quick Start Guide

Connect your Claude Desktop to the Payactiv Risk Prediction API in 5 steps.

Once set up, you can ask Claude questions like:
- "What's the EWA risk for user Mymo1002027?"
- "Who are the top high risk users this month?"
- "Is user Mymo1002233 financially distressed?"
- "Why is this user flagged as high risk?"

---

## Prerequisites

- [Claude Desktop](https://claude.ai/download) installed
- Python 3.9+ installed
- Terminal / Command Prompt

---

## Setup Steps

**Step 1 — Clone the repository**
```bash
git clone https://github.com/Mikeylove-997/payactiv-ewa-api
cd payactiv-ewa-api
```

**Step 2 — Install dependencies**
```bash
pip install mcp httpx
```

**Step 3 — Find your config file**

On Mac:
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

On Windows:
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Step 4 — Add the MCP server**

Open the config file and add the following. Replace `/your/path/to` with the actual path where you cloned the repo:

```json
{
  "mcpServers": {
    "payactiv-ewa-api": {
      "command": "python3",
      "args": ["/your/path/to/payactiv-ewa-api/mcp_server.py"]
    }
  }
}
```

**Step 5 — Restart Claude Desktop**

Quit Claude Desktop completely and reopen it. The tools will load automatically on startup.

---

## Available Tools

Once connected, Claude can call these tools automatically:

| Tool | What it does |
|---|---|
| `get_user_risk` | Predicted EWA withdrawals + demand tier for a user |
| `get_top_features` | Top SHAP drivers explaining a user's EWA demand |
| `get_high_risk_users` | All high demand users ranked by predicted withdrawals |
| `get_distress_risk` | Financial distress probability for a user |
| `get_distress_features` | Top features driving a user's distress prediction |
| `get_high_distress_users` | All financially distressed users |

---

## Example Questions to Ask Claude

**Single user lookup:**
> "What's the EWA demand for user Mymo1002027?"

> "Is user Mymo1002952 financially distressed?"

> "Why is user Mymo1002027 flagged as high risk?"

**Batch queries:**
> "Give me the top 10 high risk users this month"

> "How many users are financially distressed?"

> "Who should we prioritize for outreach this month?"

**Combined view:**
> "Give me a full risk profile for user Mymo1002027"

---

## Troubleshooting

**Claude doesn't respond with real data:**
- Make sure you restarted Claude Desktop after editing the config
- Check the file path in the config is correct

**"Module not found" error:**
- Run `pip install mcp httpx` again
- Make sure you're using the same Python that Claude Desktop uses

**"User not found" response:**
- Check the user ID is correct — IDs start with `Mymo` followed by numbers
- Example valid ID: `Mymo1002027`

---

## No Installation? Use the Dashboard Instead

If you prefer not to set up MCP, use the Gradio dashboard — no installation required:

**Dashboard:** https://huggingface.co/spaces/JoyZhan/EWAANDFD

Just enter a User ID and get instant predictions from both models.
