"""
Payactiv EWA Demand MCP Server
Exposes the Railway API as tools Claude can call directly.
"""

import httpx
import json
from mcp.server.fastmcp import FastMCP

API_BASE = "https://web-production-00c9c.up.railway.app"

mcp = FastMCP("Payactiv EWA Demand API")


@mcp.tool()
def get_user_risk(user_id: str) -> str:
    """
    Get the predicted EWA withdrawal count and demand tier for a specific user.
    Use this when asked about a specific user's EWA risk or demand level.

    Args:
        user_id: The user's ID (e.g. Mymo1002027)
    """
    try:
        response = httpx.get(f"{API_BASE}/user_risk/{user_id}", timeout=30)
        if response.status_code == 404:
            return f"User '{user_id}' not found in the system."
        data = response.json()
        return (
            f"User {data['user_id']}:\n"
            f"  Predicted EWA withdrawals: {data['predicted_ewa_count']}\n"
            f"  Demand tier: {data['demand_tier']}\n"
            f"  Risk score: {data['predicted_score']}"
        )
    except Exception as e:
        return f"Error fetching user risk: {str(e)}"


@mcp.tool()
def get_top_features(user_id: str, top_n: int = 5) -> str:
    """
    Get the top SHAP drivers explaining why a user has high or low EWA demand.
    Use this when asked why a user is flagged as high risk or what's driving their EWA demand.

    Args:
        user_id: The user's ID (e.g. Mymo1002027)
        top_n: Number of top drivers to return (default: 5)
    """
    try:
        response = httpx.get(
            f"{API_BASE}/top_features/{user_id}",
            params={"top_n": top_n},
            timeout=30
        )
        if response.status_code == 404:
            return f"User '{user_id}' not found in the system."
        data = response.json()
        drivers = data["top_drivers"]
        result = f"Top {len(drivers)} drivers for user {user_id} (predicted {data['predicted_ewa_count']} withdrawals):\n"
        for d in drivers:
            result += f"  {d['rank']}. {d['feature']} = {d['value']} → {d['direction']} (SHAP: {d['shap']})\n"
        return result
    except Exception as e:
        return f"Error fetching top features: {str(e)}"


@mcp.tool()
def get_high_risk_users(threshold: float = 0.7, limit: int = 10) -> str:
    """
    Get the list of high risk users ranked by predicted EWA demand.
    Use this when asked about which users need attention, who to prioritize, or monthly planning.

    Args:
        threshold: Percentile cutoff (0.7 = top 30%, 0.9 = top 10%). Default: 0.7
        limit: Number of users to show (default: 10)
    """
    try:
        response = httpx.get(
            f"{API_BASE}/high_risk_users",
            params={"threshold": threshold},
            timeout=30
        )
        data = response.json()
        users = data["users"][:limit]
        result = (
            f"High risk users ({data['threshold_percentile']}):\n"
            f"Total: {data['total_high_risk_users']} users\n\n"
            f"Top {len(users)}:\n"
        )
        for u in users:
            result += f"  {u['user_id']} → {u['predicted_ewa_count']} withdrawals (score: {u['predicted_score']})\n"
        return result
    except Exception as e:
        return f"Error fetching high risk users: {str(e)}"


if __name__ == "__main__":
    mcp.run()
