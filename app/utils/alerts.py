"""
alerts.py
---------
Forecast-driven Smart Alert System.

Overstock detection uses the SARIMA upper-confidence-bound total as the
demand threshold — more conservative and accurate than the avg-daily-sales
heuristic in the original app.py.
"""

import datetime as dt
from typing import Optional

import pandas as pd


def detect_alerts(
    product_summary: pd.DataFrame,
    forecast_df: pd.DataFrame,
    overstock_factor: float = 1.5,
    expiry_window: int = 7,
) -> list[dict]:
    """
    Scan product_summary against forecast_df and return a list of alert dicts.

    Parameters
    ----------
    product_summary   : columns product, current_inventory, expiry_date
    forecast_df       : columns Week, Forecast_Sales, Lower_95, Upper_95
    overstock_factor  : inventory > (Upper_95 total * factor) → overstock
    expiry_window     : days-to-expiry threshold for expiry alert

    Returns list of dicts with keys: type, product, severity, message, days_to_expiry (expiry only)
    """
    alerts = []

    if forecast_df is None or forecast_df.empty:
        return alerts

    # Total forecast demand (upper bound = conservative)
    try:
        forecast_total_upper = float(forecast_df["Upper_95"].sum())
        forecast_total_point = float(forecast_df["Forecast_Sales"].sum())
    except KeyError:
        forecast_total_upper = 0.0
        forecast_total_point = 0.0

    today = dt.date.today()

    for _, row in product_summary.iterrows():
        product = str(row["product"])
        inv = int(row.get("current_inventory", 0))
        exp_date = row.get("expiry_date", today + dt.timedelta(days=30))

        # --- Overstock ---
        threshold = forecast_total_upper * overstock_factor if forecast_total_upper > 0 else float("inf")
        if inv > threshold:
            surplus = inv - int(forecast_total_point)
            alerts.append({
                "type": "overstock",
                "product": product,
                "severity": "warning",
                "message": (
                    f"Inventory ({inv:,} units) exceeds {overstock_factor}× the forecasted "
                    f"upper-bound demand ({int(forecast_total_upper):,} units). "
                    f"Estimated surplus: {max(0, surplus):,} units."
                ),
                "current_inventory": inv,
                "forecast_upper": int(forecast_total_upper),
            })

        # --- Expiry ---
        if isinstance(exp_date, dt.date):
            days_to_expiry = (exp_date - today).days
            if days_to_expiry <= expiry_window:
                severity = "critical" if days_to_expiry <= 3 else "warning"
                label = "TODAY" if days_to_expiry == 0 else (
                    "TOMORROW" if days_to_expiry == 1 else f"in {days_to_expiry} day(s)"
                )
                alerts.append({
                    "type": "expiring",
                    "product": product,
                    "severity": severity,
                    "message": (
                        f"Expires {label} ({exp_date}). "
                        f"Current stock: {inv:,} units."
                    ),
                    "days_to_expiry": days_to_expiry,
                    "expiry_date": str(exp_date),
                })

    # Sort: critical first, then by type
    alerts.sort(key=lambda a: (0 if a["severity"] == "critical" else 1, a["type"]))
    return alerts


def alerts_to_dataframe(alerts: list[dict]) -> pd.DataFrame:
    """Flatten alert list to a display-ready DataFrame."""
    if not alerts:
        return pd.DataFrame(columns=["Product", "Type", "Severity", "Message"])
    rows = [
        {
            "Product": a["product"],
            "Type": a["type"].title(),
            "Severity": a["severity"].title(),
            "Message": a["message"],
        }
        for a in alerts
    ]
    return pd.DataFrame(rows)
