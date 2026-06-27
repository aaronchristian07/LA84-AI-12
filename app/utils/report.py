"""
report.py
---------
Demand report generator.

Produces a self-contained HTML file with embedded Plotly charts (CDN-linked JS).
No headless browser or LaTeX required.
"""

import datetime as dt
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _forecast_fig(
    historical: pd.Series,
    forecast_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=historical.index,
        y=historical.values,
        mode="lines",
        name="Historical Sales",
        line=dict(color="#4F86C6"),
    )
    if forecast_df is not None and not forecast_df.empty:
        weeks = pd.to_datetime(forecast_df["Week"])
        fig.add_scatter(
            x=weeks,
            y=forecast_df["Forecast_Sales"],
            mode="lines",
            name="Forecast",
            line=dict(color="#E05C5C", dash="dash"),
        )
        fig.add_scatter(
            x=list(weeks) + list(weeks[::-1]),
            y=list(forecast_df["Upper_95"]) + list(forecast_df["Lower_95"][::-1]),
            fill="toself",
            fillcolor="rgba(224,92,92,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    fig.update_layout(
        title="Sales History + 12-Week Forecast",
        xaxis_title="Date",
        yaxis_title="Sales (units)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _stl_fig(stl_result) -> Optional[go.Figure]:
    if stl_result is None:
        return None
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Trend", "Seasonal", "Residual"])
    fig.add_scatter(y=stl_result.trend, mode="lines", name="Trend", row=1, col=1)
    fig.add_scatter(y=stl_result.seasonal, mode="lines", name="Seasonal", row=2, col=1)
    fig.add_scatter(y=stl_result.resid, mode="markers", name="Residual",
                    marker=dict(size=3), row=3, col=1)
    fig.update_layout(height=500, title="STL Decomposition", template="plotly_white",
                      showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def generate_report(
    sales_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame],
    alerts: list[dict],
    metrics: dict,
    historical_series: Optional[pd.Series] = None,
    stl_result=None,
) -> str:
    """
    Returns a self-contained HTML string.

    Parameters
    ----------
    metrics : dict with keys total_sales, current_inventory, excess_inventory
    """
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # KPI block
    kpi_html = f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Total Historic Sales</div>
            <div class="kpi-value">{metrics.get('total_sales', 0):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Current Inventory</div>
            <div class="kpi-value">{metrics.get('current_inventory', 0):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Estimated Excess Inventory</div>
            <div class="kpi-value">{metrics.get('excess_inventory', 0):,}</div>
        </div>
    </div>
    """

    # Forecast chart
    forecast_chart_html = ""
    if historical_series is not None and not historical_series.empty:
        fig = _forecast_fig(historical_series, forecast_df)
        forecast_chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    # STL chart
    stl_chart_html = ""
    if stl_result is not None:
        stl_fig = _stl_fig(stl_result)
        if stl_fig:
            stl_chart_html = pio.to_html(stl_fig, full_html=False, include_plotlyjs=False)

    # Alerts table
    alert_rows = ""
    for a in alerts:
        color = "#c0392b" if a.get("severity") == "critical" else "#e67e22"
        alert_rows += f"""
        <tr>
            <td><strong>{a['product']}</strong></td>
            <td style="color:{color}">{a['type'].title()}</td>
            <td>{a['severity'].title()}</td>
            <td>{a['message']}</td>
        </tr>"""

    alerts_html = f"""
    <table>
        <thead>
            <tr><th>Product</th><th>Type</th><th>Severity</th><th>Details</th></tr>
        </thead>
        <tbody>{alert_rows if alert_rows else '<tr><td colspan="4">No alerts.</td></tr>'}</tbody>
    </table>
    """

    # Forecast table
    forecast_table_html = ""
    if forecast_df is not None and not forecast_df.empty:
        rows_html = "".join(
            f"<tr><td>{r['Week']}</td>"
            f"<td>{r['Forecast_Sales']:,.0f}</td>"
            f"<td>{r['Lower_95']:,.0f}</td>"
            f"<td>{r['Upper_95']:,.0f}</td></tr>"
            for _, r in forecast_df.iterrows()
        )
        forecast_table_html = f"""
        <h2>12-Week Forecast</h2>
        <table>
            <thead><tr><th>Week</th><th>Forecast</th><th>Lower 95%</th><th>Upper 95%</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Retail Demand Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 40px auto; padding: 0 24px;
         color: #222; background: #fff; }}
  h1   {{ color: #1a1a2e; border-bottom: 3px solid #4F86C6; padding-bottom: 8px; }}
  h2   {{ color: #1a1a2e; margin-top: 40px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 32px; }}
  .kpi-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 24px 0; }}
  .kpi-card {{ flex: 1; min-width: 180px; background: #f4f8ff;
               border-left: 4px solid #4F86C6; padding: 16px 20px;
               border-radius: 4px; }}
  .kpi-label {{ font-size: 0.8em; color: #555; text-transform: uppercase;
                letter-spacing: 0.05em; }}
  .kpi-value {{ font-size: 2em; font-weight: 700; color: #1a1a2e; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th    {{ background: #1a1a2e; color: #fff; padding: 10px 12px; text-align: left; }}
  td    {{ border-bottom: 1px solid #eee; padding: 8px 12px; }}
  tr:hover td {{ background: #f9f9f9; }}
  footer {{ margin-top: 60px; font-size: 0.8em; color: #999;
            border-top: 1px solid #eee; padding-top: 16px; }}
</style>
</head>
<body>
<h1>Retail Demand Report</h1>
<p class="meta">Generated: {generated_at}</p>

<h2>Key Performance Indicators</h2>
{kpi_html}

<h2>Sales History &amp; Forecast</h2>
{forecast_chart_html if forecast_chart_html else '<p>No historical series provided.</p>'}

{forecast_table_html}

<h2>Seasonal Decomposition</h2>
{stl_chart_html if stl_chart_html else '<p>STL decomposition not available for this dataset.</p>'}

<h2>Smart Alerts</h2>
{alerts_html}

<footer>Retail Demand Prediction System — prototype. Not for production use without model validation.</footer>
</body>
</html>"""

    return html
