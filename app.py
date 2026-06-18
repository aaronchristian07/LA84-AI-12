import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Retail Demand Prediction 🛒", page_icon="🛒", layout="wide")


@st.cache_data
def generate_dummy_data(start_date: dt.date = None, years: int = 3, products: Optional[list] = None):
    """Generate a dummy daily sales dataset across several products for 2-5 years.

    Returns a DataFrame with columns: date, product, sales, inventory (current), expiry_date (per product)
    """
    if start_date is None:
        end = dt.date.today()
        start = end - dt.timedelta(days=365 * years)
    else:
        start = start_date
        end = start + dt.timedelta(days=365 * years)

    if products is None:
        products = [
            "Fresh Milk",
            "Sourdough Bread",
            "Organic Salad",
            "Yogurt",
            "Orange Juice",
        ]

    dates = pd.date_range(start, end, freq="D")
    rows = []
    rng = np.random.default_rng(42)

    for p_idx, product in enumerate(products):
        base = 20 + p_idx * 5
        trend = 0.0005 * (np.arange(len(dates)))  # small upward trend
        weekly = 5 * np.sin(2 * np.pi * (dates.dayofweek) / 7 + p_idx)
        seasonal = 3 * np.sin(2 * np.pi * (dates.dayofyear) / 365 + p_idx)
        noise = rng.normal(scale=3 + p_idx, size=len(dates))

        sales = np.clip(base + trend + weekly + seasonal + noise, 0, None).round().astype(int)

        for d, s in zip(dates, sales):
            rows.append({"date": d.date(), "product": product, "sales": int(s)})

    df = pd.DataFrame(rows)

    inventory = {}
    expiry = {}
    for product in products:
        inventory[product] = int(rng.integers(50, 500))
        expiry_days = int(rng.integers(1, 45))
        expiry[product] = dt.date.today() + dt.timedelta(days=expiry_days)

    # Attach current inventory and expiry columns (one row per product summary will be derived from this)
    product_summary = pd.DataFrame(
        [
            {"product": p, "current_inventory": inventory[p], "expiry_date": expiry[p]}
            for p in products
        ]
    )

    return df, product_summary


@st.cache_data
def load_uploaded_data(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attempt to load uploaded CSV/XLSX and normalize to (sales_df, product_summary).

    Expected minimal columns: `date`, `product`, `sales`. Optional columns: `current_inventory`, `expiry_date`.
    """
    if uploaded_file is None:
        return None, None

    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            raw = pd.read_csv(uploaded_file)
        else:
            raw = pd.read_excel(uploaded_file)
    except Exception as e:
        raise e

    df = raw.copy()
    # Ensure date column exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        # Try to infer from index
        if df.shape[1] >= 1:
            df.columns = [c.strip() for c in df.columns]

    # Required: product and sales
    if "product" not in df.columns:
        # if single product file, try to set a product name
        df["product"] = "Uploaded Product"

    if "sales" not in df.columns:
        # try to guess a numeric column
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            df["sales"] = df[numeric_cols[0]]
        else:
            df["sales"] = 0

    sales_df = df[[c for c in ["date", "product", "sales"] if c in df.columns]].copy()

    # Build product summary
    if "current_inventory" in df.columns and "expiry_date" in df.columns:
        prod_summary = (
            df[["product", "current_inventory", "expiry_date"]]
            .drop_duplicates(subset=["product"])
            .copy()
        )
        prod_summary["expiry_date"] = pd.to_datetime(prod_summary["expiry_date"]).dt.date
    else:
        products = sales_df["product"].unique().tolist()
        rng = np.random.default_rng(1)
        prod_summary = pd.DataFrame(
            [
                {
                    "product": p,
                    "current_inventory": int(rng.integers(30, 300)),
                    "expiry_date": dt.date.today() + dt.timedelta(days=int(rng.integers(1, 30))),
                }
                for p in products
            ]
        )

    return sales_df, prod_summary


def aggregate_sales(sales_df: pd.DataFrame, product: Optional[str] = None) -> pd.Series:
    df = sales_df.copy()
    if product:
        df = df[df["product"] == product]
    daily = df.groupby("date")["sales"].sum().sort_index()
    # ensure continuous index
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D").date
    daily = daily.reindex(idx, fill_value=0)
    daily.index = pd.to_datetime(daily.index)
    return daily


def generate_forecast(daily_series: pd.Series, days: int = 30) -> pd.Series:
    """Create a simple forecast combining recent weekly pattern + mean trend.

    This is a placeholder for RF/LSTM/SARIMAX UI options.
    """
    if daily_series.empty:
        return pd.Series([], dtype=float)

    last = daily_series[-60:] if len(daily_series) >= 60 else daily_series

    # weekly pattern: mean sales per weekday
    weekday_means = last.groupby(last.index.weekday).mean()

    start = daily_series.index.max() + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=days, freq="D")
    preds = []
    base_mean = float(last.mean())

    # small linear trend component from last 90 days
    if len(daily_series) >= 90:
        x = np.arange(len(daily_series[-90:]))
        y = daily_series[-90:].values
        a, b = np.polyfit(x, y, 1)
        trend_func = lambda i: a * (len(x) + i) + b
    else:
        trend_func = lambda i: base_mean

    rng = np.random.default_rng(123)
    for i, d in enumerate(dates):
        weekday = d.weekday()
        season = weekday_means.get(weekday, base_mean)
        trend = trend_func(i)
        noise = rng.normal(0, scale=max(1.0, 0.05 * base_mean))
        pred = max(0, season * 0.6 + 0.4 * trend + noise)
        preds.append(pred)

    forecast = pd.Series(data=np.round(preds, 0).astype(int), index=dates)
    return forecast


def detect_alerts(product_summary: pd.DataFrame, sales_df: pd.DataFrame) -> list[dict]:
    """Detect overstock and expiring soon items and return actionable alerts."""
    alerts = []
    lead_time_days = 7
    for _, row in product_summary.iterrows():
        product = row["product"]
        inv = int(row.get("current_inventory", 0))
        exp_date = row.get("expiry_date", dt.date.today() + dt.timedelta(days=30))

        # compute avg daily sales for last 30 days
        recent = aggregate_sales(sales_df, product=product)
        recent = recent[-30:]
        avg_daily = float(recent.mean()) if len(recent) else 0.0

        # Overstock: inventory > expected_use_in_lead_time * factor
        expected_use = avg_daily * lead_time_days
        if expected_use > 0 and inv > expected_use * 1.5:
            alerts.append(
                {
                    "type": "overstock",
                    "product": product,
                    "message": f"OVERSTOCK DETECTED: {product} — inventory={inv}, expected use in {lead_time_days}d={int(expected_use)}",
                }
            )

        # Expiring soon
        if isinstance(exp_date, (dt.date,)):
            days_to_expiry = (exp_date - dt.date.today()).days
            if days_to_expiry <= 7:
                alerts.append(
                    {
                        "type": "expiring",
                        "product": product,
                        "message": f"EXPIRING SOON: {product} in {days_to_expiry} day(s) (expiry: {exp_date})",
                    }
                )

    return alerts


def format_metrics(sales_df: pd.DataFrame, product_summary: pd.DataFrame) -> tuple[int, int, int]:
    total_sales = int(sales_df["sales"].sum())
    current_inventory = int(product_summary["current_inventory"].sum())
    # estimate food waste saved (dummy): assume 5% of total sales avoided
    saved = int(total_sales * 0.05)
    return total_sales, current_inventory, saved


def main():
    st.title("Retail Demand Prediction System — Minimize Food Waste")
    st.markdown(
        "This prototype shows analytics, simple forecasting, and a Smart Alert system to help optimize stock and reduce food waste."
    )

    # Sidebar: data upload and model selection
    st.sidebar.header("Configuration & Data")
    uploaded = st.sidebar.file_uploader("Upload sales (.csv/.xlsx)", type=["csv", "xlsx"])    
    model_choice = st.sidebar.selectbox("Prediction model (placeholder)", ["Random Forest", "LSTM", "SARIMAX"])
    st.sidebar.markdown("Model is a UI placeholder — a simple forecast is generated below for demo.")

    # Load data (uploaded or dummy)
    if uploaded is not None:
        try:
            sales_df, product_summary = load_uploaded_data(uploaded)
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded file: {e}")
            sales_df, product_summary = generate_dummy_data()
    else:
        sales_df, product_summary = generate_dummy_data(years=3)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Analytical Dashboard", "Demand Predictor", "Smart Alert"])    

    # --- Tab 1: Analytical Dashboard
    with tab1:
        st.header("Analytical Dashboard")
        total_sales, current_inventory, saved_estimate = format_metrics(sales_df, product_summary)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sales (historic)", f"{total_sales:,}")
        c2.metric("Current Inventory Level", f"{current_inventory:,}")
        c3.metric("Estimated Food Waste Saved", f"{saved_estimate:,}")

        st.markdown("---")
        st.markdown("#### Historical Sales — sample time series (by product)")
        product_to_plot = st.selectbox("Choose product to visualize", options=product_summary["product"].tolist())

        daily = aggregate_sales(sales_df, product=product_to_plot)
        fig = px.line(daily, x=daily.index, y=daily.values, labels={"x": "date", "y": "sales"}, title=f"Historical sales — {product_to_plot}")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Use the `Demand Predictor` tab to produce a short-term forecast.")

    # --- Tab 2: Prediction Engine
    with tab2:
        st.header("Demand Predictor")
        st.markdown("Upload data (sidebar) or use the generated dummy dataset. Select a product and press Generate Prediction.")
        sel_product = st.selectbox("Product to predict", options=product_summary["product"].tolist(), key="pred_product")

        cols = st.columns([2, 1])
        with cols[0]:
            days_ahead = st.number_input("Days to predict", min_value=7, max_value=180, value=30, step=1)
        with cols[1]:
            generate_btn = st.button("Generate Prediction")

        if generate_btn:
            with st.spinner("Generating prediction — this may take a few seconds..."):
                hist = aggregate_sales(sales_df, product=sel_product)
                forecast = generate_forecast(hist, days=int(days_ahead))

            combined = pd.concat([hist, forecast])
            combined = combined.sort_index()

            fig2 = px.line(title=f"Historical + Forecast for {sel_product}")
            fig2.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Historical")
            fig2.add_scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast")
            st.plotly_chart(fig2, use_container_width=True)

            # Predicted quantities per product (for demo we compute for all products)
            preds = []
            for p in product_summary["product"]:
                hist_p = aggregate_sales(sales_df, product=p)
                f_p = generate_forecast(hist_p, days=int(days_ahead))
                preds.append({"product": p, "avg_daily_pred": float(f_p.mean()) if not f_p.empty else 0.0, "total_pred": int(f_p.sum()) if not f_p.empty else 0})

            pred_df = pd.DataFrame(preds).sort_values("total_pred", ascending=False)
            st.markdown("### Predicted quantity needed per product")
            st.dataframe(pred_df)

            st.success("Prediction complete. Review the forecast chart and predicted table above.")

    # --- Tab 3: Smart Alert System
    with tab3:
        st.header("Sistem Smart Alert — Actionable Flags")
        st.markdown("The system scans inventory and expiry dates and raises alerts for items needing attention.")

        alerts = detect_alerts(product_summary, sales_df)

        if not alerts:
            st.info("No alerts detected — inventory and expiry look healthy.")
        else:
            for a in alerts:
                if a["type"] == "overstock":
                    st.warning(a["message"])
                elif a["type"] == "expiring":
                    st.error(a["message"])

        st.markdown("---")
        st.markdown("### Flagged Items — actionable list")
        if alerts:
            alert_df = pd.DataFrame(alerts)
            st.table(alert_df[["product", "type", "message"]])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for demo and educational purposes. Replace the simple forecast with model training for production.")


if __name__ == "__main__":
    main()
