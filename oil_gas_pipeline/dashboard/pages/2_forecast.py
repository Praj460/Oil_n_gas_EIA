# oil_gas_pipeline | dashboard/pages/2_forecast.py
# Forecast page - shows historical prices + model forecasts with confidence bands
# Two views: Validation (2024, graded vs actuals) and Future (next 12 months)

import sys
sys.path.insert(0, "/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline")

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from dotenv import load_dotenv

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
import os
load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
# -- Data loading ------------------------------------------------------

@st.cache_data(ttl=60)
def load_history():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(
        "SELECT period, wti_price, henry_hub_price "
        "FROM gold_energy_prices ORDER BY period ASC", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df

@st.cache_data(ttl=60)
def load_forecasts():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(
        "SELECT target, model_name, forecast_period, forecast_value, "
        "lower_bound, upper_bound, created_at "
        "FROM gold_forecast_results ORDER BY forecast_period ASC", conn)
    conn.close()
    df["forecast_period"] = pd.to_datetime(df["forecast_period"])
    return df

history   = load_history()
forecasts = load_forecasts()

last_actual = history["period"].max()

# -- Sidebar controls --------------------------------------------------

st.sidebar.header("🔮 Forecast Settings")

target = st.sidebar.selectbox(
    "Target",
    ["wti_price", "henry_hub_price"],
    format_func=lambda x: "WTI Crude Oil" if x == "wti_price" else "Henry Hub Gas",
)

model = st.sidebar.selectbox("Model", ["sarima", "prophet"],
                             format_func=str.upper)

view = st.sidebar.radio(
    "Forecast View",
    ["Future (next 12 months)", "Validation (2024 vs actual)"],
)

# -- Filter forecast rows ----------------------------------------------

fc = forecasts[
    (forecasts["target"] == target) &
    (forecasts["model_name"] == model)
]

if view.startswith("Future"):
    # Future = forecast periods AFTER the last actual data point
    fc = fc[fc["forecast_period"] > last_actual]
else:
    # Validation = 2024 forecasts (we have actuals to compare)
    fc = fc[(fc["forecast_period"] >= "2024-01-01") &
            (fc["forecast_period"] <= "2024-12-01")]

# -- Header ------------------------------------------------------------

unit  = "$/barrel" if target == "wti_price" else "$/MMBtu"
title = "WTI Crude Oil" if target == "wti_price" else "Henry Hub Natural Gas"

st.title(f"🔮 {title} Forecast")
st.caption(f"Model: {model.upper()} | View: {view}")

if fc.empty:
    st.warning(
        "No forecast rows found for this view. "
        "Run: PYTHONPATH=$(pwd) python3 models/future_forecaster.py"
    )
    st.stop()

# -- Metrics row -------------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("First forecast",
            f"${fc['forecast_value'].iloc[0]:.2f}",
            f"{fc['forecast_period'].iloc[0].strftime('%b %Y')}")
col2.metric("Last forecast",
            f"${fc['forecast_value'].iloc[-1]:.2f}",
            f"{fc['forecast_period'].iloc[-1].strftime('%b %Y')}")
col3.metric("Avg forecast", f"${fc['forecast_value'].mean():.2f}")

# -- Chart -------------------------------------------------------------

fig = go.Figure()

# Historical line (last 4 years for readability)
hist_recent = history[history["period"] >= "2021-01-01"]
fig.add_trace(go.Scatter(
    x=hist_recent["period"], y=hist_recent[target],
    name="Actual", line=dict(color="#3b82f6", width=2),
))

# Confidence band
fig.add_trace(go.Scatter(
    x=fc["forecast_period"], y=fc["upper_bound"],
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=fc["forecast_period"], y=fc["lower_bound"],
    fill="tonexty", fillcolor="rgba(249, 115, 22, 0.15)",
    line=dict(width=0), name="95% Confidence",
))

# Forecast line
fig.add_trace(go.Scatter(
    x=fc["forecast_period"], y=fc["forecast_value"],
    name=f"{model.upper()} Forecast",
    line=dict(color="#f97316", width=2, dash="dash"),
))

# Vertical marker where actual data ends
fig.add_shape(
    type="line",
    x0=last_actual, x1=last_actual, y0=0, y1=1,
    yref="paper",
    line=dict(color="gray", width=1, dash="dot"),
)
fig.add_annotation(
    x=last_actual, y=1.02, yref="paper",
    text="Last actual data", showarrow=False,
    font=dict(size=11, color="gray"),
)

fig.update_layout(
    yaxis_title=unit,
    height=520,
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=60),
)

st.plotly_chart(fig, use_container_width=True)

# -- Forecast table ----------------------------------------------------

st.subheader("Forecast Values")
table = fc[["forecast_period", "forecast_value", "lower_bound", "upper_bound"]].copy()
table["forecast_period"] = table["forecast_period"].dt.strftime("%b %Y")
table.columns = ["Month", f"Forecast ({unit})", "Lower 95%", "Upper 95%"]
st.dataframe(table, use_container_width=True, hide_index=True)
