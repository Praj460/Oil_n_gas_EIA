# oil_gas_pipeline | dashboard/pages/3_model_comparison.py
# SARIMA vs Prophet side by side - metrics table + 2024 forecast vs actual chart

import sys
sys.path.insert(0, "/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline")

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

@st.cache_data(ttl=60)
def load_forecasts():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(
        "SELECT target, model_name, forecast_period, forecast_value, "
        "rmse, mae, mape FROM gold_forecast_results "
        "ORDER BY forecast_period ASC", conn)
    conn.close()
    df["forecast_period"] = pd.to_datetime(df["forecast_period"])
    return df

@st.cache_data(ttl=60)
def load_actual_2024():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(
        "SELECT period, wti_price, henry_hub_price FROM gold_energy_prices "
        "WHERE period >= '2024-01-01' AND period <= '2024-12-01' "
        "ORDER BY period ASC", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df

forecasts = load_forecasts()
actual    = load_actual_2024()

# -- Sidebar -----------------------------------------------------------

st.sidebar.header("⚖️ Comparison Settings")
target = st.sidebar.selectbox(
    "Target",
    ["wti_price", "henry_hub_price"],
    format_func=lambda x: "WTI Crude Oil" if x == "wti_price" else "Henry Hub Gas",
)

unit  = "$/barrel" if target == "wti_price" else "$/MMBtu"
title = "WTI Crude Oil" if target == "wti_price" else "Henry Hub Natural Gas"

st.title(f"⚖️ Model Comparison — {title}")
st.caption("SARIMA vs Prophet, evaluated on 2024 forecasts vs actual data")

# -- Metrics table -----------------------------------------------------

st.header("📊 Accuracy Metrics (2024 validation)")

val = forecasts[
    (forecasts["target"] == target) &
    (forecasts["forecast_period"] >= "2024-01-01") &
    (forecasts["forecast_period"] <= "2024-12-01") &
    (forecasts["rmse"].notna())
]

if val.empty:
    st.info("No metrics found. Run: python3 scripts/5_calculate_metrics.py")
else:
    metrics = (val.groupby("model_name")[["rmse", "mae", "mape"]]
                  .first().reset_index())
    metrics.columns = ["Model", f"RMSE ({unit})", f"MAE ({unit})", "MAPE (%)"]
    metrics["Model"] = metrics["Model"].str.upper()
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    best = metrics.loc[metrics["MAPE (%)"].idxmin(), "Model"]
    st.success(f"🏆 Best model on MAPE: **{best}**")

st.divider()

# -- 2024 Forecast vs Actual chart -------------------------------------

st.header("📈 2024 — Forecast vs Actual")

fig = go.Figure()

# Actual line
fig.add_trace(go.Scatter(
    x=actual["period"], y=actual[target],
    mode="lines+markers", name="Actual 2024",
    line=dict(color="#3b82f6", width=3),
))

# Each model's 2024 forecast
colors = {"sarima": "#f97316", "prophet": "#a855f7"}
for model in ["sarima", "prophet"]:
    fc = forecasts[
        (forecasts["target"] == target) &
        (forecasts["model_name"] == model) &
        (forecasts["forecast_period"] >= "2024-01-01") &
        (forecasts["forecast_period"] <= "2024-12-01")
    ]
    if not fc.empty:
        fig.add_trace(go.Scatter(
            x=fc["forecast_period"], y=fc["forecast_value"],
            mode="lines+markers", name=f"{model.upper()} Forecast",
            line=dict(color=colors[model], width=2, dash="dash"),
        ))

fig.update_layout(
    yaxis_title=unit,
    height=520,
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Blue = what actually happened in 2024. "
    "Dashed lines = what each model predicted. "
    "Closer to blue = better model."
)
