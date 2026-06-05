# oil_gas_pipeline | dashboard/pages/3_model_comparison.py
# Page 3 — SARIMA vs Prophet side-by-side model comparison
# Shows RMSE, MAPE, R² metrics and forecast value comparison

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

def get_conn():
    return psycopg2.connect(**DB)

st.title("⚖️ Model Comparison")
st.caption("SARIMA vs Prophet — accuracy metrics and forecast comparison")
st.divider()

st.sidebar.header("🔧 Settings")
target = st.sidebar.selectbox("Target",
    options=["wti_price", "henry_hub_price"],
    format_func=lambda x: "WTI Crude Oil" if x == "wti_price" else "Henry Hub Gas")

@st.cache_data(ttl=300)
def load_data(target_col):
    conn = get_conn()
    fc   = pd.read_sql(f"SELECT * FROM gold_forecast_results WHERE target='{target_col}'", conn)
    hist = pd.read_sql("SELECT * FROM gold_energy_prices ORDER BY period ASC", conn)
    conn.close()
    fc["forecast_period"] = pd.to_datetime(fc["forecast_period"])
    hist["period"]        = pd.to_datetime(hist["period"])
    return fc, hist

with st.spinner("Loading..."):
    try:
        forecast_df, historical = load_data(target)
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

if forecast_df.empty:
    st.warning("No forecast data. Run forecasting models first.")
    st.stop()

sarima_df  = forecast_df[forecast_df["model_name"] == "sarima"]
prophet_df = forecast_df[forecast_df["model_name"] == "prophet"]

st.subheader("📊 Accuracy Metrics")
col1, col2 = st.columns(2)

for col, m_name, m_df in [(col1, "SARIMA", sarima_df), (col2, "Prophet", prophet_df)]:
    with col:
        st.markdown(f"### {m_name}")
        rmse = m_df["rmse"].dropna().mean() if "rmse" in m_df.columns else None
        mape = m_df["mape"].dropna().mean() if "mape" in m_df.columns else None
        m1, m2 = st.columns(2)
        with m1:
            st.metric("RMSE", f"{rmse:.4f}" if rmse and rmse > 0 else "N/A")
        with m2:
            st.metric("MAPE", f"{mape:.2f}%" if mape and mape > 0 else "N/A")

st.divider()
st.subheader("📈 Forecast Comparison")

fig = go.Figure()
if not historical.empty:
    tail = historical.tail(36)
    fig.add_trace(go.Scatter(x=tail["period"], y=tail[target],
                             name="Actual", line=dict(color="#1f77b4", width=2.5)))

if not sarima_df.empty:
    s = sarima_df.sort_values("forecast_period")
    fig.add_trace(go.Scatter(x=s["forecast_period"], y=s["forecast_value"],
                             name="SARIMA", line=dict(color="#d62728", width=2, dash="dash")))

if not prophet_df.empty:
    p = prophet_df.sort_values("forecast_period")
    fig.add_trace(go.Scatter(x=p["forecast_period"], y=p["forecast_value"],
                             name="Prophet", line=dict(color="#ff7f0e", width=2, dash="dot")))

if not historical.empty:
    cutoff = historical["period"].iloc[-1]
    fig.add_shape(type="line",
                  x0=cutoff, x1=cutoff,
                  y0=0, y1=1,
                  yref="paper",
                  line=dict(dash="dot", color="gray", width=1))
    fig.add_annotation(x=cutoff, y=1, yref="paper",
                       text="Forecast Start", showarrow=False,
                       font=dict(size=10, color="gray"))

unit = "$/barrel" if target == "wti_price" else "$/MMBtu"
fig.update_layout(height=430, hovermode="x unified", yaxis_title=unit,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

if not sarima_df.empty and not prophet_df.empty:
    st.subheader("📉 SARIMA vs Prophet Difference")
    merged = pd.merge(
        sarima_df[["forecast_period","forecast_value"]].rename(columns={"forecast_value":"sarima"}),
        prophet_df[["forecast_period","forecast_value"]].rename(columns={"forecast_value":"prophet"}),
        on="forecast_period")
    merged["difference"] = merged["sarima"] - merged["prophet"]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in merged["difference"]]
    fig2 = go.Figure(go.Bar(x=merged["forecast_period"], y=merged["difference"], marker_color=colors))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(height=250, yaxis_title="Difference ($)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Positive = SARIMA forecasts higher | Negative = Prophet forecasts higher")

    with st.expander("📋 Raw Forecast Values"):
        merged["forecast_period"] = pd.to_datetime(merged["forecast_period"]).dt.strftime("%Y-%m")
        merged.columns = ["Period","SARIMA ($)","Prophet ($)","Difference ($)"]
        st.dataframe(merged, use_container_width=True)