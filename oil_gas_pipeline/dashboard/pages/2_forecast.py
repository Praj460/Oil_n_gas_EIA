# oil_gas_pipeline | dashboard/pages/2_forecast.py
# Page 2 — 12-month price forecast with confidence intervals
# Pulls from gold_forecast_results and gold_energy_prices


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

def get_conn():
    return psycopg2.connect(**DB)

st.title("🔮 Price Forecast")
st.caption("12-month ahead forecasts using Prophet model")
st.divider()

st.sidebar.header("🔧 Forecast Settings")
target = st.sidebar.selectbox("Forecast Target",
    options=["wti_price", "henry_hub_price"],
    format_func=lambda x: "WTI Crude Oil ($/barrel)" if x == "wti_price" else "Henry Hub Gas ($/MMBtu)")
show_history_months = st.sidebar.slider("Months of History to Show", 12, 120, 36, 6)
show_ci = st.sidebar.checkbox("Show Confidence Intervals", value=True)

@st.cache_data(ttl=300)
def load_forecast_data(target_col):
    conn = get_conn()
    df   = pd.read_sql(f"SELECT * FROM gold_forecast_results WHERE target='{target_col}' ORDER BY forecast_period ASC", conn)
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_historical(target_col, n_months):
    conn = get_conn()
    df   = pd.read_sql("SELECT * FROM gold_energy_prices ORDER BY period ASC", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df[["period", target_col]].dropna().tail(n_months)

with st.spinner("Loading forecast data..."):
    try:
        forecast_df = load_forecast_data(target)
        historical  = load_historical(target, show_history_months)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if forecast_df.empty:
    st.warning("No forecast data found. Run the forecasting models first.")
    st.stop()

forecast_df["forecast_period"] = pd.to_datetime(forecast_df["forecast_period"])

# KPI metrics
unit = "$/barrel" if target == "wti_price" else "$/MMBtu"
col1, col2, col3, col4 = st.columns(4)
latest_actual  = historical[target].iloc[-1] if not historical.empty else None
first_forecast = forecast_df["forecast_value"].iloc[0]
last_forecast  = forecast_df["forecast_value"].iloc[-1]
avg_mape       = forecast_df["mape"].dropna().mean() if "mape" in forecast_df.columns else None

with col1:
    st.metric("Last Actual", f"${latest_actual:.2f} {unit}" if latest_actual else "N/A")
with col2:
    change = ((first_forecast - latest_actual) / latest_actual * 100) if latest_actual else 0
    st.metric("Next Month Forecast", f"${first_forecast:.2f}", f"{change:+.1f}%")
with col3:
    st.metric("12M Forecast", f"${last_forecast:.2f}")
with col4:
    st.metric("Backtest MAPE", f"{avg_mape:.1f}%" if avg_mape and avg_mape > 0 else "N/A")

st.divider()

label = "WTI Crude Oil" if target == "wti_price" else "Henry Hub Natural Gas"
st.subheader(f"{label} — {show_history_months}M History + 12M Forecast")

fig = go.Figure()

if not historical.empty:
    fig.add_trace(go.Scatter(x=historical["period"], y=historical[target],
                             name="Actual", line=dict(color="#1f77b4", width=2.5)))

model_colors = {"prophet": "#ff7f0e", "sarima": "#d62728"}
for m_name in forecast_df["model_name"].unique():
    m_df  = forecast_df[forecast_df["model_name"] == m_name].sort_values("forecast_period")
    color = model_colors.get(m_name, "#9467bd")
    fig.add_trace(go.Scatter(x=m_df["forecast_period"], y=m_df["forecast_value"],
                             name=f"{m_name.upper()} Forecast",
                             line=dict(color=color, width=2, dash="dash")))
    if show_ci and "lower_bound" in m_df.columns and "upper_bound" in m_df.columns:
        x_fill = pd.concat([m_df["forecast_period"], m_df["forecast_period"][::-1]])
        y_fill = pd.concat([m_df["upper_bound"], m_df["lower_bound"][::-1]])
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself",
                                 fillcolor="rgba(255,127,14,0.15)",
                                 line=dict(color="rgba(255,255,255,0)"),
                                 name=f"{m_name.upper()} 95% CI", showlegend=True))

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

fig.update_layout(height=500, hovermode="x unified", yaxis_title=unit,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📋 Forecast Values")
display_df = forecast_df[["forecast_period","model_name","forecast_value","lower_bound","upper_bound"]].copy()
display_df["forecast_period"] = display_df["forecast_period"].dt.strftime("%Y-%m")
display_df.columns = ["Period","Model","Forecast","Lower (95%)","Upper (95%)"]
st.dataframe(display_df.style.format({
    "Forecast": "${:.2f}", "Lower (95%)": "${:.2f}", "Upper (95%)": "${:.2f}"
}), use_container_width=True, height=350)
st.download_button("⬇️ Download Forecast CSV", display_df.to_csv(index=False), "forecast.csv", "text/csv")