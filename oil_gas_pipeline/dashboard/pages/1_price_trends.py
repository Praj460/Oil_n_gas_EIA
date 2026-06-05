# oil_gas_pipeline | dashboard/pages/1_price_trends.py
# Page 1 — Historical price trends for crude oil and natural gas
# Pulls from gold_energy_prices table
# Run with: streamlit run dashboard/app.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Price Trends", page_icon="📈", layout="wide")

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

def get_conn():
    return psycopg2.connect(**DB)

st.title("📈 Energy Price Trends")
st.caption("Historical crude oil and natural gas prices from EIA OpenData API")
st.divider()

@st.cache_data(ttl=300)
def load_gold_data():
    conn = get_conn()
    df   = pd.read_sql("SELECT * FROM gold_energy_prices ORDER BY period ASC", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df

with st.spinner("Loading price data..."):
    try:
        df = load_gold_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if df.empty:
    st.warning("No data in gold_energy_prices. Run the ingestion pipeline first.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔧 Filters")
min_date   = df["period"].min().date()
max_date   = df["period"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
show_wti    = st.sidebar.checkbox("WTI Crude Oil",    value=True)
show_brent  = st.sidebar.checkbox("Brent Crude Oil",  value=True)
show_hh     = st.sidebar.checkbox("Henry Hub Gas",    value=True)
show_spread = st.sidebar.checkbox("Brent-WTI Spread", value=False)
show_3m_avg = st.sidebar.checkbox("Show 3M Avg",      value=False)

if len(date_range) == 2:
    start, end = date_range
    df = df[(df["period"].dt.date >= start) & (df["period"].dt.date <= end)]

# KPI metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    v = df["wti_price"].dropna()
    st.metric("WTI Crude", f"${v.iloc[-1]:.2f}", f"{v.iloc[-1]-v.iloc[-2]:+.2f}" if len(v)>1 else "")
with col2:
    v = df["brent_price"].dropna()
    st.metric("Brent Crude", f"${v.iloc[-1]:.2f}", f"{v.iloc[-1]-v.iloc[-2]:+.2f}" if len(v)>1 else "")
with col3:
    v = df["henry_hub_price"].dropna()
    st.metric("Henry Hub", f"${v.iloc[-1]:.2f}", f"{v.iloc[-1]-v.iloc[-2]:+.2f}" if len(v)>1 else "")
with col4:
    v = df["oil_gas_ratio"].dropna()
    st.metric("Oil/Gas Ratio", f"{v.iloc[-1]:.1f}x" if len(v)>0 else "N/A")

st.divider()
st.subheader("Crude Oil & Natural Gas Prices")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Crude Oil Prices ($/barrel)", "Natural Gas Price ($/MMBtu)"),
                    row_heights=[0.6, 0.4])

if show_wti and "wti_price" in df.columns:
    fig.add_trace(go.Scatter(x=df["period"], y=df["wti_price"], name="WTI Crude",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
if show_brent and "brent_price" in df.columns:
    fig.add_trace(go.Scatter(x=df["period"], y=df["brent_price"], name="Brent Crude",
                             line=dict(color="#ff7f0e", width=2)), row=1, col=1)
if show_spread and "price_spread" in df.columns:
    fig.add_trace(go.Scatter(x=df["period"], y=df["price_spread"], name="Brent-WTI Spread",
                             line=dict(color="#9467bd", width=1.5), fill="tozeroy",
                             fillcolor="rgba(148,103,189,0.1)"), row=1, col=1)
if show_hh and "henry_hub_price" in df.columns:
    fig.add_trace(go.Scatter(x=df["period"], y=df["henry_hub_price"], name="Henry Hub",
                             line=dict(color="#2ca02c", width=2)), row=2, col=1)

events = [
    {"date": "2016-01-01", "label": "Oil Price Crash"},
    {"date": "2020-03-01", "label": "COVID-19"},
    {"date": "2022-02-01", "label": "Russia-Ukraine War"},
]
for event in events:
    fig.add_shape(type="line",
                  x0=event["date"], x1=event["date"],
                  y0=0, y1=1, yref="paper",
                  line=dict(dash="dot", color="gray", width=1),
                  row=1, col=1)

fig.update_layout(height=550, hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  margin=dict(l=0, r=0, t=40, b=0))
fig.update_yaxes(title_text="$/barrel", row=1, col=1)
fig.update_yaxes(title_text="$/MMBtu",  row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# MoM change chart
st.subheader("Month-over-Month % Change")
if "wti_mom_change" in df.columns:
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in df["wti_mom_change"].fillna(0)]
    fig2 = go.Figure(go.Bar(x=df["period"], y=df["wti_mom_change"],
                            marker_color=colors, name="WTI MoM %"))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="% Change")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("📋 View Raw Data"):
    display_cols = [c for c in ["period","wti_price","brent_price","price_spread",
                                 "henry_hub_price","oil_gas_ratio","wti_mom_change"] if c in df.columns]
    st.dataframe(df[display_cols].sort_values("period", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=300)
    st.download_button("⬇️ Download CSV", df[display_cols].to_csv(index=False),
                       "energy_prices.csv", "text/csv")