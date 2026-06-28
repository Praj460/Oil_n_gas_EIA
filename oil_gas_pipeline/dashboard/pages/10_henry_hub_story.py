# oil_gas_pipeline | dashboard/pages/10_henry_hub_story.py
# Henry Hub / Natural Gas story — the second target, with full model results.

import os
import numpy as np
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

st.set_page_config(page_title="Henry Hub Story", page_icon="🔥", layout="wide")
st.title("🔥 Henry Hub — the natural gas side of the story")
st.caption(
    "WTI's spike grabbed the headlines. Natural gas has its own, very different "
    "story — driven by weather, storage, and seasonality."
)


@st.cache_data(ttl=300)
def load():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
        SELECT period, henry_hub_price, gas_storage, gas_production,
               hdd, cdd, wti_price
        FROM gold_features ORDER BY period
    """, conn)
    fc = pd.read_sql("""
        SELECT model_name, forecast_period, forecast_value,
               lower_bound, upper_bound, rmse, mape, trained_on_periods
        FROM gold_forecast_results
        WHERE target = 'henry_hub_price'
        ORDER BY model_name, trained_on_periods, forecast_period
    """, conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    fc["forecast_period"] = pd.to_datetime(fc["forecast_period"])
    df["month"] = df["period"].dt.month
    return df, fc


df, fc = load()

# ── Headline cards ─────────────────────────────────────────────────────────
st.markdown("### At a glance")
c1, c2, c3, c4 = st.columns(4)
clean = df.dropna(subset=["henry_hub_price"])
c1.metric("Latest price",   f"${clean.iloc[-1]['henry_hub_price']:.2f} /MMBtu",
          help=clean.iloc[-1]["period"].strftime("%b %Y"))
c2.metric("10-year low",    f"${clean['henry_hub_price'].min():.2f}")
c3.metric("10-year high",   f"${clean['henry_hub_price'].max():.2f}")
c4.metric("Average",        f"${clean['henry_hub_price'].mean():.2f}")

st.divider()

# ── Why gas is different — seasonality fingerprint ─────────────────────────
st.markdown("## Gas is fundamentally different from oil")
st.markdown(
    "Oil prices are driven by global supply/demand and macro shocks. "
    "**Gas prices are driven by weather and storage** — a much more cyclical structure. "
    "The month-of-year pattern makes this obvious."
)

seasonal = (
    df.dropna(subset=["henry_hub_price"])
      .groupby("month")["henry_hub_price"]
      .mean().reset_index()
)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig_s = go.Figure(go.Bar(
    x=month_labels, y=seasonal["henry_hub_price"],
    marker_color="#d62728",
    text=[f"${v:.2f}" for v in seasonal["henry_hub_price"]],
    textposition="outside",
))
fig_s.update_layout(height=300, yaxis_title="Avg price ($/MMBtu)",
                    margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
st.plotly_chart(fig_s, use_container_width=True)

c1, c2 = st.columns(2)
c1.info("**Winter peaks (Jan–Mar):** heating demand pulls gas out of storage.")
c2.info("**Summer secondary peaks (Jul–Aug):** cooling load → power plants burn gas.")

st.divider()

# ── Weather as demand engine ───────────────────────────────────────────────
st.markdown("## Weather drives demand")
st.markdown(
    "Heating degree days (HDD) and cooling degree days (CDD) are the textbook "
    "gas-demand signals. When HDD is high, heating burns more gas. When CDD is high, "
    "power plants work harder and burn more gas for electricity. Both lift prices."
)

fig_w = go.Figure()
fig_w.add_trace(go.Scatter(x=df["period"], y=df["henry_hub_price"],
    name="Henry Hub ($/MMBtu)", yaxis="y1", line=dict(color="#d62728", width=2)))
fig_w.add_trace(go.Scatter(x=df["period"], y=df["hdd"],
    name="HDD", yaxis="y2", line=dict(color="#1f77b4", width=1.4), opacity=0.7))
fig_w.add_trace(go.Scatter(x=df["period"], y=df["cdd"],
    name="CDD", yaxis="y2", line=dict(color="#ff7f0e", width=1.4), opacity=0.7))
fig_w.update_layout(
    height=400, hovermode="x unified",
    yaxis=dict(title="Henry Hub ($/MMBtu)", color="#d62728"),
    yaxis2=dict(title="Degree days", overlaying="y", side="right", showgrid=False),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
)
st.plotly_chart(fig_w, use_container_width=True)

st.divider()

# ── Storage as the buffer ──────────────────────────────────────────────────
st.markdown("## Storage — gas's shock absorber")
st.markdown(
    "Gas storage is the buffer. It builds in summer (injection) and draws down in "
    "winter (withdrawal). Going into a cold winter with low storage is a price-spike "
    "setup — the market has no cushion."
)

fig_st = go.Figure()
fig_st.add_trace(go.Scatter(x=df["period"], y=df["henry_hub_price"],
    name="Henry Hub", yaxis="y1", line=dict(color="#d62728", width=2)))
fig_st.add_trace(go.Scatter(x=df["period"], y=df["gas_storage"],
    name="US gas storage (Bcf)", yaxis="y2", line=dict(color="#2ca02c", width=2)))
fig_st.update_layout(
    height=360, hovermode="x unified",
    yaxis=dict(title="Henry Hub ($/MMBtu)", color="#d62728"),
    yaxis2=dict(title="Storage (Bcf)", color="#2ca02c", overlaying="y",
                side="right", showgrid=False),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
)
st.plotly_chart(fig_st, use_container_width=True)

st.divider()

# ── Model results ──────────────────────────────────────────────────────────
st.markdown("## Model results — all variants compared")

MODEL_LABELS = {
    "sarima_hh":          "SARIMA Baseline",
    "sarimax_hh":         "SARIMAX + Gas Features (HDD / Storage / WTI)",
    "xgboost_hh_strict":  "XGBoost — Fair Multi-Step Forecast",
    "xgboost_hh_tuned":   "XGBoost Tuned — Best Fair Multi-Step",
    "xgboost_hh_curated": "XGBoost — 1-Step Operational (inflated, not comparable)",
}

# Filter to HH-specific models only — drops legacy sarima/prophet rows with null MAPE
HH_MODELS = list(MODEL_LABELS.keys())
fc = fc[fc["model_name"].isin(HH_MODELS)].copy()
fc["model_label"] = fc["model_name"].map(MODEL_LABELS)

WINDOWS = {96: "2024 calm year", 105: "Oct 2024 – Sep 2025 heating cycle"}
fc["window"] = fc["trained_on_periods"].map(WINDOWS)
fc = fc.dropna(subset=["window"])
windows = list(WINDOWS.values())

# Leaderboard — one full-width table per window, stacked vertically
st.markdown("### Leaderboard (MAPE — lower is better)")
for n, win_label in WINDOWS.items():
    st.markdown(f"**{win_label}**")
    sub = (fc[fc["trained_on_periods"] == n]
           .groupby("model_label")
           .agg(mape=("mape", "first"), rmse=("rmse", "first"))
           .reset_index()
           .sort_values("mape"))
    sub["mape"] = sub["mape"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
    sub["rmse"] = sub["rmse"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    sub.columns = ["Model", "MAPE", "RMSE ($/MMBtu)"]
    st.dataframe(sub, hide_index=True, use_container_width=True)

st.caption(
    "⚠️ XGBoost 1-step operational uses pre-engineered rolling features that "
    "incorporate sequential actual prices — appropriate for one-month-ahead "
    "operational forecasting, not for fair comparison against SARIMA's multi-step forecast. "
    "Use the strict variant for apples-to-apples."
)

# Forecast-vs-actual chart
st.markdown("### Forecast vs Actual")
chosen_win = st.radio("Window:", windows, horizontal=True)
sub = fc[fc["window"] == chosen_win].copy()

# Join actuals directly from gold_features df
actual_periods = pd.to_datetime(sub["forecast_period"].unique())
actual_df = (
    df.set_index("period")["henry_hub_price"]
      .reindex(actual_periods)
      .dropna()
      .reset_index()
)
actual_df.columns = ["period", "price"]

# Colors keyed to the NEW model labels
colors = {
    "SARIMA Baseline":                              "#1f77b4",
    "SARIMAX + Gas Features (HDD / Storage / WTI)":"#ff7f0e",
    "XGBoost — Fair Multi-Step Forecast":           "#2ca02c",
    "XGBoost Tuned — Best Fair Multi-Step":         "#9467bd",
    "XGBoost — 1-Step Operational (inflated, not comparable)": "#bcbd22",
}

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=actual_df["period"], y=actual_df["price"],
    name="Actual Henry Hub", line=dict(color="#d62728", width=3),
    mode="lines+markers",
))
for label, color in colors.items():
    md = sub[sub["model_label"] == label].sort_values("forecast_period")
    if md.empty:
        continue
    fig_fc.add_trace(go.Scatter(
        x=md["forecast_period"], y=md["forecast_value"],
        name=label, line=dict(color=color, dash="dot", width=2),
        mode="lines+markers",
    ))
fig_fc.update_layout(
    height=420, hovermode="x unified",
    yaxis_title="Henry Hub price ($/MMBtu)",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
)
st.plotly_chart(fig_fc, use_container_width=True)

st.divider()

# ── Annual storage cycle ───────────────────────────────────────────────────
st.markdown("## The annual storage cycle")
st.markdown(
    "Every year follows the same pattern: injection (Apr–Oct, storage builds) "
    "→ withdrawal (Nov–Mar, storage draws). Overlaying years on a common axis "
    "shows the rhythm clearly."
)

storage_yr = df.dropna(subset=["gas_storage"]).copy()
storage_yr["year"]  = storage_yr["period"].dt.year
storage_yr["month_num"] = storage_yr["period"].dt.month

fig_cy = go.Figure()
for yr in sorted(storage_yr["year"].unique()):
    yd = storage_yr[storage_yr["year"] == yr]
    if len(yd) < 6:
        continue
    fig_cy.add_trace(go.Scatter(
        x=yd["month_num"], y=yd["gas_storage"],
        name=str(yr), mode="lines+markers", opacity=0.55,
    ))
fig_cy.update_layout(
    height=360,
    xaxis=dict(tickmode="array", tickvals=list(range(1,13)), ticktext=month_labels),
    yaxis_title="US gas storage (Bcf)", hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=-0.18),
)
st.plotly_chart(fig_cy, use_container_width=True)
st.caption("Each line is one year. The shape is consistent — build through summer, draw through winter.")

st.divider()
st.markdown(
    "**Cross-target insight:** SARIMAX improved on Henry Hub (+0.63/+1.85 pp) "
    "but worsened on WTI (-0.14/-1.2 pp). Gas demand from HDD is roughly linear "
    "across the training range, so linear models can use the signal. Oil's 2026 spike "
    "was threshold-driven (OPEC spare crossing zero) — a nonlinear effect linear models "
    "can't learn. This asymmetry is the central finding across both targets."
)
