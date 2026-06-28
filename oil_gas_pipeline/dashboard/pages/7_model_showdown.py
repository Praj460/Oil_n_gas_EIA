# oil_gas_pipeline | dashboard/pages/7_model_showdown.py
# Model Showdown — head-to-head of all model variants across both test windows.
# Reads from gold_forecast_results, joined against actual WTI from gold_features.

import os
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

st.set_page_config(page_title="Model Showdown", page_icon="🥊", layout="wide")
st.title("🥊 Model Showdown")
st.caption("Head-to-head comparison of every model variant on the same test windows. Same train, same test, same metrics.")


@st.cache_data(ttl=300)
def load_data():
    conn = psycopg2.connect(**DB)
    fc = pd.read_sql("""
        SELECT model_name, target, forecast_period, forecast_value,
               lower_bound, upper_bound, rmse, mape, trained_on_periods
        FROM gold_forecast_results
        WHERE target = 'wti_price'
        ORDER BY model_name, trained_on_periods, forecast_period
    """, conn)
    actuals = pd.read_sql("""
        SELECT period AS forecast_period, wti_price AS actual
        FROM gold_features
        ORDER BY period
    """, conn)
    conn.close()
    fc["forecast_period"] = pd.to_datetime(fc["forecast_period"])
    actuals["forecast_period"] = pd.to_datetime(actuals["forecast_period"])
    merged = fc.merge(actuals, on="forecast_period", how="left")
    return merged


df = load_data()
if df.empty:
    st.warning("No forecast results found. Run the SARIMAX and XGBoost experiment scripts first.")
    st.stop()

# Pretty model names — only the 5 variants we actually want to compare
MODEL_LABELS = {
    "sarima":           "SARIMA (baseline)",
    "sarimax":          "SARIMAX (3 exog)",
    "xgboost":          "XGBoost (192 feats)",
    "xgboost_curated":  "XGBoost (18 curated)",
    "xgboost_tuned":    "XGBoost (curated + tuned)",
}

# Keep only rows from these models AND only the two real test windows (96, 120).
# This drops the legacy prophet/sarima future-forecast rows (108/136 train, null MAPE).
df = df[df["model_name"].isin(MODEL_LABELS.keys())].copy()
df = df[df["trained_on_periods"].isin([96, 120])].copy()
df["model_label"] = df["model_name"].map(MODEL_LABELS)

# Window labels based on trained_on_periods
def window_label(n):
    if n == 96:
        return "2024 calm year"
    if n == 120:
        return "2026 spike window"
    return f"{n}-month train"
df["window"] = df["trained_on_periods"].apply(window_label)

# Fixed window order (calm first, spike second) instead of alphabetical
WINDOW_ORDER = ["2024 calm year", "2026 spike window"]

# ── Metrics summary table ──────────────────────────────────────────────────
st.markdown("### Leaderboard")
st.markdown("MAPE = mean absolute percentage error. Lower is better.")

metrics = (
    df.groupby(["window", "model_label"])
      .agg(rmse=("rmse", "first"), mape=("mape", "first"), n_months=("forecast_period", "count"))
      .reset_index()
      .sort_values(["window", "mape"])
)

# Render per-window leaderboard tables side by side
windows = [w for w in WINDOW_ORDER if w in metrics["window"].unique()]
cols = st.columns(len(windows))
for col, win in zip(cols, windows):
    with col:
        st.markdown(f"**{win}**")
        sub = metrics[metrics["window"] == win].copy()
        sub["mape"] = sub["mape"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "—")
        sub["rmse"] = sub["rmse"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        sub = sub[["model_label", "rmse", "mape", "n_months"]]
        sub.columns = ["Model", "RMSE", "MAPE", "Months"]
        st.dataframe(sub, hide_index=True, use_container_width=True)

st.divider()

# ── Forecast-vs-actual chart, filterable by window ────────────────────────
st.markdown("### Forecast vs Actual")

chosen_window = st.radio("Pick a test window:", windows, horizontal=True)
sub = df[df["window"] == chosen_window].copy()

# Common actual line
actual_line = (
    sub[["forecast_period", "actual"]]
    .drop_duplicates()
    .sort_values("forecast_period")
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=actual_line["forecast_period"], y=actual_line["actual"],
    name="Actual WTI", line=dict(color="#d62728", width=3.5),
    mode="lines+markers",
))

colors = {
    "SARIMA (baseline)":         "#1f77b4",
    "SARIMAX (3 exog)":          "#ff7f0e",
    "XGBoost (192 feats)":       "#bcbd22",
    "XGBoost (18 curated)":      "#17becf",
    "XGBoost (curated + tuned)": "#2ca02c",
}

# Optional: let user toggle which models to show
model_options = sorted(sub["model_label"].unique())
selected_models = st.multiselect(
    "Show forecasts from:", model_options, default=model_options
)

for label, color in colors.items():
    if label not in selected_models:
        continue
    md = sub[sub["model_label"] == label].sort_values("forecast_period")
    if md.empty:
        continue
    fig.add_trace(go.Scatter(
        x=md["forecast_period"], y=md["forecast_value"],
        name=label, line=dict(color=color, dash="dot", width=2),
        mode="lines+markers",
    ))

fig.update_layout(
    height=460,
    hovermode="x unified",
    yaxis_title="WTI price ($/barrel)",
    xaxis_title=None,
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ── Per-month errors, in a table ──────────────────────────────────────────
with st.expander("📋 Row-by-row predictions and errors", expanded=False):
    detail = sub[sub["model_label"].isin(selected_models)].copy()
    detail["error"] = (detail["forecast_value"] - detail["actual"]).round(2)
    pivot = (
        detail.pivot_table(
            index="forecast_period",
            columns="model_label",
            values="forecast_value",
        ).round(2)
    )
    actuals_series = detail.groupby("forecast_period")["actual"].first()
    pivot.insert(0, "Actual", actuals_series.round(2))
    pivot.index = pivot.index.strftime("%Y-%m")
    pivot.index.name = "Month"
    st.dataframe(pivot, use_container_width=True)

st.divider()
st.markdown(
    "**Story:** all five models tie on the calm year; tree-based models pull ahead on the spike. "
    "Tuned XGBoost is the only model that catches up meaningfully once the regime change is underway, "
    "though no model successfully predicts the cliff edge itself."
)