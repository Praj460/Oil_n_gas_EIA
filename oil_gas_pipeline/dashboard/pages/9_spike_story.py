# oil_gas_pipeline | dashboard/pages/9_spike_story.py
# Spike Story — guided walkthrough of the early-2026 WTI doubling.
# Built as the demo page: one scroll, one story.

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

st.set_page_config(page_title="The 2026 Spike", page_icon="📈", layout="wide")
st.title("📈 The early-2026 oil spike — story walkthrough")
st.caption("How the data caught a market regime change, and how each model handled it.")


@st.cache_data(ttl=300)
def load():
    conn = psycopg2.connect(**DB)
    feats = pd.read_sql(
        "SELECT * FROM gold_features WHERE period >= '2025-06-01' ORDER BY period", conn
    )
    fc = pd.read_sql("""
        SELECT model_name, forecast_period, forecast_value
        FROM gold_forecast_results
        WHERE target = 'wti_price' AND trained_on_periods = 120
        ORDER BY model_name, forecast_period
    """, conn)
    conn.close()
    feats["period"] = pd.to_datetime(feats["period"])
    fc["forecast_period"] = pd.to_datetime(fc["forecast_period"])
    return feats, fc


feats, fc = load()

# ── Act 1: the price move ──────────────────────────────────────────────────
st.markdown("## Act 1 — Something happened to WTI in early 2026")

c1, c2, c3, c4 = st.columns(4)
recent_4 = feats.tail(4)
for col, (_, row) in zip([c1, c2, c3, c4], recent_4.iterrows()):
    delta = None
    prev_idx = recent_4.index.get_loc(row.name) - 1
    if prev_idx >= 0:
        prev_val = recent_4.iloc[prev_idx]["wti_price"]
        if pd.notna(prev_val):
            delta = f"{((row['wti_price'] - prev_val) / prev_val * 100):+.1f}% m/m"
    col.metric(
        label=row["period"].strftime("%b %Y"),
        value=f"${row['wti_price']:.2f}",
        delta=delta,
    )

st.markdown(
    "Two months earlier oil was trading in the low $60s. By April it was above $100. "
    "Both WTI and Brent moved together — this was not a data glitch."
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=feats["period"], y=feats["wti_price"], name="WTI",
    line=dict(color="#d62728", width=2.5), mode="lines+markers",
))
fig1.add_trace(go.Scatter(
    x=feats["period"], y=feats["brent_price"], name="Brent",
    line=dict(color="#1f77b4", width=2, dash="dot"), mode="lines+markers",
))
fig1.update_layout(
    height=320, hovermode="x unified",
    yaxis_title="Price ($/barrel)",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Act 2: the supply-side mechanism ──────────────────────────────────────
st.markdown("## Act 2 — Why? OPEC's spare capacity collapsed")
st.markdown(
    "OPEC's spare crude production capacity is the market's main shock absorber. "
    "When it's high, supply disruptions are easily absorbed. When it falls to zero, "
    "the market has no buffer left and any pressure goes straight to price."
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=feats["period"], y=feats["wti_price"], name="WTI ($/bbl)",
    line=dict(color="#d62728", width=2.5), yaxis="y1",
))
fig2.add_trace(go.Scatter(
    x=feats["period"], y=feats["opec_spare"], name="OPEC spare (mbd)",
    line=dict(color="#2ca02c", width=2.5), yaxis="y2",
))
fig2.update_layout(
    height=380, hovermode="x unified",
    yaxis=dict(title="WTI ($/bbl)", color="#d62728", side="left"),
    yaxis2=dict(title="OPEC spare (mbd)", color="#2ca02c", side="right",
                overlaying="y", showgrid=False),
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig2, use_container_width=True)

st.info(
    "**OPEC spare capacity fell from ~3 million barrels per day to near zero between "
    "December 2025 and March 2026.** Exactly as WTI doubled. The mechanism is right there "
    "in the data — this is what motivated adding spare capacity as a supply-fragility feature."
)

st.divider()

# ── Act 3: how each model handled it ──────────────────────────────────────
st.markdown("## Act 3 — How each model handled the spike")
st.markdown("All five models trained on data through end of 2025, then forecasted Jan–Apr 2026.")

MODEL_LABELS = {
    "sarima":          "SARIMA",
    "sarimax":         "SARIMAX",
    "xgboost":         "XGBoost (192 feats)",
    "xgboost_curated": "XGBoost (18 curated)",
    "xgboost_tuned":   "XGBoost (curated+tuned)",
}
colors = {
    "SARIMA":                "#1f77b4",
    "SARIMAX":               "#ff7f0e",
    "XGBoost (192 feats)":   "#bcbd22",
    "XGBoost (18 curated)":  "#17becf",
    "XGBoost (curated+tuned)": "#2ca02c",
}

actual_spike = feats[feats["period"] >= "2026-01-01"][["period", "wti_price"]]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=actual_spike["period"], y=actual_spike["wti_price"],
    name="Actual", line=dict(color="black", width=3.5), mode="lines+markers",
))
for mname, label in MODEL_LABELS.items():
    md = fc[fc["model_name"] == mname].sort_values("forecast_period")
    if md.empty:
        continue
    fig3.add_trace(go.Scatter(
        x=md["forecast_period"], y=md["forecast_value"],
        name=label, line=dict(color=colors[label], dash="dot", width=2),
        mode="lines+markers",
    ))
fig3.update_layout(
    height=400, hovermode="x unified",
    yaxis_title="WTI price ($/barrel)",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
)
st.plotly_chart(fig3, use_container_width=True)

# Table view: actual vs each model prediction
pivot = (
    fc[fc["model_name"].isin(MODEL_LABELS.keys())]
      .assign(model=lambda d: d["model_name"].map(MODEL_LABELS))
      .pivot_table(index="forecast_period", columns="model", values="forecast_value")
      .round(2)
)
pivot.insert(0, "Actual", actual_spike.set_index("period")["wti_price"].round(2))
pivot.index = pivot.index.strftime("%Y-%m")
pivot.index.name = "Month"
st.dataframe(pivot, use_container_width=True)

st.divider()

# ── Act 4: the lesson ─────────────────────────────────────────────────────
st.markdown("## Act 4 — What we learned")

a, b, c = st.columns(3)
a.markdown(
    "**Linear models couldn't catch the cliff.** "
    "SARIMA and SARIMAX stayed in the $60s through April even as actual prices doubled. "
    "Linear math can't extrapolate a threshold effect it hasn't seen in training."
)
b.markdown(
    "**Trees do better but still miss March.** "
    "XGBoost models pull ahead in April once March's spike enters their lag features. "
    "No model successfully predicts the cliff edge — they only catch up after."
)
c.markdown(
    "**Spare capacity tells the story.** "
    "Even where the feature importance is small, the OPEC spare-capacity signal is the "
    "single best explanation of WHY the spike happened, even if it didn't help us predict it."
)
