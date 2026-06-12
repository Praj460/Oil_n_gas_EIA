# oil_gas_pipeline | scripts/5_calculate_metrics.py
# Compares 2024 forecasts against actual 2024 data
# Calculates RMSE, MAE, MAPE and saves to gold_forecast_results

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import psycopg2

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
def rmse(a, p):
    return round(float(np.sqrt(np.mean((a - p) ** 2))), 4)

def mae(a, p):
    return round(float(np.mean(np.abs(a - p))), 4)

def mape(a, p):
    mask = a != 0
    return round(float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100), 4)

def main():
    print("\n" + "=" * 70)
    print("METRICS - 2024 FORECAST vs ACTUAL")
    print("=" * 70)

    conn = psycopg2.connect(**DB)
    actual = pd.read_sql("""
        SELECT period, wti_price, henry_hub_price
        FROM gold_energy_prices
        WHERE period >= '2024-01-01' AND period <= '2024-12-01'
        ORDER BY period ASC
    """, conn)
    forecasts = pd.read_sql("""
        SELECT target, model_name, forecast_period, forecast_value
        FROM gold_forecast_results
        WHERE forecast_period >= '2024-01-01' AND forecast_period <= '2024-12-01'
        ORDER BY forecast_period ASC
    """, conn)
    conn.close()

    actual["period"] = pd.to_datetime(actual["period"])
    forecasts["forecast_period"] = pd.to_datetime(forecasts["forecast_period"])

    results = []
    for target in ["wti_price", "henry_hub_price"]:
        for model in ["sarima", "prophet"]:
            fc = forecasts[
                (forecasts["target"] == target) &
                (forecasts["model_name"] == model)
            ].set_index("forecast_period")["forecast_value"]

            ac = actual.set_index("period")[target]
            aligned = pd.concat(
                [ac.rename("a"), fc.rename("p")], axis=1
            ).dropna()

            if aligned.empty:
                print(f"  No overlap: {model} | {target}")
                continue

            a = aligned["a"].values.astype(float)
            p = aligned["p"].values.astype(float)

            r = {
                "target": target, "model": model,
                "rmse": rmse(a, p), "mae": mae(a, p), "mape": mape(a, p),
                "n": len(aligned),
            }
            results.append(r)

            unit = "$/bbl" if target == "wti_price" else "$/MMBtu"
            print(f"{model.upper():8} | {target:18} | "
                  f"RMSE={r['rmse']:.4f} | MAE={r['mae']:.4f} {unit} | "
                  f"MAPE={r['mape']:.2f}% | n={r['n']}")

    # Save back to DB (only on 2024 validation rows)
    if results:
        conn2 = psycopg2.connect(**DB)
        cur = conn2.cursor()
        for r in results:
            cur.execute("""
                UPDATE gold_forecast_results
                SET rmse = %s, mae = %s, mape = %s
                WHERE target = %s AND model_name = %s
                  AND forecast_period >= '2024-01-01'
                  AND forecast_period <= '2024-12-01'
            """, (r["rmse"], r["mae"], r["mape"], r["target"], r["model"]))
        conn2.commit()
        conn2.close()
        print("\nMetrics saved to gold_forecast_results")
    print("=" * 70)

if __name__ == "__main__":
    main()
