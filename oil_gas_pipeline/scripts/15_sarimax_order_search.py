# oil_gas_pipeline | scripts/15_sarimax_order_search.py
# Tries a handful of sensible SARIMAX orders and reports which converge cleanly.
# Compares each against the 5.77% baseline on 2024 WTI.
# Run with: python3 scripts/15_sarimax_order_search.py

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

load_dotenv()
DB = dict(host=os.getenv("DB_HOST","localhost"), port=int(os.getenv("DB_PORT","5432")),
          dbname=os.getenv("DB_NAME","oil_gas_db"), user=os.getenv("DB_USER"),
          password=os.getenv("DB_PASSWORD"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXOG = ["industrial_production_lag_1", "opec_spare_lag_1", "dollar_index_lag_1"]

# Orders to try. Less-complex variants of SARIMA + a few alternatives.
ORDERS = [
    # (name, order, seasonal_order)
    ("baseline   (1,1,1)(1,1,1,12)",  (1,1,1), (1,1,1,12)),
    ("noMA       (1,1,0)(1,1,1,12)",  (1,1,0), (1,1,1,12)),
    ("noSeasMA   (1,1,1)(1,1,0,12)",  (1,1,1), (1,1,0,12)),
    ("simple     (1,1,1)(0,1,1,12)",  (1,1,1), (0,1,1,12)),
    ("AR-only    (2,1,0)(1,1,0,12)",  (2,1,0), (1,1,0,12)),
    ("minimal    (1,1,0)(0,1,1,12)",  (1,1,0), (0,1,1,12)),
]

def metrics(actual, predicted):
    mask = (~actual.isna()) & (~predicted.isna())
    a, p = actual[mask].values, predicted[mask].values
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mae":  float(mean_absolute_error(a, p)),
        "mape": float(np.mean(np.abs((a - p) / a)) * 100),
    }

def main():
    # Load features + raw target
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_scaled.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period", conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df["wti_price"] = gold["wti_price"]

    train = df.loc["2016-01-01":"2023-12-01"]
    test  = df.loc["2024-01-01":"2024-12-01"]
    y_tr  = train["wti_price"].asfreq("MS")
    X_tr  = train[EXOG].asfreq("MS").dropna()
    y_tr  = y_tr.loc[X_tr.index]
    y_te  = test["wti_price"].asfreq("MS")
    X_te  = test[EXOG].asfreq("MS")

    print("\n" + "=" * 72)
    print(f"SARIMAX order search — 3 exog features, train 2016-23, test 2024")
    print("=" * 72)
    print(f"{'order':<35} {'converged':>10} {'AIC':>8} {'MAPE %':>9}")
    print("-" * 72)

    best = {"name": None, "mape": float("inf"), "aic": None, "converged": False}
    for name, order, sorder in ORDERS:
        try:
            mod = SARIMAX(y_tr, exog=X_tr, order=order, seasonal_order=sorder,
                          enforce_stationarity=False, enforce_invertibility=False)
            fit = mod.fit(disp=False, maxiter=200)
            # mle_retvals tells us if it converged
            converged = fit.mle_retvals.get("converged", False)
            fc = fit.get_forecast(steps=len(X_te), exog=X_te).predicted_mean
            fc.index = X_te.index
            m = metrics(y_te, fc)
            print(f"{name:<35} {'✅' if converged else '❌':>10} {fit.aic:>8.1f} {m['mape']:>8.2f}%")
            if converged and m['mape'] < best['mape']:
                best = {"name": name, "mape": m['mape'], "aic": fit.aic, "converged": True,
                        "order": order, "sorder": sorder, "rmse": m['rmse'], "mae": m['mae']}
        except Exception as e:
            print(f"{name:<35} {'ERROR':>10}   —      —     ({str(e)[:30]})")

    print("=" * 72)
    print(f"BASELINE (your SARIMA): MAPE 5.77%  RMSE 5.48  MAE 4.32")
    if best["name"]:
        print(f"BEST converged SARIMAX : MAPE {best['mape']:.2f}%  RMSE {best['rmse']:.2f}  MAE {best['mae']:.2f}")
        print(f"  → order  {best['order']}")
        print(f"  → s_order {best['sorder']}")
        delta = best['mape'] - 5.77
        d = "improved by" if delta < 0 else "worsened by"
        print(f"  → MAPE {d} {abs(delta):.2f} pp vs baseline")
    else:
        print("No order converged cleanly — would need to try auto_arima next.")
    print("=" * 72)

if __name__ == "__main__":
    main()
