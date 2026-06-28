# 🖥️ DEMO COMMANDS — keep this open during the interview

## ⚡ SETUP (run BEFORE the interview starts)

```bash
# Terminal 1 — go to project, activate venv (needed in EVERY new terminal)
cd /Users/prajwalanand/Oil_n_gas/oil_gas_pipeline
source venv/bin/activate
```

```bash
# Terminal 2 — start the dashboard (keep running)
cd /Users/prajwalanand/Oil_n_gas/oil_gas_pipeline
source venv/bin/activate
PYTHONPATH=$(pwd) streamlit run dashboard/data_quality.py
# → http://localhost:8501
```

```bash
# Terminal 3 — start Airflow (keep running)
cd /Users/prajwalanand/Oil_n_gas/oil_gas_pipeline
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver -p 8080 &
airflow scheduler &
# → http://localhost:8080  (admin / admin)
```

Have open in browser tabs: localhost:8501 (dashboard) + localhost:8080 (Airflow)

---

## 🛢️ THE PIPELINE — step by step (Terminal 1)

```bash
# Step 1 — Ingest core EIA data → bronze
python3 scripts/1_ingest.py

# Step 2 — Run the quality checks → logged to DB
python3 scripts/2_quality_check.py

# Step 3 — Transform bronze → gold (pandas path)
python3 scripts/3_populate_gold.py

# Step 4 — Validation forecasts (2024)
python3 scripts/4_run_forecast.py

# Step 5 — Grade them: RMSE / MAE / MAPE vs actual 2024
python3 scripts/5_calculate_metrics.py

# Step 6 — Ingest the 4 exogenous predictor series → bronze   (NEW)
PYTHONPATH=$(pwd) python3 scripts/6_ingest_exogenous.py

# Future forecast — next 12 months (trains on ALL data)
PYTHONPATH=$(pwd) python3 models/future_forecaster.py

# Architecture overview — live row counts per layer
python3 scripts/7_show_architecture.py

# OR everything at once:
python3 scripts/run_all.py
```

---

## 🌡️ EXOGENOUS DATA SOURCES (NEW — week 5/6 work)

### The 4 new EIA predictor series and why they matter

| Series | EIA code | Signal | Why it predicts price |
|---|---|---|---|
| Crude imports | MCRIMUS2 | supply | More imports = more supply = downward price pressure |
| Refinery utilization | MOPUEUS2 | crude demand | High utilization = refiners buying crude = upward pressure |
| Gasoline stocks (finished) | MGFSTUS1 | downstream demand | High inventory = oversupplied = bearish |
| Distillate stocks | MDISTUS1 | diesel/heating demand | Low winter stocks = price spikes |

### Test the 4 fetches (proves they pull cleanly)

```bash
PYTHONPATH=$(pwd) python3 -c "
from ingestion.eia_client import EIAClient
c = EIAClient()
for name, fn in [('crude imports', c.fetch_crude_imports),
                 ('refinery util', c.fetch_refinery_utilization),
                 ('gasoline stocks', c.fetch_gasoline_stocks),
                 ('distillate stocks', c.fetch_distillate_stocks)]:
    df = fn(start='2015-01')
    print(f'{name:20s} → {len(df):4d} rows | last: {df[\"period\"].max().date()} | value: {df[\"value\"].iloc[-1]}')
"
```

### How I discovered the correct EIA series codes (the debugging story)

```bash
# EIA's stocks codes are cryptic. I queried their metadata endpoint
# directly to list every valid monthly series, then filtered for the
# US-level gasoline/distillate totals.
PYTHONPATH=$(pwd) python3 -c "
import requests
from config.config import config
url='https://api.eia.gov/v2/petroleum/stoc/typ/data/'
params={'api_key':config.eia.api_key,'frequency':'monthly','data[0]':'value',
        'sort[0][column]':'period','sort[0][direction]':'desc','length':5000}
r=requests.get(url,params=params,timeout=30).json()
seen={}
for row in r['response']['data']:
    sid=row.get('series'); desc=row.get('series-description','')
    if sid and sid not in seen: seen[sid]=desc
for sid,desc in seen.items():
    d=desc.lower()
    if 'u.s.' in d and ('gasoline' in d or 'distillate' in d) and 'stock' in d:
        print(f'{sid:18s} | {desc}')
"
```

### Ingest into the 4 bronze tables

```bash
PYTHONPATH=$(pwd) python3 scripts/6_ingest_exogenous.py
```

### Glance at the new tables

```bash
# Row counts + latest period for all 4
psql -d oil_gas_db -c "
SELECT 'crude_imports'      AS series, MAX(period) AS latest, COUNT(*) AS rows FROM bronze_crude_imports
UNION ALL SELECT 'refinery_util',     MAX(period), COUNT(*) FROM bronze_refinery_utilization
UNION ALL SELECT 'gasoline_stocks',   MAX(period), COUNT(*) FROM bronze_gasoline_stocks
UNION ALL SELECT 'distillate_stocks', MAX(period), COUNT(*) FROM bronze_distillate_stocks
ORDER BY series;"

# Eyeball recent rows from one table
psql -d oil_gas_db -c "SELECT series_id, period, value, unit FROM bronze_crude_imports ORDER BY period DESC LIMIT 8;"
```

---

## 🧱 dbt (note: dbt has its OWN venv — that's the conflict fix)

```bash
cd /Users/prajwalanand/Oil_n_gas/oil_gas_pipeline/dbt
../dbt_venv/bin/dbt debug --profiles-dir .
../dbt_venv/bin/dbt run --profiles-dir .
cd ..
```

---

## 🧪 TESTS — the 55 unit tests

```bash
python3 -m pytest tests/ -v          # verbose
python3 -m pytest tests/             # summary
python3 -m pytest tests/test_preprocessor.py -v   # one file
```

---

## 🗄️ DATABASE — queries he might ask for

```bash
psql -d oil_gas_db            # \dt = list tables, \q = quit
psql -d oil_gas_db -c "\dt bronze_*"   # list all bronze tables (now 6)
```

```bash
# Row counts across all layers
psql -d oil_gas_db -c "
SELECT 'bronze_petroleum' t, COUNT(*) FROM bronze_petroleum
UNION ALL SELECT 'bronze_natural_gas', COUNT(*) FROM bronze_natural_gas
UNION ALL SELECT 'bronze_crude_imports', COUNT(*) FROM bronze_crude_imports
UNION ALL SELECT 'bronze_refinery_utilization', COUNT(*) FROM bronze_refinery_utilization
UNION ALL SELECT 'bronze_gasoline_stocks', COUNT(*) FROM bronze_gasoline_stocks
UNION ALL SELECT 'bronze_distillate_stocks', COUNT(*) FROM bronze_distillate_stocks
UNION ALL SELECT 'gold_energy_prices', COUNT(*) FROM gold_energy_prices
UNION ALL SELECT 'gold_forecast_results', COUNT(*) FROM gold_forecast_results;"
```

```bash
# Latest gold rows
psql -d oil_gas_db -c "SELECT period, wti_price, brent_price, price_spread, henry_hub_price, oil_gas_ratio FROM gold_energy_prices ORDER BY period DESC LIMIT 5;"

# Forecast results with metrics
psql -d oil_gas_db -c "SELECT model_name, target, forecast_period, forecast_value, rmse, mape FROM gold_forecast_results WHERE rmse IS NOT NULL ORDER BY target, model_name, forecast_period LIMIT 12;"

# Quality check history
psql -d oil_gas_db -c "SELECT suite_name, passed, failed, success_rate, run_at FROM data_quality_results ORDER BY run_at DESC LIMIT 5;"
```

---

## 🔧 IF SOMETHING BREAKS MID-DEMO (emergency fixes)

```bash
# "command not found" → venv not active:
source /Users/prajwalanand/Oil_n_gas/oil_gas_pipeline/venv/bin/activate

# "ModuleNotFoundError: config" → missing PYTHONPATH:
export PYTHONPATH=/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline

# Dashboard frozen → restart:
pkill -f streamlit
PYTHONPATH=$(pwd) streamlit run dashboard/data_quality.py

# Airflow not loading → restart:
pkill -f airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver -p 8080 & airflow scheduler &

# Is PostgreSQL running?
pg_isready
# if not:  brew services start postgresql@15
```

---

## 🎯 LIKELY LIVE-CHANGE REQUESTS

| He asks | You open | Change |
|---|---|---|
| "Forecast 6 months instead of 12" | config/config.py | forecast_horizon 12 → 6, rerun script 4 |
| "Stricter outlier threshold" | models/preprocessor.py | iqr_multiplier 3.0 → 2.0, run pytest |
| "Add a quality check" | great_expectations/petroleum_suite.py | copy an expect_ method, add to run() list |
| "Add another data series" | ingestion/eia_client.py | copy a fetch_ method, swap the facet code |

After ANY live change:  python3 -m pytest tests/ -v   → "all green, nothing broke."
