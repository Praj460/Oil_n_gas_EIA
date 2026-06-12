# 🛢️ Oil & Gas Forecasting Pipeline

An end-to-end data engineering project that ingests crude oil and natural gas data from the **EIA OpenData API**, validates it with custom **Great Expectations**-style quality suites, transforms it through a **medallion architecture** (bronze → silver → gold) using **dbt** and **PostgreSQL**, forecasts benchmark prices with **SARIMA** and **Prophet**, orchestrates everything with **Apache Airflow**, and visualizes results in a **Streamlit** dashboard.

---

## 📊 Architecture

```
EIA OpenData API
      │
      ▼
┌─────────────────┐
│  INGESTION      │  EIAClient (retry/rate-limit handling)
│                 │  DataIngester (validation + orchestration)
└────────┬────────┘
         ▼
┌─────────────────┐
│  BRONZE LAYER   │  bronze_petroleum (406 rows)
│  raw, append-   │  bronze_natural_gas (402 rows)
│  only           │  Full raw API responses stored as JSONB
└────────┬────────┘
         ▼
┌─────────────────┐
│  QUALITY GATES  │  PetroleumSuite — 10 checks
│                 │  GasSuite — 11 checks
│                 │  Results logged to data_quality_results
└────────┬────────┘
         ▼
┌─────────────────┐
│  SILVER LAYER   │  silver_petroleum (dbt view)
│  cleaned (dbt)  │  silver_natural_gas (dbt view)
└────────┬────────┘
         ▼
┌─────────────────┐
│  GOLD LAYER     │  gold_energy_prices (136 monthly rows)
│  analysis-ready │  gold_forecast_results (validation + future forecasts)
└────────┬────────┘
         ▼
┌─────────────────┐
│  MODELS         │  Preprocessor → SARIMA + Prophet → Evaluator
└────────┬────────┘
         ▼
┌─────────────────┐
│  DASHBOARD      │  Streamlit — 5 pages
└─────────────────┘

Orchestration: 3 Apache Airflow DAGs (daily ingest w/ quality gate,
daily transform, weekly forecast retraining)
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data Source | EIA OpenData API v2 (6 monthly series, 2015–present) |
| Database | PostgreSQL 15 (medallion: bronze / silver / gold) |
| Transformation | dbt Core 1.7 (isolated venv) + pandas |
| Data Quality | Custom Great Expectations-style suites (21 checks) |
| Forecasting | SARIMA (statsmodels) + Prophet (Meta) |
| Orchestration | Apache Airflow 2.9 (3 DAGs) |
| Dashboard | Streamlit + Plotly |
| Testing | pytest (50 unit tests) |

---

## 📈 Data

Six monthly series from the EIA OpenData API (2015-01 → present):

| Series | EIA ID | Description |
|---|---|---|
| WTI Crude | PET.RWTC.M | US benchmark oil spot price ($/barrel) |
| Brent Crude | PET.RBRTE.M | Global benchmark oil spot price ($/barrel) |
| US Oil Production | PET.MCRFPUS2.M | Monthly US crude production |
| Henry Hub | NG.RNGWHHD.M | US benchmark natural gas price ($/MMBtu) |
| Gas Storage | NG.NW2_EPG0_SWO_R48_BCF.M | US working gas in storage |
| Gas Production | NG.N9010US2.M | Monthly US gas production |

Derived features in the gold layer: **Brent-WTI price spread**, **month-over-month % changes**, **oil/gas ratio**.

---

## 🤖 Forecasting & Results

Two models forecast the two benchmark prices (WTI, Henry Hub) over a 12-month horizon:

- **Validation run** — trained on 2015–2023, forecasted 2024, graded against actual 2024 data
- **Future run** — trained on all data, forecasting the next 12 months with 95% confidence intervals

### Validation results (2024 forecast vs actual)

| Model | Target | RMSE | MAPE |
|---|---|---|---|
| **SARIMA** ✅ | WTI | **5.48 $/bbl** | **5.77%** |
| Prophet | WTI | 12.69 $/bbl | 15.51% |
| **SARIMA** ✅ | Henry Hub | **0.93 $/MMBtu** | **11.29%** |
| Prophet | Henry Hub | 1.85 $/MMBtu | 25.86% |

**SARIMA outperformed Prophet on both targets** — the 2024 market was mean-reverting, which favors SARIMA's differencing approach over Prophet's trend extrapolation.

---

## ✅ Data Quality

Custom-built validation suites (modeled on Great Expectations concepts):

- **PetroleumSuite** — 10 checks: nulls, valid series IDs, price ranges ($0–200), no future dates, no duplicates, minimum row count, date bounds
- **GasSuite** — 11 checks: same categories plus a dedicated non-negative Henry Hub check (unlike WTI, Henry Hub has never gone negative)
- Results logged to `data_quality_results` and surfaced in the dashboard
- The daily Airflow DAG uses a **BranchPythonOperator quality gate** — if quality drops below 80%, the pipeline halts before bad data propagates

Current status: **21/21 checks passing (100%)**

---

## 🌀 Orchestration (Airflow)

| DAG | Schedule | Purpose |
|---|---|---|
| `daily_ingest_dag` | Daily 06:00 | EIA pull → bronze → quality checks → branch gate |
| `transform_dag` | Daily 07:00 | dbt run (bronze → silver → gold) |
| `forecast_dag` | Sunday 08:00 | Retrain SARIMA + Prophet, save forecasts |

---

## 🖥️ Dashboard (Streamlit)

| Page | Contents |
|---|---|
| **Data Quality** (home) | Pipeline data counts, quality check results, pipeline run history |
| **Price Trends** | Historical prices, date filters, Brent-WTI spread, 3-month moving average |
| **Forecast** | Future (next 12 months) and Validation (2024 vs actual) views with 95% confidence bands |
| **Model Comparison** | SARIMA vs Prophet metrics (RMSE / MAE / MAPE) + forecast-vs-actual chart |
| **Airflow Monitor** | DAG registry and run history read from Airflow metadata |

---

## 📁 Project Structure

```
oil_gas_pipeline/
├── config/
│   ├── config.py              # Dataclass-based config, .env loading, singleton
│   └── logging_config.py
├── database/
│   ├── schema.sql             # 8 tables: bronze/silver/gold + metadata
│   └── db_manager.py          # Repository pattern — all PostgreSQL I/O
├── ingestion/
│   ├── eia_client.py          # EIA API client (retries, rate limiting)
│   └── ingester.py            # Orchestrates fetch → validate → write
├── great_expectations/
│   ├── petroleum_suite.py     # 10 quality checks
│   └── gas_suite.py           # 11 quality checks
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── stg_petroleum.sql
│       ├── stg_natural_gas.sql
│       └── mart_energy_prices.sql
├── models/
│   ├── preprocessor.py        # Resampling, gap fill, IQR outliers, features
│   ├── forecast_model.py      # SARIMA + Prophet, unified interface
│   ├── evaluator.py           # RMSE / MAE / MAPE / R², backtesting
│   └── future_forecaster.py   # Trains on all data → next 12 months
├── dags/
│   ├── daily_ingest_dag.py    # With BranchPythonOperator quality gate
│   ├── transform_dag.py
│   └── forecast_dag.py
├── dashboard/
│   ├── data_quality.py        # Main page
│   └── pages/
│       ├── 1_price_trends.py
│       ├── 2_forecast.py
│       ├── 3_model_comparison.py
│       └── 5_airflow_monitor.py
├── scripts/                   # Runnable pipeline steps
│   ├── 1_ingest.py
│   ├── 2_quality_check.py
│   ├── 3_populate_gold.py
│   ├── 4_run_forecast.py
│   ├── 5_calculate_metrics.py
│   ├── 7_show_architecture.py
│   └── run_all.py
├── tests/                     # 50 pytest unit tests
│   ├── test_ingester.py
│   ├── test_preprocessor.py
│   └── test_forecast_model.py
├── requirements.txt
└── .env.example
```

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.12, PostgreSQL 15, an EIA API key (free: https://www.eia.gov/opendata/)

### 1. Environment

```bash
git clone <repo-url>
cd oil_gas_pipeline

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # then fill in your values
```

`.env` contents:
```
EIA_API_KEY=your_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=oil_gas_db
DB_USER=your_user
DB_PASSWORD=your_password
```

### 2. Database

```bash
createdb oil_gas_db
psql -d oil_gas_db -f database/schema.sql
```

### 3. Run the pipeline

```bash
# Everything at once:
python3 scripts/run_all.py

# Or step by step:
python3 scripts/1_ingest.py            # EIA API → bronze
python3 scripts/2_quality_check.py     # quality suites → logged to DB
python3 scripts/3_populate_gold.py     # bronze → gold (pandas)
python3 scripts/4_run_forecast.py      # validation forecasts (2024)
python3 scripts/5_calculate_metrics.py # RMSE / MAE / MAPE vs actuals
PYTHONPATH=$(pwd) python3 models/future_forecaster.py   # next 12 months
```

### 4. dbt transformations (isolated venv)

```bash
python3 -m venv dbt_venv
dbt_venv/bin/pip install dbt-core==1.7.0 dbt-postgres==1.7.0 'protobuf<5'

cd dbt
../dbt_venv/bin/dbt run --profiles-dir .
```

> dbt runs in its **own virtual environment** to isolate a protobuf version conflict with Streamlit — the same isolation pattern used in production with containers.

### 5. Dashboard

```bash
PYTHONPATH=$(pwd) streamlit run dashboard/data_quality.py
```

### 6. Airflow (optional)

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow webserver -p 8080 &
airflow scheduler &
# UI: http://localhost:8080
```

### 7. Tests

```bash
python3 -m pytest tests/ -v     # 50 tests
```

---

## 🧠 Design Decisions

- **Why monthly data?** Energy companies plan on annual cycles (hedging, capex); monthly granularity matches the decision cadence and the EIA's cleanest series.
- **Why a 12-month horizon?** Annual budgeting/hedging requires a full-year outlook. Confidence intervals widen honestly with distance — near-term months are far more reliable than month 12.
- **Why both SARIMA and Prophet?** Opposite inductive biases (mean reversion vs trend following). Validation against actual 2024 data picked the winner empirically.
- **Why custom quality suites instead of the GE library?** Same concepts (named expectations, suites, reports) with zero configuration overhead — and full understanding of every check.
- **Why UUID keys, NUMERIC prices, JSONB raw responses?** Global uniqueness, exact decimal arithmetic for financial data, and a complete audit trail of every API response.
- **Chronological train/test split** — never shuffled. Shuffling time series leaks future information into training.

## 🔮 Future Improvements

- Hyperparameter search for SARIMA (auto_arima) and Prophet changepoints
- Add a gradient-boosted model (XGBoost) using the engineered lag/rolling features
- Dockerize: separate containers for pipeline, dbt, Airflow, and dashboard
- CI/CD with automated test runs and dbt build on merge
- Swap SQLite Airflow metadata for Postgres + CeleryExecutor

---

*Built as a 5-week data engineering project — every component designed, debugged, and explainable end to end.*
