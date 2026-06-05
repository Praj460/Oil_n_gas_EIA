# Oil & Gas Forecasting Pipeline

A production-grade data engineering project that ingests crude oil and 
natural gas data from the EIA API, processes it through a medallion 
architecture (bronze → silver → gold) using PostgreSQL and dbt, runs 
SARIMA and Prophet forecasting models, validates data quality with 
Great Expectations, orchestrates everything with Apache Airflow, and 
visualizes results in a Streamlit dashboard.

## Stack
- Data Source: EIA OpenData API (petroleum + natural gas)
- Database: PostgreSQL (bronze/silver/gold layers)
- Transformation: dbt Core
- Data Quality: Great Expectations
- Orchestration: Apache Airflow
- Forecasting: SARIMA (statsmodels) + Prophet
- Dashboard: Streamlit + Plotly
- Language: Python 3.11

## Architecture
Ingestion → Bronze tables → dbt transforms → Silver tables → 
Gold tables → Forecast models → Streamlit dashboard

## Folder Structure
- ingestion/   → EIA API client, Kaggle loader, DataIngester class
- database/    → PostgreSQL connection, schema, DatabaseManager class
- dbt/         → SQL transformation models (staging + mart layer)
- great_expectations/ → Data quality suites
- models/      → Preprocessor, ForecastModel, Evaluator classes
- dags/        → Airflow DAGs for scheduling
- dashboard/   → Streamlit multipage app
- config/      → Environment variables, logging
- tests/       → Unit tests