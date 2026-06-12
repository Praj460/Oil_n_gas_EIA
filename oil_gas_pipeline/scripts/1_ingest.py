# oil_gas_pipeline | scripts/1_ingest.py
# Step 1 — Pull fresh data from EIA API and load into bronze tables
# Run with: python3 scripts/1_ingest.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="ingestion.log")

import logging
logger = logging.getLogger(__name__)

from ingestion.ingester import DataIngester

def main():
    logger.info("=" * 60)
    logger.info("STEP 1 — DATA INGESTION STARTED")
    logger.info("=" * 60)

    ingester = DataIngester()

    results = ingester.run_full_ingestion(
        start="2015-01",
        include_kaggle=False,
    )

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"WTI Price rows        : {results['petroleum']['wti']}")
    print(f"Brent Price rows      : {results['petroleum']['brent']}")
    print(f"US Oil Production rows: {results['petroleum']['production']}")
    print(f"Henry Hub rows        : {results['natural_gas']['henry_hub']}")
    print(f"Gas Storage rows      : {results['natural_gas']['storage']}")
    print(f"Gas Production rows   : {results['natural_gas']['production']}")

    pet_failed = results['petroleum']['failed']
    gas_failed = results['natural_gas']['failed']

    if pet_failed or gas_failed:
        print(f"\n⚠️  Failed series: petroleum={pet_failed} | gas={gas_failed}")
    else:
        print("\n✅ All series ingested successfully")
    print("=" * 60)

if __name__ == "__main__":
    main()
