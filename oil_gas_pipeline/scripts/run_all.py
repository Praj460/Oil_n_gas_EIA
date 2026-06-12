# oil_gas_pipeline | scripts/run_all.py
# Runs the entire pipeline end to end in the correct order
# Run with: python3 scripts/run_all.py

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_step(step_number, script_name, description):
    print(f"\n{'='*60}")
    print(f"RUNNING STEP {step_number} — {description}")
    print(f"{'='*60}")
    start = time.time()

    import importlib.util
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    spec        = importlib.util.spec_from_file_location("module", script_path)
    module      = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    elapsed = round(time.time() - start, 1)
    print(f"\n✅ Step {step_number} completed in {elapsed}s")

def main():
    print("\n" + "🛢️  " * 20)
    print("OIL & GAS FORECASTING PIPELINE — FULL RUN")
    print("🛢️  " * 20)

    start_total = time.time()

    steps = [
        (1, "1_ingest.py",            "Ingest EIA API data → bronze tables"),
        (2, "2_quality_check.py",     "Run Great Expectations quality checks"),
        (3, "3_populate_gold.py",     "Transform bronze → gold table"),
        (4, "4_run_forecast.py",      "Train SARIMA + Prophet, save forecasts"),
        (5, "5_calculate_metrics.py", "Calculate RMSE and MAPE metrics"),
    ]

    failed_steps = []

    for step_num, script, description in steps:
        try:
            run_step(step_num, script, description)
        except Exception as e:
            print(f"\n❌ Step {step_num} FAILED: {e}")
            failed_steps.append((step_num, description, str(e)))
            response = input(f"\nContinue to next step? (y/n): ").strip().lower()
            if response != "y":
                print("Pipeline stopped.")
                break

    total_time = round(time.time() - start_total, 1)

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time}s")

    if failed_steps:
        print(f"\n⚠️  {len(failed_steps)} step(s) failed:")
        for step_num, desc, err in failed_steps:
            print(f"   Step {step_num} — {desc}")
            print(f"   Error: {err}")
    else:
        print("\n✅ All 5 steps completed successfully")
        print("\nYour dashboard is ready at http://localhost:8501")
        print("Run: PYTHONPATH=$(pwd) streamlit run dashboard/app.py")

    print("=" * 60)

if __name__ == "__main__":
    main()
