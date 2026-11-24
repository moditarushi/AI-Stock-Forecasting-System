
import subprocess
import sys


def run_pipeline(selected_company):

    scripts = [
        "01_data_loading.py",
        "02_data_preprocessing.py",
        "07_feature_engineering.py",
        "10_xgboost.py",
        "12_ml_future_predictions.py",
        "06_future_predictions.py"
    ]

    print(f"\nStarting pipeline for {selected_company}")

    for script in scripts:

        print(f"\nRunning {script}...")

        subprocess.run(
            [sys.executable, script, selected_company],
            check=True
        )

        print(f"{script} completed successfully!")

    print("\n================================")
    print("STOCK FORECASTING PIPELINE DONE")
    print(f"Company: {selected_company}")
    print("================================")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide a company name."
        )

    selected_company = sys.argv[1]

    run_pipeline(selected_company)
