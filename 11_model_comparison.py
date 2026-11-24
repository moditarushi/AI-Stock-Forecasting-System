import pandas as pd
import os
import config

print(" MODEL COMPARISON ")


# Load metric files

rf_file = f"data/{config.COMPANY_NAME}_Direction_Metrics.csv"
xgb_file = f"data/{config.COMPANY_NAME}_XGBoost_Metrics.csv"


if not os.path.exists(rf_file):
    print("Random Forest metrics not found.")
    print("Run 09_direction_prediction.py first.")
    exit()


if not os.path.exists(xgb_file):
    print("XGBoost metrics not found.")
    print("Run 10_xgboost.py first.")
    exit()


rf = pd.read_csv(rf_file)
xgb = pd.read_csv(xgb_file)


# Create comparison table

baseline_accuracy = rf["Baseline Accuracy"].iloc[0]

comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        baseline_accuracy,
        rf["Accuracy"].iloc[0],
        xgb["Accuracy"].iloc[0]
    ]
})


comparison["Accuracy (%)"] = (
    comparison["Accuracy"] * 100
)


# Display

print("\nModel Performance:")
print(comparison[["Model", "Accuracy (%)"]].to_string(index=False))


# Find best model

best_model = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("BEST MODEL")

print(f"Model    : {best_model['Model']}")
print(f"Accuracy : {best_model['Accuracy (%)']:.2f}%")

# Save comparison

comparison_file = (
    f"data/{config.COMPANY_NAME}_Model_Comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


print("\nComparison saved as:")
print(comparison_file)
