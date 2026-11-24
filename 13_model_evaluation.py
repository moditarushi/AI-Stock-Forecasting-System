import os
import pandas as pd
import matplotlib.pyplot as plt
import config

# Load prediction data
file_path = f"data/{config.COMPANY_NAME}_XGBoost_Predictions.csv"

if not os.path.exists(file_path):
    print("Prediction file not found!")
    print("Run 08_ml_models.py first.")
    exit()

data = pd.read_csv(file_path)

print("\nMODEL EVALUATION")
print(f"Company: {config.COMPANY_NAME}")

print("\nColumns:")
print(data.columns.tolist())

# Convert date
data["Date"] = pd.to_datetime(data["Date"])

# Display first 5 rows
print("\nFirst 5 Rows:")
print(data.head())

# ACTUAL VS PREDICTED DIRECTION

data["Actual Direction"] = data["Actual Direction"].map({"DOWN": 0, "UP": 1})
data["Predicted Direction"] = data["Predicted Direction"].map({"DOWN": 0, "UP": 1})

plt.figure(figsize=(12, 6))

plt.plot(
    data["Date"],
    data["Actual Direction"],
    label="Actual Direction"
)

plt.plot(
    data["Date"],
    data["Predicted Direction"],
    label="Predicted Direction"
)

plt.xlabel("Date")
plt.ylabel("Direction")
plt.title(
    f"{config.COMPANY_NAME} - Actual vs Predicted Direction"
)

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Create results folder
os.makedirs("results", exist_ok=True)

# Save graph
graph_path = (
    f"results/{config.COMPANY_NAME}_Actual_vs_Predicted.png"
)

plt.savefig(graph_path)

print("\nActual vs Predicted graph saved as:")
print(graph_path)

plt.show()
