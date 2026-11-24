import pandas as pd
import numpy as np
import os
import config

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load Feature Dataset

input_file = f"data/{config.COMPANY_NAME}_Features.csv"

if not os.path.exists(input_file):
    print("Feature dataset not found.")
    print("Run 07_feature_engineering.py first.")
    exit()

data = pd.read_csv(input_file)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)


# Create Direction Target

data["Direction"] = (
    data["Target_Return"] > 0
).astype(int)


# Features

feature_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "Return_5D",
    "Return_20D",
    "MA7",
    "MA20",
    "MA50",
    "EMA20",
    "Volatility_7",
    "Volatility_20",
    "High_Low_Pct",
    "Open_Close_Pct",
    "Volume_Change",
    "RSI",
    "MACD",
    "MACD_Signal"
]

# Remove missing values

data = data.dropna().reset_index(drop=True)


# Time-Series Split

split_index = int(len(data) * 0.8)

train = data.iloc[:split_index]
test = data.iloc[split_index:]


X_train = train[feature_columns]
X_test = test[feature_columns]

y_train = train["Direction"]
y_test = test["Direction"]


print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# Baseline

baseline_prediction = np.zeros(len(y_test))

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)


print("Baseline Performance")

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)

# Random Forest Classifier

print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("Random Forest Direction Model")

print(f"Company  : {config.COMPANY_NAME}")
print(f"Accuracy : {accuracy * 100:.2f}%")


# Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["DOWN", "UP"]
    )
)


# Confusion Matrix

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# Predictions DataFrame
prediction_df = pd.DataFrame({
    "Date": test["Date"].values,
    "Actual Direction": y_test.values,
    "Predicted Direction": y_pred,
    "UP Probability": y_probability
})


prediction_df["Actual Direction"] = (
    prediction_df["Actual Direction"]
    .map({
        0: "DOWN",
        1: "UP"
    })
)

prediction_df["Predicted Direction"] = (
    prediction_df["Predicted Direction"]
    .map({
        0: "DOWN",
        1: "UP"
    })
)


# Save Predictions

prediction_file = (
    f"data/{config.COMPANY_NAME}_Direction_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# Save Metrics

metrics_df = pd.DataFrame({
    "Company": [config.COMPANY_NAME],
    "Model": ["Random Forest Classifier"],
    "Baseline Accuracy": [baseline_accuracy],
    "Accuracy": [accuracy]
})


metrics_file = (
    f"data/{config.COMPANY_NAME}_Direction_Metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


print("\nPredictions saved as:")
print(prediction_file)

print("\nMetrics saved as:")
print(metrics_file)
