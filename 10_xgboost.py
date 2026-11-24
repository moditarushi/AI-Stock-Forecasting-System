import pandas as pd
import os
import config
import sys

# Get selected company
if len(sys.argv) > 1:
    selected_company = sys.argv[1]

    if selected_company not in config.COMPANIES:
        raise ValueError(f"Unknown company: {selected_company}")

    COMPANY_NAME, TICKER = config.COMPANIES[selected_company]
else:
    COMPANY_NAME = config.COMPANY_NAME

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Load Feature Dataset

input_file = f"data/{COMPANY_NAME}_Features.csv"

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
    "MACD_Signal",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Volume_Lag1",
    "Close_Lag1",
    "Close_Lag5",
    "Momentum_5",
    "Momentum_10"
]


# Remove missing values

data = data.dropna().reset_index(drop=True)


# Time-Series Train/Test Split

split_index = int(len(data) * 0.8)

train = data.iloc[:split_index]
test = data.iloc[split_index:]


X_train = train[feature_columns]
X_test = test[feature_columns]

y_train = train["Direction"]
y_test = test["Direction"]


print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# XGBoost Model
print("\nTraining XGBoost Classifier...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("\nSelected XGBoost Parameters:")
print({
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 1.0
})


# Prediction

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("XGBoost Direction Model")

print(f"Company  : {COMPANY_NAME}")
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


# Save Predictions

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


prediction_file = (
    f"data/{COMPANY_NAME}_XGBoost_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


# Save Metrics

metrics_df = pd.DataFrame({
    "Company": [COMPANY_NAME],
    "Model": ["XGBoost Classifier"],
    "Accuracy": [accuracy]
})


metrics_file = (
    f"data/{COMPANY_NAME}_XGBoost_Metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

# Feature Importance

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print("\nTop Feature Importances:")

print(
    importance_df.head(10).to_string(index=False)
)


print("\nPredictions saved as:")
print(prediction_file)

print("\nMetrics saved as:")
print(metrics_file)
