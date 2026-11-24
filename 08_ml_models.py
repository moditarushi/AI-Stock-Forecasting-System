import pandas as pd
import numpy as np
import os
import config

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Feature Dataset

input_file = f"data/{config.COMPANY_NAME}_Features.csv"

if not os.path.exists(input_file):
    print("Feature dataset not found.")
    print("Run 07_feature_engineering.py first.")
    exit()

data = pd.read_csv(input_file)

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values("Date").reset_index(drop=True)

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

y_train = train["Target_Return"]
y_test = test["Target_Return"]


print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# Random Forest

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)

# Convert predicted return to price

current_price = test["Close"].values
predicted_price = current_price * (1 + y_pred)
actual_price = current_price * (1 + y_test)

# Evaluation
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

non_zero = y_test != 0

mape = np.mean(
    np.abs(
        (y_test[non_zero] - y_pred[non_zero])
        / y_test[non_zero]
    )
) * 100

r2 = r2_score(
    y_test,
    y_pred
)

print("Random Forest Performance")

print(f"Company : {config.COMPANY_NAME}")
print(f"RMSE    : {rmse:.2f}")
print(f"MAE     : {mae:.2f}")
print(f"MAPE    : {mape:.2f}%")
print(f"R2 Score: {r2:.4f}")


# Save Predictions

prediction_df = pd.DataFrame({
    "Date": test["Date"].values,
    "Actual Price": actual_price,
    "Predicted Price": predicted_price
})

prediction_file = (
    f"data/{config.COMPANY_NAME}_RandomForest_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


# Save Metrics

metrics_df = pd.DataFrame({
    "Company": [config.COMPANY_NAME],
    "Model": ["Random Forest"],
    "RMSE": [rmse],
    "MAE": [mae],
    "MAPE": [mape],
    "R2": [r2]
})

metrics_file = (
    f"data/{config.COMPANY_NAME}_RandomForest_Metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


print("\nPredictions saved as:")
print(prediction_file)

print("\nMetrics saved as:")
print(metrics_file)


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
print(importance_df.head(10).to_string(index=False))
