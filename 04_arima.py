import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import numpy as np
import config

file_name = f"data/{config.COMPANY_NAME}_Stock_Clean.csv"
data = pd.read_csv(file_name)

# Display First rows
print(data.head())

# Convert Date
data["Date"] = pd.to_datetime(data["Date"])

print("\nData Types")
print(data.dtypes)

# Create Quarter columns

data["Q1"] = (data["Date"].dt.quarter == 1).astype(int)
data["Q2"] = (data["Date"].dt.quarter == 2).astype(int)
data["Q3"] = (data["Date"].dt.quarter == 3).astype(int)
data["Q4"] = (data["Date"].dt.quarter == 4).astype(int)

# Month Features
for i in range(12):

    month_column = "m" + str(i + 1)

    data[month_column] = (
        data["Date"].dt.month == i + 1
    ).astype(int)

# Day Features
for i in range(31):

    day_column = "d" + str(i + 1)

    data[day_column] = (
        data["Date"].dt.day == i + 1
    ).astype(int)

# Week Features
for i in range(52):

    week_column = "w" + str(i + 1)

    data[week_column] = (
        data["Date"].dt.isocalendar().week == i + 1
    ).astype(int)

# Trend
trend_values = []

for i in range(len(data)):
    trend_values.append(i + 1)

data["trend"] = trend_values

# Train-Test split
test_begin_date = "2024-01-01"
data_train = data[data["Date"] < test_begin_date]
data_test = data[data["Date"] >= test_begin_date]

# Create X_train
feature_columns = (
    ["trend"]
    + [f"m{i}" for i in range(1, 13)]
    + [f"Q{i}" for i in range(1, 5)]
    + [f"d{i}" for i in range(1, 32)]
    + [f"w{i}" for i in range(1, 53)]
)

Xtrain = data_train[feature_columns]
Xtest = data_test[feature_columns]

# Create Y_train
Ytrain = data_train["Close"]
Ytest = data_test["Close"]

# ARIMA model
model = ARIMA(
    endog=Ytrain,
    order=(5, 1, 0)
)

model_fit = model.fit()
print(model_fit.summary())

# Make predictions
forecast = model_fit.forecast(steps=len(Ytest))
len(Ytest)

# Build Arima model
data_endog = Ytrain
data_exog = Xtrain

Trend = "t"

Enforce_Stationarity = False
Concentrate_Scale = False

p = 0
d = 1
q = 0

model = ARIMA(
    endog=data_endog,
    exog=data_exog,
    order=(p, d, q),
    trend=Trend,
    enforce_stationarity=Enforce_Stationarity,
    concentrate_scale=Concentrate_Scale
)

model_fit = model.fit()
print(model_fit.summary())

exog_data = Xtest
y_pred = model_fit.predict(
    start=len(Ytrain),
    end=len(Ytrain) + len(Ytest) - 1,
    exog=exog_data
)
print("Ytrain:", len(Ytrain))
print("Ytest :", len(Ytest))
print("Prediction:", len(y_pred))

# Calculate error metrics
rmse = np.sqrt(mean_squared_error(Ytest, y_pred))
mae = mean_absolute_error(Ytest, y_pred)
mape = np.mean(np.abs((Ytest - y_pred) / Ytest)) * 100
r2 = r2_score(Ytest, y_pred)

print("\nModel Performance")
print("=" * 40)

print(f"Company : {config.COMPANY_NAME}")
print(f"RMSE    : {rmse:.2f}")
print(f"MAE     : {mae:.2f}")
print(f"MAPE    : {mape:.2f}%")
print(f"R² Score: {r2:.4f}")

# Save predictions
prediction_df = pd.DataFrame({
    "Date": data_test["Date"],
    "Actual Price": Ytest.values,
    "Predicted Price": y_pred.values
})

prediction_file = f"data/{config.COMPANY_NAME}_Predictions.csv"

prediction_df.to_csv(
    prediction_file,
    index=False
)
print(f"\nPredictions saved as {prediction_file}")

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    data_test["Date"],
    Ytest,
    label="Actual Price",
    color="blue"
)

plt.plot(
    data_test["Date"],
    y_pred,
    label="Predicted Price",
    color="red"
)

plt.title(f"{config.COMPANY_NAME} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()

# Save metrics
metrics_df = pd.DataFrame({
    "Metric": [
        "RMSE",
        "MAE",
        "MAPE",
        "R2 Score"
    ],
    "Value": [
        rmse,
        mae,
        mape,
        r2
    ]
})

metrics_file = f"data/{config.COMPANY_NAME}_Model_Metrics.csv"

metrics_df.to_csv(
    metrics_file,
    index=False
)

print(f"Metrics saved as {metrics_file}")
