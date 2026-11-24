import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import config

# Load Dataset
print(f"Auto ARIMA Model - {config.COMPANY_NAME}")

file_name = f"data/{config.COMPANY_NAME}_Stock_Clean.csv"
data = pd.read_csv(file_name)

data["Date"] = pd.to_datetime(data["Date"])

print(data.head())
print()
print(data.dtypes)

# Quarter Features
data["Q1"] = (data["Date"].dt.quarter == 1).astype(int)
data["Q2"] = (data["Date"].dt.quarter == 2).astype(int)
data["Q3"] = (data["Date"].dt.quarter == 3).astype(int)
data["Q4"] = (data["Date"].dt.quarter == 4).astype(int)

# Month Features
for i in range(12):
    month_col = "m" + str(i + 1)
    data[month_col] = (data["Date"].dt.month == i + 1).astype(int)

# Day Features
for i in range(31):
    day_col = "d" + str(i + 1)
    data[day_col] = (data["Date"].dt.day == i + 1).astype(int)

# Week Features
for i in range(52):
    week_col = "w" + str(i + 1)
    data[week_col] = (
        data["Date"].dt.isocalendar().week == i + 1
    ).astype(int)

# Trend
trend_values = []

for i in range(len(data)):
    trend_values.append(i + 1)

data["trend"] = trend_values

# Training


def Training(
    data_train,
    data_test,
    Trend="t"
):

    # Feature Columns
    feature_columns = (
        ["trend"]
        + [f"m{i}" for i in range(1, 13)]
        + [f"Q{i}" for i in range(1, 5)]
        + [f"d{i}" for i in range(1, 32)]
        + [f"w{i}" for i in range(1, 53)]
    )

    Xtrain = data_train[feature_columns]
    Xtest = data_test[feature_columns]

    Ytrain = data_train["Close"]
    Ytest = data_test["Close"]

    data_endog = Ytrain
    data_exog = Xtrain

    # Auto ARIMA
    auto_model = auto_arima(
        data_endog,
        exogenous=data_exog,
        seasonal=False,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=0,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True
    )

    best_order = auto_model.order

    AR = best_order[0]
    DIFF = best_order[1]
    MA = best_order[2]

    print("\nBest ARIMA Order :", best_order)

    # Final ARIMA Model
    model = ARIMA(
        endog=data_endog,
        exog=data_exog,
        order=(AR, DIFF, MA),
        trend=Trend,
        enforce_stationarity=False,
        concentrate_scale=False
    )

    model_fit = model.fit()

    print(model_fit.summary())

    # Prediction
    y_pred = model_fit.predict(
        start=len(Ytrain),
        end=len(Ytrain) + len(Ytest) - 1,
        exog=Xtest
    )
    # Metrics
    rmse = np.sqrt(mean_squared_error(Ytest, y_pred))
    mae = mean_absolute_error(Ytest, y_pred)
    mape = np.mean(np.abs((Ytest - y_pred) / Ytest)) * 100
    r2 = r2_score(Ytest, y_pred)

    print("\nModel Performance")
    print("=" * 40)
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R2   : {r2:.4f}")

    # Prediction DataFrame
    prediction_df = pd.DataFrame({
        "Date": data_test["Date"],
        "Actual Price": Ytest.values,
        "Predicted Price": y_pred.values
    })

    prediction_file = f"data/{config.COMPANY_NAME}_Auto_ARIMA_Predictions.csv"

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    metrics_df = pd.DataFrame({
        "Company": [config.COMPANY_NAME],
        "Best Order": [best_order],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE": [mape],
        "R2": [r2]
    })

    metrics_file = f"data/{config.COMPANY_NAME}_Auto_ARIMA_Metrics.csv"

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print(f"\nPredictions saved as {prediction_file}")
    print(f"Metrics saved as {metrics_file}")

    return model_fit, prediction_df


# split data
split_index = int(len(data) * 0.8)

data_train = data.iloc[:split_index]
data_test = data.iloc[split_index:]

model_fit, prediction_df = Training(
    data_train,
    data_test
)

# plot
plt.figure(figsize=(12, 6))

plt.plot(
    prediction_df["Date"],
    prediction_df["Actual Price"],
    label="Actual Price",
    color="blue"
)

plt.plot(
    prediction_df["Date"],
    prediction_df["Predicted Price"],
    label="Predicted Price",
    color="red"
)

plt.title(f"Auto ARIMA Prediction - {config.COMPANY_NAME}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
