import pandas as pd
import matplotlib.pyplot as plt

from pandas.tseries.offsets import BDay
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
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

# LOAD DATA

file_name = f"data/{COMPANY_NAME}_Stock_Clean.csv"
data = pd.read_csv(file_name)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data = data.dropna(subset=["Close"])


print("ARIMA FUTURE PRICE FORECAST")
print(f"Company: {COMPANY_NAME}")
print(f"Observations: {len(data)}")

# CLOSE PRICE SERIES

y = data["Close"].astype(float)


# AUTO ARIMA

print("\nSearching for best ARIMA model...")

auto_model = auto_arima(
    y,
    seasonal=False,
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5,
    d=None,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

best_order = auto_model.order

print("\nBest ARIMA Order:", best_order)


# FINAL ARIMA MODEL

model = ARIMA(
    y,
    order=best_order
)

model_fit = model.fit()


# NEXT BUSINESS DAY

last_date = data.iloc[-1]["Date"]

forecast_date = (
    pd.to_datetime(last_date) + BDay(1)
)

# FORECAST

forecast = model_fit.forecast(
    steps=1
)

predicted_price = float(forecast.iloc[0])


# DISPLAY

print("\n====================================")
print("ARIMA FORECAST")
print("====================================")

print(
    f"Last available date : "
    f"{last_date.strftime('%d %b %Y')}"
)

print(
    f"Last closing price  : "
    f"₹{y.iloc[-1]:.2f}"
)

print(
    f"Forecast date       : "
    f"{forecast_date.strftime('%d %b %Y')}"
)

print(
    f"Predicted price     : "
    f"₹{predicted_price:.2f}"
)

# SAVE FORECAST

future_df = pd.DataFrame({
    "Date": [forecast_date],
    "Predicted Price": [predicted_price],
    "Model": [f"ARIMA{best_order}"]
})

output_file = (
    f"data/{COMPANY_NAME}_ARIMA_Future_Prediction.csv"
)

future_df.to_csv(
    output_file,
    index=False
)

print("\nForecast saved as:")
print(output_file)


# PLOT

plt.figure(figsize=(12, 6))

plt.plot(
    data["Date"],
    data["Close"],
    label="Historical Price"
)

plt.scatter(
    forecast_date,
    predicted_price,
    s=100,
    label="ARIMA Forecast"
)

plt.annotate(
    f"₹{predicted_price:.2f}",
    (forecast_date, predicted_price),
    xytext=(10, 10),
    textcoords="offset points"
)

plt.title(
    f"{COMPANY_NAME} ARIMA Future Forecast"
)

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
