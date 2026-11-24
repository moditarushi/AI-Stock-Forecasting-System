import pandas as pd
import os
import config
import sys
import numpy as np

# Get selected company
if len(sys.argv) > 1:
    selected_company = sys.argv[1]

    if selected_company not in config.COMPANIES:
        raise ValueError(f"Unknown company: {selected_company}")

    COMPANY_NAME, TICKER = config.COMPANIES[selected_company]
else:
    COMPANY_NAME = config.COMPANY_NAME

print(f"Feature Engineering - {COMPANY_NAME}")

# Load cleaned data

input_file = f"data/{COMPANY_NAME}_Stock_Clean.csv"
output_file = f"data/{COMPANY_NAME}_Features.csv"

if not os.path.exists(input_file):
    print("Clean dataset not found.")
    print("Run 02_data_preprocessing.py first.")
    exit()

data = pd.read_csv(input_file)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# Returns

data["Return"] = data["Close"].pct_change()
data["Return_5D"] = data["Close"].pct_change(5)
data["Return_20D"] = data["Close"].pct_change(20)


# Moving Averages

data["MA7"] = data["Close"].rolling(7).mean()
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()


# Exponential Moving Average

data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

# Volatility

data["Volatility_7"] = data["Return"].rolling(7).std()
data["Volatility_20"] = data["Return"].rolling(20).std()


# Price-based features

data["High_Low_Pct"] = (
    (data["High"] - data["Low"]) / data["Close"]
)

data["Open_Close_Pct"] = (
    (data["Close"] - data["Open"]) / data["Open"]
)


# Volume change

data["Volume_Change"] = data["Volume"].pct_change()

# RSI

delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# MACD

ema12 = data["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = data["Close"].ewm(
    span=26,
    adjust=False
).mean()

data["MACD"] = ema12 - ema26

data["MACD_Signal"] = data["MACD"].ewm(
    span=9,
    adjust=False
).mean()

# Lag Features
data["Return_Lag1"] = data["Return"].shift(1)
data["Return_Lag2"] = data["Return"].shift(2)
data["Return_Lag3"] = data["Return"].shift(3)
data["Return_Lag5"] = data["Return"].shift(5)
data["Return_Lag10"] = data["Return"].shift(10)
data["Return_Lag20"] = data["Return"].shift(20)

data["Volume_Lag1"] = data["Volume"].shift(1)

data["Close_Lag1"] = data["Close"].shift(1)
data["Close_Lag5"] = data["Close"].shift(5)

# Momentum Features
data["Momentum_5"] = data["Close"] / data["Close"].shift(5) - 1
data["Momentum_10"] = data["Close"] / data["Close"].shift(10) - 1

# Target: Next Day Return9          1


data["Target_Return"] = (
    data["Close"].shift(-1) / data["Close"]
) - 1


# Handle infinite values
data.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Remove rows with missing feature values.
# Keep the final row even though Target_Return is NaN.
required_columns = [
    col for col in data.columns
    if col != "Target_Return"
]

data.dropna(
    subset=required_columns,
    inplace=True
)

data.reset_index(drop=True, inplace=True)

# Display features

print("\nFeature Columns:")
print(data.columns.tolist())

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())


# Save

data.to_csv(output_file, index=False)
print("\nFeature Engineering Completed!")
print(f"Saved As: {output_file}")
