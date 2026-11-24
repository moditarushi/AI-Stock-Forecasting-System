import os
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"Exploratory Data Analysis - {config.COMPANY_NAME}")

# Load Clean Data
file_name = f"data/{config.COMPANY_NAME}_Stock_Clean.csv"

if not os.path.exists(file_name):
    print("Clean dataset not found.")
    print("Run 02_data_preprocessing.py first.")
    exit()

data = pd.read_csv(file_name)

# Convert Date
data["Date"] = pd.to_datetime(data["Date"])

print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nStatistical Summary:")
print(data.describe())

# Graph 1 : Closing Price
plt.figure(figsize=(14, 6))

plt.plot(data["Date"], data["Close"])

plt.title(f"{config.COMPANY_NAME} Closing Price")

plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.grid(True)

plt.show()

# Graph 2 : Trading Volume
plt.figure(figsize=(14, 6))

plt.plot(data["Date"], data["Volume"])
plt.title(f"{config.COMPANY_NAME} Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)

plt.show()

# Moving Averages

data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

plt.figure(figsize=(14, 6))
plt.plot(data["Date"], data["Close"], label="Close")
plt.plot(data["Date"], data["MA20"], label="20 Day MA")
plt.plot(data["Date"], data["MA50"], label="50 Day MA")
plt.plot(data["Date"], data["MA200"], label="200 Day MA")
plt.title(f"{config.COMPANY_NAME} Moving Averages")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
plt.boxplot(data["Close"])
plt.title(f"{config.COMPANY_NAME} Closing Price Box Plot")
plt.ylabel("Price")
plt.show()

# Histogram
plt.figure(figsize=(10, 6))

plt.hist(
    data["Close"],
    bins=30
)

plt.title(f"{config.COMPANY_NAME} Closing Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Daily Returns

data["Daily Return"] = data["Close"].pct_change()

plt.figure(figsize=(14, 6))

plt.plot(
    data["Date"],
    data["Daily Return"]
)

plt.title(f"{config.COMPANY_NAME} Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.show()

# Volatility
data["Volatility"] = data["Daily Return"].rolling(20).std()
plt.figure(figsize=(14, 6))
plt.plot(
    data["Date"],
    data["Volatility"]
)
plt.title(f"{config.COMPANY_NAME} 20-Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(f"{config.COMPANY_NAME} Correlation Heatmap")
plt.show()
print("\nEDA Completed Successfully.")
