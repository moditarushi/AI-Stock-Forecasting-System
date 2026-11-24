
import yfinance as yf
import pandas as pd
import os
import sys

import config

# Get selected company
if len(sys.argv) > 1:
    selected_company = sys.argv[1]

    if selected_company not in config.COMPANIES:
        raise ValueError(f"Unknown company: {selected_company}")

    COMPANY_NAME, TICKER = config.COMPANIES[selected_company]
else:
    COMPANY_NAME = config.COMPANY_NAME
    TICKER = config.TICKER

START_DATE = config.START_DATE
END_DATE = config.END_DATE

# Create data folder
os.makedirs("data", exist_ok=True)

print(f"Downloading data for {COMPANY_NAME} ({TICKER})...")

df = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)

if df.empty:
    raise ValueError(f"No stock data found for {TICKER}")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)
df.columns.name = None

output_file = f"data/{COMPANY_NAME}_Stock.csv"
df.to_csv(output_file, index=False)

print("Download completed!")
print(f"Latest available date: {df['Date'].max()}")
print(f"Total records: {len(df)}")
print(f"Saved to: {output_file}")
