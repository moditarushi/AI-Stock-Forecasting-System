
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
    selected_company = "Reliance Industries"
    COMPANY_NAME, TICKER = config.COMPANIES[selected_company]

# Download recent 1-minute stock data
print(f"Fetching 1-minute data for {COMPANY_NAME}...")

df = yf.download(
    TICKER,
    period="5d",
    interval="1m",
    auto_adjust=False,
    progress=False,
    prepost=False
)

if df.empty:
    raise ValueError(
        f"No intraday data available for {TICKER}. "
        "Check market hours or Yahoo Finance availability."
    )

# Handle MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Prepare data
df = df.reset_index()
df.columns.name = None

# Save intraday data separately
os.makedirs("data/intraday", exist_ok=True)

output_file = f"data/intraday/{COMPANY_NAME}_1m.csv"

df.to_csv(output_file, index=False)

print("Intraday download completed!")
print("Latest available record:", df.iloc[-1, 0])
print("Total records:", len(df))
print("Saved to:", output_file)
