
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

# File paths
input_file = f"data/{COMPANY_NAME}_Stock.csv"
output_file = f"data/{COMPANY_NAME}_Stock_Clean.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"{input_file} not found. Run 01_data_loading.py first."
    )

# Load data
df = pd.read_csv(input_file)

# Clean data
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.drop_duplicates(subset=["Date"])
df = df.ffill()
df = df.dropna(subset=["Open", "High", "Low", "Close"])

# Save cleaned data
df.to_csv(output_file, index=False)

print("Data preprocessing completed!")
print(f"Company: {COMPANY_NAME}")
print(f"Latest date: {df['Date'].max()}")
print(f"Total records: {len(df)}")
print(f"Saved to: {output_file}")
