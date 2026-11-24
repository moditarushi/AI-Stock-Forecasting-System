
import pandas as pd
import os
import numpy as np

from xgboost import XGBClassifier
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
    TICKER = config.TICKER

# LOAD FEATURE DATA
input_file = f"data/{COMPANY_NAME}_Features.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"{input_file} not found. Run feature engineering first."
    )

data = pd.read_csv(input_file)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)


# FEATURES
feature_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "Return_5D", "Return_20D",
    "MA7", "MA20", "MA50", "EMA20",
    "Volatility_7", "Volatility_20",
    "High_Low_Pct", "Open_Close_Pct",
    "Volume_Change", "RSI", "MACD", "MACD_Signal"
]

# CHECK REQUIRED COLUMNS
required_columns = feature_columns + ["Target_Return"]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")


# PREPARE TRAINING DATA
data["Direction"] = (data["Target_Return"] > 0).astype(int)

training_data = data.dropna(
    subset=required_columns
).copy()

training_data = training_data.replace(
    [np.inf, -np.inf], np.nan
).dropna(subset=required_columns)

if len(training_data) < 100:
    raise ValueError("Not enough valid training data.")

X_train = training_data[feature_columns].astype(float)
y_train = training_data["Direction"]

if y_train.nunique() < 2:
    raise ValueError("Training data contains only one direction class.")


# LATEST AVAILABLE DATA
prediction_data = data.dropna(
    subset=feature_columns
).copy()

prediction_data = prediction_data.replace(
    [np.inf, -np.inf], np.nan
).dropna(subset=feature_columns)

if prediction_data.empty:
    raise ValueError("No valid data available for prediction.")

latest_row = prediction_data.iloc[-1]

X_future = pd.DataFrame(
    [latest_row[feature_columns].values],
    columns=feature_columns
).astype(float)


# TRAIN XGBOOST
print("Training XGBoost...")
print(f"Company: {COMPANY_NAME}")
print(f"Training rows: {len(X_train)}")

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


# FUTURE PREDICTION
prediction = model.predict(X_future)[0]
probabilities = model.predict_proba(X_future)[0]

down_probability = probabilities[0]
up_probability = probabilities[1]

predicted_direction = "UP" if prediction == 1 else "DOWN"

latest_date = latest_row["Date"]
latest_price = float(latest_row["Close"])


# DISPLAY RESULT
print("\nNEXT TRADING DAY PREDICTION")
print(f"Last available date: {latest_date.strftime('%d %b %Y')}")
print(f"Last available price: ₹{latest_price:.2f}")
print(f"Predicted direction: {predicted_direction}")
print(f"UP probability: {up_probability * 100:.2f}%")
print(f"DOWN probability: {down_probability * 100:.2f}%")


# SAVE PREDICTION
future_prediction = pd.DataFrame({
    "Company": [COMPANY_NAME],
    "Last Available Date": [latest_date],
    "Last Available Price": [latest_price],
    "Predicted Direction": [predicted_direction],
    "UP Probability": [up_probability],
    "DOWN Probability": [down_probability],
    "Model": ["XGBoost"]
})

output_file = f"data/{COMPANY_NAME}_Future_Prediction.csv"

future_prediction.to_csv(output_file, index=False)

print(f"\nFuture prediction saved: {output_file}")
