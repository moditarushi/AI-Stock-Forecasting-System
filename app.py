import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import sys

from streamlit_autorefresh import st_autorefresh


# Page Configuration

st.set_page_config(
    page_title="AI Stock Forecasting",
    page_icon="📈",
    layout="wide"
)

# PROFESSIONAL DASHBOARD STYLING

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #0b1120;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1500px;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    /* Subtitle */
    .main-subtitle {
        font-size: 17px;
        color: #94a3b8;
        margin-bottom: 25px;
    }

    /* Section headings */
    h2, h3 {
        color: #f8fafc !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.20);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 27px;
        font-weight: 650;
    }

    /* Select box */
    [data-testid="stSelectbox"] {
        margin-bottom: 20px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Divider */
    hr {
        border-color: #1e293b;
    }

    /* Caption */
    .stCaption {
        color: #64748b;
    }

</style>
""", unsafe_allow_html=True)

# Title

st.markdown(
    '<div class="main-title">📈 AI Stock Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered market analytics, technical indicators & price forecasting'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ANALYSIS CONTROLS

st.markdown("### 🎯 Analysis Controls")

company_map = {
    "Reliance Industries": "Reliance",
    "Tata Consultancy Services": "TCS",
    "Infosys": "Infosys",
    "HDFC Bank": "HDFC_Bank",
    "ICICI Bank": "ICICI_Bank",
    "State Bank of India": "SBI",
    "Wipro": "Wipro",
    "HCL Technologies": "HCLTech"
}

col_control, col_status = st.columns([2.5, 1])

with col_control:

    selected_company = st.selectbox(
        "Select Stock",
        list(company_map.keys()),
        index=0
    )

company = company_map[selected_company]


# LOAD INTRADAY STOCK DATA

intraday_file = f"data/intraday/{company}_1m.csv"

if os.path.exists(intraday_file):
    intraday_data = pd.read_csv(intraday_file)

    # Convert timestamp
    timestamp_column = intraday_data.columns[0]
    intraday_data[timestamp_column] = pd.to_datetime(
        intraday_data[timestamp_column]
    )

else:
    intraday_data = pd.DataFrame()

# AUTOMATIC MARKET DATA REFRESH (Every 5 Minutes)

refresh_counter = st_autorefresh(
    interval=5 * 60 * 1000,  # 5 minutes in milliseconds
    key="market_data_refresh"
)

# Run pipeline only when the automatic refresh counter changes
if "last_refresh_counter" not in st.session_state:
    st.session_state.last_refresh_counter = refresh_counter

elif refresh_counter != st.session_state.last_refresh_counter:

    st.session_state.last_refresh_counter = refresh_counter

    with st.spinner("Automatically updating stock data and predictions..."):
        try:
            subprocess.run(
                [
                    sys.executable,
                    "pipeline.py",
                    selected_company
                ],
                check=True
            )

            st.success("Automatic market data update completed!")
            st.rerun()

        except subprocess.CalledProcessError:
            st.error("Automatic update failed. Check the terminal.")


# Refresh Market Data
if st.button("🔄 Refresh Market Data", type="primary"):

    with st.spinner(
        f"Updating {selected_company} data and generating predictions..."
    ):

        import subprocess
        import sys

        try:

            subprocess.run(
                [
                    sys.executable,
                    "pipeline.py",
                    selected_company
                ],
                check=True
            )

            # Refresh 1-minute intraday data
            subprocess.run(
                [
                    sys.executable,
                    "intraday_data.py",
                    selected_company
                ],
                check=True
            )

            st.success("Market data and predictions updated successfully!")

            st.rerun()

        except subprocess.CalledProcessError:
            st.error(
                "Pipeline failed. Please check the terminal for details."
            )

with col_status:

    st.markdown("**Data Status**")

    st.success(
        "● Dataset Ready"
    )

st.caption(
    f"Selected asset: **{selected_company}**  |  "
    f"Forecast model: **XGBoost + ARIMA**"
)

st.divider()

# File Paths

feature_file = (
    f"data/{company}_Features.csv"
)

direction_file = (
    f"data/{company}_XGBoost_Predictions.csv"
)

future_file = (
    f"data/{company}_Future_Prediction.csv"
)

comparison_file = (
    f"data/{company}_Model_Comparison.csv"
)


# Check Files

if not os.path.exists(feature_file):

    st.error(
        "Feature dataset not found. "
        "Run 07_feature_engineering.py first."
    )

    st.stop()


if not os.path.exists(direction_file):

    st.error(
        "Direction prediction file not found. "
        "Run 09_direction_prediction.py first."
    )

    st.stop()


# Load Data

data = pd.read_csv(feature_file)
predictions = pd.read_csv(direction_file)
future_prediction = pd.read_csv(future_file)
data["Date"] = pd.to_datetime(data["Date"])
predictions["Date"] = pd.to_datetime(
    predictions["Date"]
)


# Latest Data

latest = data.iloc[-1]
latest_rsi = latest["RSI"]
latest_macd = latest["MACD"]


# Future ML Prediction

future = future_prediction.iloc[0]

latest_price = future["Last Available Price"]

latest_date = pd.to_datetime(
    future["Last Available Date"]
)

predicted_direction = (
    future["Predicted Direction"]
)

up_probability = (
    future["UP Probability"] * 100
)

down_probability = (
    future["DOWN Probability"] * 100
)

# ARIMA Price Forecast
arima_file = f"data/{company}_ARIMA_Future_Prediction.csv"

arima_prediction = pd.read_csv(arima_file)

arima_price = arima_prediction.iloc[0]["Predicted Price"]
arima_date = arima_prediction.iloc[0]["Date"]

# MARKET OVERVIEW

st.subheader("📊 Market Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Current Price",
        f"₹{latest_price:,.2f}"
    )

with col2:
    signal_icon = "🟢" if predicted_direction == "UP" else "🔴"

    st.metric(
        "Next-Day Signal",
        f"{signal_icon} {predicted_direction}"
    )

with col3:
    st.metric(
        "UP Probability",
        f"{up_probability:.2f}%"
    )

with col4:
    st.metric(
        "ARIMA Forecast",
        f"₹{arima_price:,.2f}"
    )

with col5:
    st.metric(
        "RSI",
        f"{latest_rsi:.2f}"
    )

st.caption(
    f"Market data available through "
    f"{latest_date.strftime('%d %b %Y')} "
    f"| Forecast generated using XGBoost and ARIMA"
)


st.divider()

# DASHBOARD TABS

tab_overview, tab_technical, tab_forecast, tab_models = st.tabs(
    [
        "📊 Overview",
        "📈 Technical Analysis",
        "🔮 Forecast",
        "🤖 Model Performance"
    ]
)

with tab_overview:
    
    # ONE-MINUTE INTRADAY PRICE MONITORING

    st.subheader("⚡ Intraday Stock Price (1-Minute Data)")

    if not intraday_data.empty:

        fig_intraday = go.Figure()

        fig_intraday.add_trace(
            go.Scatter(
                x=intraday_data[intraday_data.columns[0]],
                y=intraday_data["Close"],
                mode="lines",
                name="1-Minute Price",
                line=dict(width=2)
            )
        )

        fig_intraday.update_layout(
            title=f"{selected_company} — Intraday Price",
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            height=400,
            template="plotly_dark",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_intraday,
            use_container_width=True
        )

        latest_intraday = intraday_data.iloc[-1]

        st.metric(
            "Latest Available Intraday Price",
            f"₹{latest_intraday['Close']:,.2f}"
        )

        st.caption(
            f"Last available update: "
            f"{latest_intraday[intraday_data.columns[0]]}"
        )

    else:
        st.warning(
            "Intraday data is not available. "
            "Run the intraday data downloader first."
        )

    # HISTORICAL PRICE ANALYSIS

    st.subheader("📈 Historical Price Analysis")

    fig_price = go.Figure()

    fig_price.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Closing Price",
            line=dict(width=2),
            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>Price:</b> ₹%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig_price.add_hline(
        y=latest_price,
        line_dash="dash",
        annotation_text=f"Current ₹{latest_price:,.2f}",
        annotation_position="top right"
    )

    fig_price.update_layout(
        title=f"{selected_company} — Closing Price",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(
                        count=1,
                        label="1M",
                        step="month",
                        stepmode="backward"
                    ),
                    dict(
                        count=6,
                        label="6M",
                        step="month",
                        stepmode="backward"
                    ),
                    dict(
                        count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward"
                    ),
                    dict(
                        label="All",
                        step="all"
                    )
                ]
            ),
            rangeslider=dict(visible=True)
        )
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True
        }
    )

# MOVING AVERAGE ANALYSIS
with tab_technical:
    st.subheader("📊 Moving Average Analysis")

ma_data = data[
    [
        "Date",
        "Close",
        "MA7",
        "MA20",
        "MA50"
    ]
].dropna()

fig_ma = go.Figure()

fig_ma.add_trace(
    go.Scatter(
        x=ma_data["Date"],
        y=ma_data["Close"],
        mode="lines",
        name="Closing Price",
        line=dict(width=2),
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y}<br>"
            "<b>Close:</b> ₹%{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=ma_data["Date"],
        y=ma_data["MA7"],
        mode="lines",
        name="MA 7",
        line=dict(width=1.5),
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y}<br>"
            "<b>MA7:</b> ₹%{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=ma_data["Date"],
        y=ma_data["MA20"],
        mode="lines",
        name="MA 20",
        line=dict(width=1.5),
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y}<br>"
            "<b>MA20:</b> ₹%{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_ma.add_trace(
    go.Scatter(
        x=ma_data["Date"],
        y=ma_data["MA50"],
        mode="lines",
        name="MA 50",
        line=dict(width=1.5),
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y}<br>"
            "<b>MA50:</b> ₹%{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

fig_ma.update_layout(
    title=f"{selected_company} — Moving Average Trends",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    height=500,
    hovermode="x unified",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=60, b=20),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),

    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(
                    count=1,
                    label="1M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=6,
                    label="6M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward"
                ),
                dict(
                    label="All",
                    step="all"
                )
            ]
        ),
        rangeslider=dict(
            visible=True
        )
    )
)

st.plotly_chart(
    fig_ma,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True
    }
)


# RSI ANALYSIS

st.subheader("📉 Relative Strength Index (RSI)")

# RSI interpretation
if latest_rsi >= 70:
    rsi_status = "Overbought"
    rsi_icon = "🔴"
elif latest_rsi <= 30:
    rsi_status = "Oversold"
    rsi_icon = "🟢"
else:
    rsi_status = "Neutral"
    rsi_icon = "🟡"

rsi_col1, rsi_col2 = st.columns([3, 1])

with rsi_col1:

    fig_rsi = go.Figure()

    fig_rsi.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(width=2),
            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>RSI:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Overbought level
    fig_rsi.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Overbought — 70",
        annotation_position="top right"
    )

    # Neutral level
    fig_rsi.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="Neutral — 50",
        annotation_position="bottom right"
    )

    # Oversold level
    fig_rsi.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Oversold — 30",
        annotation_position="bottom right"
    )

    fig_rsi.update_layout(
        title=f"{selected_company} — RSI Trend",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=450,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),

        yaxis=dict(
            range=[0, 100]
        ),

        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(
                        count=1,
                        label="1M",
                        step="month",
                        stepmode="backward"
                    ),
                    dict(
                        count=6,
                        label="6M",
                        step="month",
                        stepmode="backward"
                    ),
                    dict(
                        count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward"
                    ),
                    dict(
                        label="All",
                        step="all"
                    )
                ]
            )
        )
    )

    st.plotly_chart(
        fig_rsi,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True
        }
    )

with rsi_col2:

    st.markdown("### Current RSI")

    st.metric(
        "RSI Value",
        f"{latest_rsi:.2f}"
    )

    st.markdown(
        f"### {rsi_icon} {rsi_status}"
    )

    st.caption(
        "RSI above 70 may indicate "
        "overbought conditions. "
        "RSI below 30 may indicate "
        "oversold conditions."
    )


# MACD ANALYSIS

st.subheader("📊 Moving Average Convergence Divergence (MACD)")

macd_data = data[
    [
        "Date",
        "MACD",
        "MACD_Signal"
    ]
].dropna().copy()

# Calculate MACD histogram
macd_data["Histogram"] = (
    macd_data["MACD"] -
    macd_data["MACD_Signal"]
)

latest_macd_value = macd_data["MACD"].iloc[-1]
latest_signal_value = macd_data["MACD_Signal"].iloc[-1]
latest_histogram = macd_data["Histogram"].iloc[-1]

# MACD interpretation
if latest_macd_value > latest_signal_value:
    macd_status = "Bullish"
    macd_icon = "🟢"
else:
    macd_status = "Bearish"
    macd_icon = "🔴"

macd_col1, macd_col2 = st.columns([3, 1])

with macd_col1:

    fig_macd = go.Figure()

    # MACD line
    fig_macd.add_trace(
        go.Scatter(
            x=macd_data["Date"],
            y=macd_data["MACD"],
            mode="lines",
            name="MACD",
            line=dict(width=2),
            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>MACD:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Signal line
    fig_macd.add_trace(
        go.Scatter(
            x=macd_data["Date"],
            y=macd_data["MACD_Signal"],
            mode="lines",
            name="Signal",
            line=dict(width=2),
            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>Signal:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Histogram
    fig_macd.add_trace(
        go.Bar(
            x=macd_data["Date"],
            y=macd_data["Histogram"],
            name="Histogram",
            opacity=0.45,
            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}<br>"
                "<b>Histogram:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Zero line
    fig_macd.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_macd.update_layout(
        title=f"{selected_company} — MACD Trend",
        xaxis_title="Date",
        yaxis_title="MACD",
        height=450,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),

        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(
                        count=1,
                        label="1M",
                        step="month",
                        stepmode="backward"
                    ),
                    dict(
                        count=6,
                        label="6M",
                        step="month",
                        stepmode="backward"
                    ),
                    dict(
                        count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward"
                    ),
                    dict(
                        label="All",
                        step="all"
                    )
                ]
            )
        )
    )

    st.plotly_chart(
        fig_macd,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True
        }
    )

with macd_col2:

    st.markdown("### Current MACD")

    st.metric(
        "MACD",
        f"{latest_macd_value:.2f}"
    )

    st.metric(
        "Signal",
        f"{latest_signal_value:.2f}"
    )

    st.metric(
        "Histogram",
        f"{latest_histogram:.2f}"
    )

    st.markdown(
        f"### {macd_icon} {macd_status}"
    )

    st.caption(
        "MACD above the signal line "
        "indicates bullish momentum, while "
        "MACD below the signal line indicates "
        "bearish momentum."
    )

# Prediction Signal

with tab_forecast:

    st.subheader("🔮 Next Trading Day Prediction")
    st.caption("Model used: XGBoost")

    if predicted_direction == "UP":

        st.success(
            f"🟢 Predicted Direction: UP\n\n"
            f"Probability of UP: {up_probability:.2f}%\n\n"
            f"Probability of DOWN: {down_probability:.2f}%"
        )

    else:

        st.error(
            f"🔴 Predicted Direction: DOWN\n\n"
            f"Probability of DOWN: {down_probability:.2f}%\n\n"
            f"Probability of UP: {up_probability:.2f}%"
        )


st.subheader("📈 ARIMA Price Forecast")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Predicted Price",
        f"₹{arima_price:,.2f}"
    )

with col2:
    st.metric(
        "Forecast Date",
        str(arima_date)
    )


# Model Comparison

st.subheader("🏆 Model Comparison")

if os.path.exists(comparison_file):

    comparison = pd.read_csv(
        comparison_file
    )

    comparison["Accuracy (%)"] = (
        comparison["Accuracy"] * 100
    )

    st.dataframe(
        comparison[
            [
                "Model",
                "Accuracy (%)"
            ]
        ],
        use_container_width=True
    )

    fig_model = px.bar(
        comparison,
        x="Model",
        y="Accuracy (%)",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True
    )


# Recent Predictions

st.subheader("📊 Recent XGBoost Backtest Predictions")

recent_predictions = predictions.tail(10).copy()

recent_predictions["UP Probability"] = (
    recent_predictions["UP Probability"] * 100
).round(2)

recent_predictions["UP Probability"] = (
    recent_predictions["UP Probability"].astype(str) + "%"
)

st.dataframe(
    recent_predictions,
    use_container_width=True
)


# Download

csv = predictions.to_csv(
    index=False
)

st.download_button(
    "⬇️ Download Predictions",
    csv,
    file_name=f"{company}_XGBoost_Predictions.csv",
    mime="text/csv"
)


# Footer

st.divider()
st.caption(
    "AI Stock Forecasting Project | "
    "ARIMA + Random Forest + XGBoost"
)
