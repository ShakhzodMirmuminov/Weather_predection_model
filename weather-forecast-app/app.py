# app.py

import streamlit as st
import pandas as pd
from ml_model import run_ml_pipeline # Import the master function

# Set the title and icon for the browser tab
st.set_page_config(
    page_title="ML Weather Forecast Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 1. DATA LOADING & CACHING
# ----------------------------------------------------

# Use st.cache_data or st.cache_resource to avoid retraining the model
# and re-fetching API data every time the page refreshes.
@st.cache_data(ttl=600) # Cache the result for 10 minutes
def load_and_predict_data():
    """Wrapper function to run the ML pipeline."""
    with st.spinner('⏳ Training Machine Learning Model and Fetching Data...'):
        data = run_ml_pipeline()
    return data

# Load all the processed data
data = load_and_predict_data()

# Check if data loading was successful
if data is None:
    st.error("🚨 **Error:** Failed to load weather data or train the ML model. Ensure the API key is valid and the historical data file (`./data/tashkent_weather_20years.csv`) exists.")
    st.stop() # Stop the app if there's an error

current_weather = data['current_weather']
rain_prediction = data['rain_prediction']
forecasts_data = data['forecasts']
model_metrics = data['model_metrics']

# Convert forecast list to DataFrame for better display
forecast_df = pd.DataFrame(forecasts_data)
# ----------------------------------------------------

# ----------------------------------------------------
# 2. PAGE LAYOUT AND DESIGN
# ----------------------------------------------------

st.title("🤖 ML-Powered Weather Forecast Demo")
st.markdown("### Predicting Weather for **Tashkent, Uzbekistan** using Linear Regression")

# --- Current Weather Dashboard ---
st.header(f"☀️ Current Weather: {current_weather['city']}, {current_weather['country']}")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

# Column 1: Main Temp
with col1:
    st.metric(
        label="🌡️ Current Temperature",
        value=f"{current_weather['current_temp']} °C",
        delta=f"Feels Like {current_weather['feels_like']} °C"
    )
    st.caption(f"Status: **{current_weather['description'].title()}**")

# Column 2: Wind & Pressure
with col2:
    st.metric(
        label="💨 Wind Speed",
        value=f"{current_weather['wind_speed']} m/s",
        delta=f"Direction: {current_weather['compass_direction']}"
    )
    st.metric(
        label="🌀 Pressure",
        value=f"{current_weather['pressure']} hPa"
    )

# Column 3: Humidity & Rain
with col3:
    st.metric(
        label="💧 Humidity",
        value=f"{current_weather['humidity']} %"
    )
    st.metric(
        label="🌧️ Rain Prediction",
        value=f"Likely: {rain_prediction}"
    )

# Column 4: Min/Max & Model Metrics
with col4:
    st.metric(
        label="🔥 Max / ❄️ Min",
        value=f"{current_weather['temp_max']} °C / {current_weather['temp_min']} °C"
    )
    st.info(f"**Model Performance (Test Set):**\n\n**MAE:** {model_metrics['MAE']}\n\n**R²:** {model_metrics['R2']}")
    
st.markdown("---")

# --- 2-Day Forecast Section ---
st.header("📅 2-Day Temperature Forecast (ML Prediction)")
st.markdown("Forecasted temperature for key periods using the trained Linear Regression Model.")

# Split forecasts into Today and Tomorrow
today_df = forecast_df[forecast_df['day'] == 'Today'].drop(columns=['day'])
tomorrow_df = forecast_df[forecast_df['day'] == 'Tomorrow'].drop(columns=['day'])

forecast_col1, forecast_col2 = st.columns(2)

with forecast_col1:
    st.subheader("Today's Forecast")
    st.dataframe(
        today_df,
        use_container_width=True,
        hide_index=True,
        column_order=('hour', 'period', 'temperature'),
        column_config={
            "hour": "Time",
            "period": "Period",
            "temperature": st.column_config.ProgressColumn(
                "Temperature (°C)",
                format="%.1f °C",
                min_value=min(forecast_df['temperature']) - 2,
                max_value=max(forecast_df['temperature']) + 2,
            ),
        }
    )
    
with forecast_col2:
    st.subheader("Tomorrow's Forecast")
    st.dataframe(
        tomorrow_df,
        use_container_width=True,
        hide_index=True,
        column_order=('hour', 'period', 'temperature'),
        column_config={
            "hour": "Time",
            "period": "Period",
            "temperature": st.column_config.ProgressColumn(
                "Temperature (°C)",
                format="%.1f °C",
                min_value=min(forecast_df['temperature']) - 2,
                max_value=max(forecast_df['temperature']) + 2,
            ),
        }
    )

# --- Interactive Plot ---
st.header("📈 Interactive Temperature Prediction Plot")
st.line_chart(
    forecast_df,
    x='hour',
    y='temperature',
    color='day',
    height=400
)

st.markdown("---")
st.caption(f"Last updated: {current_weather['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} (Data is cached for 10 minutes)")

# ----------------------------------------------------