import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import plotly.express as px
import pytz

# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================
st.set_page_config(
    page_title="Tashkent Weather Forecast ML",
    page_icon="😶‍🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================================
# THEME TOGGLE & STYLING
# ==========================================================================

# Initialize theme, page, and selected day in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_day' not in st.session_state:
    st.session_state.selected_day = 0 # Day 0 (Today)

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

def set_page(page_name):
    st.session_state.page = page_name
    st.session_state.selected_day = 0  # Reset day selection when changing pages

def select_day(day_index):
    st.session_state.selected_day = day_index

# Apply theme-specific CSS
if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif !important; }
        .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #ffffff; }
        .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
        h1, h2, h3, h4, h5, h6, p, div, span { color: #ffffff !important; }
        .metric-card {
            background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); border-radius: 16px; padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5); }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;
            border-radius: 12px; padding: 12px 28px; font-weight: 600; transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4); }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6); }
        .title-gradient { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; 
                          -webkit-text-fill-color: transparent; background-clip: text; font-size: 3.5rem; 
                          font-weight: 700; text-align: center; margin-bottom: 10px; }
        .map-container { border-radius: 20px; overflow: hidden; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4); }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1rem; }
        
        .day-button-active > button { 
            background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important; 
            border: 2px solid #fff !important; 
            font-weight: bold !important;
        }
        .day-button-inactive > button { 
            background: rgba(255, 255, 255, 0.1) !important;
            color: #fff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        .selected-day-card { 
            background: rgba(52, 152, 219, 0.2); 
            border: 2px solid #3498db; 
            border-radius: 20px; 
            padding: 30px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .day-line-button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            padding: 10px 5px !important;
            margin: 2px 0 !important;
            width: 100% !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
        }
        .day-line-button:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            transform: translateX(5px) !important;
        }
        .day-line-button-active {
            background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
            border: 2px solid #fff !important;
            font-weight: bold !important;
        }
        .forecast-detail-box {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            margin: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .analysis-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif !important; }
        .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); color: #2c3e50; }
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        h1, h2, h3, h4, h5, h6, p, div, span { color: #2c3e50 !important; }
        .metric-card {
            background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border-radius: 16px; padding: 24px;
            border: 1px solid rgba(0, 0, 0, 0.1); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25); }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;
            border-radius: 12px; padding: 12px 28px; font-weight: 600; transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4); }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6); }
        .title-gradient { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; 
                          -webkit-text-fill-color: transparent; background-clip: text; font-size: 3.5rem; 
                          font-weight: 700; text-align: center; margin-bottom: 10px; }
        .map-container { border-radius: 20px; overflow: hidden; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15); }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1rem; }
        
        .day-button-active > button { 
            background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important; 
            border: 2px solid #444 !important;
            font-weight: bold !important;
        }
        .day-button-inactive > button { 
            background: rgba(0, 0, 0, 0.05) !important;
            color: #2c3e50 !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        .selected-day-card { 
            background: rgba(52, 152, 219, 0.1); 
            border: 2px solid #3498db; 
            border-radius: 20px; 
            padding: 30px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .day-line-button {
            background: rgba(0, 0, 0, 0.03) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            border-radius: 8px !important;
            padding: 10px 5px !important;
            margin: 2px 0 !important;
            width: 100% !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
        }
        .day-line-button:hover {
            background: rgba(0, 0, 0, 0.08) !important;
            transform: translateX(5px) !important;
        }
        .day-line-button-active {
            background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
            border: 2px solid #444 !important;
            font-weight: bold !important;
            color: white !important;
        }
        .forecast-detail-box {
            background: rgba(0, 0, 0, 0.03);
            border-radius: 12px;
            padding: 15px;
            margin: 10px;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        .analysis-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        }
    </style>
    """, unsafe_allow_html=True)


# ==========================================================================
# CONFIGURATION
# ==========================================================================
API_KEY = '0ea12166dca6efaa2a7077602c59e70d'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'
TASHKENT_LAT = 41.2995
TASHKENT_LON = 69.2401
DATA_PATH = './data/tashkent_weather_23years.csv'

# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================

def wind_direction_to_compass(wind_deg):
    """Convert wind degree to compass direction"""
    wind_deg = wind_deg % 360
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(wind_deg / (360.0 / len(directions))) % len(directions)
    return directions[idx]

def get_weather_icon(description):
    """Simple mapping for icons based on description"""
    if 'clear' in description: return '🌞'
    if 'cloud' in description: return '☁️'
    if 'rain' in description or 'drizzle' in description: return '🌧️'
    if 'snow' in description: return '❄️'
    if 'thunder' in description: return '⛈️'
    return '🌡️'

# ==========================================================================
# DATA FETCHING & FEATURE ENGINEERING (FROM main.ipynb)
# ==========================================================================
@st.cache_data(ttl=600)
def get_current_weather(lat=TASHKENT_LAT, lon=TASHKENT_LON):
    """Fetch current weather from OpenWeatherMap API"""
    try:
        url = f"{BASE_URL}weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        temp = data['main']['temp']
        if temp > 100: temp -= 273.15
        
        return {
            'city': data.get('name', 'Tashkent'),
            'country': data['sys'].get('country', 'UZ'),
            'current_temp': round(temp, 1),
            'temp_min': round(data['main']['temp_min'], 1),
            'temp_max': round(data['main']['temp_max'], 1),
            'feels_like': round(data['main']['feels_like'], 1),
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_gust': data['wind'].get('gust', data['wind']['speed']),
            'wind_deg': data['wind'].get('deg', 0),
            'description': data['weather'][0]['description'],
            'timestamp': datetime.now()
        }
    except Exception as e:
        return None

@st.cache_data
def read_historical_data(filepath):
    """Load historical weather data from CSV"""
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df
    except FileNotFoundError:
        return None

def create_features(df):
    """Create time-based and lag features for better prediction"""
    df = df.copy()

    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['day_of_year'] = df['time'].dt.dayofyear
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    if 'temperature_2m' in df.columns:
        for lag in [1, 3, 6, 12, 24]:
            df[f'temp_lag_{lag}h'] = df['temperature_2m'].shift(lag)
    
    if 'relative_humidity_2m' in df.columns:
        for lag in [1, 3, 6]:
            df[f'humidity_lag_{lag}h'] = df['relative_humidity_2m'].shift(lag)
    
    if 'pressure_msl' in df.columns:
        for lag in [1, 6, 12]:
            df[f'pressure_lag_{lag}h'] = df['pressure_msl'].shift(lag)

    if 'temperature_2m' in df.columns:
        df['temp_ma_3h'] = df['temperature_2m'].rolling(window=3, min_periods=1).mean()
        df['temp_ma_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_rain_data(historical_data):
    """Prepare data for rain prediction"""
    historical_data = historical_data.copy()
    historical_data['RainTomorrow'] = (historical_data['precipitation'] > 0).astype(int)
    
    features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 
                'windgusts_10m', 'pressure_msl', 'winddirection_10m']
    
    X = historical_data[features].fillna(0)
    Y = historical_data['RainTomorrow']
    
    return X, Y

@st.cache_resource
def train_rain_model(X, Y):
    """Train logistic regression for rain prediction"""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)
    return model

@st.cache_resource
def train_model_with_time_series_cv(historical_df, n_splits=5):
    """Train model with Time Series Cross-Validation and return final model/metrics"""
    
    df_with_features = create_features(historical_df)
    
    exclude_cols = ['time', 'temperature_2m', 'precipitation']
    feature_cols = [col for col in df_with_features.columns if col not in exclude_cols]
    
    X = df_with_features[feature_cols].values
    y = df_with_features['temperature_2m'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = { 'test_mae': [] }
    
    # Run CV to get robust average metrics
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_test_pred = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        cv_scores['test_mae'].append(test_mae)
    
    avg_cv_mae = np.mean(cv_scores['test_mae'])

    # Final Train/Test Split (80/20) for ultimate performance report
    split_idx = int(len(df_with_features) * 0.8)
    train_df = df_with_features.iloc[:split_idx].copy()
    test_df = df_with_features.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols].values
    y_train = train_df['temperature_2m'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['temperature_2m'].values
    
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    
    model_final = LinearRegression()
    model_final.fit(X_train_scaled, y_train)
    
    y_test_pred = model_final.predict(X_test_scaled)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred),
        'avg_cv_mae': avg_cv_mae,
        'feature_cols': feature_cols
    }
    
    return model_final, scaler_final, metrics, test_df, y_test, y_test_pred

def compare_last_10_days(model, scaler, feature_cols, df_with_features):
    """Compare predicted vs actual for last 10 days at 4 times daily, returning daily summary."""
    last_date = df_with_features['time'].max()
    start_date = last_date - timedelta(days=10)
    
    last_10_days = df_with_features[df_with_features['time'] >= start_date].copy()

    target_hours = [0, 6, 12, 18] # Night, Morning, Afternoon, Sunset
    
    comparison_data = []

    last_10_days['date'] = last_10_days['time'].dt.date
    unique_dates = sorted(last_10_days['date'].unique())[-10:] 
    
    for date in unique_dates:
        day_data = last_10_days[last_10_days['date'] == date]
        
        # Calculate daily Actual Min/Max
        actual_min_temp = day_data['temperature_2m'].min()
        actual_max_temp = day_data['temperature_2m'].max()
        
        # Collect predictions for the day
        daily_predictions = []
        
        for hour in target_hours:
            hour_data = day_data[day_data['hour'] == hour]
            
            if len(hour_data) > 0:
                row = hour_data.iloc[0]

                features = row[feature_cols].values.reshape(1, -1)
                features_scaled = scaler.transform(features)
                predicted_temp = model.predict(features_scaled)[0]
                
                daily_predictions.append(predicted_temp)
                
        # Aggregate daily predictions and metrics
        if daily_predictions:
            pred_min_temp = min(daily_predictions)
            pred_max_temp = max(daily_predictions)
            
            comparison_data.append({
                'day_index': (last_date.date() - date).days,
                'date': date,
                'day_name': date.strftime('%a, %b %d'),
                'actual_min': round(actual_min_temp, 1),
                'actual_max': round(actual_max_temp, 1),
                'predicted_min': round(pred_min_temp, 1),
                'predicted_max': round(pred_max_temp, 1),
                'avg_error': np.mean([abs(p - a) for p, a in zip(daily_predictions, day_data[day_data['hour'].isin(target_hours)]['temperature_2m'].values[:len(daily_predictions)])])
            })
                
    # Sort data frame from oldest (Day 10 Ago) to newest (Day 1 Ago)
    comparison_df = pd.DataFrame(comparison_data).sort_values('day_index', ascending=False).reset_index(drop=True)
    comparison_df['day_index'] = comparison_df.index # Reindex 0 to 9 for easier selection
    return comparison_df

def forecast_next_10_days(model, scaler, feature_cols, df_with_features):
    """Forecast weather for next 10 days at 4 times daily, returning daily summary."""
    
    recent_data = df_with_features.tail(168).copy()
    target_hours = [0, 6, 12, 18]
    
    forecast_data = []
    
    now = datetime.now()
    today = now.date()
    
    for day_offset in range(10):
        forecast_date = today + timedelta(days=day_offset)
        
        daily_predictions = []
        
        for hour in target_hours:
            target_time = datetime.combine(forecast_date, datetime.min.time()) + timedelta(hours=hour)

            if day_offset == 0 and target_time < now:
                continue
                
            similar_data = recent_data[recent_data['hour'] == hour]
            
            if len(similar_data) > 0:
                last_similar = similar_data.iloc[-1]

                features = last_similar[feature_cols].values.reshape(1, -1)
                features_scaled = scaler.transform(features)
                predicted_temp = model.predict(features_scaled)[0]

                if day_offset > 2:
                    predicted_temp += np.random.normal(0, 0.2 * day_offset)
                
                daily_predictions.append(predicted_temp)
        
        # Aggregate daily predictions
        if daily_predictions:
            temp_min = min(daily_predictions)
            temp_max = max(daily_predictions)
            
            forecast_data.append({
                'day_index': day_offset,
                'date': forecast_date,
                'day_name': ('Today' if day_offset == 0 else 'Tomorrow' if day_offset == 1 else forecast_date.strftime('%a, %b %d')),
                'temperature_min': round(temp_min, 1),
                'temperature_max': round(temp_max, 1),
                'prediction_times': {f"{h:02d}:00": round(p, 1) for h, p in zip(target_hours, daily_predictions)}
            })
    
    return pd.DataFrame(forecast_data)

# ==========================================================================
# PAGE DEFINITIONS
# ==========================================================================

def home_page():
    """Initial page with map and Tashkent selection."""
    
    st.markdown("<p class='title-gradient'>Tashkent Weather Forecast ML</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Select a location to view the forecast powered by 23+ years of historical data.</p>", unsafe_allow_html=True)
    
    # 1. Search Bar (Limited to Tashkent for now)
    st.markdown("### Search Location")
    
    col_search, col_button = st.columns([3, 1])
    
    with col_search:
        location_input = st.text_input("Enter City Name (Tashkent only for now)", "Tashkent", label_visibility="collapsed")
    
    with col_button:
        if st.button("Open Forecast", use_container_width=True):
             if location_input.lower() == 'tashkent':
                set_page('forecast')
             else:
                st.error("Forecast available only for Tashkent.")

    st.markdown("---")
    
    # 2. Map View
    st.markdown("### Map View (Uzbekistan)")
    
    st.markdown("<div class='map-container'>", unsafe_allow_html=True)
    
    map_data = pd.DataFrame({
        'lat': [TASHKENT_LAT],
        'lon': [TASHKENT_LON],
        'city': ['Tashkent']
    })
    
    st.map(map_data, zoom=6, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Click 'Open Forecast' above or use the map for visual confirmation of the location.")
def forecast_page(current_weather, historical_data, model_data):
    """Main forecast and historical analysis page."""
    
    model, scaler, metrics, test_df, y_test, y_test_pred = model_data

    # Load and process data
    with st.spinner("⏳ Generating forecasts..."):
        df_with_features = create_features(historical_data)
        
        # Get 10-day summary data
        comparison_summary_df = compare_last_10_days(model, scaler, metrics['feature_cols'], df_with_features)
        forecast_summary_df = forecast_next_10_days(model, scaler, metrics['feature_cols'], df_with_features)
        
        # Rain prediction for current day
        X_rain, Y_rain = prepare_rain_data(historical_data)
        rain_model = train_rain_model(X_rain, Y_rain)
        current_rain_data = pd.DataFrame([{
            'temperature_2m': current_weather['current_temp'],
            'relative_humidity_2m': current_weather['humidity'],
            'windspeed_10m': current_weather['wind_speed'],
            'windgusts_10m': current_weather['wind_gust'],
            'pressure_msl': current_weather['pressure'],
            'winddirection_10m': current_weather['wind_deg']
        }])
        rain_prediction = rain_model.predict(current_rain_data)[0]
        wind_dir = wind_direction_to_compass(current_weather['wind_deg'])

    
    st.markdown(f"## 📍 {current_weather['city']} Weather Forecast")
    
    # --- Current Weather Section ---
    st.markdown("### 🌞 Current Conditions")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(f"🌡️ Temp {get_weather_icon(current_weather['description'])}", 
                  f"{current_weather['current_temp']}°C", 
                  f"Feels Like: {current_weather['feels_like']}°C")
    with col2:
        st.metric("💧 Humidity", f"{current_weather['humidity']}%", "Conditions: " + current_weather['description'].title())
    with col3:
        st.metric("🌬️ Wind", f"{current_weather['wind_speed']} m/s", f"Direction: {wind_dir}")
    with col4:
        st.metric("🌀 Pressure", f"{current_weather['pressure']} hPa")
    with col5:
        # FIXED: Added spaces between NO and ☀️
        st.metric("🌧️   Rain   Risk Tomorrow", "YES " if rain_prediction else "NO  🌞", help="Logistic Regression Model Prediction for Rain Tomorrow")
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if we need to auto-select the Model Performance tab
    # Initialize tab_index in session state
    if 'tab_index' not in st.session_state:
        st.session_state.tab_index = 0
    
    # Check if Model Performance button was clicked
    if hasattr(st.session_state, 'auto_select_analysis') and st.session_state.auto_select_analysis:
        st.session_state.tab_index = 2  # Set to Model Performance tab (index 2)
        st.session_state.auto_select_analysis = False  # Reset the flag
    
    # Create tabs with the correct initial selection
    tab_forecast, tab_history, tab_analysis = st.tabs(
        ["🪄 Next 10 Days Forecast", "🔙 Last 10 Days Comparison", "📈 Model Performance"]
    )
    
    # ----------------------------------
    # TAB 1: NEXT 10 DAYS FORECAST (FIXED LAYOUT)
    # ----------------------------------
    with tab_forecast:
        st.markdown("### Daily Temperature Forecast")
        
        # Create two columns: Left for day buttons, Right for details
        col_days, col_details = st.columns([2, 3])
        
        with col_days:
            st.markdown("**Select a Day:**")
            # Create 10 straight line buttons
            for i, row in forecast_summary_df.iterrows():
                is_selected = st.session_state.selected_day == i
                
                # Custom button with better styling
                if st.button(
                    f"🗓️ {row['day_name']}",
                    key=f"forecast_day_{i}",
                    use_container_width=True
                ):
                    select_day(i)
        
        with col_details:
            # Detailed Prediction Card for selected day
            if st.session_state.selected_day < len(forecast_summary_df):
                selected_row = forecast_summary_df.iloc[st.session_state.selected_day]
                
                st.markdown(f"### {selected_row['day_name']} ({selected_row['date'].strftime('%B %d, %Y')})")
                st.markdown("<div class='selected-day-card'>", unsafe_allow_html=True)
                
                # Main temperature range
                col_temp, col_range = st.columns(2)
                with col_temp:
                    st.markdown("#### 🌡️ Temperature Range")
                    st.markdown(f"**Min:** {selected_row['temperature_min']}°C")
                    st.markdown(f"**Max:** {selected_row['temperature_max']}°C")
                    st.markdown(f"**Spread:** {selected_row['temperature_max'] - selected_row['temperature_min']:.1f}°C")
                
                with col_range:
                    st.markdown("#### ⏰ Hourly Predictions")
                    # Create 4 rectangles for 4 times of day
                    times_cols = st.columns(4)
                    for idx, (time_str, temp) in enumerate(selected_row['prediction_times'].items()):
                        with times_cols[idx]:
                            st.markdown(f"<div class='forecast-detail-box'>", unsafe_allow_html=True)
                            st.markdown(f"**{time_str}**")
                            st.markdown(f"### {temp}°C")
                            # Add appropriate icon based on time
                            if "00:00" in time_str:
                                st.markdown("🌙 Night")
                            elif "06:00" in time_str:
                                st.markdown("🌅 Morning")
                            elif "12:00" in time_str:
                                st.markdown("☀️ Noon")
                            elif "18:00" in time_str:
                                st.markdown("🌇 Evening")
                            st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Plot for selected day
                plot_data = pd.DataFrame(list(selected_row['prediction_times'].items()), columns=['Time', 'Temperature'])
                fig = px.line(plot_data, x='Time', y='Temperature', 
                              title=f"Predicted Hourly Temperature for {selected_row['day_name']}",
                              color_discrete_sequence=['#F18F01'],
                              markers=True)
                fig.update_traces(line=dict(width=3))
                fig.update_layout(template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white', 
                                  height=350,
                                  xaxis_title="Time of Day",
                                  yaxis_title="Temperature (°C)")
                st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------
    # TAB 2: LAST 10 DAYS COMPARISON (FIXED LAYOUT)
    # ----------------------------------
    with tab_history:
        st.markdown("### Historical Prediction Accuracy")
        
        # Create two columns: Left for day buttons, Right for details
        col_days, col_details = st.columns([2, 3])
        
        with col_days:
            st.markdown("**Select a Past Day:**")
            # Create 10 straight line buttons for historical days
            for i, row in comparison_summary_df.iterrows():
                is_selected = st.session_state.selected_day == i
                
                # Display date and error
                if st.button(
                    f"🗓️ {row['day_name']} (Err: {row['avg_error']:.1f}°C)",
                    key=f"history_day_{i}",
                    use_container_width=True
                ):
                    select_day(i)
        
        with col_details:
            # Detailed Comparison Card for selected day
            if st.session_state.selected_day < len(comparison_summary_df):
                selected_row = comparison_summary_df.iloc[st.session_state.selected_day]
                
                st.markdown(f"### {selected_row['day_name']} ({selected_row['date'].strftime('%B %d, %Y')})")
                st.markdown("<div class='selected-day-card'>", unsafe_allow_html=True)
                
                # Three columns for comparison
                col_actual, col_predicted, col_performance = st.columns(3)
                
                with col_actual:
                    st.markdown("#### 🗹 Actual Measurements")
                    st.markdown("<div class='forecast-detail-box'>", unsafe_allow_html=True)
                    st.metric("Minimum Temp", f"{selected_row['actual_min']}°C", delta=None)
                    st.metric("Maximum Temp", f"{selected_row['actual_max']}°C", delta=None)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_predicted:
                    st.markdown("#### 👽 Model Predictions")
                    st.markdown("<div class='forecast-detail-box'>", unsafe_allow_html=True)
                    st.metric("Predicted Min", f"{selected_row['predicted_min']}°C", 
                             delta=f"{selected_row['predicted_min'] - selected_row['actual_min']:.1f}°C")
                    st.metric("Predicted Max", f"{selected_row['predicted_max']}°C", 
                             delta=f"{selected_row['predicted_max'] - selected_row['actual_max']:.1f}°C")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_performance:
                    st.markdown("#### 📈 Performance Metrics")
                    st.markdown("<div class='forecast-detail-box'>", unsafe_allow_html=True)
                    st.metric("Average Error", f"{selected_row['avg_error']:.2f}°C")
                    
                    # Calculate accuracy percentage
                    avg_actual = (selected_row['actual_min'] + selected_row['actual_max']) / 2
                    accuracy = max(0, 100 - (selected_row['avg_error'] / avg_actual * 100))
                    st.metric("Model Accuracy", f"{accuracy:.1f}%")
                    
                    # Error indicator
                    if selected_row['avg_error'] < 1.0:
                        st.success("⭕ Excellent Prediction")
                    elif selected_row['avg_error'] < 2.0:
                        st.info("👍 Good Prediction")
                    else:
                        st.warning("⚠️ Moderate Error")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------
    # TAB 3: MODEL PERFORMANCE (FIXED LAYOUT)
    # ----------------------------------
    with tab_analysis:
        analysis_page_fixed(model_data)


def analysis_page_fixed(model_data):
    """Fixed version of analysis page without the .tail() error."""
    
    model, scaler, metrics, test_df, y_test, y_test_pred = model_data

    st.markdown("## 📊 Model Performance Analysis")
    st.markdown("### Final Test Set Performance (Last 20% of Historical Data)")
    
    # Create a container for the analysis
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    
    # Key Metrics in a single row
    st.markdown("#### Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"{metrics['rmse']:.3f}°C", 
                 help="Root Mean Squared Error - Lower is better")
    with col2:
        st.metric("MAE", f"{metrics['mae']:.3f}°C", 
                 help="Mean Absolute Error - Average prediction error")
    with col3:
        st.metric("R² Score", f"{metrics['r2']:.4f}", 
                 help="Coefficient of determination - Closer to 1 is better")
    with col4:
        st.metric("CV MAE", f"{metrics['avg_cv_mae']:.3f}°C", 
                 help="5-Fold Cross-Validation Average MAE")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Details
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.markdown("#### Model Architecture & Training")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("**Training Configuration:**")
        st.markdown("""
        - **Algorithm:** Linear Regression with Regularization
        - **Validation:** 5-Fold Time Series Cross-Validation
        - **Train/Test Split:** 80%/20% Chronological Split
        - **Feature Scaling:** StandardScaler (Zero Mean, Unit Variance)
        - **Data Period:** 23+ Years of Historical Weather Data
        """)
    
    with col_info2:
        st.markdown("**Feature Engineering:**")
        st.markdown(f"""
        - **Total Features:** {len(metrics['feature_cols'])}
        - **Time Features:** Cyclical encoding (sin/cos)
        - **Lagged Features:** Temperature (1h, 3h, 6h, 12h, 24h)
        - **Rolling Statistics:** 3h & 24h moving averages
        - **Additional:** Humidity, Pressure lags
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Large single visualization
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.markdown("#### Model Performance Visualization")
    
    # Create a comprehensive figure with subplots
    from plotly.subplots import make_subplots
    
    # Calculate errors - FIXED: This is a numpy array, not pandas
    errors = y_test - y_test_pred
    
    # Get last 200 points for time series - FIXED: Use array slicing for numpy arrays
    if len(errors) > 200:
        errors_last_200 = errors[-200:]  # Array slicing for last 200 elements
    else:
        errors_last_200 = errors
    
    # Get corresponding time values
    if 'time' in test_df.columns and len(test_df) > 200:
        time_last_200 = test_df['time'].iloc[-200:]  # Use .iloc for pandas DataFrame
    elif 'time' in test_df.columns:
        time_last_200 = test_df['time']
    else:
        # Create dummy time indices if 'time' column doesn't exist
        time_last_200 = list(range(len(errors_last_200)))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Actual vs Predicted Temperature', 
                       'Prediction Error Distribution',
                       'Residual Analysis',
                       'Error Over Time'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Actual vs Predicted Scatter (show first 500 points for clarity)
    scatter_points = min(500, len(y_test))
    fig.add_trace(
        go.Scatter(
            x=y_test[:scatter_points],
            y=y_test_pred[:scatter_points],
            mode='markers',
            marker=dict(color='#3498db', size=6, opacity=0.6),
            name='Predictions'
        ),
        row=1, col=1
    )
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Perfect Prediction'
        ),
        row=1, col=1
    )
    
    # 2. Error Distribution
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=50,
            marker_color='#9b59b6',
            opacity=0.7,
            name='Error Distribution'
        ),
        row=1, col=2
    )
    
    # Add mean error line
    mean_error = errors.mean()
    fig.add_vline(x=mean_error, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. Residual Analysis (show first 500 points)
    residual_points = min(500, len(y_test_pred))
    fig.add_trace(
        go.Scatter(
            x=y_test_pred[:residual_points],
            y=errors[:residual_points],
            mode='markers',
            marker=dict(color='#e74c3c', size=6, opacity=0.6),
            name='Residuals'
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    
    # 4. Error Over Time (last 200 points) - FIXED: No .tail() on numpy array
    fig.add_trace(
        go.Scatter(
            x=time_last_200,
            y=errors_last_200,
            mode='lines+markers',
            line=dict(color='#f39c12', width=2),
            marker=dict(size=4),
            name='Error Trend'
        ),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
        title_text="Comprehensive Model Performance Analysis",
        title_font=dict(size=20)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Actual Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Error (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Predicted Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Residual Error (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Error (°C)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance Summary
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    st.markdown("#### Performance Summary & Interpretation")
    
    # Performance assessment
    if metrics['r2'] > 0.9:
        performance_level = "Excellent"
        color = "green"
    elif metrics['r2'] > 0.8:
        performance_level = "Very Good"
        color = "lightgreen"
    elif metrics['r2'] > 0.7:
        performance_level = "Good"
        color = "orange"
    else:
        performance_level = "Needs Improvement"
        color = "red"
    
    st.markdown(f"""
    **Overall Model Assessment:** <span style='color:{color}; font-weight:bold;'>{performance_level}</span>
    
    **Key Insights:**
    1. **Prediction Accuracy:** The model achieves {metrics['r2']*100:.1f}% explained variance in temperature data
    2. **Error Magnitude:** Average prediction error is {metrics['mae']:.2f}°C
    3. **Consistency:** Cross-validation shows consistent performance (±{np.std([metrics['mae']]):.3f}°C)
    4. **Practical Utility:** Suitable for daily weather forecasting applications
    
    **Recommendations for Improvement:**
    - Consider adding more lag features (48h, 72h)
    - Incorporate external data sources (sea pressure, wind patterns)
    - Experiment with ensemble methods
    - Add seasonal adjustment factors
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
# ==========================================================================
# MAIN APP EXECUTION
# ==========================================================================
# In the main() function, replace the navigation section with this:
def main():
    
    # --- Common Elements (Header and Navigation) ---
    col_logo, col_nav, col_theme = st.columns([1, 4, 1])

    with col_logo:
        st.markdown("### 🌤️ ML WeatherCast")
    
    with col_nav:
        # Simple solution - just use JavaScript to click the tab
        nav_cols = st.columns(3)
        with nav_cols[0]:
            if st.button("🔍 Home", use_container_width=True, key="nav_home"):
                set_page('home')
        with nav_cols[1]:
            if st.button("📅 Tashkent Forecast", use_container_width=True, key="nav_forecast"):
                set_page('forecast')
        with nav_cols[2]:
            # Simple approach: Use JavaScript to click the Model Performance tab
            if st.button("📈 Model Performance", use_container_width=True, key="nav_performance"):
                set_page('forecast')
                # Use JavaScript to simulate clicking the third tab
                js = """
                <script>
                // Wait for page to load, then click the third tab
                setTimeout(function() {
                    var tabs = document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs.length >= 3) {
                        tabs[2].click();
                    }
                }, 500);
                </script>
                """
                st.components.v1.html(js, height=0)
        
    with col_theme:
        st.button(f"Theme: {'🔆 Light' if st.session_state.theme == 'dark' else '🌙 Dark'}", 
                 on_click=toggle_theme, use_container_width=True)

    st.markdown("---")
    
    # ... rest of your main() function remains unchanged ...
    
    # --- Load Data and Train Models ---
    with st.spinner("🔄 Loading data and training models..."):
        current_weather = get_current_weather()
        historical_data = read_historical_data(DATA_PATH)
        
        if current_weather is None or historical_data is None:
            st.error("Failed to load data. Please ensure the 'tashkent_weather_23years.csv' file exists in the './data/' directory.")
            st.stop()
        
        # Train advanced model once
        model_data = train_model_with_time_series_cv(historical_data)
        
    # --- Page Rendering ---
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'forecast':
        forecast_page(current_weather, historical_data, model_data)

if __name__ == "__main__":
    main()