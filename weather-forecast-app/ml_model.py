# ml_model.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import requests
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================================================
# CONFIGURATION (Use secrets management for real deployment!)
# ==========================================================================
API_KEY = '0ea12166dca6efaa2a7077602c59e70d'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# ==========================================================================
# HELPER FUNCTIONS (Copy-pasted from your original code)
# ==========================================================================

def get_current_weather(lat=41.2995, lon=69.2401):
    # ... (Your get_current_weather function goes here) ...
    """Fetch current weather from OpenWeatherMap API"""
    try:
        url = f"{BASE_URL}weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert temperature if needed 
        temp = data['main']['temp']
        temp_min = data['main']['temp_min']
        temp_max = data['main']['temp_max']
        feels_like = data['main']['feels_like']
        
        if temp > 100:
            temp = temp - 273.15
        if temp_min > 100:
            temp_min = temp_min - 273.15
        if temp_max > 100:
            temp_max = temp_max - 273.15
        if feels_like > 100:
            feels_like = feels_like - 273.15
        
        return {
            'city': data.get('name', 'Tashkent'),
            'country': data['sys'].get('country', 'UZ'),
            'current_temp': round(temp, 2),
            'temp_min': round(temp_min, 2),
            'temp_max': round(temp_max, 2),
            'feels_like': round(feels_like, 2),
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_gust': data['wind'].get('gust', data['wind']['speed']),
            'wind_deg': data['wind'].get('deg', 0),
            'description': data['weather'][0]['description'],
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"❌ Error fetching weather: {e}")
        return None

def read_historical_data(filepath):
    # ... (Your read_historical_data function goes here) ...
    """Load historical weather data from CSV"""
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: Historical data file not found at {filepath}")
        return pd.DataFrame()

def create_features(df):
    # ... (Your create_features function goes here) ...
    """Create time-based and lag features for better prediction"""
    df = df.copy()
    
    # Time-based features
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        df['day_of_year'] = df['time'].dt.dayofyear
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lag features
    if 'temperature_2m' in df.columns:
        for lag in [1, 3, 6, 12, 24]:
            df[f'temp_lag_{lag}h'] = df['temperature_2m'].shift(lag)
    
    if 'relative_humidity_2m' in df.columns:
        for lag in [1, 3, 6]:
            df[f'humidity_lag_{lag}h'] = df['relative_humidity_2m'].shift(lag)
    
    if 'pressure_msl' in df.columns:
        for lag in [1, 6, 12]:
            df[f'pressure_lag_{lag}h'] = df['pressure_msl'].shift(lag)
    
    # Rolling statistics
    if 'temperature_2m' in df.columns:
        df['temp_ma_3h'] = df['temperature_2m'].rolling(window=3, min_periods=1).mean()
        df['temp_ma_24h'] = df['temperature_2m'].rolling(window=24, min_periods=1).mean()
    
    df = df.dropna().reset_index(drop=True)
    return df

def train_advanced_model(historical_df):
    # ... (Your train_advanced_model function goes here) ...
    """Train model with train/test split and evaluation (simplified output)"""
    
    df_with_features = create_features(historical_df)
    
    split_idx = int(len(df_with_features) * 0.8)
    train_df = df_with_features.iloc[:split_idx].copy()
    test_df = df_with_features.iloc[split_idx:].copy()
    
    exclude_cols = ['time', 'temperature_2m']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['temperature_2m'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['temperature_2m'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Calculate test metrics for display
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return model, scaler, feature_cols, test_mae, test_r2

def prepare_rain_data(historical_data):
    # ... (Your prepare_rain_data function goes here) ...
    """Prepare data for rain prediction"""
    historical_data['RainTomorrow'] = (historical_data['precipitation'] > 0).astype(int)
    
    features = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 
                'windgusts_10m', 'pressure_msl', 'winddirection_10m']
    
    X = historical_data[features].fillna(0)
    Y = historical_data['RainTomorrow']
    
    return X, Y

def train_rain_model(X, Y):
    # ... (Your train_rain_model function goes here) ...
    """Train logistic regression for rain prediction"""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)
    return model

def predict_future_temp(model, current_weather, historical_df, scaler, feature_cols):
    """Predict Today and Tomorrow temperatures for display"""
    
    # Train/Scale is already done in train_advanced_model, so we use the historical_df 
    # and the trained model/scaler to generate forecast features.
    
    df_with_features = create_features(historical_df)
    recent_data = df_with_features.tail(100).copy()  # Use recent data for lag consistency
    
    forecast_hours = [6, 12, 18, 0]
    hour_names = {6: "Morning", 12: "Afternoon", 18: "Evening", 0: "Night"}
    
    now = datetime.now()
    today = now.date()
    tomorrow = today + timedelta(days=1)
    
    forecasts = []
    
    for day_name, day_date in [("Today", today), ("Tomorrow", tomorrow)]:
        for hour in forecast_hours:
            target_time = datetime.combine(day_date, datetime.min.time()) + timedelta(hours=hour)
            
            # --- Key Prediction Logic ---
            # 1. Create a dummy row for the future time, using the characteristics of the 
            #    most recent historical data point that shares the same hour.
            target_hour = target_time.hour
            same_hour_data = recent_data[recent_data['hour'] == target_hour]
            
            if len(same_hour_data) > 0:
                last_row = same_hour_data.iloc[-1].to_dict()
                
                # Update time-dependent features
                last_row['time'] = target_time
                last_row['hour'] = target_time.hour
                last_row['day_of_year'] = target_time.timetuple().tm_yday
                last_row['month'] = target_time.month
                last_row['day_of_week'] = target_time.weekday()
                last_row['hour_sin'] = np.sin(2 * np.pi * last_row['hour'] / 24)
                last_row['hour_cos'] = np.cos(2 * np.pi * last_row['hour'] / 24)
                last_row['day_sin'] = np.sin(2 * np.pi * last_row['day_of_year'] / 365)
                last_row['day_cos'] = np.cos(2 * np.pi * last_row['day_of_year'] / 365)
                
                # Create DataFrame and prepare features
                future_features = pd.DataFrame([last_row])[feature_cols].values.reshape(1, -1)
                future_features_scaled = scaler.transform(future_features)
                predicted_temp = model.predict(future_features_scaled)[0]
                
                forecasts.append({
                    'date': day_date.strftime('%Y-%m-%d'),
                    'day': day_name,
                    'hour': target_time.strftime("%H:00"),
                    'period': hour_names[hour],
                    'temperature': round(predicted_temp, 1),
                })
    
    return forecasts

def wind_direction_to_compass(wind_deg):
    # ... (Your wind_direction_to_compass function goes here) ...
    """Convert wind degree to compass direction"""
    wind_deg = wind_deg % 360
    
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    
    for point, start, end in compass_points:
        if start <= wind_deg < end:
            return point
    return "N"

# ==========================================================================
# MASTER FUNCTION
# ==========================================================================

def run_ml_pipeline():
    """Runs the entire ML pipeline and returns data for the web app."""
    
    # 1. Get current weather
    current_weather = get_current_weather()
    if not current_weather:
        return None
    
    # 2. Load historical data
    historical_data = read_historical_data('./data/tashkent_weather_20years.csv')
    if historical_data.empty:
        return None
    
    # 3. Train temperature model
    temp_model, temp_scaler, feature_cols, test_mae, test_r2 = train_advanced_model(historical_data)
    
    # 4. Train rain model
    X_rain, Y_rain = prepare_rain_data(historical_data)
    rain_model = train_rain_model(X_rain, Y_rain)
    
    # 5. Predict rain
    current_rain_data = pd.DataFrame([{
        'temperature_2m': current_weather['current_temp'],
        'relative_humidity_2m': current_weather['humidity'],
        'windspeed_10m': current_weather['wind_speed'],
        'windgusts_10m': current_weather['wind_gust'],
        'pressure_msl': current_weather['pressure'],
        'winddirection_10m': current_weather['wind_deg']
    }])
    rain_prediction = rain_model.predict(current_rain_data)[0]
    
    # 6. Predict 2-day forecast
    two_day_forecast = predict_future_temp(temp_model, current_weather, historical_data, temp_scaler, feature_cols)
    
    # 7. Add compass direction
    current_weather['compass_direction'] = wind_direction_to_compass(current_weather['wind_deg'])
    
    return {
        'current_weather': current_weather,
        'rain_prediction': 'Yes' if rain_prediction else 'No',
        'forecasts': two_day_forecast,
        'model_metrics': {'MAE': f"{test_mae:.3f}°C", 'R2': f"{test_r2:.4f}"}
    }

if __name__ == "__main__":
    # Test the ML pipeline
    data = run_ml_pipeline()
    if data:
        print("ML Pipeline executed successfully. Data ready for Streamlit app.")
    else:
        print("ML Pipeline failed to run.")