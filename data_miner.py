import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Configuration ---
LATITUDE = 41.2995
LONGITUDE = 69.2401
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Ensure the path is correct (corrected to './data' based on common structure)
DATA_PATH = "./data/tashkent_weather_23years.csv" 

# The first possible date in your dataset (used if the file doesn't exist)
INITIAL_START_DATE = datetime(2002, 1, 1).date()

def update_weather_data():
    """Fetches new weather data from the last recorded date up to yesterday and appends it."""
    
    # 1. Determine the end date (up to yesterday to ensure data completeness)
    today = datetime.now().date()
    end_date_fetch = today - timedelta(days=1) 
    
    start_date_fetch = INITIAL_START_DATE

    # 2. Determine the start date by checking the existing CSV
    if os.path.exists(DATA_PATH):
        try:
            # Read only the 'time' column for efficiency
            df_existing = pd.read_csv(DATA_PATH, usecols=['time'])
            df_existing['time'] = pd.to_datetime(df_existing['time'])
            
            # Find the date of the last recorded hourly data point
            last_recorded_time = df_existing['time'].max()
            
            # Set the new fetch start date to the day after the last recorded date
            start_date_fetch = (last_recorded_time + timedelta(days=1)).date()
            
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting full fetch from {INITIAL_START_DATE}.")
    else:
        print(f"File not found at {DATA_PATH}. Starting initial fetch.")
        
    # Check if there is new data to fetch
    if start_date_fetch > end_date_fetch:
        print(f"Data is already up-to-date until {end_date_fetch}. No update needed.")
        return

    print(f"Fetching data from {start_date_fetch} to {end_date_fetch}...")
    
    # 3. Prepare API parameters
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date_fetch.strftime('%Y-%m-%d'),
        "end_date": end_date_fetch.strftime('%Y-%m-%d'),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,windgusts_10m,pressure_msl,winddirection_10m",
        "timezone": "Asia/Tashkent"
    }
    
    # 4. Fetch data
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return

    # 5. Process and Append/Save
    if "hourly" in data and data["hourly"].get("time"):
        df_new = pd.DataFrame(data["hourly"])
        df_new["time"] = pd.to_datetime(df_new["time"])
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        if os.path.exists(DATA_PATH) and start_date_fetch > INITIAL_START_DATE:
            # Append new data to existing file (header=False is essential)
            df_new.to_csv(DATA_PATH, mode='a', header=False, index=False)
            print(f"Successfully appended {len(df_new)} new hourly records to {DATA_PATH}")
        else:
            # Overwrite (initial setup or error recovery)
            df_new.to_csv(DATA_PATH, index=False)
            print(f"Successfully created/overwrote {DATA_PATH} with {len(df_new)} hourly records.")
    else:
        print("No new hourly data received.")

if __name__ == "__main__":
    update_weather_data()