import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_data(latitude, longitude, start_date, end_date, filename):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,windgusts_10m,pressure_msl,winddirection_10m",
        "timezone": "Asia/Tashkent"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if "hourly" in data:
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data received for the given period")
#TAshkent Coordinates
latitude = 41.2995
longitude = 69.2401

start_date = "2005-01-01"
end_date = "2025-12-04"
filename = "tashkent_weather_20years.csv"

fetch_weather_data(latitude, longitude, start_date, end_date, filename)
