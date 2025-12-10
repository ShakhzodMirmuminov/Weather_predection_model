import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = "./data/uzbekistan_cities/"
INITIAL_START_DATE = datetime(2002, 1, 1).date()

UZBEKISTAN_CITIES = {
    "Tashkent": {"lat": 41.2995, "lon": 69.2401},
    "Samarkand": {"lat": 39.6270, "lon": 66.9750},
    "Bukhara": {"lat": 39.7750, "lon": 64.4286},
    "Andijan": {"lat": 40.7821, "lon": 72.3442},
    "Namangan": {"lat": 40.9983, "lon": 71.6726},
    "Fergana": {"lat": 40.3864, "lon": 71.7864},
    "Nukus": {"lat": 42.4531, "lon": 59.6103},
    "Qarshi": {"lat": 38.8606, "lon": 65.7892},
    "Kokand": {"lat": 40.5283, "lon": 70.9428},
    "Margilan": {"lat": 40.4717, "lon": 71.7244},
    "Urgench": {"lat": 41.5500, "lon": 60.6333},
    "Jizzakh": {"lat": 40.1158, "lon": 67.8422},
    "Termez": {"lat": 37.2242, "lon": 67.2783},
    "Navoiy": {"lat": 40.0844, "lon": 65.3792},
    "Angren": {"lat": 41.0167, "lon": 70.1436},
    "Chirchiq": {"lat": 41.4686, "lon": 69.5822},
    "Gulistan": {"lat": 40.4892, "lon": 68.7844},
    "Bekabad": {"lat": 40.2214, "lon": 69.1953},
    "Denov": {"lat": 38.2667, "lon": 67.9000},
    "Shahrisabz": {"lat": 39.0500, "lon": 66.8333}
}


def get_city_coordinates(city_name, country="Uzbekistan"):
    params = {
        "name": city_name,
        "count": 10,
        "language": "en",
        "format": "json"
    }
    
    try:
        response = requests.get(GEOCODING_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data:
            for result in data["results"]:
                if result.get("country") == country or result.get("country_code") == "UZ":
                    return {
                        "name": result.get("name"),
                        "latitude": result.get("latitude"),
                        "longitude": result.get("longitude"),
                        "elevation": result.get("elevation"),
                        "timezone": result.get("timezone"),
                        "country": result.get("country"),
                        "admin1": result.get("admin1"),  # Region/Province
                        "population": result.get("population", 0)
                    }
        
        print(f"City '{city_name}' not found in Uzbekistan")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching coordinates for {city_name}: {e}")
        return None


def update_city_weather_data(city_info):

    city_name = city_info["name"]
    latitude = city_info["latitude"]
    longitude = city_info["longitude"]

    safe_city_name = city_name.replace(" ", "_").lower()
    city_data_path = os.path.join(DATA_DIR, f"{safe_city_name}_weather.csv")
    
    today = datetime.now().date()
    end_date_fetch = today - timedelta(days=1)
    start_date_fetch = INITIAL_START_DATE

    if os.path.exists(city_data_path):
        try:
            df_existing = pd.read_csv(city_data_path, usecols=['time'])
            df_existing['time'] = pd.to_datetime(df_existing['time'])
            last_recorded_time = df_existing['time'].max()
            start_date_fetch = (last_recorded_time + timedelta(days=1)).date()
            print(f"{city_name}: Updating from {start_date_fetch} to {end_date_fetch}")
        except Exception as e:
            print(f"{city_name}: Error reading existing CSV: {e}. Starting full fetch.")
    else:
        print(f"{city_name}: Starting initial fetch from {INITIAL_START_DATE}")
    
    if start_date_fetch > end_date_fetch:
        print(f"{city_name}: Already up-to-date until {end_date_fetch}")
        return True

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_fetch.strftime('%Y-%m-%d'),
        "end_date": end_date_fetch.strftime('%Y-%m-%d'),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m,windgusts_10m,pressure_msl,winddirection_10m",
        "timezone": city_info.get("timezone", "Asia/Tashkent")
    }
    
    try:
        print(f"{city_name}: Fetching data from API...")
        response = requests.get(ARCHIVE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" in data and data["hourly"].get("time"):
            df_new = pd.DataFrame(data["hourly"])
            df_new["time"] = pd.to_datetime(df_new["time"])
            df_new["city"] = city_name

            os.makedirs(DATA_DIR, exist_ok=True)
            
            if os.path.exists(city_data_path) and start_date_fetch > INITIAL_START_DATE:
                df_new.to_csv(city_data_path, mode='a', header=False, index=False)
                print(f"{city_name}: Appended {len(df_new)} records")
            else:
                df_new.to_csv(city_data_path, index=False)
                print(f"{city_name}: Created file with {len(df_new)} records")
            
            return True
        else:
            print(f"{city_name}: No data received from API")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"{city_name}: API request failed: {e}")
        return False


def save_city_metadata(cities_data):

    metadata_path = os.path.join(DATA_DIR, "cities_metadata.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(cities_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCity metadata saved to {metadata_path}")


def main():

    print("=" * 60)
    print("Uzbekistan Weather Data Collector")
    print("=" * 60)

    print("\nStep 1: Fetching city coordinates...")
    cities_data = []
    
    for city_name in UZBEKISTAN_CITIES.keys():
        print(f"Looking up: {city_name}...")
        city_info = get_city_coordinates(city_name)
        
        if city_info:
            cities_data.append(city_info)
            print(f"   Found: {city_info['name']} ({city_info['latitude']:.2f}, {city_info['longitude']:.2f})")
        else:
            print(f"   Not found: {city_name}")
        
        time.sleep(0.5)  
    
    print(f"\nSuccessfully found {len(cities_data)} cities")

    save_city_metadata(cities_data)

    print("\n" + "=" * 60)
    print("Step 2: Fetching weather data for each city...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, city_info in enumerate(cities_data, 1):
        print(f"\n[{i}/{len(cities_data)}] Processing {city_info['name']}...")
        
        success = update_city_weather_data(city_info)
        
        if success:
            successful += 1
        else:
            failed += 1

        if i < len(cities_data):
            time.sleep(2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cities processed: {len(cities_data)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nData saved in: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()