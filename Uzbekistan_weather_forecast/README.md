# Weather Forecasting For Uzbekistan
## Abstract
Weather Forecast ML - All Uzbekistan Cities. We have a plan to expand our work through Uzbekistan so we did it you can use it easily.
This enhanced version provides weather forecasts for all major cities in Uzbekistan: 20 Major cities.
## New Features
**Compared to the Tashkent-only version, this includes:**

- 20 Cities Support - Complete coverage of major Uzbekistan cities
- Interactive Map - Visual city selection on Uzbekistan map
-  Smart Search - Quick city lookup with autocomplete
- City Grid - Fast access button grid for all cities
- Separate Data Files - Individual datasets for each city
  
## Installation

**Navigate to the Uzbekistan folder**
```bash
cd Uzbekistan_weather_forecast
```
## Install dependencies
```bash 
pip install -r requirements.txt
```
## Collect data for all cities
**Important: This will take 5-7 minutes as it fetches 23 years of data for 20 cities**
```bash
python3 data_miner_uz.py
```
## Launch the application
```bash
streamlit run main_uz.py
```
[Demo video how it works](../vd/Uzbekistan.mp4)
## Data Collection Details
**API Usage
The data collection script uses:**

- Open-Meteo Geocoding API - Find city coordinates
- Open-Meteo Archive API - Historical weather data  
 
`Thank you `