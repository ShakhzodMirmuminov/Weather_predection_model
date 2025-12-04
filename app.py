from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Import your weather_view functions here

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

@app.route('/api/weather', methods=['GET'])
def get_weather():
    # Run your weather prediction
    current_weather = get_current_weather()
    historical_data = read_historical_data('./data/tashkent_weather_20years.csv')
    
    # Get predictions
    model, scaler, feature_cols = train_advanced_model(historical_data)
    forecasts = predict_today_tomorrow(model, scaler, feature_cols, historical_data)
    
    # Format response
    response = {
        'current': current_weather,
        'forecasts': forecasts,
        # Add other data here
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)