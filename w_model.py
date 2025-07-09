import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import pytz

# Fetch from environment variable
VC_API_KEY = os.getenv("VC_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def fetch_historical_weather(city, days=7):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}?unitGroup=metric&include=days&key={VC_API_KEY}&contentType=json"
    response = requests.get(url)
    data = response.json()
    
    records = []
    for day in data['days']:
        records.append({
            'date': day['datetime'],
            'MinTemp': day['tempmin'],
            'MaxTemp': day['tempmax'],
            'Temp': day['temp'],
            'Humidity': day['humidity'],
            'Pressure': day['pressure'],
            'WindGustSpeed': day.get('windgust', day.get('windspeed', 0)),
            'Conditions': day.get('conditions', 'Unknown'),
            'RainTomorrow': 1 if 'rain' in day.get('conditions', '').lower() else 0
        })

    return pd.DataFrame(records)

def prepare_data(data):
    le = LabelEncoder()
    if 'Conditions' in data.columns:
        data['Conditions'] = le.fit_transform(data['Conditions'])
    X = data[['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp', 'Conditions']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(X).reshape(-1, 1), np.array(y)

def predict_future(model, current_value, steps=5):
    predictions = [current_value]
    for _ in range(steps):
        next_val = model.predict(np.array([[predictions[-1]]]))[0]
        predictions.append(next_val)
    return predictions[1:]

def get_current_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'country': data['sys']['country'],
        'current_temp': data['main']['temp'],
        'feels_like': data['main']['feels_like'],
        'temp_min': data['main']['temp_min'],
        'temp_max': data['main']['temp_max'],
        'humidity': data['main']['humidity'],
        'description': data['weather'][0]['description'],
        'wind_speed': data['wind']['speed'],
        'pressure': data['main']['pressure'],
        'lat': data['coord']['lat'],
        'lon': data['coord']['lon']
    }
