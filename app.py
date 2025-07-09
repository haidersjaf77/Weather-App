import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import pydeck as pdk
from geopy.geocoders import Nominatim
from w_model import (
    fetch_historical_weather,
    get_current_weather,
    prepare_data,
    train_rain_model,
    prepare_regression_data,
    train_regression_model,
    predict_future
)

st.set_page_config(page_title="🌦️ WeatherML Dashboard", layout="wide")

# Dark Mode Toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# Sidebar Enhancements
st.sidebar.markdown("---")
st.sidebar.markdown("## 📍 Weather Search")
st.sidebar.info("Enter any city name to view its current & predicted weather.")

# Today's Date
st.sidebar.markdown(f"🗓️ **Today:** {date.today().strftime('%A, %d %B %Y')}")

# Timezone
st.sidebar.markdown("🌐 **Timezone:** Asia/Karachi")

# About
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption("This app uses machine learning to forecast temperature, humidity, and rain based on real-time and historical weather data.")
st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [OpenWeather API](https://openweathermap.org/api).")

# Themes
if dark_mode:
    background = "#0e1117"
    text_color = "#fafafa"
    plot_template = "plotly_dark"
    card_bg = "linear-gradient(135deg, #2b2b2b, #3a3a3a)"
    card_shadow = "0 4px 20px rgba(0, 255, 255, 0.2)"
else:
    background = "linear-gradient(135deg, #e0f7fa, #ffffff)"
    text_color = "#000000"
    plot_template = "plotly_white"
    card_bg = "linear-gradient(135deg, #ffffff, #d0f0f8)"
    card_shadow = "0 4px 20px rgba(0, 0, 0, 0.1)"

st.markdown(f"""
    <style>
        .main {{ background: {background}; }}
        .title {{
            font-size: 40px;
            font-weight: 800;
            color: {text_color};
            text-align: center;
            margin-bottom: 10px;
            animation: glow 2s ease-in-out infinite alternate;
        }}
        @keyframes glow {{
            from {{ text-shadow: 0 0 10px #00e6e6; }}
            to {{ text-shadow: 0 0 20px #00e6e6, 0 0 30px #00e6e6; }}
        }}
        .metric-card {{
            background: {card_bg};
            border-radius: 16px;
            text-align: center;
            padding: 20px;
            color: {text_color};
            box-shadow: {card_shadow};
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, background 0.3s ease;
        }}
        .metric-card:hover {{
            transform: scale(1.05);
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            color: white;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌍 Real-Time Weather Forecasting</div>", unsafe_allow_html=True)

# City Input
city = st.text_input("🔍 Enter a city name", placeholder="e.g., Karachi")

if city.strip():
    try:
        with st.spinner("Fetching weather data..."):
            historical_df = fetch_historical_weather(city)
            current = get_current_weather(city)

            if historical_df.empty or len(historical_df) < 5:
                st.warning("⚠️ Not enough historical data found.")
            else:
                X, y, le = prepare_data(historical_df)
                rain_model = train_rain_model(X, y)

                X_temp, y_temp = prepare_regression_data(historical_df, 'Temp')
                X_hum, y_hum = prepare_regression_data(historical_df, 'Humidity')
                temp_model = train_regression_model(X_temp, y_temp)
                hum_model = train_regression_model(X_hum, y_hum)

                condition_encoded = le.transform([current['description']])[0] if current['description'] in le.classes_ else -1
                X_live = pd.DataFrame([{
                    'MinTemp': current['temp_min'],
                    'MaxTemp': current['temp_max'],
                    'WindGustSpeed': current['wind_speed'],
                    'Humidity': current['humidity'],
                    'Pressure': current['pressure'],
                    'Temp': current['current_temp'],
                    'Conditions': condition_encoded
                }])

                rain_pred = rain_model.predict(X_live)[0]
                future_temp = predict_future(temp_model, current['current_temp'])
                future_humidity = predict_future(hum_model, current['humidity'])

                timezone = pytz.timezone('Asia/Karachi')
                now = datetime.now(timezone)
                future_times = [(now + timedelta(hours=i)).strftime("%I %p") for i in range(1, 6)]

                st.subheader(f"🌡️ Weather in {city.title()}, {current['country']}")
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>🌡️ Temp<div class='metric-value'>{round(current['current_temp'], 1)}°C</div></div>
                        <div class='metric-card'>💧 Humidity<div class='metric-value'>{current['humidity']}%</div></div>
                        <div class='metric-card'>📈 Pressure<div class='metric-value'>{current['pressure']} mb</div></div>
                        <div class='metric-card'>☁️ Condition<div class='metric-value'>{current['description'].capitalize()}</div></div>
                        <div class='metric-card'>🌧️ Rain Forecast<div class='metric-value'>{'Yes ☔' if rain_pred else 'No ☀️'}</div></div>
                    """, unsafe_allow_html=True)

                with col2:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(x=future_times, y=future_temp, mode='lines+markers', name='Temp',
                                                  line=dict(color='#FF6F61', width=3)))
                    fig_temp.update_layout(title='🌡️ Temp Forecast (Next 5 Hrs)', template=plot_template)

                    fig_hum = go.Figure()
                    fig_hum.add_trace(go.Scatter(x=future_times, y=future_humidity, mode='lines+markers', name='Humidity',
                                                 line=dict(color='#0099ff', width=3)))
                    fig_hum.update_layout(title='💧 Humidity Forecast (Next 5 Hrs)', template=plot_template)

                    st.plotly_chart(fig_temp, use_container_width=True)
                    st.plotly_chart(fig_hum, use_container_width=True)

                st.markdown("---")
                st.subheader("🗺️ Location Map")

                lat, lon = current.get('lat'), current.get('lon')
                if not lat or not lon:
                    geolocator = Nominatim(user_agent="weather_app")
                    location = geolocator.geocode(city)
                    if location:
                        lat, lon = location.latitude, location.longitude

                if lat and lon:
                    st.pydeck_chart(pdk.Deck(
                        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=10, pitch=50),
                        layers=[
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=pd.DataFrame([{'lat': lat, 'lon': lon}]),
                                get_position='[lon, lat]',
                                get_color='[0, 128, 255, 160]',
                                get_radius=5000
                            )
                        ]
                    ))
                else:
                    st.warning("⚠️ Location not found.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("📍 Please enter a city name to view forecast.")
