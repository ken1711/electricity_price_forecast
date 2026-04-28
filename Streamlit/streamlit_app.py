import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import os
from pathlib import Path


API_URL = os.getenv("API_URL", "http://localhost:8000")
FINAL_PREDICTION = f"{API_URL}/predict"
DATA_PATH = Path(__file__).parent / "actual_predictions_error.csv"

st.set_page_config(
    page_title="Electricity Price Forecasting Dashboard",
    layout="wide"
)


# Header
st.title("Electricity Price Forecast")

st.markdown(
"""
Predict **day-ahead electricity prices (EUR/MWh)** for Germany/Luxembourg  
using a machine learning model.

Use the sidebar to enter feature values and click **Predict load** to generate a forecast.
"""
)


# API Documentation
st.subheader("API Documentation")
st.markdown(
    f"""
    **Open Swagger UI**
    [View API Documentation]({API_URL}/docs)
    """
)
# Backend URL 
st.subheader("Backend API Endpoint")
st.markdown(f"Connected: {API_URL}")



# Sidebar - Simple Inputs
st.sidebar.header("Forecast Setup")

date = st.sidebar.date_input("Date", datetime.date.today())
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# Auto features
day_of_week = date.weekday()
is_weekend = 1 if day_of_week >= 5 else 0
month = date.month
year = date.year
day_of_year = date.timetuple().tm_yday
quarter = (month - 1) // 3 + 1


# Advanced Inputs
with st.sidebar.expander("Energy Market Inputs", expanded=False):

    grid_load = st.slider("Grid Load (MWh)", 20000, 80000, 50000)
    residual_load = st.slider("Residual Load (MWh)", 10000, 60000, 30000)

    total_gen = st.slider("Total Generation (MWh)", 20000, 80000, 55000)
    renewables = st.slider("Renewables (MWh)", 5000, 40000, 20000)

    wind_on = st.slider("Wind Onshore (MWh)", 0, 30000, 10000)
    wind_off = st.slider("Wind Offshore (MWh)", 0, 15000, 5000)
    solar = st.slider("Solar (MWh)", 0, 20000, 8000)
    other = st.slider("Other Generation (MWh)", 0, 20000, 7000)

# Hidden Features (Backend Simulation)
def generate_hidden_features():
    return {
        "Actual_grid_load_1h_ago": grid_load * 0.98,
        "Air_temperature_2_meters_above_ground_1h_ago": 15,
        "Wind_speed_1h_ago": 5,
        "Germany_Luxembourg_price_1h_ago": 80,

        "Actual_grid_load_2h_ago": grid_load * 0.97,
        "Air_temperature_2_meters_above_ground_2h_ago": 15,
        "Wind_speed_2h_ago": 5,
        "Germany_Luxembourg_price_2h_ago": 78,

        "Actual_grid_load_6h_ago": grid_load * 0.95,
        "Air_temperature_2_meters_above_ground_6h_ago": 14,
        "Wind_speed_6h_ago": 6,
        "Germany_Luxembourg_price_6h_ago": 75,

        "Actual_grid_load_24h_ago": grid_load * 0.93,
        "Air_temperature_2_meters_above_ground_24h_ago": 13,
        "Wind_speed_24h_ago": 4,
        "Germany_Luxembourg_price_24h_ago": 70,

        "Actual_grid_load_1_week_ago": grid_load * 0.90,
        "Air_temperature_2_meters_above_ground_1_week_ago": 12,
        "Wind_speed_1_week_ago": 5,
        "Germany_Luxembourg_price_1_week_ago": 65,

        "Average_Germany_Luxembourg_EUR_price_over_last_24h": 75,
        "Variation_Germany_Luxembourg_price_over_last_24h": 5,
        "Average_Germany_Luxembourg_price_over_last_week": 70,
        "Variation_Germany_Luxembourg_price_over_last_week": 8,
    }

# Payload
payload = {
    "Grid_load_predicted": grid_load,
    "Residual_load_predicted": residual_load,
    "Total_generation": total_gen,
    "Renewables_total": renewables,
    "Wind_offshore": wind_off,
    "Wind_onshore": wind_on,
    "Solar_generation": solar,
    "Other_generation": other,
    "Hour_of_day": hour,
    "Day_of_week": day_of_week,
    "Is_weekend": is_weekend,
    "Quarter_of_year": quarter,
    "Month": month,
    "Year": year,
    "Day_of_year": day_of_year,
}

payload.update(generate_hidden_features())

# Prediction Button
if st.button("Predict Price"):

    with st.spinner("Running model inference..."):

        try:
            response = requests.post(FINAL_PREDICTION, json=payload)

            if response.status_code == 200:

                result = response.json()
                predicted_price = result["predicted_price"]

                # KPI
                st.metric(
                    label="💰 Predicted Electricity Price",
                    value=f"{predicted_price} EUR/MWh"
                )

                # Load REAL predictions.csv
                st.subheader("📊 Model Evaluation")

                if DATA_PATH.exists():

                    df = pd.read_csv(DATA_PATH)

                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)

                    df["Date"] = df["Timestamp"].dt.date

                    df = df.sort_values("Timestamp")

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df["Date"],
                        y=df["Actual"],
                        mode="lines",
                        name="Actual"
                    ))

                    fig.add_trace(go.Scatter(
                        x=df["Date"],
                        y=df["Predicted"],
                        mode="lines",
                        name="Predicted"
                    ))

                    fig.add_trace(go.Scatter(
                        x=df["Date"],
                        y=df["Error"],
                        mode="lines",
                        name="Error",
                        line=dict(dash="dot")
                    )) 

                    fig.update_layout(
                        title="Actual vs Predicted Electricity Prices",
                        xaxis_title="Date",
                        yaxis_title="EUR/MWh",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("actual_predictions_error.csv not found. Please place it in the project folder.")

                # Insight
                st.info(
                    """
                    📌 **Insight:**  
                    This chart compares actual electricity prices with model predictions and includes the error (difference between predicted and actual values.
                    The error line highlights where and by how much the model deviates, making it easier to spot periods of high volatility and reduced forecasting accuracy.
                    """
                )

            else:
                st.error(f"API Error: {response.text}")

        except Exception as e:
            st.error(f"Connection error: {e}")