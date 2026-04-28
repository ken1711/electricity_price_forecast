# ⚡Electricity Price Forecasting

![Python](https://img.shields.io/badge/Python-3.13-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange) ![LightGBM](https://img.shields.io/badge/Lightgbm-Model-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)![Docker](https://img.shields.io/badge/Docker-Containerized-blue) ![Azure](https://img.shields.io/badge/Azure-Container_Apps-lightblue)

A machine learning system that predicts day-ahead hourly electricity prices for Germany/Luxembourg, using SMARD and Copernicus Climate data.
Built with FastAPI (backend) and Streamlit (frontend), containerized with Docker, and deployed on Azure Container Apps.


## 📌Project Overview

This project develops an end-to-end machine learning system for forecasting hourly electricity prices in Germany and Luxembourg. Data is sourced from the SMARD and Copernicus Climate platforms, then cleaned, preprocessed, and enhanced through feature engineering.
Two models—XGBoost and LightGBM—are trained and evaluated, with LightGBM achieving slightly better performance.
The system covers the complete workflow, including data collection, preprocessing, model training, and model persistence. It also features an interactive web interface for real-time visualization and forecasting, along with containerized deployment to support scalable and efficient operation.


## 📌 Problem Statement

Electricity prices fluctuate due to demand, weather, renewable energy production, and market conditions. Accurate forecasting helps: Grid operators, Energy traders, Renewable energy planners, and Consumers.

This project predicts hourly electricity prices using historical and weather-based features.

Why Germany/Luxembourg (DE/LU)?
Germany and Luxembourg operate as a single electricity bidding zone (DE/LU), meaning a common wholesale electricity price is applied across both countries.

Historically, this zone also included Austria (DE/AT/LU). However, in 2018 Austria separated from the joint market due to significant transmission congestion and unscheduled loop flows that were impacting neighboring countries such as the Czech Republic and Poland.

This project mainly analyzes data from Germany and Luxembourg (DE/LU) covering the years 2015 to 2025. To ensure completeness, it also includes data from Germany, Austria, and Luxembourg (DE/AT/LU) for the period up to 2018, when Austria was still part of the shared bidding zone.

Today, Germany and Luxembourg continue to operate as a single bidding zone, reflecting their high level of grid integration.

In this shared zone, a unified wholesale electricity price applies across both countries. This arrangement is rooted in historical market structures and integration efforts.

Electricity prices are determined using the merit-order principle: power plants are dispatched based on their marginal costs, and the most expensive unit required to meet demand sets the market price.

| Source | Marginal Cost|
| :------| :------------|
| Wind/Solar   | ~€0 | 
| Nuclear / Lignite    | Low |                            
| Coal   | Medium |                            
| Gas     | High (Often price-setter) | 

When renewables cover most demand → prices fall. 

When gas plants are needed → prices rise.


## 🔗 Project Workflow

 Data Collection

- Smart.de:
  - Germany/Luxembourg_EUR/MWh_Calculated_resolution
  - Grid_load_MWh_Calculated_resolutions_actual        
  - Grid_load_MWh_Calculated_resolutions_predicted      
  - Predicted_Residual_load_MWh_Calculated_resolutions  
  - Total_MWh_Calculated_resolutions                    
  - Photovoltaics_and_wind_MWh_Calculated_resolutions   
  - Offshore_wind_MWh_Calculated_resolutions            
  - Wind_Onshore_MWh_Calculated_resolutions             
  - Photovoltaics_MWh_Calculated_resolutions           
  - Other_MWh_Calculated_resolutions                    

- CDS Climate Data Store:
  - t2m_degree_celsius
  - wind_speed_m/s  

EDA & Feature Engineering

- Time-series analysis

- Lag features

- Rolling statistics

- Weather variables

Model Development

- XGBoost regression

- Hyperparameter tuning

- Evaluation (RMSE, MAE, R2)

- Model saved as model.joblib

Backend (FastAPI)

- /predict endpoint

- Loads joblib model

- Swagger UI documentation

Frontend (Streamlit)

- User input form

- Calls FastAPI

- Displays prediction

Containerization

- Dockerfiles for both apps

- docker-compose.yml

- Pushed to Docker Hub

Cloud Deployment

- Azure Container Apps

- Scale-to-zero

- Public URLs 


## 🏗️Architecture

```bash

                       ┌──────────────────────────────┐
                       │           Dataset            │
                       │       Feature Engineering +  │
                       │       Prediction Pipeline    │
                       └───────────────┬──────────────┘
                                       │
                                       ▼
                       ┌──────────────────────────────┐
                       │    XGBoost Model Training    │
                       │    LightGBM Model Training   │
                       └───────────────┬──────────────┘
                                       │  
                                       ▼
                       ┌──────────────────────────────┐
                       │   Saved Model via Joblib     |
                       └──────────────────────────────┘
                                       |
                                       │  
                                       ▼
                       ┌──────────────────────────────┐
                       │       FastAPI Backend        │
                       │       Docker Image           │
                       │       Docker Hub:            │
                       └───────────────┬──────────────┘
                                       │  Docker Compose
                                       ▼
                       ┌──────────────────────────────┐
                       │       Streamlit Frontend     |
                       |       Docker Image           |
                       │       Docker Hub             │
                       └───────────────┬──────────────┘
                                       │  
                                       ▼
        ┌──────────────────────────────┬──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌────────────────┐           ┌────────────────┐            ┌────────────────┐
│   FastAPI API  │◄────────► │ Azure Container│◄──────────►│ Streamlit UI   │
│ (Predictions)  │           │     Apps       │            │ (User Facing)  │
└────────────────┘           └────────────────┘            └────────────────┘

```


## 🗂️Project Structure

```bash
electricity-load-forecasting/
│
│
├── data/
|   ├── 2015_2025_de_Market_Wholesale_Prices.csv
|   ├── 2015_2025_de_PC_AEC.csv
|   ├── 2015_2025_de_PC_PPC.csv
|   ├── 2015_2025_de_PG_FG.csv
|   ├── 2015_2025_de_weather_tem2m_cds.csv
|   ├── 2015_2025_de_windspeed_uw_cds.csv
│   └── actual_predictions_error.csv
│
├── notebook/
│   ├── 01_eda_electricity_price.ipynb
│   ├── 02_feature_engineering_electricity_price.ipynb
│   ├── 03_modelTraining_evaluationResult_electricity_price.ipynb
│   ├── forecast_electricity_price_load.html
│   └── full_workflow_electricity_price.ipynb
│
├── electricity_price_lgbm_model.joblib
│  
│
├── plots/
│   ├── Electricity_Loads.png
│   ├── Germany_Luxembourg_EUR_MWh_Calculated_resolution.png
│   ├── Germany_Luxembourg_EUR_MWh_Calculated_resolution_lag_features.png
│   ├── Germany_Luxembourg_EUR_MWh_Calculated_resolution_over_time.png
│   ├── Grid_load_MWh_Calculated_resolutions_actual.png
│   ├── Grid_load_MWh_Calculated_resolutions_actual_Lag_Features.png
│   ├── Grid_load_MWh_Calculated_resolutions_predicted.png
|   ├── Time_series_cross_validation_visualization.png
|   ├── Total_MWh_Calculated_resolutions_over_time.png
|   ├── photovoltaics-wind.png
|   ├── t2m_degree_celsius.png
│   └── t2m_degree_celsius_lag_features, wind_speed_ms_lag_features.png
│
├── fastapi/
│   ├── Dockerfile.fastapi
│   ├── fastapi.gif.gif
│   ├── fastapi_app.py
│   └── fastapi_requirements.txt
│
├── streamlit/
│   ├── Dockerfile.streamlit.py
│   ├── streamlit.gif.gif
│   ├── streamlit_app.py
│   └── streamlit_requirements.txt
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
└── README.md
```


## 📊Dataset

The dataset used in this project is publicly available: SMARD Data from 2015 - 2025 

https://www.smard.de/home/downloadcenter/download-marktdaten/?downloadAttributes=%7B%22selectedCategory%22:2,%22selectedSubCategory%22:5,%22selectedRegion%22:false,%22selectedFileType%22:%22CSV%22,%22from%22:1514761200000,%22to%22:1609455599999%7D

Climate Data Store - from 2015 - 2025

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download

[Download direct dataset](https://github.com/ken1711/electricity_price_forecast/tree/main/data)

Dataset Notes

 - Market Wholesale Prices
   - Germany/Luxembourg_EUR/MWh_Calculated_resolution
 - Actual Electricity Consumption
   - Grid_load_MWh_Calculated_resolutions_actual 
 - Power Consumption Predicted Power Consumption       
   - Grid_load_MWh_Calculated_resolutions_predicted      
   - Predicted_Residual_load_MWh_Calculated_resolutions
 - Power Generation Forecast Generation day-ahead
   - Total_MWh_Calculated_resolutions                    
   - Photovoltaics_and_wind_MWh_Calculated_resolutions   
   - Offshore_wind_MWh_Calculated_resolutions            
   - Wind_Onshore_MWh_Calculated_resolutions             
   - Photovoltaics_MWh_Calculated_resolutions           
   - Other_MWh_Calculated_resolutions                    

- CDS Climate Data Store:
  - t2m_degree_celsius
  - wind_speed_m/s 

Features Used

- Hourly electricity load values
- Calendar-based features (hour of day, day of week, is weekend, quarter, month, year, day of year)
- Rolling statistics (Mean and Standard deviation)
- Lag features (e.g., 1 hour, 24 hours, 48 hours, 168 hours)


## 🧠Modeling Approach

The modeling pipeline includes:

- Time Series Cross Validation split  
- Feature engineering (date-time, Lag features, Rolling window statistics)  
- Model selection (XGBRegressor, LGBMRegressor)  
- Time-series cross‑validation  
- Error analysis  

[Download Model.joblib](https://github.com/ken1711/electricity_price_forecast/blob/main/electricity_price_lgbm_model.joblib)

- Results

| Metric | XGB Value | LGBM Value|
| :------| :---------|-----------|
| RMSE   | 13.58 MWh | 13.14 MWh |
| MAE    | 7.66 MWh  | 7.67 MWh  |                               
| R2     | 0.93      | 0.94      |


## 🔗Jupyter Notebooks

All data science work is documented in the notebooks listed below.

- **01 – Exploratory Data Analysis**

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/Total_MWh_Calculated_resolutions_over_time.png)

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/Germany_Luxembourg_EUR_MWh_Calculated_resolution_over_time.png)

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/Germany_Luxembourg_EUR_MWh_Calculated_resolution.png)

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/photovoltaics-wind.png)

[Open Notebook](https://github.com/ken1711/electricity_price_forecast/blob/main/notebook/01_eda_electricity_price.ipynb)

- **02 – Feature Engineering** 

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/Grid_load_MWh_Calculated_resolutions_actual.png)

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/Grid_load_MWh_Calculated_resolutions_predicted.png)

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/plots/t2m_degree_celsius.png)

[Open Notebook](https://github.com/ken1711/electricity_price_forecast/blob/main/notebook/02_feature_engineering_electricity_price.ipynb)

- **03 – Model Training / Evaluation & Results**  

[Open Notebook](https://github.com/ken1711/electricity_price_forecast/blob/main/notebook/03_modelTraining_evaluationResult_electricity_price.ipynb)

- **Full Workflow**  

[Open Notebook](https://github.com/ken1711/electricity_price_forecast/blob/main/notebook/full_workflow_electricity_price.ipynb)

These notebooks demonstrate the full workflow from raw data to final model.


## 🚀How to Install and Run the project 

1. Clone repository

git clone https://github.com/ken1711/electricity_price_forecast.git

cd electricity_price_forecast

2. Install all dependencies

pip install -r requirements.txt

3. Run FastAPI

uvicorn fastapi_app:app --reload

4. Run Streamlit

streamlit run streamlit_app.py


## 🐳 Docker and Containerization

### Docker Compose 

To build and start all services, run:

docker-compose up --build

Once running, open your browser:

FastAPI (Frontend): http://localhost:8501

Streamlit (Backend API): http://localhost:8000

![FastAPI Live Demo](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/fastapi/fastapi.gif.gif)

![Streamlit Live Demo](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/Streamlit/streamlit.gif.gif)

Docker Hub repositories:

Backend (FastAPI): kenwhite17/backend-app:latest

Frontend (Streamlit): kenwhite17/frontend-app:latest


## ☁️Deployment (Azure Container Apps)

🔹 Step 1 — Open Container Apps

Go to 👉 https://portal.azure.com
Login
Start by navigating to “Container Apps”

and Click ➕ Create
🔹 Step 1 — Basics tab

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/fastapi/4.png)

Next

🔹 Step 2 — Container tab

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/fastapi/55.png)

🔹 Step 3 — Ingress (VERY IMPORTANT)

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/fastapi/6.png)

🔹 Step 4 — Review + Create

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/fastapi/7.png)

🔹 Step 5 — Copy URL and check

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/fastapi/1010.png)

[FastAPI Backend Cloud URL:](https://backend-app.greenhill-0fd68809.germanywestcentral.azurecontainerapps.io)

Now deploy Streamlit (the same process)

🔹 Step 1 — Create another Container App

Click Create again

🔹 Step 2 — Basics tab

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/streamlit/1.png)

🔹 Step 3 — Container tab

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/streamlit/2.png)

🔹 Step 4 — Ingress 

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/streamlit/3.png)

🔹 Step 5 — Review + Create

![Screenshots](https://raw.githubusercontent.com/ken1711/electricity_price_forecast/main/azure_deployment_screenshot/streamlit/4.png)

🔹 Step 5 — Once finished, copy URL and check

![Screenshots](https://github.com/ken1711/electricity_price_forecast/blob/main/azure_deployment_screenshot/streamlit/7.png)

[Streamlit Frontend Cloud URL:](https://frontend-app.greenhill-0fd68809.germanywestcentral.azurecontainerapps.io)

### Live Cloud links (Live Demo):

Final result you’ll get:

FastAPI Backend:
https://backend-app.greenhill-0fd68809.germanywestcentral.azurecontainerapps.io

Swagger API Docs:
https://backend-app.greenhill-0fd68809.germanywestcentral.azurecontainerapps.io/docs

Streamlit Frontend:
https://frontend-app.greenhill-0fd68809.germanywestcentral.azurecontainerapps.io


## 🛠️Tech Stack

| Layer       | Technology|
| :-----------| :----------|
| Language    | Python | 
| ML Model    | XGBoost, LightGBM, Pandas, NumPy, Scikit-learn| 
| Backend API | FastAPI + Uvicorn|       
| Frontend    | Streamlit|  
| Containerization | Docker| 
| Cloud | Azure Container Apps|


## 🧪 Future Improvements

What can be improved:

- Build automated model retraining pipelines with live data.
- Extend forecasting to multi-zone coverage across Europe.
- Experiment with deep learning approaches (LSTM, Temporal Fusion Transformer).


## 📚 References
- [Smard Dataset](https://www.smard.de/home/downloadcenter/download-marktdaten/?downloadAttributes=%7B%22selectedCategory%22:2,%22selectedSubCategory%22:5,%22selectedRegion%22:false,%22selectedFileType%22:%22CSV%22,%22from%22:1514761200000,%22to%22:1609455599999%7D)

- [Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download) 

- [Sciki-Learn Documentaion](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor)

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html)

- [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/)

- [Streamlit Documentation](https://docs.streamlit.io/)

- [Docker Documentation](https://docs.docker.com/get-started/docker-concepts/building-images/build-tag-and-publish-an-image/)

- [Azure Cloud](https://learn.microsoft.com/en-us/azure/container-apps/overview)


## 📜 License

This project is released under the MIT License.


## 📬Contact

github: https://github.com/ken1711
                           
