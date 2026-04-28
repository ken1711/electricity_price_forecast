from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
import logging



# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Input Schema
class ForecastInput(BaseModel):

    Grid_load_predicted: float
    Residual_load_predicted: float
    Total_generation: float
    Renewables_total: float                                   
    Wind_offshore: float                                
    Wind_onshore: float                             
    Solar_generation: float                                  
    Other_generation: float                               
    Hour_of_day: int = Field(..., ge = 0, le = 23)                                     
    Day_of_week: int = Field(..., ge = 0, le = 6)
    Is_weekend: int = Field(..., ge = 0, le = 1)                                           
    Quarter_of_year: int = Field(..., ge = 1, le = 4)                                  
    Month: int = Field(..., ge = 1, le = 12)                                      
    Year: int = Field(..., ge = 2000, le = 2100)                                             
    Day_of_year: int = Field(..., ge = 1, le = 366)                                  
    Actual_grid_load_1h_ago: float                          
    Air_temperature_2_meters_above_ground_1h_ago: float       
    Wind_speed_1h_ago: float                                 
    Germany_Luxembourg_price_1h_ago: float                     
    Actual_grid_load_2h_ago: float                             
    Air_temperature_2_meters_above_ground_2h_ago: float
    Wind_speed_2h_ago: float         
    Germany_Luxembourg_price_2h_ago: float                         
    Actual_grid_load_6h_ago: float                            
    Air_temperature_2_meters_above_ground_6h_ago: float       
    Wind_speed_6h_ago: float                                        
    Germany_Luxembourg_price_6h_ago: float                 
    Actual_grid_load_24h_ago: float                          
    Air_temperature_2_meters_above_ground_24h_ago: float       
    Wind_speed_24h_ago: float                                
    Germany_Luxembourg_price_24h_ago: float                     
    Actual_grid_load_1_week_ago: float                       
    Air_temperature_2_meters_above_ground_1_week_ago: float         
    Wind_speed_1_week_ago: float                                     
    Germany_Luxembourg_price_1_week_ago: float                      
    Average_Germany_Luxembourg_EUR_price_over_last_24h: float
    Variation_Germany_Luxembourg_price_over_last_24h: float        
    Average_Germany_Luxembourg_price_over_last_week: float          
    Variation_Germany_Luxembourg_price_over_last_week: float   


# Fearure Mapping
BUSINESSFEATURE_TO_INTERNALFEATURE = {
    "Grid_load_predicted":                                       "Grid_load_MWh_Calculated_resolutions_predicted",
    "Residual_load_predicted":                                   "Predicted_Residual_load_MWh_Calculated_resolutions",
    "Total_generation":                                          "Total_MWh_Calculated_resolutions",
    "Renewables_total":                                          "Photovoltaics_and_wind_MWh_Calculated_resolutions",
    "Wind_offshore":                                             "Offshore_wind_MWh_Calculated_resolutions",
    "Wind_onshore":                                              "Wind_Onshore_MWh_Calculated_resolutions",
    "Solar_generation":                                          "Photovoltaics_MWh_Calculated_resolutions",
    "Other_generation":                                          "Other_MWh_Calculated_resolutions",
    "Hour_of_day":                                               "hour",
    "Day_of_week":                                               "dayofweek",
    "Is_weekend":                                                "is_weekend",
    "Quarter_of_year":                                           "quarter",
    "Month":                                                     "month",
    "Year":                                                      "year",
    "Day_of_year":                                               "dayofyear",
    "Actual_grid_load_1h_ago":                                   "Grid_load_MWh_Calculated_resolutions_actual_lag_1",
    "Air_temperature_2_meters_above_ground_1h_ago":              "t2m_degree_celsius_lag_1",
    "Wind_speed_1h_ago":                                         "wind_speed_m/s_lag_1",
    "Germany_Luxembourg_price_1h_ago":                           "Germany/Luxembourg_EUR/MWh_Calculated_resolution_lag_1",
    "Actual_grid_load_2h_ago":                                   "Grid_load_MWh_Calculated_resolutions_actual_lag_2",
    "Air_temperature_2_meters_above_ground_2h_ago":              "t2m_degree_celsius_lag_2",
    "Wind_speed_2h_ago":                                         "wind_speed_m/s_lag_2",
    "Germany_Luxembourg_price_2h_ago":                           "Germany/Luxembourg_EUR/MWh_Calculated_resolution_lag_2",
    "Actual_grid_load_6h_ago":                                   "Grid_load_MWh_Calculated_resolutions_actual_lag_6",
    "Air_temperature_2_meters_above_ground_6h_ago":              "t2m_degree_celsius_lag_6",
    "Wind_speed_6h_ago":                                         "wind_speed_m/s_lag_6",
    "Germany_Luxembourg_price_6h_ago":                           "Germany/Luxembourg_EUR/MWh_Calculated_resolution_lag_6",
    "Actual_grid_load_24h_ago":                                  "Grid_load_MWh_Calculated_resolutions_actual_lag_24",
    "Air_temperature_2_meters_above_ground_24h_ago":             "t2m_degree_celsius_lag_24",
    "Wind_speed_24h_ago":                                        "wind_speed_m/s_lag_24",
    "Germany_Luxembourg_price_24h_ago":                          "Germany/Luxembourg_EUR/MWh_Calculated_resolution_lag_24",
    "Actual_grid_load_1_week_ago":                               "Grid_load_MWh_Calculated_resolutions_actual_lag_168",
    "Air_temperature_2_meters_above_ground_1_week_ago":          "t2m_degree_celsius_lag_168",
    "Wind_speed_1_week_ago":                                     "wind_speed_m/s_lag_168",
    "Germany_Luxembourg_price_1_week_ago":                       "Germany/Luxembourg_EUR/MWh_Calculated_resolution_lag_168",
    "Average_Germany_Luxembourg_EUR_price_over_last_24h":        "Germany/Luxembourg_EUR/MWh_Calculated_resolution_roll_mean_24",
    "Variation_Germany_Luxembourg_price_over_last_24h":          "Germany/Luxembourg_EUR/MWh_Calculated_resolution_roll_std_24",
    "Average_Germany_Luxembourg_price_over_last_week":           "Germany/Luxembourg_EUR/MWh_Calculated_resolution_roll_mean_168",
    "Variation_Germany_Luxembourg_price_over_last_week":         "Germany/Luxembourg_EUR/MWh_Calculated_resolution_roll_std_168",
}



MODEL_FEATURE_ORDER = list(BUSINESSFEATURE_TO_INTERNALFEATURE.values())



VISIBLE_FEATURES = {
    "Grid_load_predicted",                               
    "Residual_load_predicted",                                  
    "Total_generation",                                          
    "Renewables_total",                                     
    "Wind_offshore",                                        
    "Wind_onshore",                                           
    "Solar_generation",                                         
    "Other_generation",                                       
    "Hour_of_day",                                         
    "Day_of_week",                                              
    "Is_weekend",                                              
    "Quarter_of_year", 
    "Month",                                         
    "Year",                                                      
    "Day_of_year",   
}



def build_model_input(user_input: dict) -> pd.DataFrame:
    """
    Maps user-facing field names to the models internal column names
    and returns a single-row DataFrame with the correct column order.
    """
    col_values = {BUSINESSFEATURE_TO_INTERNALFEATURE[k]: v for k, v in user_input.items()}
    return pd.DataFrame(
        [[col_values[col] for col in MODEL_FEATURE_ORDER]], 
        columns = MODEL_FEATURE_ORDER,
    )



MODEL_PATH = Path(__file__).parent / "electricity_price_lgbm_model.joblib"

# Lifespan: Load Model
@asynccontextmanager
async def lifespan(app: FastAPI):
        app.state.model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully FROM {MODEL_PATH}")
        yield  
        logger.info("API shutting down.")


app = FastAPI(
    title = "Electricity Price Forecast API",
    description = "Predicts day-ahead electricity price (EUR/MWh) using a LightGBM model.",
    version = "1.0",
    lifespan = lifespan,
)


@app.get("/", summary = "Root Endpoint")
def root():
    return {"Message": "Electricity Price Forecasting API is running"}



@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}


@app.get("/model-info", summary="Model Metadata")
def model_info():
    return {
        "model_type": "LightGBM",
        "target": "Electricity price (EUR/MWh)",
        "market": "Germany/Luxembourg",
        "number_of_features": len(MODEL_FEATURE_ORDER),
    }


@app.post("/predict", summary="Predict Electricity Price")
def predict_price(input_data: ForecastInput, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
 
    try:
        user_input = input_data.model_dump()
        df = build_model_input(user_input)
        predicted_price = model.predict(df)[0]
 
        logger.info("Prediction made successfully.")
 
        # Only show the clean, readable fields in the response
        visible_input = {k: v for k, v in user_input.items() if k in VISIBLE_FEATURES}
 
        return {
            "input_summary": visible_input,
            "predicted_price": round(float(predicted_price), 2),
            "unit": "EUR/MWh",
            "market": "Germany/Luxembourg",
        }
 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong during prediction.")
