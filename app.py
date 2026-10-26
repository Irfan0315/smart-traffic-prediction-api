
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# 1. Create the FastAPI application
app = FastAPI(
    title="Smart Traffic Prediction API",
    description="Predict traffic congestion using Machine Learning",
    version="1.0"
)

# 2. Load the trained model
model = joblib.load("models/traffic_model.joblib")


# 3. Define the traffic input
class TrafficData(BaseModel):
    City: str
    Vehicle_Type: str
    Junction_Type: str
    Peak_Hour: str
    Weather_Condition: str
    Lane_Count: int = Field(ge=1)
    Accident_Reported: str
    Traffic_Light_Status: str
    Day_Type: str
    Vehicle_Count: int = Field(ge=0)
    Avg_Speed_km_h: float = Field(ge=0, le=200)
    Road_Type: str
    Hour: int = Field(ge=0, le=23)
    Minute: int = Field(ge=0, le=59)


# 4. Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Traffic Prediction API",
        "docs": "/docs"
    }


# 5. Health-check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# 6. Prediction endpoint
@app.post("/predict")
def predict_traffic(data: TrafficData):

    # Convert the received data into a DataFrame
    traffic = pd.DataFrame([data.model_dump()])

    # Predict the congestion level
    prediction = model.predict(traffic)[0]

    return {
        "predicted_congestion": str(prediction)
    }
