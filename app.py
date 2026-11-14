
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Smart Traffic Prediction API",
    description="Predict traffic congestion using Machine Learning",
    version="1.0"
)

model = joblib.load("models/traffic_model.joblib")


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


@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Traffic Prediction API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_traffic(data: TrafficData):

    traffic = pd.DataFrame([data.model_dump()])

    prediction = model.predict(traffic)[0]

    return {
        "predicted_congestion": str(prediction)
    }
