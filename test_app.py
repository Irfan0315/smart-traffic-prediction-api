
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

traffic_data = {
    "City": "Sonipat",
    "Vehicle_Type": "Car",
    "Junction_Type": "Crossroad",
    "Peak_Hour": "Yes",
    "Weather_Condition": "Clear",
    "Lane_Count": 4,
    "Accident_Reported": "No",
    "Traffic_Light_Status": "Green",
    "Day_Type": "Weekday",
    "Vehicle_Count": 300,
    "Avg_Speed_km_h": 20,
    "Road_Type": "Highway",
    "Hour": 18,
    "Minute": 30
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    response = client.post("/predict", json=traffic_data)
    assert response.status_code == 200
    assert "predicted_congestion" in response.json()


def test_invalid_vehicle_count():
    data = {**traffic_data, "Vehicle_Count": -5}
    response = client.post("/predict", json=data)
    assert response.status_code == 422
