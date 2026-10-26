
from fastapi.testclient import TestClient
from app import app

# Create a test client for our API
client = TestClient(app)


# Test 1: Check the home page
def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


# Test 2: Check the health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


# Test 3: Check traffic prediction
def test_predict():
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

    response = client.post("/predict", json=traffic_data)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_congestion"] in [
        "Low", "Medium", "High", "Severe"
    ]


# Test 4: Check invalid input
def test_invalid_vehicle_count():
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
        "Vehicle_Count": -5,
        "Avg_Speed_km_h": 20,
        "Road_Type": "Highway",
        "Hour": 18,
        "Minute": 30
    }

    response = client.post("/predict", json=traffic_data)

    assert response.status_code == 422
