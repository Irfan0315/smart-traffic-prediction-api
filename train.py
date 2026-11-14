
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/traffic.csv")

print("Dataset loaded successfully!")
print(df.head())

time = pd.to_numeric(df["Time"], errors="coerce").fillna(0)

X = df[[
    "City",
    "Vehicle_Type",
    "Junction_Type",
    "Peak_Hour",
    "Weather_Condition",
    "Lane_Count",
    "Accident_Reported",
    "Traffic_Light_Status",
    "Day_Type",
    "Vehicle_Count",
    "Avg_Speed_km_h",
    "Road_Type"
]].copy()

X["Hour"] = time.astype(int)
X["Minute"] = ((time - X["Hour"]) * 100).round().astype(int)

y = df["Congestion_Level"]

valid = y.notna() & y.astype("string").str.strip().ne("")

X = X[valid]
y = y[valid].astype(str)

text_cols = X.select_dtypes(include=["object", "string"]).columns
number_cols = X.select_dtypes(include=["number"]).columns

text_process = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

number_process = Pipeline([
    ("fill", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("text", text_process, text_cols),
    ("number", number_process, number_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")

model.fit(X_train, y_train)

print("Model trained successfully!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/traffic_model.joblib")

print("Model saved successfully!")
