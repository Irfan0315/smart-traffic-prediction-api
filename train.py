
# Smart Traffic Prediction API
# Step 1: Train the machine learning model

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("data/traffic.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# 2. Display the first five rows
print("\nFirst five rows:")
print(df.head())

# 3. Display the congestion classes
print("\nCongestion categories:")
print(df["Congestion_Level"].value_counts())

# 4. Prepare the time feature
# In this dataset, 6.15 means 6:15, not 6.15 decimal hours.
time_value = pd.to_numeric(df["Time"], errors="coerce")
hour = time_value.fillna(0).astype(int)

minute = (
    (time_value.fillna(0) - hour) * 100
).round().astype(int)

# 5. Select input features
# We exclude Traffic_Situation because it may overlap
# with the congestion target and create misleading results.
X = df[
    [
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
    ]
].copy()

X["Hour"] = hour
X["Minute"] = minute

# 6. Select the output to predict
y = df["Congestion_Level"].astype(str)

# Remove rows without a target label
valid_rows = y.notna() & (y.str.strip() != "")
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows]

# 7. Identify text and numeric columns
text_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

number_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 8. Prepare the data
text_pipeline = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

number_pipeline = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(transformers=[
    ("text", text_pipeline, text_columns),
    ("numbers", number_pipeline, number_columns)
])

# 9. Create the complete ML pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 10. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 11. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 12. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 13. Save the trained pipeline
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/traffic_model.joblib")

print("\nModel saved successfully!")
print("Location: models/traffic_model.joblib")
