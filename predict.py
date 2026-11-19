import pandas as pd
import joblib

# Load the trained model
model = joblib.load("crop_model.joblib")

# Example agricultural inputs
farm_data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.88,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.94
}])

# Predict the crop
prediction = model.predict(farm_data)

print("Recommended crop:", prediction[0])