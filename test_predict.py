import joblib
import numpy as np
import pandas as pd

# Load the saved Random Forest model
model = joblib.load("rf_crop_model.pkl")

def predict_crops(n, p, k, temp, humidity, ph, rainfall):
    # Create DataFrame with exact column names
    features = pd.DataFrame(
        [[n, p, k, temp, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    # Get probabilities for all crops
    probs = model.predict_proba(features)[0]
    crops = model.classes_
    top_3 = np.argsort(probs)[::-1][:3]

    return [
        {"rank": i + 1, "crop": crops[idx], "feasibility": f"{probs[idx] * 100:.1f}%"}
        for i, idx in enumerate(top_3)
    ]

print("Top 3 Crop Recommendations:", predict_crops(90, 42, 43, 200.8, 82.0, 6.5, 202.9))