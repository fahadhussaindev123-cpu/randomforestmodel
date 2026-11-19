import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# 2. Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# 5. Train the model
print("Training KrishiVision crop recommendation model...")

model.fit(X_train, y_train)

print("Training completed!")

# 6. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Save the trained model
joblib.dump(model, "crop_model.joblib")

print("\nModel saved as crop_model.joblib")