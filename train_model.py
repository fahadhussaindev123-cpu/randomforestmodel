import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load your dataset
df = pd.read_csv("Crop_recommendation.csv")
print("Dataset Loaded:", df.shape)

# 2. Split features and target label
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# 3. 80% Training, 20% Testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    random_state=42
)
rf_model.fit(X_train, y_train)

# 5. Evaluate Accuracy
y_pred = rf_model.predict(X_test)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 6. Save the trained model
joblib.dump(rf_model, "rf_crop_model.pkl")
print("Success: Model saved as 'rf_crop_model.pkl'")