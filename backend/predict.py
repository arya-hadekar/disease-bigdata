import joblib
import pandas as pd

# Load
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")
features = joblib.load("model/features.pkl")

# Example input
input_data = {
    "fever": 1,
    "cough": 1,
    "headache": 0
}

# Fill missing features
full_data = {col: input_data.get(col, 0) for col in features}

df = pd.DataFrame([full_data])

# Predict
pred = model.predict(df)[0]
disease = le.inverse_transform([pred])[0]

print("Predicted Disease:", disease)