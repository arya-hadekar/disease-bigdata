from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# -------------------------------
# Load model once
# -------------------------------
model = joblib.load("model/model.pkl")
le = joblib.load("model/label_encoder.pkl")
features = joblib.load("model/features.pkl")

# -------------------------------
# API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Fill missing features
        full_data = {col: data.get(col, 0) for col in features}

        df = pd.DataFrame([full_data])

        pred = model.predict(df)[0]
        disease = le.inverse_transform([pred])[0]

        return jsonify({
            "prediction": disease
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)