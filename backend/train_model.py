import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/cleaned_dataset.csv")

target = "diseases"

# Drop nulls
df = df.dropna()

# -------------------------------
# Encode labels
# -------------------------------
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# -------------------------------
# Features & Target
# -------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model
# -------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -------------------------------
# Accuracy
# -------------------------------
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(list(X.columns), "model/features.pkl")

print("Model saved successfully")