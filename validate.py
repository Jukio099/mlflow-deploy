import os
import sys
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

THRESHOLD = 5000.0

# --- Load dataset (same split as train.py) ---
X, y = load_diabetes(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Load model ---
model_path = os.path.abspath("model.pkl")
if not os.path.exists(model_path):
    print(f"ERROR: model.pkl not found at {model_path}")
    sys.exit(1)

model = joblib.load(model_path)

# --- Predict & evaluate ---
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"MSE: {mse:.4f} (umbral: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ Modelo válido.")
    sys.exit(0)
else:
    print("❌ MSE supera el umbral.")
    sys.exit(1)
