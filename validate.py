"""
Validación del modelo registrado en MLflow.
Carga el modelo por run_id y evalúa con datos externos de Supabase.
"""
import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils import fetch_subastas_general, encode_features, FEATURES, TARGET

# Umbrales de calidad aceptables para el modelo sin precio_base_kg
MSE_THRESHOLD = 5_000_000.0   # RMSE ≈ 2236 COP/kg
R2_THRESHOLD = 0.30

# --- Cargar modelo desde MLflow usando run_id ---
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

if not os.path.exists("run_id.txt"):
    print("ERROR: run_id.txt no encontrado. Ejecuta make train primero.")
    sys.exit(1)

with open("run_id.txt") as f:
    run_id = f.read().strip()

print(f"Cargando modelo desde MLflow (run_id: {run_id})...")
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# --- Datos externos desde Supabase ---
print("Cargando datos de validación desde Supabase...")
df = fetch_subastas_general()

# Cargar encoders entrenados para codificar igual que en train
import joblib
encoders = joblib.load("encoders.pkl")
df, _ = encode_features(df, encoders=encoders)
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Evaluación ---
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\nResultados de validación:")
print(f"  RMSE : {mse**0.5:,.2f} COP/kg  (umbral MSE: {MSE_THRESHOLD:,})")
print(f"  R2   : {r2:.4f}              (umbral R2 : {R2_THRESHOLD})")

if mse <= MSE_THRESHOLD and r2 >= R2_THRESHOLD:
    print("\n✅ Modelo válido — cumple criterios de calidad.")
    sys.exit(0)
else:
    print("\n❌ Modelo no cumple los umbrales de calidad.")
    sys.exit(1)
