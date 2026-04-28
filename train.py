"""
Entrenamiento del modelo de predicción de precios para la Subasta General Ganadera de Casanare.
Dataset: subastas_casanare (Supabase) — datos reales de Yopal, Casanare.
Target: precio_final_kg (COP/kg)
"""
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models import infer_signature

from utils import fetch_subastas_tradicional, encode_features, FEATURES, TARGET

# --- Datos ---
print("Cargando datos de Supabase (subastas_casanare - Tradicional)...")
df = fetch_subastas_tradicional()
df, encoders = encode_features(df)
df = df.dropna(subset=FEATURES + [TARGET])
print(f"Registros listos para entrenamiento: {len(df)}")

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Modelo ---
model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# --- MLflow ---
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("subastas-casanare-general")

with mlflow.start_run() as run:
    mlflow.log_params({
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "tipo_subasta": "Tradicional",
        "n_registros": len(df),
    })
    mlflow.log_metrics({"mse": mse, "rmse": mse ** 0.5, "r2": r2})

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )
    run_id = run.info.run_id

# --- Persistencia local ---
os.makedirs("Model", exist_ok=True)
joblib.dump(model, "model.pkl")
joblib.dump(model, "Model/model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(encoders, "Model/encoders.pkl")

with open("run_id.txt", "w") as f:
    f.write(run_id)

os.makedirs("Results", exist_ok=True)
with open("Results/metrics.txt", "w") as f:
    f.write(f"MSE: {mse:.2f}\nRMSE: {mse**0.5:.2f}\nR2: {r2:.4f}\nRegistros: {len(df)}\nRun ID: {run_id}\n")

print(f"✅ Entrenamiento completo — RMSE: {mse**0.5:.2f} COP/kg | R2: {r2:.4f}")
print(f"   Run ID: {run_id}")
