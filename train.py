import os
import requests
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models import infer_signature

SUPABASE_URL = "https://peiuuworaqxesmxfowkf.supabase.co/rest/v1"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def fetch_data():
    headers = {"Authorization": f"Bearer {SUPABASE_KEY}", "apikey": SUPABASE_KEY}
    records, offset, limit = [], 0, 1000
    while True:
        r = requests.get(
            f"{SUPABASE_URL}/subastas_casanare",
            headers=headers,
            params={
                "select": "fecha_subasta,sexo_codigo,cantidad_animales,peso_promedio_kg,precio_final_kg,tipo_subasta",
                "precio_final_kg": "not.is.null",
                "peso_promedio_kg": "not.is.null",
                "limit": limit,
                "offset": offset,
            },
        )
        batch = r.json()
        if not batch:
            break
        records.extend(batch)
        offset += limit
    return pd.DataFrame(records)

print("Cargando datos de Supabase...")
df = fetch_data()
print(f"Registros cargados: {len(df)}")

# Convertir tipos
for col in ["peso_promedio_kg", "cantidad_animales", "precio_base_kg", "precio_final_kg"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filtrar outliers: precios realistas para ganado en COP/kg (3000 - 25000)
# Solo subasta GENERAL
df = df[df["tipo_subasta"].str.upper() == "GENERAL"]
df = df[(df["precio_final_kg"] >= 3000) & (df["precio_final_kg"] <= 25000)]
print(f"Registros tras filtrar outliers: {len(df)}")

# Features temporales
df["fecha_subasta"] = pd.to_datetime(df["fecha_subasta"], errors="coerce")
df["mes"] = df["fecha_subasta"].dt.month
df["anio"] = df["fecha_subasta"].dt.year

# Codificar categóricas
encoders = {}
le = LabelEncoder()
df["sexo_codigo_enc"] = le.fit_transform(df["sexo_codigo"].fillna("Desconocido").astype(str))
encoders["sexo_codigo"] = le

features = ["peso_promedio_kg", "cantidad_animales", "sexo_codigo_enc", "mes", "anio"]

df = df.dropna(subset=features + ["precio_final_kg"])
X = df[features]
y = df["precio_final_kg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("subastas-casanare")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("tipo_subasta", "GENERAL")
    mlflow.log_param("n_registros", len(df))
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", mse ** 0.5)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
    run_id = run.info.run_id

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

print(f"✅ Entrenamiento completo. RMSE: {mse**0.5:.2f} COP/kg | R2: {r2:.4f}")
