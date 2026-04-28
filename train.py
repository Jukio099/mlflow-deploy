import os
import sys
import requests
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
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
                "select": "tipo_subasta,sexo_codigo,cantidad_animales,peso_promedio_kg,precio_base_kg,precio_final_kg",
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

# Codificar variables categóricas
le_tipo = LabelEncoder()
le_sexo = LabelEncoder()
df["tipo_subasta_enc"] = le_tipo.fit_transform(df["tipo_subasta"].fillna("Desconocido"))
df["sexo_codigo_enc"] = le_sexo.fit_transform(df["sexo_codigo"].fillna("Desconocido"))

features = ["peso_promedio_kg", "cantidad_animales", "precio_base_kg", "tipo_subasta_enc", "sexo_codigo_enc"]
target = "precio_final_kg"

df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("subastas-casanare")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("n_registros", len(df))
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
    run_id = run.info.run_id

# Guardar modelo y encoders
os.makedirs("Model", exist_ok=True)
joblib.dump(model, "model.pkl")
joblib.dump(model, "Model/model.pkl")
joblib.dump({"tipo": le_tipo, "sexo": le_sexo}, "Model/encoders.pkl")
joblib.dump({"tipo": le_tipo, "sexo": le_sexo}, "encoders.pkl")

with open("run_id.txt", "w") as f:
    f.write(run_id)

os.makedirs("Results", exist_ok=True)
with open("Results/metrics.txt", "w") as f:
    f.write(f"MSE: {mse:.2f}\nR2: {r2:.4f}\nRegistros: {len(df)}\nRun ID: {run_id}\n")

print(f"✅ Entrenamiento completo. MSE: {mse:.2f} | R2: {r2:.4f}")
