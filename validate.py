import os
import sys
import requests
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SUPABASE_URL = "https://peiuuworaqxesmxfowkf.supabase.co/rest/v1"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MSE_THRESHOLD = 5_000_000.0  # RMSE ~2236 COP/kg sin precio_base
R2_THRESHOLD = 0.30

def fetch_data():
    headers = {"Authorization": f"Bearer {SUPABASE_KEY}", "apikey": SUPABASE_KEY}
    records, offset, limit = [], 0, 1000
    while True:
        r = requests.get(
            f"{SUPABASE_URL}/subastas_casanare",
            headers=headers,
            params={
                "select": "fecha_subasta,tipo_subasta,sexo_codigo,cantidad_animales,peso_promedio_kg,precio_final_kg",
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

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

df = fetch_data()
for col in ["peso_promedio_kg", "cantidad_animales", "precio_base_kg", "precio_final_kg"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["tipo_subasta"].str.upper() == "GENERAL"]
df = df[(df["precio_final_kg"] >= 3000) & (df["precio_final_kg"] <= 25000)]

df["fecha_subasta"] = pd.to_datetime(df["fecha_subasta"], errors="coerce")
df["mes"] = df["fecha_subasta"].dt.month
df["anio"] = df["fecha_subasta"].dt.year

df["sexo_codigo_enc"] = encoders["sexo_codigo"].transform(df["sexo_codigo"].fillna("Desconocido").astype(str))

features = ["peso_promedio_kg", "cantidad_animales", "sexo_codigo_enc", "mes", "anio"]

df = df.dropna(subset=features + ["precio_final_kg"])
X, y = df[features], df["precio_final_kg"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"RMSE: {mse**0.5:.2f} COP/kg | R2: {r2:.4f}")
print(f"Umbrales → MSE: {MSE_THRESHOLD} | R2 mínimo: {R2_THRESHOLD}")

if mse <= MSE_THRESHOLD and r2 >= R2_THRESHOLD:
    print("✅ Modelo válido.")
    sys.exit(0)
else:
    print("❌ Modelo no cumple umbrales.")
    sys.exit(1)
