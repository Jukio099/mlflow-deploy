import os
import sys
import requests
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

SUPABASE_URL = "https://peiuuworaqxesmxfowkf.supabase.co/rest/v1"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
THRESHOLD = 5_000_000_000.0  # MSE en COP² (precios ~6000-8000 COP/kg)

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

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

df = fetch_data()
df["tipo_subasta_enc"] = encoders["tipo"].transform(df["tipo_subasta"].fillna("Desconocido"))
df["sexo_codigo_enc"] = encoders["sexo"].transform(df["sexo_codigo"].fillna("Desconocido"))

features = ["peso_promedio_kg", "cantidad_animales", "precio_base_kg", "tipo_subasta_enc", "sexo_codigo_enc"]
df = df.dropna(subset=features + ["precio_final_kg"])
X = df[features]
y = df["precio_final_kg"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"MSE: {mse:.2f} (umbral: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ Modelo válido.")
    sys.exit(0)
else:
    print("❌ MSE supera el umbral.")
    sys.exit(1)
