"""Utilidades compartidas para el pipeline de subastas ganaderas."""
import os
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SUPABASE_URL = "https://peiuuworaqxesmxfowkf.supabase.co/rest/v1"

# Features usadas por el modelo (variables conocidas antes de la subasta)
FEATURES = ["peso_promedio_kg", "cantidad_animales", "sexo_codigo_enc", "mes", "anio"]
TARGET = "precio_final_kg"


def fetch_subastas_tradicional() -> pd.DataFrame:
    """Descarga registros de Subasta Tradicional desde Supabase con filtro server-side."""
    key = os.getenv("SUPABASE_KEY")
    headers = {"Authorization": f"Bearer {key}", "apikey": key}
    records, offset = [], 0

    while True:
        r = requests.get(
            f"{SUPABASE_URL}/subastas_casanare",
            headers=headers,
            params={
                "select": "fecha_subasta,sexo_codigo,cantidad_animales,peso_promedio_kg,precio_final_kg",
                "tipo_subasta": "eq.Tradicional",
                "precio_final_kg": "not.is.null",
                "peso_promedio_kg": "not.is.null",
                "limit": 1000,
                "offset": offset,
            },
        )
        batch = r.json()
        if not batch or not isinstance(batch, list):
            break
        records.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    df = pd.DataFrame(records)

    # Convertir tipos numéricos
    for col in ["peso_promedio_kg", "cantidad_animales", "precio_final_kg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtrar precios realistas para ganado en COP/kg
    df = df[(df[TARGET] >= 3000) & (df[TARGET] <= 25000)]

    # Features temporales desde fecha
    df["fecha_subasta"] = pd.to_datetime(df["fecha_subasta"], errors="coerce")
    df["mes"] = df["fecha_subasta"].dt.month
    df["anio"] = df["fecha_subasta"].dt.year

    return df


def encode_features(df: pd.DataFrame, encoders: dict = None):
    """Codifica variables categóricas. Entrena encoders si no se pasan."""
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    le = encoders.get("sexo_codigo", LabelEncoder())
    if fit_mode:
        df["sexo_codigo_enc"] = le.fit_transform(df["sexo_codigo"].fillna("Desconocido").astype(str))
        encoders["sexo_codigo"] = le
    else:
        df["sexo_codigo_enc"] = le.transform(df["sexo_codigo"].fillna("Desconocido").astype(str))

    return df, encoders
