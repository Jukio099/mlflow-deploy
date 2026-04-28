# Subasta Ganadera Casanare — Pipeline MLflow + GitHub Actions

Pipeline de CI/CD para entrenar, validar y desplegar un modelo de predicción de precios de ganado en la **Subasta General de Yopal, Casanare**.

## Dataset

- **Fuente:** Base de datos propia en Supabase (`subastas_casanare`)
- **Registros:** ~42,000 lotes de ganado subastados en Yopal, Casanare
- **Filtro:** Solo `tipo_subasta = GENERAL`
- **Target:** `precio_final_kg` — precio final pagado por kg de ganado (COP)

### Variables del modelo

| Feature | Descripción |
|---|---|
| `peso_promedio_kg` | Peso promedio del lote en kg |
| `cantidad_animales` | Número de animales en el lote |
| `sexo_codigo` | Categoría del ganado (NG, HV, VC, MC...) |
| `mes` | Mes de la subasta |
| `anio` | Año de la subasta |

> `precio_base_kg` fue excluido deliberadamente — es el precio que fija la subasta, no una variable disponible antes del evento.

## Modelo

**GradientBoostingRegressor** (200 estimadores, max_depth=5, lr=0.05)

## Estructura

```
mlflow-deploy/
├── train.py          # Entrenamiento y registro en MLflow
├── validate.py       # Validación cargando modelo desde MLflow
├── utils.py          # Funciones compartidas (fetch, encode)
├── requirements.txt
├── Makefile
├── App/              # App Gradio para HuggingFace Spaces
├── Model/            # Modelo y encoders serializados
├── Results/          # Métricas generadas
└── .github/workflows/
    ├── mlflow-ci.yml # CI: train → validate → eval → update
    └── cd.yml        # CD: deploy a HuggingFace Spaces
```

## Ejecución local

```bash
pip install -r requirements.txt
export SUPABASE_KEY=tu_clave
make train
make validate
```
