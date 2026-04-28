import gradio as gr
import joblib
import numpy as np
import os

base = os.path.dirname(__file__)
model = joblib.load(os.path.join(base, "..", "Model", "model.pkl"))
encoders = joblib.load(os.path.join(base, "..", "Model", "encoders.pkl"))

tipos = list(encoders["tipo_subasta"].classes_)
sexos = list(encoders["sexo_codigo"].classes_)
martillos = list(encoders["martillo"].classes_)

def predecir(tipo_subasta, sexo_codigo, martillo, cantidad_animales, peso_promedio_kg, mes, anio):
    tipo_enc = encoders["tipo_subasta"].transform([tipo_subasta])[0]
    sexo_enc = encoders["sexo_codigo"].transform([sexo_codigo])[0]
    martillo_enc = encoders["martillo"].transform([martillo])[0]
    X = np.array([[peso_promedio_kg, cantidad_animales, tipo_enc, sexo_enc, martillo_enc, mes, anio]])
    precio = model.predict(X)[0]
    return f"${precio:,.0f} COP/kg"

demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Dropdown(choices=tipos, label="Tipo de Subasta"),
        gr.Dropdown(choices=sexos, label="Sexo del Ganado"),
        gr.Dropdown(choices=martillos, label="Martillo"),
        gr.Number(label="Cantidad de Animales", value=10),
        gr.Number(label="Peso Promedio (kg)", value=350),
        gr.Slider(minimum=1, maximum=12, step=1, value=6, label="Mes"),
        gr.Slider(minimum=2020, maximum=2026, step=1, value=2025, label="Año"),
    ],
    outputs=gr.Text(label="Precio Final Estimado (COP/kg)"),
    title="Predictor de Precio - Subasta Ganadera Casanare",
    description="Predice el precio final por kg de ganado en subastas de Yopal, Casanare. Modelo GradientBoosting entrenado con datos reales.",
)

if __name__ == "__main__":
    demo.launch()
