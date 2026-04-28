import gradio as gr
import joblib
import numpy as np
import os

base = os.path.dirname(__file__)
model = joblib.load(os.path.join(base, "..", "Model", "model.pkl"))
encoders = joblib.load(os.path.join(base, "..", "Model", "encoders.pkl"))

sexos = list(encoders["sexo_codigo"].classes_)

def predecir(sexo_codigo, cantidad_animales, peso_promedio_kg, mes, anio):
    sexo_enc = encoders["sexo_codigo"].transform([sexo_codigo])[0]
    X = np.array([[peso_promedio_kg, cantidad_animales, sexo_enc, mes, anio]])
    precio = model.predict(X)[0]
    return f"${precio:,.0f} COP/kg"

demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Dropdown(choices=sexos, label="Sexo del Ganado"),
        gr.Number(label="Cantidad de Animales", value=10),
        gr.Number(label="Peso Promedio (kg)", value=350),
        gr.Slider(minimum=1, maximum=12, step=1, value=6, label="Mes"),
        gr.Slider(minimum=2020, maximum=2026, step=1, value=2025, label="Año"),
    ],
    outputs=gr.Text(label="Precio Final Estimado (COP/kg)"),
    title="Predictor de Precio - Subasta General Ganadera Casanare",
    description="Predice el precio final por kg de ganado en la Subasta General de Yopal, Casanare. Ingresa las características del ganado que llevarías a subastar.",
)

if __name__ == "__main__":
    demo.launch()
