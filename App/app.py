import gradio as gr
import joblib
import numpy as np
import os

base = os.path.dirname(__file__)
model = joblib.load(os.path.join(base, "..", "Model", "model.pkl"))
encoders = joblib.load(os.path.join(base, "..", "Model", "encoders.pkl"))

tipos = list(encoders["tipo"].classes_)
sexos = list(encoders["sexo"].classes_)

def predecir(tipo_subasta, sexo_codigo, cantidad_animales, peso_promedio_kg, precio_base_kg):
    tipo_enc = encoders["tipo"].transform([tipo_subasta])[0]
    sexo_enc = encoders["sexo"].transform([sexo_codigo])[0]
    X = np.array([[peso_promedio_kg, cantidad_animales, precio_base_kg, tipo_enc, sexo_enc]])
    precio = model.predict(X)[0]
    return f"${precio:,.0f} COP/kg"

demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Dropdown(choices=tipos, label="Tipo de Subasta"),
        gr.Dropdown(choices=sexos, label="Sexo del Ganado"),
        gr.Number(label="Cantidad de Animales", value=10),
        gr.Number(label="Peso Promedio (kg)", value=350),
        gr.Number(label="Precio Base (COP/kg)", value=6000),
    ],
    outputs=gr.Text(label="Precio Final Estimado (COP/kg)"),
    title="Predictor de Precio - Subasta Ganadera Casanare",
    description="Predice el precio final por kg de ganado en subastas de Yopal, Casanare, basado en datos reales.",
)

if __name__ == "__main__":
    demo.launch()
