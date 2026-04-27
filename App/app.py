import gradio as gr
import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "Model", "model.pkl")
model = joblib.load(model_path)

feature_names = [
    "age", "sex", "bmi", "bp",
    "s1", "s2", "s3", "s4", "s5", "s6"
]

def predict(*args):
    features = np.array(args).reshape(1, -1)
    prediction = model.predict(features)[0]
    return round(float(prediction), 2)

inputs = [gr.Number(label=name) for name in feature_names]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Number(label="Diabetes Progression Score"),
    title="Diabetes Progression Predictor",
    description="Modelo de regresión lineal entrenado con el dataset de diabetes de scikit-learn.",
)

if __name__ == "__main__":
    demo.launch()
