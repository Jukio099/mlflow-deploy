import os
import sys
import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

# --- Tracking URI: Supabase PostgreSQL (set via env var) or local fallback ---
tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file://" + os.path.abspath(os.path.join(os.getcwd(), "mlruns")),
)
mlflow.set_tracking_uri(tracking_uri)
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# --- Experiment ---
experiment_name = "diabetes-linear-regression"
mlflow.set_experiment(experiment_name)

# --- Data & Model ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- MLflow Run ---
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("mse", mse)

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

# Save run_id for validate step
with open("run_id.txt", "w") as f:
    f.write(run_id)

# Save model locally for validate.py
joblib.dump(model, "model.pkl")

# Save metrics for report
os.makedirs("Results", exist_ok=True)
with open("Results/metrics.txt", "w") as f:
    f.write(f"MSE: {mse:.4f}\nRun ID: {run_id}\n")

print(f"✅ Entrenamiento completo. MSE: {mse:.4f}")
