import mlflow
from fastapi import FastAPI
import pandas as pd 
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

app = FastAPI(
    title="Water Potability Prediction",
    description = "An API to predict whether water is potable(safe to drink) or not."
)

# Set the MLFlow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "sfgrahman"
repo_name = "ci_mlops_complete_end_to_end_project"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
model_name ="Best Model"
# Load the latest model from MLFlow
def load_model():
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    run_id = versions[0].run_id
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/{model_name}")

model = load_model()

class Water(BaseModel):
    ph : float
    Hardness : float
    Solids : float
    Chloramines: float
    Sulfate : float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def index():
    return "Welcome to Water Potability Prediction  FastAPI"

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame({
        'ph' : [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' : [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate' : [water.Chloramines],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes':[water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })
    predicted_value = model.predict(sample)
    
    if predicted_value[0] ==1:
        return {"result":"Water is Consumable"}
    else:
        return {"result": "water is not Consumable"}
    
    