import json
from mlflow.tracking import MlflowClient
import mlflow
import unittest
import os
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
os.environ["MLFLOW_TRACKING_USERNAME"]= dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "sfgrahman"
repo_name = "ci_mlops_complete_end_to_end_project"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
model_name = "Best Model"

class TestModelLoading(unittest.TestCase):
    
    def test_model_in_staging(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        self.assertGreater(len(versions), 0, "No model found in the 'Staging' stage.")
    
    
    def test_model_loading(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping model loading test.") 
        latest_version = versions[0].version
        run_id = versions[0].run_id
        
        logged_model = f"runs:/{run_id}/{model_name}"
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load the model: {e}")
        
        self.assertIsNotNone(loaded_model, "The loaded model is None.")
        print(f"Model successfully loaded from {logged_model}")

if __name__=="__main__":
    unittest.main()
        
