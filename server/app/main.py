from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.app.naive_bayes_controller import NaiveBayesController
import numpy as np

app = FastAPI()

# הגדרת CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = NaiveBayesController()
controller.load_and_prepare_data()
controller.train_model()

def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(i) for i in obj]
    return obj

@app.post("/classify")
def classify(request: dict):
    try:
        features = request.get("features", {})
        if not features:
            return {"error": "Missing input data. No features provided."}
        print(f"Received features for classification: {features}")
        result = controller.classifier.classify_customer(features, controller.model)
        reliability = controller.reliability
        
        response = {
            "predicted_class": result["class"],
            "probability": result["probability"],
            "reliability": reliability
        }
        print(f"Classification result: {response}")
        return convert_numpy_to_python(response)
    except Exception as e:
        return {"error": str(e)}

@app.get("/features")
def get_features():
    try:
        features_data = controller.trainer.get_unique_values_dict(
            controller.cleaned_data, controller.feature_columns
        )
        return convert_numpy_to_python(features_data)
    except Exception as e:
        return {"error": str(e)}
