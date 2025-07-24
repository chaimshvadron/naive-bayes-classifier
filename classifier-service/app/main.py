from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .classifier import Classifier
from .model_client import ModelClient
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

# יצירת לקוח המודל והמסווג
model_client = ModelClient()
classifier = Classifier()

print("Starting to get trained model from training service...")
success = model_client.get_model_from_trainer()

if not success:
    print("Could not get model! Service will work in limited mode.")

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
    """Classify new data"""
    try:
        if not model_client.model_data:
            return {"error": "No model available. Try refreshing the model."}
            
        features = request.get("features", {})
        if not features:
            return {"error": "Missing input data. No features provided."}
            
        print(f"Got request for classification: {features}")
        
        # use model for classification
        model = model_client.model_data['model']
        result = classifier.classify_customer(features, model)
        reliability = model_client.model_data['reliability']
        
        response = {
            "predicted_class": result["class"],
            "probability": result["probability"],
            "reliability": reliability
        }
        print(f"Classification result: {response}")
        return convert_numpy_to_python(response)
        
    except Exception as e:
        print(f"Error in classification: {e}")
        return {"error": str(e)}

@app.get("/features")
def get_features():
    """Get available features"""
    try:
        features_data = model_client.get_features_from_trainer()
        if features_data:
            return convert_numpy_to_python(features_data)
        else:
            return {"error": "Could not get features from training service"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/refresh-model")
def refresh_model():
    """Refresh model from training service"""
    try:
        success = model_client.get_model_from_trainer()
        if success:
            return {"message": "Model refreshed successfully"}
        else:
            return {"error": "Could not refresh model"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Check if service is working"""
    model_status = "loaded" if model_client.model_data else "not_loaded"
    return {
        "status": "healthy", 
        "service": "classifier",
        "model_status": model_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
