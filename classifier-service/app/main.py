from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.classifier import Classifier
from app.model_client import ModelClient
import numpy as np
import requests

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

print("Starting classifier service...")
print("Model will be loaded when needed.")


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

def check_training_status():
    """Check what's happening with the training service"""
    try:
        response = requests.get(f"{model_client.trainer_service_url}/health", timeout=5)
        if response.status_code != 200:
            return {"error": "Cannot connect to training service", "status": "service_error"}
            
        trainer_info = response.json()
        training_status = trainer_info.get('training_status', 'unknown')
        training_message = trainer_info.get('training_message', 'Unknown status')
        
        return {
            "training_status": training_status,
            "training_message": training_message,
            "is_completed": training_status == "completed"
        }
        
    except Exception as e:
        return {"error": f"Training service is offline: {str(e)}", "status": "offline"}

@app.post("/classify")
def classify(request: dict):
    """Classify new data"""
    try:
        # Check if we need to load the model
        if not model_client.model_data:
            print("Model not loaded yet, checking training status...")
            
            status_info = check_training_status()
            
            # If there's an error connecting to trainer
            if "error" in status_info:
                return {
                    "error": status_info["error"],
                    "status": status_info.get("status", "unknown"),
                    "message": "Cannot check training progress"
                }
            
            # If training is not completed yet
            if not status_info["is_completed"]:
                return {
                    "error": f"Model is still being trained: {status_info['training_message']}",
                    "status": "training",
                    "training_stage": status_info["training_status"],
                    "message": status_info["training_message"],
                    "suggestion": "Please wait and try again in a few minutes"
                }
            
            # Training is completed, try to load model
            print("Training completed, trying to load model...")
            success = model_client.get_model_from_trainer()
            if not success:
                return {
                    "error": "Model training completed but failed to load",
                    "status": "load_error",
                    "message": "There's a technical problem loading the trained model"
                }
        
        # Validate input
        features = request.get("features", {})
        if not features:
            return {"error": "Missing input data. No features provided."}
            
        print(f"Got request for classification: {features}")
        
        # Perform classification
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
