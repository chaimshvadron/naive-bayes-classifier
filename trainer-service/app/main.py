from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from trainer_controller import TrainerController
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

controller = TrainerController()
print("Starting to load and prepare data...")
controller.load_and_prepare_data()
print("Starting to train model...")
controller.train_model()
print("Training service is ready!")

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

@app.get("/model")
def get_trained_model():
    """Get the trained model"""
    try:
        model_data = controller.get_trained_model()
        print("Sending trained model to classifier service")
        return convert_numpy_to_python(model_data)
    except Exception as e:
        print(f"Error getting model: {e}")
        return {"error": str(e)}

@app.get("/features")
def get_features():
    """Get available features"""
    try:
        features_data = controller.trainer.get_unique_values_dict(
            controller.cleaned_data, controller.feature_columns
        )
        return convert_numpy_to_python(features_data)
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Check if service is working"""
    return {"status": "healthy", "service": "trainer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
