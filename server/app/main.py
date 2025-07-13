from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from naive_bayes_controller import NaiveBayesController

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

@app.post("/classify")
def classify(request: dict):
    try:
        features = request.get("features", {})
        predicted = controller.classifier.classify_customer(features, controller.model)
        reliability = controller.tester.test_model(
            controller.model,
            controller.cleaned_test_data,
            controller.feature_columns,
            controller.target_column
        )
        return {"predicted_class": predicted, "reliability": reliability}
    except Exception as e:
        return {"error": str(e)}

@app.get("/features")
def get_features():
    try:
        return controller.trainer.get_unique_values_dict(controller.cleaned_data, controller.feature_columns)
    except Exception as e:
        return {"error": str(e)}