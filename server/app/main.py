from fastapi import FastAPI
from naive_bayes_controller import NaiveBayesController

app = FastAPI()

controller = NaiveBayesController()
controller.load_and_prepare_data()
controller.train_model()

@app.post("/classify")
def classify(request: dict):
    try:
        features = request.get("features", {})
        predicted = controller.classifier.classify_customer(features, controller.model)
        reliability = controller.tester.test_model(controller.model, controller.cleaned_test_data)
        return {"predicted_class": predicted, "reliability": reliability}
    except Exception as e:
        return {"error": str(e)}

@app.get("/features")
def get_features():
    try:
        return controller.trainer.get_unique_values_dict(controller.cleaned_data)
    except Exception as e:
        return {"error": str(e)}
