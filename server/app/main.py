from fastapi import FastAPI
from naive_bayes_controller import NaiveBayesController

app = FastAPI()

@app.post("/classify")
def classify(request: dict):
    try:
        controller = NaiveBayesController()
        controller.load_and_prepare_data()
        controller.train_model()
        predicted = controller.classifier.classify_customer(request.get("features", {}), controller.model)
        reliability = controller.tester.test_model(controller.model, controller.cleaned_test_data)
        return {"predicted_class": predicted, "reliability": reliability}
    except Exception as e:
        return {"error": str(e)}
 
@app.get("/features")
def get_features():
    try:
        controller = NaiveBayesController()
        controller.load_and_prepare_data()
        # compute unique feature values without full training
        features = controller.trainer.get_unique_values_dict(controller.cleaned_data)
        return features
    except Exception as e:
        return {"error": str(e)}
