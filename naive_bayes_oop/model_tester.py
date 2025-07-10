from classifier import Classifier

class ModelTester:
    
    def test_model(self, model, test_data):
        classifier = Classifier()
        correct = 0
        total = len(test_data)
        features = test_data.columns[:-1]
        for _, row in test_data.iterrows():
            customer_choices = {feature: row[feature] for feature in features}
            predicted = classifier.classify_customer(customer_choices, model)
            actual = row[test_data.columns[-1]]
            if predicted == actual:
                correct += 1
        return correct / total * 100.0
