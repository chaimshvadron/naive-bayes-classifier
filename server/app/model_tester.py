from classifier import Classifier

class ModelTester:
    
    def test_model(self, model, test_data, feature_columns, target_column):
        classifier = Classifier()

        def is_correct_prediction(row):
            customer_choices = row[feature_columns].to_dict()
            predicted = classifier.classify_customer(customer_choices, model)
            actual = row[target_column]
            return predicted == actual

        correct_predictions = test_data.apply(is_correct_prediction, axis=1).sum()
        total = len(test_data)
        return correct_predictions / total * 100.0
                                                           