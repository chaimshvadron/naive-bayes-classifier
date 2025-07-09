class Classifier:
    
    def classify_customer(self, customer_choices, model):
        best_class = None
        best_score = 0
        
        conditionals = model['conditionals']
        priors = model['priors']
        
        for class_name in conditionals:
            score = priors[class_name]
            
            for feature, value in customer_choices.items():
                score *= conditionals[class_name][feature][value]
            
            if score > best_score:
                best_score = score
                best_class = class_name
        
        return best_class
