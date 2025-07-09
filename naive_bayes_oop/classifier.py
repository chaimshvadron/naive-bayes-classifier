class Classifier:
    
    def classify_customer(self, customer_choices, model):
        best_class = None
        best_score = 0
        
        for class_name in model:
            score = 1 
            
            for feature, value in customer_choices.items():
                score *= model[class_name][feature][value]
            
            if score > best_score:
                best_score = score
                best_class = class_name
        
        return best_class
