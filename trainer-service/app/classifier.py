# classifier.py

class Classifier:
    
    def classify_customer(self, customer_choices, model):
        best_class = None
        best_score = 0
        class_scores = {} 
        
        conditionals = model['conditionals']
        priors = model['priors']
        
        for class_name in conditionals:
            score = priors[class_name]
            
            for feature, value in customer_choices.items():
                score *= conditionals[class_name][feature][value]
            print(f"Class: {class_name}, Score: {score}")
            class_scores[class_name] = score  # save the score
            
            if score > best_score:
                best_score = score
                best_class = class_name
        
        total_score = sum(class_scores.values())
        normalized_scores = {}
        if total_score > 0:
            for class_name, score in class_scores.items():
                normalized = score / total_score
                normalized_scores[class_name] = normalized * 100  # percentages
                print(f"Class: {class_name}, Probability: {normalized:.4f}")
        
        return {
            "class": best_class,
            "probability": normalized_scores.get(best_class, 0)
        }
