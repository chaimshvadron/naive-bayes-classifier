class UserInterface:
    def show_welcome_message(self):
        print("Welcome to the Naive Bayes Classifier!")

    def show_step_message(self, step, message):
        print(f"\nStep {step}: {message}")

    def show_feature_options(self, unique_values):
        print("\nAvailable feature values:")
        for feature, values in unique_values.items():
            print(f" - {feature}: {', '.join(values)}")

    def interactive_classification(self, classify_fn, unique_values):
        print("\nStarting interactive classification via API:")
        while True:
            customer_choices = {}
            for feature, values in unique_values.items():
                prompt = f"Enter value for {feature} ({'/'.join(values)}): "
                while True:
                    choice = input(prompt).strip()
                    if choice in values:
                        customer_choices[feature] = choice
                        break
                    else:
                        print(f"Invalid choice. Choose from {values}.")
            # Call the API classify function
            response = classify_fn(customer_choices)
            predicted = response.get("predicted_class")
            reliability = response.get("reliability", 0)
            probability = response.get("probability", 0)
            
            print(f"Predicted class: {predicted}, probability: {probability:.2f}%, reliability: {reliability:.2f}%")
            cont = input("Classify another? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting interactive mode.")
                break
