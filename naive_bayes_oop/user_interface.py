class UserInterface:
    def show_welcome_message(self):
        print("Welcome to the Naive Bayes Classifier!")

    def show_step_message(self, step, message):
        print(f"\nStep {step}: {message}")

    def show_feature_options(self, unique_values):
        print("\nAvailable feature values:")
        for feature, values in unique_values.items():
            print(f" - {feature}: {', '.join(values)}")

    def interactive_classification(self, classifier, model, unique_values):
        print("\nStarting interactive classification:")
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
            result = classifier.classify_customer(customer_choices, model)
            print(f"Predicted class: {result}")
            cont = input("Classify another? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting interactive mode.")
                break
