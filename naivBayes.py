import pandas as pd

def load_data_csv(file_path):
    data = pd.read_csv(file_path)
    for col in data.columns:
        if data[col].dtype == 'bool':
            data[col] = data[col].astype(str)
    return data

def get_unique_values_dict(data):
    features = data.columns[:-1]
    return {feature: data[feature].unique() for feature in features}

def calculate_conditional_counts(data):
    target_column = data.columns[-1]
    features = data.columns[:-1]
    
    all_possible_values = get_unique_values_dict(data)
    result = {}

    grouped = data.groupby(target_column)
    # Group the data by the target column
    for class_name, group in grouped:
        result[class_name] = {}
        class_count = len(group)
        # Count the number of instances for each class
        for feature in features:
            result[class_name][feature] = {}
            value_counts = group[feature].value_counts()
            num_possible_values = len(all_possible_values[feature])
            # Calculate the conditional probabilities
            for value in all_possible_values[feature]:
                count = value_counts.get(value, 0)
                result[class_name][feature][value] = (count + 1) / (class_count + num_possible_values)

    return result


def classify_customer(customer_choices, model):

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
if __name__ == "__main__":
    data = load_data_csv('data/PlayTennis.csv')
    
    # 1. Get unique values for each feature
    unique_values = get_unique_values_dict(data)
    print("Options for each feature:")
    for feature, values in unique_values.items():
        print(f"{feature}: {values}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Create the model
    model = calculate_conditional_counts(data)
    print("Model created successfully!")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Example: classify a customer
    customer_choices = {
        'Outlook': 'Sunny',
        'Temperature': 'Hot', 
        'Humidity': 'High',
        'Windy': 'False'
    }
    
    print(f"Customer choices: {customer_choices}")
    
    best_class = classify_customer(customer_choices, model)
    
    print(f"Recommendation for the customer: {best_class}")
    
    # Another example
    print("\n" + "-"*30 + "\n")
    
    customer_choices2 = {
        'Outlook': 'Overcast',
        'Temperature': 'Mild',
        'Humidity': 'Normal', 
        'Windy': 'False'
    }
    
    print(f"Another customer's choices: {customer_choices2}")
    best_class2 = classify_customer(customer_choices2, model)
    print(f"Recommendation for the second customer: {best_class2}")
