import pandas as pd

def load_data_csv(file_path):
    data = pd.read_csv(file_path)
    for col in data.columns:
        if data[col].dtype == 'bool':
            data[col] = data[col].astype(str)
    return data

def calculate_conditional_counts(data):
    target_column = data.columns[-1]
    features = data.columns[:-1]
    classes = data[target_column].unique()

    all_possible_values = {
        feature: data[feature].unique()
        for feature in features
    }

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

if __name__ == "__main__":
    data = load_data_csv('data/PlayTennis.csv')
    conditional_counts = calculate_conditional_counts(data)
    print(conditional_counts)
