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
    classes = data[target_column].value_counts()

    all_possible_values = {
        feature: data[feature].unique()
        for feature in features
    }

    result = {}

    for class_name, class_count in classes.items():
        class_subset = data[data[target_column] == class_name]
        result[class_name] = {}

        for feature in features:
            result[class_name][feature] = {}

            values = all_possible_values[feature]
            num_possible_values = len(values)

            for value in values:
                count = len(class_subset[class_subset[feature] == value])
                result[class_name][feature][value] = (count + 1) / (class_count + num_possible_values)

    return result



if __name__ == "__main__":
    data = load_data_csv('data/PlayTennis.csv')
    conditional_counts = calculate_conditional_counts(data)
    print(conditional_counts)
