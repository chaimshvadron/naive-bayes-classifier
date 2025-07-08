import pandas as pd

def load_data_csv(file_path):
    return pd.read_csv(file_path)

def calculate_conditional_counts(data):
    target_column = data.columns[-1]
    features = data.columns[:-1]
    classes = data[target_column].value_counts()   
    result = {}
    for class_name, class_count in classes.items():
        class_subset = data[data[target_column] == class_name]     
        if class_name not in result:
            result[class_name] = {}
        for feature in features:
            value_counts = class_subset[feature].value_counts()
            for value, count in value_counts.items():
                if feature not in result[class_name]:
                    result[class_name][feature] = {}
                result[class_name][feature][value] = count / class_count
    return result


if __name__ == "__main__":
    data = load_data_csv('data/PlayTennis.csv')
    conditional_counts = calculate_conditional_counts(data)
    print(conditional_counts)
