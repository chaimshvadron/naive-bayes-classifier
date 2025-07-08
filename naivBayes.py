import pandas as pd

def load_data_csv(file_path):
    return pd.read_csv(file_path)

def calculate_conditional_counts(data):
    target_column = data.columns[-1]
    features = data.columns[:-1]
    classes = data[target_column].unique()
    
    result = {}

    for class_value in classes:
        class_subset = data[data[target_column] == class_value]
        
        if class_value not in result:
            result[class_value] = {}

        for feature in features:
            value_counts = class_subset[feature].value_counts()
            result[class_value][feature] = value_counts.to_dict()
    
    return result


if __name__ == "__main__":
    data = load_data_csv('data/PlayTennis.csv')
    conditional_counts = calculate_conditional_counts(data)
    print(conditional_counts)
