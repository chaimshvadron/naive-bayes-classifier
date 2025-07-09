class ModelTrainer:
    
    def get_unique_values_dict(self, data):
        features = data.columns[:-1]
        return {feature: data[feature].unique() for feature in features}
    
    def calculate_conditional_counts(self, data):
        target_column = data.columns[-1]
        features = data.columns[:-1]
        
        all_possible_values = self.get_unique_values_dict(data)
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