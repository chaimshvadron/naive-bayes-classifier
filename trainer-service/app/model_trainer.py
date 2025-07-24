class ModelTrainer:
    
    def get_unique_values_dict(self, data, feature_columns):
        return {feature: list(data[feature].unique()) for feature in feature_columns}
    
    def calculate_prior_probabilities(self, data, target_column):
        total_count = len(data)
        class_counts = data[target_column].value_counts()
        return {class_name: count / total_count for class_name, count in class_counts.items()}
    
    def calculate_conditional_counts(self, data, feature_columns, target_column):
        all_possible_values = self.get_unique_values_dict(data, feature_columns)
        result = {}

        grouped = data.groupby(target_column)
        for class_name, group in grouped:
            result[class_name] = {}
            class_count = len(group)
            for feature in feature_columns:
                result[class_name][feature] = {}
                value_counts = group[feature].value_counts()
                num_possible_values = len(all_possible_values[feature])
                for value in all_possible_values[feature]:
                    count = value_counts.get(value, 0)
                    result[class_name][feature][value] = (count + 1) / (class_count + num_possible_values)

        return result
    
    def train_model(self, data, feature_columns, target_column):
        return {
            'conditionals': self.calculate_conditional_counts(data, feature_columns, target_column),
            'priors': self.calculate_prior_probabilities(data, target_column)
        }
