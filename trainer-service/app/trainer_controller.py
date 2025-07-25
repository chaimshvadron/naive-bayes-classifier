from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .model_trainer import ModelTrainer
from .model_tester import ModelTester
from .classifier import Classifier
import json

class TrainerController:
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.trainer = ModelTrainer()
        self.tester = ModelTester()
        self.classifier = Classifier()
        
        self.data = None
        self.cleaned_data = None
        self.test_data = None
        self.cleaned_test_data = None
        self.model = None
        self.unique_values = None
        self.feature_columns = None
        self.target_column = None
        self.reliability = None
        
        # מעקב אחר מצב האימון
        self.training_status = "not_started"  # not_started, loading_data, training, testing, completed
    
    def load_and_prepare_data(self, file_path: str = './data/phishing.csv'):
        self.training_status = "loading_data"
        train_df, test_df = self.data_loader.load_and_split_csv(file_path, test_size=0.3, random_state=42)
        self.data = train_df
        self.test_data = test_df
        print(f"Training data shape: {self.data.shape}, Test data shape: {self.test_data.shape}")
        
        self.cleaned_data = self.data_cleaner.clean_data(self.data)
        self.cleaned_test_data = self.data_cleaner.clean_data(self.test_data)
        self.feature_columns = self.cleaned_data.columns[:-1]
        self.target_column = self.cleaned_data.columns[-1]
        print("Data cleaning completed for train and test sets!")
    
    def train_model(self):
        self.training_status = "training"
        self.unique_values = self.trainer.get_unique_values_dict(self.cleaned_data, self.feature_columns)
        self.model = self.trainer.train_model(self.cleaned_data, self.feature_columns, self.target_column)
        print("Model training completed!")

        self.training_status = "testing"
        reliability = self.tester.test_model(
            self.model,
            self.cleaned_test_data,
            self.feature_columns,
            self.target_column
        )
        self.reliability = reliability
        print(f"Model reliability on test data: {reliability:.2f}%")
        
        self.training_status = "completed"
    
    def get_trained_model(self):
        """Return the trained model with all needed info for classification"""
        model_data = {
            'model': self.model,
            'feature_columns': list(self.feature_columns),
            'target_column': self.target_column,
            'unique_values': self.unique_values,
            'reliability': self.reliability
        }
        return model_data
