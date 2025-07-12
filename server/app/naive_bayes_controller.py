from data_loader import DataLoader
from data_cleaner import DataCleaner
from model_trainer import ModelTrainer
from model_tester import ModelTester
from classifier import Classifier

class NaiveBayesController:
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
    
    def load_and_prepare_data(self, file_path: str = './data/PlayTennis.csv'):
        train_df, test_df = self.data_loader.load_and_split_csv(file_path, test_size=0.3, random_state=42)
        self.data = train_df
        self.test_data = test_df
        print(f"Training data shape: {self.data.shape}, Test data shape: {self.test_data.shape}")
        
        self.cleaned_data = self.data_cleaner.clean_data(self.data)
        self.cleaned_test_data = self.data_cleaner.clean_data(self.test_data)
        print("Data cleaning completed for train and test sets!")
    
    def train_model(self):
        self.unique_values = self.trainer.get_unique_values_dict(self.cleaned_data)
        self.model = self.trainer.train_model(self.cleaned_data)
        print("Model training completed!")

        reliability = self.tester.test_model(self.model, self.cleaned_test_data)
        print(f"Model reliability on test data: {reliability:.2f}%")
            
