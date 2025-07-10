from data_loader import DataLoader
from data_cleaner import DataCleaner
from model_trainer import ModelTrainer
from model_tester import ModelTester
from classifier import Classifier
from user_interface import UserInterface

class NaiveBayesController:
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.trainer = ModelTrainer()
        self.tester = ModelTester()
        self.classifier = Classifier()
        self.ui = UserInterface()
        
        self.data = None
        self.cleaned_data = None
        self.test_data = None
        self.cleaned_test_data = None
        self.model = None
        self.unique_values = None
    
    def load_and_prepare_data(self, file_path='./data/PlayTennis.csv'):
        # Step 1: Load and split data into train and test sets
        self.ui.show_step_message(1, "LOADING & SPLITTING DATA")
        train_df, test_df = self.data_loader.load_and_split_csv(file_path, test_size=0.3, random_state=42)
        self.data = train_df
        self.test_data = test_df
        print(f"Training data shape: {self.data.shape}, Test data shape: {self.test_data.shape}")
        
        # Step 2: Clean both train and test data
        self.ui.show_step_message(2, "CLEANING DATA")
        self.cleaned_data = self.data_cleaner.clean_data(self.data)
        self.cleaned_test_data = self.data_cleaner.clean_data(self.test_data)
        print("Data cleaning completed for train and test sets!")
    
    def train_model(self):
        # Step 3: Train the model
        self.ui.show_step_message(3, "TRAINING MODEL")
        self.unique_values = self.trainer.get_unique_values_dict(self.cleaned_data)
        self.model = self.trainer.train_model(self.cleaned_data)
        print("Model training completed!")

        # Evaluate model reliability on test set
        reliability = self.tester.test_model(self.model, self.cleaned_test_data)
        print(f"Model reliability on test data: {reliability:.2f}%")
        
        # Step 4: Model ready for interactive classification
        self.ui.show_step_message(4, "MODEL READY")
        self.ui.show_feature_options(self.unique_values)
    
    def start_interactive_mode(self):
        """Start interactive mode"""
        print("\n" + "="*50 + "\n")
        self.ui.show_step_message(5, "INTERACTIVE MODE")
        self.ui.interactive_classification(self.classifier, self.model, self.unique_values)
    
    def run_full_pipeline(self):
        """Run the complete process"""
        self.ui.show_welcome_message()
        
        try:
            self.load_and_prepare_data()
            self.train_model()
            self.start_interactive_mode()
            
        except Exception as e:
            print(f"\n rror: {str(e)}")
            print("Process stopped.")
