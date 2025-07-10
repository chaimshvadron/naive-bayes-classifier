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
        self.model = None
        self.unique_values = None
    
    def load_and_prepare_data(self, file_path='./data/PlayTennis.csv'):
        """Load and prepare the data"""
        self.ui.show_step_message(1, "LOADING DATA")
        self.data = self.data_loader.load_data_csv(file_path)
        print(f"Data loaded successfully! Shape: {self.data.shape}")
        
        self.ui.show_step_message(2, "CLEANING DATA")
        self.cleaned_data = self.data_cleaner.clean_data(self.data)
        print("Data cleaning completed!")
    
    def train_model(self):
        """Train the model"""
        self.ui.show_step_message(3, "TRAINING MODEL")
        
        self.unique_values = self.trainer.get_unique_values_dict(self.cleaned_data)
        self.model = self.trainer.train_model(self.cleaned_data)
        
        print("Model training completed!")
        
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
            print(f"\n❌ Error: {str(e)}")
            print("Process stopped.")
    
    # Future functions
    def save_model(self, file_path):
        """Save model to file"""
        # TODO: implement model saving
        pass
    
    def load_model(self, file_path):
        """Load model from file"""
        # TODO: implement model loading
        pass
    
    def evaluate_model(self):
        """Evaluate the model"""
        # TODO: implement model evaluation
        pass
