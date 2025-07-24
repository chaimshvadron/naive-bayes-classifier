import requests
import json

class ModelClient:
    def __init__(self, trainer_service_url="http://trainer-service:8001"):
        self.trainer_service_url = trainer_service_url
        self.model_data = None
        
    def get_model_from_trainer(self):
        """Get trained model from training service"""
        try:
            print(f"Asking for model from: {self.trainer_service_url}/model")
            response = requests.get(f"{self.trainer_service_url}/model")
            
            if response.status_code == 200:
                self.model_data = response.json()
                print("Got trained model successfully!")
                print(f"Model accuracy: {self.model_data.get('reliability', 'unknown')}%")
                return True
            else:
                print(f"Error getting model: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error connecting to training service: {e}")
            return False
    
    def get_features_from_trainer(self):
        """Get features list from training service"""
        try:
            response = requests.get(f"{self.trainer_service_url}/features")
            
            if response.status_code == 200:
                features_data = response.json()
                print("Got features list successfully!")
                return features_data
            else:
                print(f"Error getting features: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error connecting to training service: {e}")
            return None
