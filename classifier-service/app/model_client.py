import requests
import json
import time

class ModelClient:
    def __init__(self, trainer_service_url="http://trainer-service:8001"):
        self.trainer_service_url = trainer_service_url
        self.model_data = None
        
    def get_model_from_trainer(self, max_retries=5, retry_delay=2):
        """Get trained model from training service with retry mechanism"""
        for attempt in range(max_retries):
            try:
                print(f"Asking for model from: {self.trainer_service_url}/model (attempt {attempt + 1}/{max_retries})")
                response = requests.get(f"{self.trainer_service_url}/model", timeout=10)
                
                if response.status_code == 200:
                    self.model_data = response.json()
                    print("Got trained model successfully!")
                    print(f"Model accuracy: {self.model_data.get('reliability', 'unknown')}%")
                    return True
                else:
                    print(f"Error getting model: {response.status_code}")
                    
            except Exception as e:
                print(f"Error connecting to training service (attempt {attempt + 1}): {e}")
                
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                
        print("Failed to get model after all attempts")
        return False
    
    def get_features_from_trainer(self):
        """Get features list from training service"""
        try:
            response = requests.get(f"{self.trainer_service_url}/features", timeout=10)
            
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
