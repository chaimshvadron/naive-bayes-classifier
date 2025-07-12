# api_client.py

import requests

SERVER_URL = "http://localhost:8000"

def classify(features: dict) -> dict:
    """
    Send features to the FastAPI server for classification
    """
    url = f"{SERVER_URL}/classify"
    response = requests.post(url, json={"features": features})
    response.raise_for_status()
    return response.json()

def get_features() -> dict:
    """
    Retrieve available feature values from the server
    """
    url = f"{SERVER_URL}/features"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
