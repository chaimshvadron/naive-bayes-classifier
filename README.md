# naive-bayes-classifier
 
## Quick Start

### Option 1: Docker Multi-Container (Recommended)
```powershell
# Build and run both services with Docker Compose
docker-compose up --build
```

This will start:
- **Trainer Service**: http://localhost:8001 - trains the model
- **Classifier Service**: http://localhost:8000 - classifies new data

Access the API at: http://localhost:8000/docs

### Option 2: Docker Single Services
```powershell
# Run trainer service
docker build -t naive-bayes-trainer ./trainer-service
docker run -p 8001:8001 naive-bayes-trainer

# Run classifier service (in another terminal)
docker build -t naive-bayes-classifier ./classifier-service
docker run -p 8000:8000 naive-bayes-classifier
```

### Option 3: Local Development
#### Trainer Service
```powershell
cd trainer-service
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

#### Classifier Service
```powershell
cd classifier-service
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Client
Run the interactive client:
```powershell
python client/main.py
```

## Architecture

The system is split into two separate services:

1. **Trainer Service** (Port 8001): responsible for training the model
   - loads data
   - cleans data  
   - trains Naive Bayes model
   - tests model accuracy
   - returns trained model

2. **Classifier Service** (Port 8000): responsible for classifying new data
   - gets trained model from training service
   - classifies new data
   - returns classification results
