# naive-bayes-classifier
 
## Quick Start

### Option 1: Docker (Recommended)
```powershell
# Build and run with Docker
docker build -t naive-bayes-server .
docker run -p 8000:8000 naive-bayes-server
```
Access the API at: http://localhost:8000/docs

### Option 2: Local Development
#### Server
1. Install requirements and start development server:
   ```powershell
   pip install -r requirements.txt
   fastapi dev .\server\app\main.py
   ```

#### Client
Run the interactive client:
```powershell
python client/main.py
```
