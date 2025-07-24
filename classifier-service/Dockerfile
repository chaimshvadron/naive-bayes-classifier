# Use official Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app/ ./app/

# Expose the port where server will run
EXPOSE 8000

# Command to start classifier service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
