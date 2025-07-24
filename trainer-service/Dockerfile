# Use official Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files and data files
COPY app/ ./app/
COPY data/ ./data/

# Expose the port where server will run
EXPOSE 8001

# Command to start training service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
