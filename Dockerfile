# Bachelor thesis FIT VUT
# Author: Martin Kováčik (xkovacm01)
# Date: 20.4.2026
# 
# Docker file for this thesis project.

# Use an official Python runtime as the base image
FROM python:3.13.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code into the container
COPY . .

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
