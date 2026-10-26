
# Use Python as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency list
COPY requirements.txt .

# Install the required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and trained model
COPY app.py .
COPY models/ models/

# Expose the API port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
