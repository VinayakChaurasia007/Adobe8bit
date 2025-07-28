# Use a slim Buster-based Python image
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies for spaCy and pypdf
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model
RUN python -m spacy download en_core_web_sm

# Create directories for models and documents
RUN mkdir -p /app/model_weights/bge-base-en-v1.5 /app/documents

# Copy your application code into the container
COPY . .

# Command to run your application (this can be overridden by docker run)
ENTRYPOINT ["python", "run.py"]