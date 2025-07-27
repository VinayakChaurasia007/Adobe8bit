FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies & Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Run the processor
CMD ["python", "process_pdfs.py"]
