    # Use a slim Bullseye-based Python image
    FROM python:3.9-slim-bullseye

    # Set the working directory inside the container
    WORKDIR /app

    # Install system-level dependencies required for building Python packages
    # build-essential: provides compilers (gcc, g++) and make
    # python3-dev: provides Python header files needed for compiling C extensions
    # libatlas-base-dev: often needed for numpy and scipy (linear algebra libraries)
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libatlas-base-dev && \
        rm -rf /var/lib/apt/lists/*

    # Copy the requirements file into the container
    COPY requirements.txt .

    # Install Python dependencies
    # --no-cache-dir: Prevents pip from storing cache, reducing image size
    RUN pip install --no-cache-dir -r requirements.txt

    # Download the spaCy language model
    # This is crucial for the script's functionality
    RUN python -m spacy download en_core_web_sm

    # Copy the model download script and run it to get the sentence transformer model
    # This assumes download_model.py populates the model_weights directory
    COPY download_model.py .
    RUN python3 download_model.py

    # Copy the rest of your application files into the container
    # This includes:
    # - run.py (your main script)
    # - challenge1b_input.json (the input JSON file)
    # - input/ (directory containing config.json)
    # - PDFs/ (directory containing your PDF documents)
    # - The now populated model_weights/ directory
    COPY . .

    # Define the command to run your main script
    # We will output to a specific 'output' directory inside /app
    CMD ["python3", "run.py", "--input_file", "input/challenge1b_input.json", "--pdf_dir", "PDFs/", "--output", "/app/output/challenge1b_output.json"]
    