# Intelligent Document Analyst: Methodology Explanation

This document outlines the methodology employed by the Intelligent Document Analyst, a system designed to extract and prioritize relevant information from PDF documents based on a user's specific persona and job-to-be-done. The approach combines robust document preprocessing with advanced natural language processing (NLP) techniques, including semantic search capabilities, to deliver highly targeted insights.

## Core Methodology

The system operates in four primary stages:

### 1. Document Preprocessing and Chunking

The first step involves ingesting PDF documents and transforming their raw text content into manageable and semantically meaningful chunks.
* **Text Extraction:** Each page of the input PDF is read, and its text content is extracted. Empty or malformed pages are skipped.
* **Section Segmentation:** The extracted text is then segmented into logical sections or passages. The primary method for this is splitting text based on multiple consecutive newline characters, which often signifies paragraph breaks or new sections.
* **Length Filtering:** Each identified section is validated against minimum and maximum length constraints to ensure it contains sufficient content to be meaningful while preventing overly large chunks from diluting relevance.
* **Section Title Candidate Generation:** A preliminary section title is generated for each chunk, typically based on its page and section number, or by identifying potential titles from the first line of the section based on capitalization or common numerical/alphanumeric patterns.

### 2. Query Understanding with Dynamic Aspects

Beyond a simple search query, the system aims to understand the nuanced intent behind the user's request, considering both what they are looking for (positive aspects) and what they want to avoid (negative aspects).
* **Main Query Formulation:** The user's provided persona description and job-to-be-done statement are combined to form a comprehensive main query.
* **Semantic Embedding (Primary):** A pre-trained Sentence Transformer model (specifically BGE-base-en-v1.5) is used to generate a high-dimensional vector (embedding) for the main query. Specific prefixes are applied to optimize the embedding quality for search.
* **Dynamic Aspect Extraction:** The combined query text is analyzed using pre-defined patterns and categorical definitions to identify explicit positive or negative semantic concepts. For instance, phrases indicating exclusion (e.g., "not applicable") trigger negative aspects, while terms related to specific domains (e.g., "cost-effectiveness" within a business context) trigger positive aspects.
* **Aspect Embedding:** If identified, these positive and negative aspects are also converted into their own distinct semantic embeddings.
* **Keyword Fallback:** In cases where the advanced semantic model is unavailable (e.g., due to loading errors), the system gracefully degrades to a keyword-based extraction using an NLP library for lemmatization and noun chunking, or basic regex if that library is also unavailable.

### 3. Information Extraction & Ranking

This stage evaluates the relevance of each preprocessed document section against the user's comprehensive query.
* **Section Embedding:** Each document section's text content is also transformed into a semantic embedding using the same Sentence Transformer model, applying a document-specific prefix.
* **Cosine Similarity:** The core of the ranking is based on cosine similarity, a metric that measures the angular distance between two vectors. Higher cosine similarity indicates greater semantic resemblance.
    * **Main Query Similarity:** Each section's embedding is compared to the main query embedding to establish a base relevance score.
    * **Positive Aspect Boosting:** If a positive aspect was identified, sections highly similar to this positive aspect receive a weighted boost to their score, emphasizing desired characteristics.
    * **Negative Aspect Penalization:** Conversely, if a negative aspect was identified, sections highly similar to this negative aspect receive a weighted penalty, diminishing the score of undesirable content.
* **Score Aggregation & Ranking:** The main similarity, positive boost, and negative penalty are combined using adjustable weights to produce a final importance score for each section. Sections are then sorted by this score in descending order, with sections below a certain relevance threshold filtered out.
* **Keyword Fallback Ranking:** If the semantic model is unavailable, a keyword density-based scoring mechanism is used, where keyword counts from positive aspects boost scores and negative aspects reduce them.

### 4. Sub-Section Analysis for Refined Text Extraction

For the top-ranked sections, a more granular analysis is performed to extract the most pertinent sentences.
* **Sentence Tokenization:** Each top section is broken down into individual sentences using an NLP library (or regex as a fallback).
* **Sentence-Level Scoring:** Each sentence within these top sections is scored against the main, positive, and negative query representations using the same semantic or keyword-based methodology as in step 3. This allows for precise identification of relevant phrases within a larger section.
* **Top Sentence Selection:** A predefined number of the highest-scoring sentences are selected from each top section.
* **Output Refinement:** These selected sentences are then concatenated to form a "refined text" summary for that section, providing a concise answer directly relevant to the user's query and aspects.

This multi-stage approach ensures that the system provides not just broad relevant documents, but highly specific and contextually aware information, dynamically adapting to the user's nuanced needs.

---

## Libraries Used

The Intelligent Document Analyst leverages the following key Python libraries:

* **`pypdf`**: For robust PDF text extraction.
* **`spacy`**: An industrial-strength NLP library used for efficient sentence tokenization and keyword extraction (lemmatization, noun chunking).
* **`sentence-transformers`**: For generating high-quality semantic embeddings of queries and document passages, powered by models like BGE-base-en-v1.5.
* **`scikit-learn`**: Specifically for `cosine_similarity` to calculate semantic resemblance between embeddings.
* **`numpy`**: For efficient numerical operations, especially with vector embeddings.
* **`argparse`**: For parsing command-line arguments to specify input and output paths.
* **`os`, `glob`, `time`, `json`, `re`, `heapq`, `datetime`**: Core Python modules for file system operations, string manipulation, timing, and data structures.

---

## 🚀 Execution Methods

The Intelligent Document Analyst can be run in two primary ways: via Docker for a consistent, isolated environment, or directly on your local machine for development and native execution.

### 🐳 Dockerized Execution

The Intelligent Document Analyst is designed to be easily deployed and run within a Docker container, ensuring a consistent environment and simplifying dependency management.

#### ⚙️ Project Structure & Working Directory

* Your terminal's **working directory** when executing `docker build` or `docker run` commands should be `your_project_root/`. This ensures Docker can find the `Dockerfile` and correctly map local paths.

**NOTE**

1. Run the download_mode.py first before running Docker/
2. Put the important pdfs into the input/documents along with the persona and job


#### 🔧 How to Build the Docker Image

1.  **Ensure Docker is Installed:** Make sure you have Docker Desktop (for Windows/macOS) or Docker Engine (for Linux) installed and running on your system.

2.  **Navigate to the Project Root:** Open your terminal or command prompt and navigate to the `your_project_root/` directory.

3.  **Build the Image:** Execute the following command to build the Docker image. This process might take a few minutes as it downloads the base Python image, installs dependencies, and downloads the spaCy model.

    ```bash
    docker build -t intelligent-document-analyst .
    ```
    * `-t intelligent-document-analyst`: Tags the image with a user-friendly name (`intelligent-document-analyst`).
    * `.`: Specifies that the build context (where Docker looks for files like the `Dockerfile`, `requirements.txt`, and `model_weights`) is the current directory (`your_project_root/`). This ensures all necessary files are copied into the Docker image.

    Upon successful completion, you will have a Docker image ready to run your application.


### 💻 Local Execution Guide

This guide explains how to set up and run the Intelligent Document Analyst directly on your local machine, without relying on Docker containers.

#### Prerequisites

1.  **Python 3.9+**: Ensure you have Python 3.9 or a newer version installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
2.  **Internet Connection**: Required for downloading Python packages and the spaCy language model.

#### Project Setup & Working Directory

First, ensure your project directory is structured as follows on your local machine:

* Your terminal's **working directory** when executing commands should be `your_project_root/`.

#### Install Dependencies

1.  **Navigate to Project Root**: Open your terminal or command prompt and change your directory to `your_project_root/`.

2.  **Install Python Packages**: Install all the required Python libraries using `pip`. It's highly recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html) to manage your project's dependencies cleanly.

    ```bash
    # (Optional) Create and activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Download spaCy Model**: After installing the Python packages, you need to download the specific spaCy language model (`en_core_web_sm`) used by the application.

    ```bash
    python3 -m spacy download en_core_web_sm
    ```

#### How to Run the Application

Once the prerequisites are met, and dependencies are installed, you can run the `run.py` script.

1.  **Ensure Input Data is Prepared**: As described in the "Project Setup & Working Directory" section, make sure your `input/` directory contains all necessary PDFs, `job.txt`, `persona.txt`, and `config.json`.

2.  **Execute from Project Root**: From your `your_project_root/` directory in the terminal, run the following command:

    ```bash
    python3 run.py --input_dir ./input --output ./output/results.json
    ```
    * `python3 run.py`: Invokes your Python script.
    * `--input_dir ./input`: This argument tells the script that all its input files (documents, job, persona, config) are located in the local `./input` directory relative to where you run the command.
    * `--output ./output/results.json`: This argument tells the script to save the final JSON output to a file named `results.json` inside the local `output/` directory.

The script will begin processing, and you will see output in your terminal indicating its progress. Upon successful completion, `results.json` will be generated in your `output/` directory.