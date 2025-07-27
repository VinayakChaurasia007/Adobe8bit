# Challenge 1A – Adobe India Hackathon 2025

## 🧠 Problem Statement
The goal is to extract a structured outline from PDF documents. This includes:
- Detecting the **document title**
- Extracting **headings and subheadings** across pages
- Formatting the results into a **JSON file** as per the given schema

## 🧩 Approach

The solution uses a rule-based and font-feature-based scoring system to identify section headings from PDF files. Key aspects:
- Uses **PyMuPDF** to extract all text blocks from each page
- Scores potential headings using:
  - Font size ratios
  - Boldness
  - Capitalization
  - Presence of stop words
  - Pattern matching for headings (e.g. `Chapter 1`, `1.1`, `ALL CAPS`, `Title Case`)
- Uses **multiprocessing** for speed on multi-page documents
- Extracts the document title from metadata or by font size on the first page

## 📦 Libraries Used

- `PyMuPDF` (fitz) – For PDF parsing
- `multiprocessing` – For parallelized processing
- `jsonschema` – For output schema validation
- `json`, `re`, `logging` – Core Python modules

## 🐳 Dockerized Execution

### 🔧 How to Build

```bash
docker build --platform linux/amd64 -t adobe-outline-extractor:test .
