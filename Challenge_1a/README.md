# Challenge 1A – Adobe India Hackathon 2025

## 🧠 Problem Statement  
The goal is to build a PDF processing solution that extracts a structured outline from documents. This includes:
- Detecting the **document title**
- Extracting **headings and subheadings**
- Outputting the extracted structure in a standardized **JSON format**

The solution must meet performance and resource constraints and be fully containerized via Docker.

---

## 🧩 Approach  
Our approach follows a rule-based, layout-aware strategy to detect document structure. It uses:
- Text extraction from PDFs via `PyMuPDF`
- Font-based and positional features (e.g., size, boldness, alignment)
- Simple heuristics for detecting potential section titles
- Parallel processing for fast multi-page PDF handling
- A fallback mechanism to handle PDFs with inconsistent formatting

This allows us to reliably extract outlines even from unstructured or noisy documents.

---

## 📦 Libraries Used

- `PyMuPDF` (fitz) – for parsing and layout extraction  
- `multiprocessing` – to speed up page-wise operations  
- `jsonschema` – for validating output structure  
- `json`, `re`, `collections`, `logging` – core Python utilities  

---

## 📁 Directory Structure
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # Output JSON files
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Container setup
├── process_pdfs.py      # Main processing script
├── pdf_parser.py        # Supporting logic
└── README.md            # This file
```

---

## 🐳 Dockerized Execution

### 🔧 How to Build
```bash
docker build --platform linux/amd64 -t adobe-outline-extractor:test .
```

### ▶️ How to Run
```bash
docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/input:ro \
  -v $(pwd)/sample_dataset/outputs:/app/output \
  --network none adobe-outline-extractor:test
```

> Ensure the `/app/input` folder contains `.pdf` files  
> The processed output `.json` files will be written to `/app/output`

---

## 📤 Output Format

For every PDF in the input folder, a corresponding `filename.json` is generated in the output folder.  
Each output must conform to the schema defined in:

```
sample_dataset/schema/output_schema.json
```

---

## ✅ Validation Checklist

- [x] All `.pdf` files are parsed  
- [x] JSONs are written for each file  
- [x] Output structure follows provided schema  
- [x] Execution finishes under 10 seconds for 50-page files  
- [x] Memory usage ≤ 16 GB  
- [x] No internet access required at runtime  
- [x] Compatible with AMD64 CPU (8 cores)
- [x] Multilingual support 

---

## 💡 Notes

- This implementation is built entirely on **open-source libraries**  
- Works offline — all dependencies are self-contained  
- No proprietary packages or external APIs used  
