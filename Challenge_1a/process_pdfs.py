import os
import json
from pathlib import Path
from pdf_parser import PDFOutlineExtractor

def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFOutlineExtractor()
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            # Extract outline
            result = extractor.extract_outline(str(pdf_file))
            
            # Write result to JSON
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Processed {pdf_file.name} -> {output_file.name}")
        
        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    print("Starting PDF processing...")
    process_pdfs()
    print("Completed all PDF processing.")
