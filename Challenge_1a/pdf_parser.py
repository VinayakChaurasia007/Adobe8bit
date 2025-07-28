"""
PDF parsing and outline extraction logic
"""
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial
import logging
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # make language detection deterministic
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    """Extract structured outlines from PDF documents"""
    
    def __init__(self):
        self.heading_patterns = [
            r'^(\d+\.?\s+.+)',  # 1. Title or 1 Title
            r'^([A-Z][A-Z\s]{2,})',  # ALL CAPS headings
            r'^(Chapter\s+\d+.*)',  # Chapter headings
            r'^(Section\s+\d+.*)',  # Section headings
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\s$',  # Title Case
        ]
    
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract outline from PDF using multiprocessing
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with title and headings
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Extract document title
            title = self._extract_title(doc)
            
            # Process pages in parallel (limit to 8 workers)
            num_workers = min(8, mp.cpu_count(), len(doc))
            
            if len(doc) == 1:
                # Single page - process directly
                page_data = self._process_page(doc, 0)
                all_blocks = [page_data] if page_data else []
            else:
                # Multiple pages - use multiprocessing
                with mp.Pool(processes=num_workers) as pool:
                    process_func = partial(self._process_page_wrapper, pdf_path)
                    page_results = pool.map(process_func, range(len(doc)))
                    all_blocks = [block for block in page_results if block]
            
            doc.close()
            
            # Detect headings from all blocks
            headings = self._detect_headings(all_blocks)
            
            return {
                "title": title,
                "outline": headings
            }
            
        except Exception as e:
            logger.error(f"Error extracting outline: {str(e)}")
            raise
    
    def _process_page_wrapper(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Wrapper for multiprocessing - opens document per process"""
        doc = fitz.open(pdf_path)
        result = self._process_page(doc, page_num)
        doc.close()
        return result
    
    def _process_page(self, doc: fitz.Document, page_num: int) -> List[Dict]:
        """Process a single page and extract text blocks with formatting"""
        try:
            # page = doc[page_num]
            # blocks = page.get_text("dict")["blocks"]
            page = doc[page_num]
            text = page.get_text("text", flags=1)  # Extract using better encoding
            
            # You can still get blocks if needed like this:
            blocks = page.get_text("dict", flags=1)["blocks"]

            text_blocks = []
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        # Extract formatting information
                        font_size = span["size"]
                        font_flags = span["flags"]  # Bold, italic flags
                        bbox = span["bbox"]
                        
                        text_blocks.append({
                            "text": text,
                            "font_size": font_size,
                            "is_bold": bool(font_flags & 2**4),  # Bold flag
                            "is_italic": bool(font_flags & 2**1),  # Italic flag
                            "bbox": bbox,
                            "page": page_num + 1,
                            "y_position": bbox[1]  # Top y coordinate
                        })
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return []
    
    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract document title from metadata or first page"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get("title"):
            return metadata["title"].strip()
        
        # Try first page - look for largest text
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            
            largest_text = ""
            largest_size = 0
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        size = span["size"]
                        
                        if size > largest_size and len(text) > 5:
                            largest_size = size
                            largest_text = text
            
            if largest_text:
                return largest_text
        
        return "Untitled Document"
    
    def _detect_headings(self, all_blocks: List[List[Dict]]) -> List[Dict[str, Any]]:
         """Detect headings using a weighted feature‑based scoring system."""
         if not all_blocks:
             return []
 
         # Flatten
         flat = [span for page in all_blocks for span in page]
 
         if not flat:
             return []
         # Detect language from a sample of text
         sample_text = " ".join([b["text"] for b in flat[:50]])  # take first 50 spans
         try:
             lang = detect(sample_text)
         except Exception:
             lang = "en"  # default to English if detection fails

         # Estimate median body font size for ratio
         sizes = [b["font_size"] for b in flat]
         median_size = sorted(sizes)[len(sizes)//2]
         
         stopwords_all = {
            "en": {"the","and","of","in","to","with","a","for","on","is","this","that","by","an","be","are","as","at","from","it"},
            "fr": {"le","la","et","de","des","en","pour","sur","avec","que"},
            "es": {"el","la","y","de","en","para","con","que"},
            "de": {"der","die","und","von","zu","mit","das","ist","für"},
            "pt": {"o","a","e","de","em","para","com","que"},
            "ru": {"и","в","не","на","с","что","по","для","как","это"},
            "hi": {"और","के","का","में","को","यह","से","कि","है"},
            "bn": {"এবং","এর","এ","কে","কে","এই","থেকে","যে","হয়"},
            "zh": set(),  # Chinese headings won’t use stopwords
            "ja": set(),
         }

         # per-language heading patterns
         lang_headings = {
             "en":    [r'^\d+(\.\d+)*\s+', r'^(Chapter|Section)\s+\d+'],
             "fr":    [r'^Chapitre\s+\d+', r'^Section\s+\d+'],
             "es":    [r'^Cap[ií]tulo\s+\d+', r'^Secci[oó]n\s+\d+'],
             "de":    [r'^Kapitel\s+\d+', r'^Abschnitt\s+\d+'],
             "pt":    [r'^Cap[ií]tulo\s+\d+', r'^Se[cç][ãa]o\s+\d+'],
             "ru":    [r'^Глава\s+\d+', r'^Раздел\s+\d+'],
             "hi":    [r'^अध्याय\s+\d+', r'^अनुभाग\s+\d+'],
             "bn":    [r'^(অধ্যায়|অধ্যায়)\s+\d+', r'^অনুচ্ছেদ\s+\d+'],
             "zh": [
                       r'^第[一二三四五六七八九十百千万0-9]+章',
                       r'^第[一二三四五六七八九十百千万0-9]+节',
                       r'^[\u4e00-\u9fff]{2,10}$'  # 2–10 Chinese characters line (common for section titles)
                   ],

             "ja":    [r'^第\d+章',    r'^第\d+節'], 
         }
         # Helpers
         sw_list = stopwords_all.get(lang, stopwords_all["en"])

 
         headings = []
         seen = set()
 
         for blk in flat:
             text = blk["text"]
             if lang == "zh":
               wc = len(text)  # Approximate: each character is a word
             else:
                 words = text.split()
                 wc = len(words)

             if wc == 0:
                 continue

             # Skip if the entire line is just a number or common math-like pattern
             if text.strip().isdigit():
                 continue
             if re.match(r'^\d+(\.\d+)*$', text.strip()):
                 continue

             font_size = blk["font_size"]
             is_bold = blk["is_bold"]
             y_norm = blk["y_position"] / blk["bbox"][3]  # y / page_height approx
             lvl = None
             score = 0
             # Penalize if text has no alphabetic characters (likely a number, symbol, or formula)
             if not any(c.isalpha() for c in text):
                 score -= 10
 
             # 1) Font‑size ratio
             ratio = font_size / median_size
             if ratio > 1.5:
                 score += 3
             elif ratio > 1.2:
                 score += 1
 
             # 2) Bold
             if is_bold:
                 score += 1
             else:
                 score-=1
             # 3) Numbered / chapter / section
             if re.match(r'^\d+(\.\d+)*\s+', text):
                 score += 1
                 lvl = "H2"
             if re.match(r'^(Chapter|Section)\s+\d+', text, re.IGNORECASE):
                 score += 3
                 lvl = "H1"
             # Multilingual heading patterns
             for pattern in lang_headings.get(lang, lang_headings["en"]):
                 if re.match(pattern, text, re.IGNORECASE):
                     score += 3
                     lvl = lvl or "H1"
                     break
             # 4) ALL CAPS or Title Case
             if text.isupper() and wc <= 6:
                 score += 1
                 lvl = lvl or "H2"
             elif text.istitle() and wc <= 5:
                 score += 1
                 lvl = lvl or "H3"
             
             # 5) Stop‑word ratio
            #  sw_count = sum(1 for w in words if w.lower() in stopwords)
            #  if sw_count / wc < 0.3:
            #      score += 1
             sw_count = sum(1 for w in words if w.lower() in sw_list)
             if wc and sw_count / wc < 0.3:
                score += 1
 
             # 6) Word count between 2 and 10
             if 2 <= wc <= 10:
                 score += 1
             else:
                 score -= 1
 
             # 7) No trailing punctuation
             if not text.endswith(('.',':',',','?','+','-','*','/','%','<','>','%','[',']','(',')','=','@','!','#','^','_','≤','≥')):
                 score += 1
             else:
                 score-=2
             
             if text.startswith(('.',':',',','?','+','-','*','/','%','<','>','%','[',']','(',')','=','@','!','#','^','_','≤','≥')):
                 score-=2
             # 8) Page position bonus/penalty
            #  if y_norm < 0.2:
            #      score += 1
            #  elif y_norm > 0.9:
            #      score -= 1
             # 9) Penalize if the first character is lowercase (likely a sentence)
             if text and text[0].islower():
                 score -= 2
             # 10) Penalize if followed by text on the same line (likely part of paragraph)
            #  next_same_line = any(
            #      b["page"] == blk["page"] and
            #      abs(b["y_position"] - blk["y_position"]) < 1 and
            #      b["bbox"][0] > blk["bbox"][2]  # horizontally to the right
            #      for b in flat if b != blk
            #  )
            #  if next_same_line:
            #      score -= 2
             
             # Assign default level by font ratio if still none
             if not lvl:
                 if ratio > 1.8:
                     lvl = "H1"
                 elif ratio > 1.4:
                     lvl = "H2"
                 else:
                     lvl = "H3"
 
             # Final threshold
             key = (text.lower(), blk["page"])
             if score >= 4 and key not in seen:
                 seen.add(key)
                 headings.append({
                     "level": lvl,
                     "text": text,
                     "page": blk["page"]
                 })
 
         # Sort by page then by vertical position
         headings.sort(key=lambda h: (h["page"], next(
             blk["y_position"] for blk in flat 
             if blk["text"] == h["text"] and blk["page"] == h["page"]
         )))
 
         return headings
