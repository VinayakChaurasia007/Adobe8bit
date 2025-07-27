"""
Tests for PDF parsing functionality
"""
import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from pdf_parser import PDFOutlineExtractor

class TestPDFOutlineExtractor:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = PDFOutlineExtractor()
    
    def test_heading_patterns(self):
        """Test heading pattern matching"""
        patterns = self.extractor.heading_patterns
        
        # Test numbered headings
        assert any(re.match(pattern, "1. Introduction") for pattern in patterns)
        assert any(re.match(pattern, "2.1 Overview") for pattern in patterns)
        
        # Test all caps
        assert any(re.match(pattern, "CHAPTER ONE") for pattern in patterns)
        
        # Test chapter headings
        assert any(re.match(pattern, "Chapter 1: Getting Started") for pattern in patterns)
    
    @patch('fitz.open')
    def test_extract_title_from_metadata(self, mock_fitz_open):
        """Test title extraction from PDF metadata"""
        # Mock PDF document
        mock_doc = Mock()
        mock_doc.metadata = {"title": "Test Document Title"}
        mock_fitz_open.return_value = mock_doc
        
        title = self.extractor._extract_title(mock_doc)
        assert title == "Test Document Title"
    
    @patch('fitz.open')
    def test_extract_title_from_first_page(self, mock_fitz_open):
        """Test title extraction from first page when metadata is empty"""
        # Mock PDF document with no metadata title
        mock_doc = Mock()
        mock_doc.metadata = {"title": ""}
        mock_doc.__len__ = Mock(return_value=1)
        
        # Mock page with text blocks
        mock_page = Mock()
        mock_page.get_text.return_value = {
            "blocks": [{
                "lines": [{
                    "spans": [{
                        "text": "Large Title Text",
                        "size": 24.0
                    }]
                }]
            }]
        }
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        title = self.extractor._extract_title(mock_doc)
        assert title == "Large Title Text"
    
    def test_detect_headings_font_size(self):
        """Test heading detection based on font size"""
        # Mock text blocks with different font sizes
        blocks = [[
            {
                "text": "Main Title",
                "font_size": 18.0,
                "is_bold": True,
                "page": 1,
                "y_position": 50
            },
            {
                "text": "Regular text content here",
                "font_size": 12.0,
                "is_bold": False,
                "page": 1,
                "y_position": 100
            },
            {
                "text": "Subtitle",
                "font_size": 14.0,
                "is_bold": True,
                "page": 1,
                "y_position": 150
            }
        ]]
        
        headings = self.extractor._detect_headings(blocks)
        
        # Should detect the larger font sizes as headings
        assert len(headings) >= 1
        assert any(h["text"] == "Main Title" for h in headings)
    
    def test_detect_headings_patterns(self):
        """Test heading detection using regex patterns"""
        blocks = [[
            {
                "text": "1. Introduction",
                "font_size": 12.0,
                "is_bold": False,
                "page": 1,
                "y_position": 50
            },
            {
                "text": "CHAPTER TWO",
                "font_size": 12.0,
                "is_bold": False,
                "page": 1,
                "y_position": 100
            }
        ]]
        
        headings = self.extractor._detect_headings(blocks)
        
        # Should detect pattern-based headings
        assert len(headings) >= 2
        heading_texts = [h["text"] for h in headings]
        assert "1. Introduction" in heading_texts
        assert "CHAPTER TWO" in heading_texts

if __name__ == "__main__":
    pytest.main([__file__])
