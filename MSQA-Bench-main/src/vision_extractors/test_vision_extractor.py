#!/usr/bin/env python3
"""Smoke tests for the PDF Vision Extractor.

These tests skip cleanly when the optional Ollama service or sample PDF is not
available, so `python -m pytest` remains useful in a fresh code checkout.
"""

import os
import sys
from pathlib import Path

import pytest
import requests

try:
    import fitz  # noqa: F401
    HAVE_PYMUPDF = True
except ImportError:
    HAVE_PYMUPDF = False

PDFVisionExtractor = None
if HAVE_PYMUPDF:
    try:
        from .vision_extractor import PDFVisionExtractor
    except ImportError:  # pragma: no cover - direct script execution fallback
        from vision_extractor import PDFVisionExtractor


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
SAMPLE_PDF = Path(os.environ.get("MSQA_TEST_PDF", "input/dd03a3b2551ce2921e8ae7fe7c9dc0f145767277.pdf"))


def ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


@pytest.mark.skipif(not HAVE_PYMUPDF, reason="PyMuPDF is not installed")
@pytest.mark.skipif(not ollama_available(), reason="Ollama is not running")
def test_single_pdf():
    """Test extraction from a single PDF file."""
    print("=== Testing Single PDF Extraction ===")
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found: {SAMPLE_PDF}")
    
    try:
        # Initialize extractor
        extractor = PDFVisionExtractor(
            model="llava:7b",  # Use smaller model for testing
            output_dir="test_output",
            log_level="INFO",
            ollama_url=OLLAMA_URL,
        )
        
        # Extract first 2 pages only for testing
        success = extractor.extract_from_pdf(
            pdf_path=str(SAMPLE_PDF),
            output_filename="test_extraction.txt",
            start_page=1,
            end_page=2
        )
        
        if success:
            print("Single PDF test passed.")
            
            # Check if output file was created
            output_file = Path("test_output/test_extraction.txt")
            if output_file.exists():
                print(f"Output file created: {output_file}")
                print(f"  File size: {output_file.stat().st_size} bytes")
                
                # Show first few lines
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')[:5]
                    print("  First few lines:")
                    for i, line in enumerate(lines, 1):
                        if line.strip():
                            print(f"    {i}: {line[:80]}...")
            return True
        else:
            pytest.fail("Single PDF extraction returned False")
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")


@pytest.mark.skipif(not HAVE_PYMUPDF, reason="PyMuPDF is not installed")
@pytest.mark.skipif(not ollama_available(), reason="Ollama is not running")
def test_connection():
    """Test Ollama connection."""
    print("=== Testing Ollama Connection ===")
    
    try:
        extractor = PDFVisionExtractor(
            model="llava:7b",
            output_dir="test_output",
            log_level="INFO",
            ollama_url=OLLAMA_URL,
        )
        print("Ollama connection test passed.")
        return True
        
    except SystemExit:
        pytest.fail("Ollama connection failed; ensure `ollama serve` and `ollama pull llava:7b` have run")
    except Exception as e:
        pytest.fail(f"Connection test failed with error: {e}")


def main():
    """Run through pytest when invoked directly."""
    return pytest.main([__file__])


if __name__ == "__main__":
    sys.exit(main())
