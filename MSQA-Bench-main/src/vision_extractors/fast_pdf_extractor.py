#!/usr/bin/env python3
"""
Fast PDF Text Extractor for Large-Scale Processing

This script provides efficient text extraction from thousands of PDFs using
direct text extraction methods, with OCR fallback only when necessary.

Key improvements over agentic_vision_extractor.py:
- 10-100x faster using direct text extraction
- Parallel processing support
- Minimal memory footprint
- Simple, maintainable code
- Proper error handling for bulk processing
"""

import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, asdict
import time

import fitz  # PyMuPDF
try:
    import pytesseract
    from PIL import Image
    from io import BytesIO
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class ExtractionResult:
    """Result of a PDF extraction."""
    success: bool
    pdf_path: str
    output_path: str
    pages_extracted: int
    total_pages: int
    extraction_time: float
    method: str  # 'direct', 'ocr', 'mixed'
    error: Optional[str] = None


class FastPDFExtractor:
    """Fast, scalable PDF text extractor for bulk processing."""

    def __init__(
        self,
        output_dir: str = "extracted_text",
        log_file: str = "logs/extraction.log",
        progress_file: str = "logs/progress.json",
        min_text_length: int = 10,
        ocr_threshold: float = 0.1,  # Min text/page ratio to avoid OCR
        max_workers: Optional[int] = None,
        use_ocr_fallback: bool = False,
        clean_text: bool = True,
        save_as_markdown: bool = False
    ):
        """
        Initialize the Fast PDF Extractor.

        Args:
            output_dir: Directory for extracted text files
            log_file: Path to log file
            progress_file: Path to progress tracking file
            min_text_length: Minimum characters per page to consider valid
            ocr_threshold: Ratio of text to page size below which to try OCR
            max_workers: Number of parallel workers (None = CPU count)
            use_ocr_fallback: Whether to use OCR for pages with no text
            clean_text: Whether to clean and normalize extracted text
            save_as_markdown: Save output as .md instead of .txt
        """
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        # Processing settings
        self.min_text_length = min_text_length
        self.ocr_threshold = ocr_threshold
        self.max_workers = max_workers or mp.cpu_count()
        self.use_ocr_fallback = use_ocr_fallback and OCR_AVAILABLE
        self.clean_text = clean_text
        self.save_as_markdown = save_as_markdown

        # Setup logging
        self._setup_logging()

        # Load progress
        self.progress = self._load_progress()

        if self.use_ocr_fallback and not OCR_AVAILABLE:
            self.logger.warning(
                "OCR fallback requested but pytesseract not available. "
                "Install with: pip install pytesseract pillow"
            )
            self.use_ocr_fallback = False

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_progress(self) -> Dict[str, Any]:
        """Load progress tracking data."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load progress: {e}")
        return {"completed": [], "failed": [], "stats": {}}

    def _save_progress(self) -> None:
        """Save progress tracking data."""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for tracking."""
        return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip empty lines and very short lines (likely noise)
            if len(line) > 3:
                # Normalize internal whitespace
                line = ' '.join(line.split())
                cleaned_lines.append(line)

        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)

        # Remove common extraction artifacts
        artifacts = [
            '©', '®', '™',  # Copyright symbols
            '\x00', '\x01', '\x02',  # Control characters
        ]
        for artifact in artifacts:
            cleaned_text = cleaned_text.replace(artifact, '')

        return cleaned_text

    def _extract_page_text(self, page: fitz.Page) -> Tuple[str, str]:
        """
        Extract text from a single page.

        Returns:
            (text, method) where method is 'direct' or 'ocr'
        """
        # Try direct text extraction first
        text = page.get_text()

        if self.clean_text:
            text = self._clean_text(text)

        # Check if we got meaningful text
        if len(text) > self.min_text_length:
            return text, "direct"

        # Try OCR if enabled and text is insufficient
        if self.use_ocr_fallback:
            try:
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))

                # Run OCR
                ocr_text = pytesseract.image_to_string(img)

                if self.clean_text:
                    ocr_text = self._clean_text(ocr_text)

                if len(ocr_text) > len(text):
                    return ocr_text, "ocr"

            except Exception as e:
                self.logger.debug(f"OCR failed for page: {e}")

        return text, "direct"

    def extract_from_pdf(
        self,
        pdf_path: Path,
        force: bool = False
    ) -> ExtractionResult:
        """
        Extract text from a single PDF file.

        Args:
            pdf_path: Path to PDF file
            force: Force re-extraction even if already completed

        Returns:
            ExtractionResult with extraction details
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()

        # Check if already processed
        file_hash = self._get_file_hash(pdf_path)
        if not force and file_hash in self.progress.get("completed", []):
            self.logger.info(f"Skipping already processed: {pdf_path.name}")
            return ExtractionResult(
                success=True,
                pdf_path=str(pdf_path),
                output_path="",
                pages_extracted=0,
                total_pages=0,
                extraction_time=0,
                method="skipped"
            )

        # Determine output path
        ext = ".md" if self.save_as_markdown else ".txt"
        output_path = self.output_dir / f"{pdf_path.stem}{ext}"

        try:
            # Open PDF
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            # Extract text from all pages
            all_text = []
            methods_used = set()
            pages_with_text = 0

            for page_num in range(total_pages):
                page = doc[page_num]
                text, method = self._extract_page_text(page)
                methods_used.add(method)

                if text and len(text) > self.min_text_length:
                    if self.save_as_markdown:
                        all_text.append(f"## Page {page_num + 1}\n")
                    else:
                        all_text.append(f"\n--- Page {page_num + 1} ---\n")
                    all_text.append(text)
                    pages_with_text += 1

            doc.close()

            # Save extracted text
            if all_text:
                final_text = '\n'.join(all_text)

                if self.save_as_markdown:
                    # Add markdown header
                    header = f"# {pdf_path.stem}\n\n"
                    header += (
                        f"*Extracted: "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
                    header += (
                        f"*Pages: {pages_with_text}/{total_pages}*\n\n---\n")
                    final_text = header + final_text

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)

                # Update progress
                if file_hash not in self.progress.get("completed", []):
                    self.progress.setdefault("completed", []).append(file_hash)

                # Determine extraction method
                if "ocr" in methods_used and "direct" in methods_used:
                    method = "mixed"
                elif "ocr" in methods_used:
                    method = "ocr"
                else:
                    method = "direct"

                extraction_time = time.time() - start_time

                self.logger.info(
                    f"✓ {pdf_path.name}: {pages_with_text}/{total_pages} pages "
                    f"({method}, {extraction_time:.1f}s)"
                )

                return ExtractionResult(
                    success=True,
                    pdf_path=str(pdf_path),
                    output_path=str(output_path),
                    pages_extracted=pages_with_text,
                    total_pages=total_pages,
                    extraction_time=extraction_time,
                    method=method
                )
            else:
                raise ValueError("No text extracted from PDF")

        except Exception as e:
            # Log failure
            error_msg = str(e)
            self.logger.error(f"✗ {pdf_path.name}: {error_msg}")

            # Update progress
            if file_hash not in self.progress.get("failed", []):
                self.progress.setdefault("failed", []).append({
                    "hash": file_hash,
                    "path": str(pdf_path),
                    "error": error_msg
                })

            return ExtractionResult(
                success=False,
                pdf_path=str(pdf_path),
                output_path="",
                pages_extracted=0,
                total_pages=0,
                extraction_time=time.time() - start_time,
                method="failed",
                error=error_msg
            )

    def _extract_worker(self, pdf_path: Path) -> ExtractionResult:
        """Worker function for parallel extraction."""
        return self.extract_from_pdf(pdf_path)

    def batch_extract(
        self,
        input_dir: str,
        pattern: str = "*.pdf",
        force: bool = False,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from multiple PDFs in parallel.

        Args:
            input_dir: Directory containing PDFs
            pattern: Glob pattern for PDF files
            force: Force re-extraction of already processed files
            parallel: Use parallel processing

        Returns:
            Dictionary with extraction statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {input_path}")
            return {}

        # Find all PDF files
        pdf_files = list(input_path.glob(pattern))
        if not pdf_files:
            self.logger.warning(f"No PDF files found matching {pattern} in {input_path}")
            return {}

        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Statistics
        stats = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_pages": 0,
            "total_time": 0,
            "methods": {"direct": 0, "ocr": 0, "mixed": 0},
            "start_time": datetime.now().isoformat()
        }

        results = []

        if parallel and len(pdf_files) > 1:
            # Process in parallel
            self.logger.info(f"Processing with {self.max_workers} workers")

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_pdf = {
                    executor.submit(self.extract_from_pdf, pdf, force): pdf
                    for pdf in pdf_files
                }

                # Process completed tasks
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        result = future.result(timeout=300)  # 5 min timeout
                        results.append(result)

                        # Update stats
                        if result.success:
                            if result.method == "skipped":
                                stats["skipped"] += 1
                            else:
                                stats["successful"] += 1
                                stats["total_pages"] += result.pages_extracted
                                stats["total_time"] += result.extraction_time
                                if result.method in stats["methods"]:
                                    stats["methods"][result.method] += 1
                        else:
                            stats["failed"] += 1

                    except Exception as e:
                        self.logger.error(f"Worker failed for {pdf.name}: {e}")
                        stats["failed"] += 1
                        results.append(
                            ExtractionResult(
                            success=False,
                            pdf_path=str(pdf),
                            output_path="",
                            pages_extracted=0,
                            total_pages=0,
                            extraction_time=0,
                            method="failed",
                            error=str(e)
                            ))

                    # Save progress periodically
                    if len(results) % 10 == 0:
                        self._save_progress()
        else:
            # Process sequentially
            self.logger.info("Processing sequentially")

            for pdf in pdf_files:
                result = self.extract_from_pdf(pdf, force)
                results.append(result)

                # Update stats
                if result.success:
                    if result.method == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["successful"] += 1
                        stats["total_pages"] += result.pages_extracted
                        stats["total_time"] += result.extraction_time
                        if result.method in stats["methods"]:
                            stats["methods"][result.method] += 1
                else:
                    stats["failed"] += 1

                # Save progress periodically
                if len(results) % 10 == 0:
                    self._save_progress()

        # Final save
        self._save_progress()

        # Calculate summary statistics
        stats["end_time"] = datetime.now().isoformat()
        if stats["successful"] > 0:
            stats["avg_time_per_file"] = (
                stats["total_time"] / stats["successful"])
            stats["avg_pages_per_file"] = (
                stats["total_pages"] / stats["successful"])

        # Save statistics
        self.progress["stats"] = stats
        self._save_progress()

        # Print summary
        self._print_summary(stats, results)

        return {
            "stats": stats,
            "results": [asdict(r) for r in results]
        }

    def _print_summary(self, stats: Dict[str, Any], results: List[ExtractionResult]) -> None:
        """Print extraction summary."""
        print("\n" + "=" * 60)
        print("PDF EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total files: {stats['total_files']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Skipped: {stats['skipped']}")

        if stats['successful'] > 0:
            print("\nExtraction methods used:")
            for method, count in stats['methods'].items():
                if count > 0:
                    print(f"  - {method}: {count} files")

            print("\nPerformance:")
            print(f"  - Total pages extracted: {stats['total_pages']}")
            print(
                f"  - Avg pages per file: "
                f"{stats['avg_pages_per_file']:.1f}")
            print(
                f"  - Avg time per file: "
                f"{stats['avg_time_per_file']:.1f}s")

            if stats['total_time'] > 0:
                files_per_second = (
                    stats['successful'] / stats['total_time'])
                print(f"  - Processing speed: "
                      f"{files_per_second:.2f} files/second")

        if stats['failed'] > 0:
            print("\nFailed files:")
            for result in results:
                if not result.success:
                    print(f"  ✗ {Path(result.pdf_path).name}: {result.error}")

        print("=" * 60)

    def verify_extraction(self, pdf_path: Path) -> bool:
        """
        Verify that a PDF has been successfully extracted.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if extraction exists and is valid
        """
        ext = ".md" if self.save_as_markdown else ".txt"
        output_path = self.output_dir / f"{pdf_path.stem}{ext}"

        if not output_path.exists():
            return False

        # Check if file has content
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return len(content) > self.min_text_length
        except Exception:
            return False


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast PDF text extraction for large-scale processing"
    )
    parser.add_argument(
        "input_path",
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "-o", "--output",
        default="extracted_text",
        help="Output directory for extracted text (default: extracted_text)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR fallback for pages without text"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable text cleaning and normalization"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Save output as markdown files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of already processed files"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern for batch processing (default: *.pdf)"
    )
    parser.add_argument(
        "--log-file",
        default="logs/extraction.log",
        help="Path to log file"
    )
    parser.add_argument(
        "--progress-file",
        default="logs/progress.json",
        help="Path to progress tracking file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create extractor
    extractor = FastPDFExtractor(
        output_dir=args.output,
        log_file=args.log_file,
        progress_file=args.progress_file,
        max_workers=args.workers,
        use_ocr_fallback=args.ocr,
        clean_text=not args.no_clean,
        save_as_markdown=args.markdown
    )

    input_path = Path(args.input_path)

    if input_path.is_file():
        # Single file extraction
        if input_path.suffix.lower() != '.pdf':
            print(f"Error: {input_path} is not a PDF file")
            sys.exit(1)

        result = extractor.extract_from_pdf(input_path, force=args.force)

        if result.success:
            print("\n✓ Extraction completed successfully!")
            print(f"  Output: {result.output_path}")
            print(f"  Pages: {result.pages_extracted}/{result.total_pages}")
            print(f"  Time: {result.extraction_time:.1f}s")
            print(f"  Method: {result.method}")
        else:
            print(f"\n✗ Extraction failed: {result.error}")
            sys.exit(1)

    elif input_path.is_dir():
        # Batch extraction
        results = extractor.batch_extract(
            input_dir=str(input_path),
            pattern=args.pattern,
            force=args.force,
            parallel=not args.sequential
        )

        # Exit with error if any files failed
        if results.get("stats", {}).get("failed", 0) > 0:
            sys.exit(1)

    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
