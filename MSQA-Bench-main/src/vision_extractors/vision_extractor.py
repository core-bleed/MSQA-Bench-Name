#!/usr/bin/env python3
"""
PDF Vision Text Extractor using Ollama

This script uses Ollama's vision models to extract and clean text from PDF files.
It converts PDF pages to images and uses vision models for more accurate text extraction.
Supports resuming from where it stopped if interrupted.
"""

import sys
import json
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict
import hashlib

import fitz  # PyMuPDF
import requests
from PIL import Image
from unidecode import unidecode


class PDFVisionExtractor:
    """Extract text from PDFs using Ollama vision models."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_level: Optional[str] = None,
        log_file: Optional[str] = None,
        progress_file: Optional[str] = None
    ):
        """
        Initialize the PDF Vision Extractor.

        Args:
            config_path: Path to config JSON file (optional)
            ollama_url: Base URL for Ollama API (overrides config)
            model: Vision model to use (overrides config)
            output_dir: Directory to save extracted text files (overrides config)
            log_level: Logging level (overrides config)
            log_file: Path to log file (overrides config)
            progress_file: Path to progress tracking file (overrides config)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        processing_config = self.config.get('processing', {})
        logging_config = self.config.get('logging', {})

        # Use provided args or fall back to config
        default_url = 'http://localhost:11434'
        self.ollama_url = (
            ollama_url or
            self.config.get('ollama', {}).get('url', default_url)
        ).rstrip('/')
        self.model = (
            model or
            self.config.get('ollama', {}).get('model', 'llava:13b')
        )
        default_output = 'extracted_text'
        self.output_dir = Path(
            output_dir or
            processing_config.get('output_dir', default_output)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        progress_path = progress_file or processing_config.get(
            'progress_file', 'extraction_progress.json'
        )
        self.progress_file = Path(progress_path)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load_progress()
        
        # Logging configuration
        log_file_path = (
            log_file or
            logging_config.get('log_file') or
            logging_config.get('file') or
            'pdf_extraction.log'
        )
        self.log_file = Path(log_file_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Processing settings from config
        self.image_dpi = processing_config.get('image_dpi', 200)
        self.timeout = self.config.get('ollama', {}).get('timeout', 120)
        self.temperature = self.config.get('ollama', {}).get('temperature', 0.1)
        self.top_p = self.config.get('ollama', {}).get('top_p', 0.9)
        
        # Text cleaning settings
        text_cleaning = self.config.get('text_cleaning', {})
        self.apply_unidecode = text_cleaning.get('apply_unidecode', True)
        self.remove_extra_whitespace = (
            text_cleaning.get('remove_extra_whitespace', True)
        )
        self.min_text_length = text_cleaning.get('min_text_length', 10)

        # Setup logging
        resolved_log_level = (
            log_level or
            logging_config.get('log_level') or
            logging_config.get('level') or
            processing_config.get('log_level') or
            'INFO'
        )
        self._setup_logging(resolved_log_level)

        # Verify Ollama connection
        self._verify_ollama_connection()

        # Text cleaning prompt
        self.cleaning_prompt = self._get_cleaning_prompt()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
            # Try to find config file in common locations
            possible_paths = [
                Path("config/config.json"),
                Path("../config/config.json"),
                Path(__file__).parent.parent.parent / "config" / "config.json"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logging.info(f"Loaded configuration from: {config_path}")
                    return config
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
                return {}
        else:
            logging.info("No config file found. Using default settings.")
            return {}
    
    def _load_progress(self) -> Dict:
        """Load progress tracking data."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    logging.info(f"Loaded progress from: {self.progress_file}")
                    return progress
            except Exception as e:
                logging.warning(f"Failed to load progress file: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save_progress(self) -> None:
        """Save progress tracking data."""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for tracking (based on path and size)."""
        try:
            stat = file_path.stat()
            # Use path and modification time as identifier
            identifier = f"{file_path.absolute()}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(identifier.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    
    def _get_progress_key(self, pdf_path: Path) -> str:
        """Get progress tracking key for a PDF file."""
        return self._get_file_hash(pdf_path)
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Reset existing handlers to avoid duplicate logs
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _verify_ollama_connection(self) -> None:
        """Verify Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                self.logger.warning(f"Model '{self.model}' not found. Available models: {model_names}")
                self.logger.info(f"You may need to pull the model: ollama pull {self.model}")
            else:
                self.logger.info(f"Successfully connected to Ollama. Using model: {self.model}")
                
        except requests.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama at {self.ollama_url}: {e}")
            self.logger.error("Make sure Ollama is running: ollama serve")
            sys.exit(1)
    
    def _get_cleaning_prompt(self) -> str:
        """Get the text cleaning prompt."""
        return """
        You are analyzing an image of a PDF page. Extract and clean the text content following these strict rules:
        
        1. REMOVE ALL:
        - References/citations (e.g., [1], (Smith 2020), Fig. 1)
        - Figure/table captions and mentions (e.g., "Figure 1:", "Table 2 shows...")
        - Headers, footers, page numbers
        - Footnotes, marginalia, watermarks
        - Mathematical equations or formulas (unless they are part of main text)
        - Stray characters, symbols, or OCR artifacts
        
        2. PRESERVE ONLY:
        - Main body text paragraphs
        - Natural paragraph structure with proper breaks
        - Correct obvious OCR errors only when absolutely certain
        - Proper sentence structure and punctuation
        
        3. FORMATTING RULES:
        - Single spaces between words
        - Single line breaks between paragraphs
        - No leading/trailing whitespace
        - Maintain natural reading flow
        
        4. ABSOLUTELY DO NOT:
        - Add commentary, explanations, or metadata
        - Summarize or rephrase content
        - Include placeholder text like "[...]"
        - Add your own interpretations
        
        Extract the clean text from this image and return ONLY the cleaned text content.
        """
    
    def _pdf_page_to_image(self, page: fitz.Page, dpi: Optional[int] = None) -> str:
        """
        Convert PDF page to base64 encoded image.
        
        Args:
            page: PyMuPDF page object
            dpi: Image resolution (higher = better quality but larger file)
            
        Returns:
            Base64 encoded image string
        """
        if dpi is None:
            dpi = self.image_dpi
        # Convert page to image
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 is default DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
    
    def _extract_text_from_image(self, image_base64: str) -> Optional[str]:
        """
        Extract text from image using Ollama vision model.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Extracted and cleaned text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": self.cleaning_prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            extracted_text = result.get('response', '').strip()
            
            if extracted_text:
                # Additional cleaning
                extracted_text = self._post_process_text(extracted_text)
                return extracted_text
            else:
                self.logger.warning("Empty response from vision model")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during text extraction: {e}")
            return None
    
    def _post_process_text(self, text: str) -> str:
        """
        Additional post-processing of extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Post-processed text
        """
        # Apply unidecode to handle special characters
        if self.apply_unidecode:
            text = unidecode(text)
        
        if self.remove_extra_whitespace:
            # Remove extra whitespace
            lines = [line.strip() for line in text.split('\n')]
            lines = [line for line in lines if line]  # Remove empty lines
            
            # Join with single newlines
            processed_text = '\n'.join(lines)
            
            # Replace multiple spaces with single space
            processed_text = ' '.join(processed_text.split())
        else:
            processed_text = text
        
        # Filter out text that's too short
        if len(processed_text.strip()) < self.min_text_length:
            return ""
        
        return processed_text
    
    def extract_from_pdf(
        self,
        pdf_path: str, 
        output_filename: Optional[str] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        resume: bool = True
    ) -> bool:
        """
        Extract text from PDF using vision model with resume capability.
        
        Args:
            pdf_path: Path to PDF file
            output_filename: Custom output filename (optional)
            start_page: First page to process (1-indexed)
            end_page: Last page to process (optional, processes all if None)
            resume: Whether to resume from previous progress
            
        Returns:
            True if successful, False otherwise
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Generate output filename
        if output_filename is None:
            output_filename = f"{pdf_path.stem}_extracted.txt"
        
        output_path = self.output_dir / output_filename
        progress_key = self._get_progress_key(pdf_path)
        
        # Load existing progress if resuming
        file_progress = None
        if resume and progress_key in self.progress:
            file_progress = self.progress[progress_key]
            self.logger.info(
                f"Resuming extraction for {pdf_path.name}"
            )
            prev_pages = file_progress.get('processed_pages', [])
            self.logger.info(f"Previously processed pages: {prev_pages}")
        
        try:
            # Open PDF
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            if end_page is None:
                end_page = total_pages
            
            self.logger.info(f"Processing PDF: {pdf_path.name}")
            self.logger.info(
                f"Total pages: {total_pages}, "
                f"Processing pages: {start_page}-{end_page}"
            )
            
            # Initialize progress tracking for this file
            if progress_key not in self.progress:
                self.progress[progress_key] = {
                    'file_path': str(pdf_path.absolute()),
                    'output_path': str(output_path.absolute()),
                    'processed_pages': [],
                    'extracted_texts': {}
                }
            
            file_progress = self.progress[progress_key]
            processed_pages = set(file_progress.get('processed_pages', []))
            extracted_texts = file_progress.get('extracted_texts', {})
            
            # Determine which pages to process
            pages_to_process = []
            for page_num in range(start_page - 1, min(end_page, total_pages)):
                page_index = page_num + 1  # 1-indexed
                if page_index not in processed_pages:
                    pages_to_process.append((page_num, page_index))
            
            if not pages_to_process and resume:
                msg = "All pages already processed. Loading existing output."
                self.logger.info(msg)
                # All pages processed, just verify output exists
                if output_path.exists():
                    return True
                else:
                    # Reconstruct from saved progress
                    self.logger.info(
                        "Reconstructing output from saved progress..."
                    )
                    end = min(end_page, total_pages)
                    pages_to_process = [
                        (i, i+1) for i in range(start_page - 1, end)
                    ]
            
            # Process each page
            for page_num, page_index in pages_to_process:
                self.logger.info(f"Processing page {page_index}/{total_pages}")
                
                try:
                    page = doc[page_num]
                    
                    # Convert page to image
                    image_base64 = self._pdf_page_to_image(page)
                    
                    # Extract text using vision model
                    extracted_text = self._extract_text_from_image(image_base64)
                    
                    if extracted_text:
                        extracted_texts[str(page_index)] = extracted_text
                        processed_pages.add(page_index)
                        msg = (
                            f"Successfully extracted text from page {page_index}"
                        )
                        self.logger.info(msg)
                        
                        # Save progress after each page
                        file_progress['processed_pages'] = sorted(
                            list(processed_pages)
                        )
                        file_progress['extracted_texts'] = extracted_texts
                        self._save_progress()
                        
                        # Also save partial output
                        self._save_partial_output(
                            output_path, extracted_texts, total_pages
                        )
                    else:
                        self.logger.warning(f"No text extracted from page {page_index}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing page {page_index}: {e}")
                    # Save progress even on error
                    self._save_progress()
                    continue
            
            # Combine all extracted text (including previously processed)
            all_pages = sorted([int(p) for p in extracted_texts.keys()])
            if all_pages:
                final_pages = []
                for page_num in range(1, total_pages + 1):
                    if str(page_num) in extracted_texts:
                        final_pages.append(extracted_texts[str(page_num)])
                
                final_text = '\n\n'.join(final_pages)
                
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                
                # Mark as complete
                file_progress['completed'] = True
                file_progress['total_pages'] = total_pages
                file_progress['processed_count'] = len(processed_pages)
                self._save_progress()
                
                msg = f"Text extraction completed. Output saved to: {output_path}"
                self.logger.info(msg)
                msg = f"Total pages processed: {len(processed_pages)}/{total_pages}"
                self.logger.info(msg)
                return True
            else:
                self.logger.error("No text was extracted from any page")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            # Save progress on error
            self._save_progress()
            return False
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _save_partial_output(
        self, output_path: Path, extracted_texts: Dict[str, str],
        total_pages: int
    ) -> None:
        """Save partial output as pages are processed."""
        try:
            final_pages = []
            for page_num in range(1, total_pages + 1):
                if str(page_num) in extracted_texts:
                    final_pages.append(extracted_texts[str(page_num)])
                else:
                    # Add placeholder for missing pages
                    msg = f"[Page {page_num} - Not yet processed]"
                    final_pages.append(msg)
            
            partial_text = '\n\n'.join(final_pages)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(partial_text)
        except Exception as e:
            self.logger.warning(f"Failed to save partial output: {e}")
    
    def batch_extract(
        self, input_dir: str, file_pattern: str = "*.pdf",
        resume: bool = True
    ) -> Dict[str, bool]:
        """
        Batch extract text from multiple PDFs with resume capability.
        
        Args:
            input_dir: Directory containing PDF files
            file_pattern: File pattern to match (default: "*.pdf")
            resume: Whether to resume from previous progress
            
        Returns:
            Dictionary mapping filenames to success status
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {input_path}")
            return {}
        
        pdf_files = list(input_path.glob(file_pattern))
        if not pdf_files:
            msg = (
                f"No PDF files found in {input_path} "
                f"matching pattern {file_pattern}"
            )
            self.logger.warning(msg)
            return {}
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Check which files are already completed
        completed_files = set()
        if resume:
            for progress_key, file_data in self.progress.items():
                if file_data.get('completed', False):
                    completed_path = Path(file_data.get('file_path', ''))
                    if completed_path.exists():
                        completed_files.add(completed_path.absolute())
                        msg = f"Skipping already completed file: {completed_path.name}"
                        self.logger.info(msg)
        
        results = {}
        for pdf_file in pdf_files:
            # Skip if already completed
            if pdf_file.absolute() in completed_files:
                results[pdf_file.name] = True
                continue
            
            self.logger.info(f"Processing: {pdf_file.name}")
            try:
                success = self.extract_from_pdf(str(pdf_file), resume=resume)
                results[pdf_file.name] = success
                
                if success:
                    self.logger.info(f"✓ Successfully processed: {pdf_file.name}")
                else:
                    self.logger.error(f"✗ Failed to process: {pdf_file.name}")
            except KeyboardInterrupt:
                self.logger.warning("Processing interrupted by user. Progress saved.")
                self._save_progress()
                raise
            except Exception as e:
                self.logger.error(f"✗ Error processing {pdf_file.name}: {e}")
                results[pdf_file.name] = False
                self._save_progress()
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        msg = (
            f"Batch processing completed: {successful}/{total} "
            f"files processed successfully"
        )
        self.logger.info(msg)
        
        return results


def main():
    """Main function to run the PDF Vision Extractor."""
    import argparse
    
    desc = "Extract text from PDFs using Ollama vision models"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "pdf_path", nargs='?',
        help="Path to PDF file or directory"
    )
    parser.add_argument(
        "-c", "--config", help="Path to config JSON file"
    )
    parser.add_argument(
        "-m", "--model",
        help="Ollama vision model to use (overrides config)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output filename (for single file) or directory"
    )
    parser.add_argument(
        "-u", "--url",
        help="Ollama server URL (overrides config)"
    )
    parser.add_argument(
        "--start-page", type=int, default=1,
        help="Start page (1-indexed)"
    )
    parser.add_argument(
        "--end-page", type=int, help="End page (optional)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all PDFs in directory"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable resume functionality"
    )
    parser.add_argument(
        "--progress-file",
        help="Progress tracking file"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    log_level = "DEBUG" if args.verbose else None
    
    # Initialize extractor with config
    extractor = PDFVisionExtractor(
        config_path=args.config,
        ollama_url=args.url,
        model=args.model,
        output_dir=args.output,
        log_level=log_level,
        progress_file=args.progress_file,
        log_file=args.log_file
    )
    
    # Determine if batch processing
    if args.pdf_path:
        is_batch = args.batch or Path(args.pdf_path).is_dir()
    else:
        # Try to get input directory from config
        config_input = extractor.config.get('processing', {}).get('input_dir')
        if config_input and Path(config_input).exists():
            args.pdf_path = config_input
            is_batch = True
            msg = f"Using input directory from config: {config_input}"
            extractor.logger.info(msg)
        else:
            msg = (
                "pdf_path is required or input_dir must be set in config"
            )
            parser.error(msg)
    
    resume = not args.no_resume
    
    # Process files
    if is_batch:
        # Batch processing
        try:
            results = extractor.batch_extract(args.pdf_path, resume=resume)
            
            # Print summary
            successful = sum(results.values())
            total = len(results)
            print(f"\n{'='*50}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Total files: {total}")
            print(f"Successful: {successful}")
            print(f"Failed: {total - successful}")
            
            if total - successful > 0:
                print("\nFailed files:")
                for filename, success in results.items():
                    if not success:
                        print(f"  ✗ {filename}")
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted. Progress has been saved.")
            print("You can resume by running the same command again.")
            sys.exit(0)
        
    else:
        # Single file processing
        is_dir = Path(args.output).is_dir() if args.output else False
        output_filename = args.output if args.output and not is_dir else None
        try:
            success = extractor.extract_from_pdf(
                args.pdf_path,
                output_filename=output_filename,
                start_page=args.start_page,
                end_page=args.end_page,
                resume=resume
            )
            
            if success:
                print("\n✓ Text extraction completed successfully!")
            else:
                print("\n✗ Text extraction failed!")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted. Progress has been saved.")
            print("You can resume by running the same command again.")
            sys.exit(0)


if __name__ == "__main__":
    main()
