#!/usr/bin/env python3
"""
Agentic PDF Vision Text Extractor using LangGraph and Ollama

This script uses LangGraph to create a robust agentic workflow for PDF text
extraction with self-correction, quality validation, and retry mechanisms.
"""

import sys
import json
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Annotated, TypedDict, Literal
import hashlib
from datetime import datetime

import fitz  # PyMuPDF
import requests
from PIL import Image
from unidecode import unidecode

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOllama


class ExtractionState(TypedDict):
    """State for the extraction workflow."""
    page_num: int
    total_pages: int
    image_base64: str
    extracted_text: str
    quality_score: float
    retry_count: int
    max_retries: int
    errors: List[str]
    validation_feedback: str
    extraction_strategy: str  # 'standard', 'detailed', 'ocr-focused'
    temperature: float
    is_complete: bool
    needs_retry: bool


class AgenticPDFVisionExtractor:
    """Agentic PDF extractor with LangGraph workflow."""
    
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
        """Initialize the Agentic PDF Vision Extractor."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        processing_config = self.config.get('processing', {})
        logging_config = self.config.get('logging', {})
        agentic_config = self.config.get('agentic', {})

        # Ollama configuration
        default_url = 'http://localhost:11434'
        self.ollama_url = (
            ollama_url or
            self.config.get('ollama', {}).get('url', default_url)
        ).rstrip('/')
        self.model = (
            model or
            self.config.get('ollama', {}).get('model', 'llama3.2-vision:latest')
        )
        
        # Output configuration
        default_output = 'extracted_text/vision'
        self.output_dir = Path(
            output_dir or
            processing_config.get('output_dir', default_output)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        progress_path = progress_file or processing_config.get(
            'progress_file', 'logs/extraction_progress.json'
        )
        self.progress_file = Path(progress_path)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.progress = self._load_progress()
        
        # Logging configuration
        log_file_path = (
            log_file or
            logging_config.get('log_file') or
            'logs/pdf_extraction.log'
        )
        self.log_file = Path(log_file_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Processing settings
        self.image_dpi = processing_config.get('image_dpi', 200)
        self.timeout = self.config.get('ollama', {}).get('timeout', 500)
        self.temperature = self.config.get('ollama', {}).get('temperature', 0.1)
        self.top_p = self.config.get('ollama', {}).get('top_p', 0.9)
        
        # Agentic settings
        self.max_retries = agentic_config.get('max_retries', 3)
        self.quality_threshold = agentic_config.get('quality_threshold', 0.6)
        self.min_text_length = agentic_config.get('min_text_length', 50)
        self.max_repetition_ratio = agentic_config.get(
            'max_repetition_ratio', 0.3
        )
        self.enable_reflection = agentic_config.get('enable_reflection', True)
        
        # Text cleaning settings
        text_cleaning = self.config.get('text_cleaning', {})
        self.apply_unidecode = text_cleaning.get('apply_unidecode', True)
        self.remove_extra_whitespace = text_cleaning.get(
            'remove_extra_whitespace', True
        )

        # Setup logging
        resolved_log_level = (
            log_level or
            logging_config.get('log_level') or
            processing_config.get('log_level') or
            'INFO'
        )
        self._setup_logging(resolved_log_level)

        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        
        # Verify Ollama connection
        self._verify_ollama_connection()

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
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
                    logging.info(f"Loaded config from: {config_path}")
                    return config
            except Exception as e:
                logging.warning(f"Failed to load config: {e}")
                return {}
        else:
            logging.info("No config file found. Using defaults.")
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
                logging.warning(f"Failed to load progress: {e}")
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
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        level = getattr(logging, log_level.upper(), logging.INFO)
        
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
        """Verify Ollama server is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                self.logger.warning(
                    f"Model '{self.model}' not found. "
                    f"Available: {model_names}"
                )
                self.logger.info(f"Pull model: ollama pull {self.model}")
            else:
                self.logger.info(f"Connected to Ollama. Model: {self.model}")
                
        except requests.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.logger.error("Make sure Ollama is running: ollama serve")
            sys.exit(1)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agentic extraction."""
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("retry_strategy", self._retry_strategy_node)
        
        # Define edges
        workflow.set_entry_point("extract")
        
        # After extraction, validate
        workflow.add_edge("extract", "validate")
        
        # After validation, decide next step
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {
                "reflect": "reflect",
                "retry": "retry_strategy",
                "complete": END
            }
        )
        
        # After reflection, retry with new strategy
        workflow.add_edge("reflect", "retry_strategy")
        
        # After retry strategy, extract again
        workflow.add_edge("retry_strategy", "extract")
        
        return workflow.compile()
    
    def _extract_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Extract text from image using vision model."""
        self.logger.info(
            f"Extracting page {state['page_num']} "
            f"(attempt {state['retry_count'] + 1}/{state['max_retries']})"
        )
        
        try:
            # Get prompt based on strategy
            prompt = self._get_extraction_prompt(state['extraction_strategy'])
            
            # Call Ollama vision API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [state['image_base64']],
                "stream": False,
                "options": {
                    "temperature": state['temperature'],
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
                # Basic post-processing
                extracted_text = self._post_process_text(extracted_text)
                state['extracted_text'] = extracted_text
                self.logger.info(
                    f"Extracted {len(extracted_text)} characters"
                )
            else:
                state['extracted_text'] = ""
                state['errors'].append("Empty response from vision model")
                self.logger.warning("Empty response from model")
                
        except requests.RequestException as e:
            state['errors'].append(f"API request failed: {e}")
            state['extracted_text'] = ""
            self.logger.error(f"API request failed: {e}")
        except Exception as e:
            state['errors'].append(f"Extraction error: {e}")
            state['extracted_text'] = ""
            self.logger.error(f"Extraction error: {e}")
        
        return state
    
    def _validate_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Validate quality of extracted text."""
        text = state['extracted_text']
        
        if not text:
            state['quality_score'] = 0.0
            state['validation_feedback'] = "No text extracted"
            return state
        
        # Calculate quality metrics
        quality_score = 1.0
        issues = []
        
        # Check 1: Minimum length
        if len(text) < self.min_text_length:
            quality_score -= 0.3
            issues.append(
                f"Text too short ({len(text)} < {self.min_text_length})"
            )
        
        # Check 2: Repetition detection
        repetition_ratio = self._calculate_repetition_ratio(text)
        if repetition_ratio > self.max_repetition_ratio:
            quality_score -= 0.5
            issues.append(
                f"High repetition ratio ({repetition_ratio:.2f})"
            )
        
        # Check 3: Diversity of words
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                quality_score -= 0.3
                issues.append(
                    f"Low word diversity ({unique_ratio:.2f})"
                )
        
        # Check 4: Excessive special characters
        special_char_ratio = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        ) / len(text) if len(text) > 0 else 0
        
        if special_char_ratio > 0.3:
            quality_score -= 0.2
            issues.append(
                f"Too many special characters ({special_char_ratio:.2f})"
            )
        
        # Check 5: Contains common error patterns
        error_patterns = [
            "please provide the problem",
            "i cannot",
            "unable to",
            "error occurred"
        ]
        
        text_lower = text.lower()
        for pattern in error_patterns:
            if pattern in text_lower:
                quality_score -= 0.4
                issues.append(f"Contains error pattern: '{pattern}'")
                break
        
        state['quality_score'] = max(0.0, quality_score)
        state['validation_feedback'] = "; ".join(issues) if issues else "OK"
        
        self.logger.info(
            f"Quality score: {state['quality_score']:.2f} - "
            f"{state['validation_feedback']}"
        )
        
        return state
    
    def _reflect_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Reflect on failures and provide feedback."""
        if not self.enable_reflection:
            return state
        
        self.logger.info("Reflecting on extraction quality...")
        
        feedback_parts = []
        
        if state['quality_score'] < self.quality_threshold:
            feedback_parts.append(
                f"Quality score {state['quality_score']:.2f} is below "
                f"threshold {self.quality_threshold}"
            )
        
        if state['validation_feedback']:
            feedback_parts.append(f"Issues: {state['validation_feedback']}")
        
        if state['errors']:
            feedback_parts.append(
                f"Errors encountered: {'; '.join(state['errors'][-3:])}"
            )
        
        # Suggest improvements
        if "repetition" in state['validation_feedback'].lower():
            feedback_parts.append(
                "Consider reducing temperature or using different prompt"
            )
        
        state['validation_feedback'] = " | ".join(feedback_parts)
        self.logger.info(f"Reflection: {state['validation_feedback']}")
        
        return state
    
    def _retry_strategy_node(self, state: ExtractionState) -> ExtractionState:
        """Node: Determine retry strategy."""
        state['retry_count'] += 1
        
        self.logger.info(
            f"Planning retry {state['retry_count']}/{state['max_retries']}"
        )
        
        # Adjust strategy based on previous failures
        if "repetition" in state['validation_feedback'].lower():
            # High repetition: reduce temperature
            state['temperature'] = max(0.0, state['temperature'] - 0.1)
            state['extraction_strategy'] = 'detailed'
            self.logger.info(
                f"Lowering temperature to {state['temperature']}"
            )
        
        elif "too short" in state['validation_feedback'].lower():
            # Too short: try more detailed extraction
            state['extraction_strategy'] = 'detailed'
            state['temperature'] = min(0.3, state['temperature'] + 0.05)
        
        elif "special characters" in state['validation_feedback'].lower():
            # OCR issues: focus on text extraction
            state['extraction_strategy'] = 'ocr-focused'
            state['temperature'] = 0.0
        
        else:
            # Default: cycle through strategies
            strategies = ['standard', 'detailed', 'ocr-focused']
            current_idx = strategies.index(state['extraction_strategy'])
            next_idx = (current_idx + 1) % len(strategies)
            state['extraction_strategy'] = strategies[next_idx]
        
        self.logger.info(f"Retry strategy: {state['extraction_strategy']}")
        
        # Clear previous extraction
        state['extracted_text'] = ""
        state['errors'] = []
        
        return state
    
    def _should_retry(
        self, state: ExtractionState
    ) -> Literal["reflect", "retry", "complete"]:
        """Decide whether to retry, reflect, or complete."""
        # Check if extraction is good enough
        if (state['quality_score'] >= self.quality_threshold and
            state['extracted_text']):
            return "complete"
        
        # Check if we've exhausted retries
        if state['retry_count'] >= state['max_retries']:
            self.logger.warning(
                f"Max retries reached for page {state['page_num']}"
            )
            return "complete"
        
        # Reflect before retry if enabled
        if self.enable_reflection and state['retry_count'] > 0:
            return "reflect"
        else:
            return "retry"
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate how repetitive the text is."""
        if len(text) < 50:
            return 0.0
        
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Check for repeated sequences
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find most common word
        max_count = max(word_counts.values())
        repetition_ratio = max_count / len(words)
        
        return repetition_ratio
    
    def _get_extraction_prompt(self, strategy: str) -> str:
        """Get extraction prompt based on strategy."""
        base_rules = """
You are analyzing a PDF page image. Extract text following these rules:

REMOVE:
- References, citations, figure captions
- Headers, footers, page numbers
- Footnotes, watermarks
- Tables
- Images
- Mathematical equations (unless in main text)

PRESERVE:
- Main body text paragraphs
- Natural paragraph structure
- Proper punctuation and sentences

OUTPUT:
- Clean text only, no commentary
- No placeholders or metadata
"""
        
        if strategy == 'detailed':
            return base_rules + """
Focus on capturing ALL text content from the image, even small details.
Read carefully and extract every word from paragraphs.
"""
        
        elif strategy == 'ocr-focused':
            return base_rules + """
Focus on accurate character recognition.
Carefully transcribe exactly what you see.
Avoid interpreting or paraphrasing.
"""
        
        else:  # standard
            return base_rules + """
Extract the main text content clearly and accurately.
"""
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text."""
        if self.apply_unidecode:
            text = unidecode(text)
        
        if self.remove_extra_whitespace:
            lines = [line.strip() for line in text.split('\n')]
            lines = [line for line in lines if line]
            processed_text = '\n'.join(lines)
            processed_text = ' '.join(processed_text.split())
        else:
            processed_text = text
        
        return processed_text
    
    def _pdf_page_to_image(
        self, page: fitz.Page, dpi: Optional[int] = None
    ) -> str:
        """Convert PDF page to base64 encoded image."""
        if dpi is None:
            dpi = self.image_dpi
        
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        output_filename: Optional[str] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        resume: bool = True
    ) -> bool:
        """Extract text from PDF using agentic workflow."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        if output_filename is None:
            output_filename = f"{pdf_path.stem}_extracted.md"
        
        output_path = self.output_dir / output_filename
        progress_key = self._get_file_hash(pdf_path)
        
        # Load existing progress
        if resume and progress_key in self.progress:
            file_progress = self.progress[progress_key]
            self.logger.info(f"Resuming extraction for {pdf_path.name}")
            prev_pages = file_progress.get('processed_pages', [])
            self.logger.info(f"Previously processed: {prev_pages}")
        
        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            
            if end_page is None:
                end_page = total_pages
            
            self.logger.info(f"Processing PDF: {pdf_path.name}")
            self.logger.info(
                f"Total pages: {total_pages}, "
                f"Processing: {start_page}-{end_page}"
            )
            
            # Initialize progress
            if progress_key not in self.progress:
                self.progress[progress_key] = {
                    'file_path': str(pdf_path.absolute()),
                    'output_path': str(output_path.absolute()),
                    'processed_pages': [],
                    'extracted_texts': {},
                    'quality_scores': {}
                }
            
            file_progress = self.progress[progress_key]
            processed_pages = set(file_progress.get('processed_pages', []))
            extracted_texts = file_progress.get('extracted_texts', {})
            quality_scores = file_progress.get('quality_scores', {})
            
            # Process each page with agentic workflow
            for page_num in range(start_page - 1, min(end_page, total_pages)):
                page_index = page_num + 1
                
                if page_index in processed_pages:
                    self.logger.info(
                        f"Skipping already processed page {page_index}"
                    )
                    continue
                
                self.logger.info(
                    f"\n{'='*60}\n"
                    f"Processing page {page_index}/{total_pages}\n"
                    f"{'='*60}"
                )
                
                try:
                    page = doc[page_num]
                    image_base64 = self._pdf_page_to_image(page)
                    
                    # Initialize state for this page
                    initial_state: ExtractionState = {
                        'page_num': page_index,
                        'total_pages': total_pages,
                        'image_base64': image_base64,
                        'extracted_text': '',
                        'quality_score': 0.0,
                        'retry_count': 0,
                        'max_retries': self.max_retries,
                        'errors': [],
                        'validation_feedback': '',
                        'extraction_strategy': 'standard',
                        'temperature': self.temperature,
                        'is_complete': False,
                        'needs_retry': False
                    }
                    
                    # Run agentic workflow
                    final_state = self.workflow.invoke(initial_state)
                    
                    # Store results
                    if final_state['extracted_text']:
                        extracted_texts[str(page_index)] = (
                            final_state['extracted_text']
                        )
                        quality_scores[str(page_index)] = (
                            final_state['quality_score']
                        )
                        processed_pages.add(page_index)
                        
                        self.logger.info(
                            f"✓ Page {page_index} extracted successfully "
                            f"(quality: {final_state['quality_score']:.2f}, "
                            f"attempts: {final_state['retry_count'] + 1})"
                        )
                    else:
                        self.logger.warning(
                            f"✗ Failed to extract page {page_index} after "
                            f"{final_state['retry_count'] + 1} attempts"
                        )
                    
                    # Save progress after each page
                    file_progress['processed_pages'] = sorted(
                        list(processed_pages)
                    )
                    file_progress['extracted_texts'] = extracted_texts
                    file_progress['quality_scores'] = quality_scores
                    self._save_progress()
                    
                    # Save partial output
                    self._save_partial_output(
                        output_path, extracted_texts, total_pages, quality_scores
                    )
                    
                except Exception as e:
                    self.logger.error(
                        f"Error processing page {page_index}: {e}"
                    )
                    self._save_progress()
                    continue
            
            # Combine all extracted text in markdown format
            if extracted_texts:
                final_text = self._format_as_markdown(
                    pdf_path=pdf_path,
                    extracted_texts=extracted_texts,
                    quality_scores=quality_scores,
                    total_pages=total_pages,
                    include_metadata=True
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                
                # Calculate average quality
                avg_quality = sum(quality_scores.values()) / len(
                    quality_scores
                ) if quality_scores else 0.0
                
                file_progress['completed'] = True
                file_progress['total_pages'] = total_pages
                file_progress['processed_count'] = len(processed_pages)
                file_progress['average_quality'] = avg_quality
                self._save_progress()
                
                self.logger.info(
                    f"\n{'='*60}\n"
                    f"✓ Extraction completed: {output_path}\n"
                    f"Pages processed: {len(processed_pages)}/{total_pages}\n"
                    f"Average quality: {avg_quality:.2f}\n"
                    f"{'='*60}"
                )
                return True
            else:
                self.logger.error("No text was extracted from any page")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            self._save_progress()
            return False
        finally:
            if 'doc' in locals():
                doc.close()
    
    def _save_partial_output(
        self, output_path: Path, extracted_texts: Dict[str, str],
        total_pages: int, quality_scores: Dict[str, float] = None
    ) -> None:
        """Save partial output in markdown format."""
        try:
            # Get PDF path from output path
            pdf_path = Path(output_path.stem.replace('_extracted', ''))
            
            # Use markdown formatting for partial output
            partial_text = self._format_as_markdown(
                pdf_path=output_path.parent.parent / pdf_path.name,
                extracted_texts=extracted_texts,
                quality_scores=quality_scores or {},
                total_pages=total_pages,
                include_metadata=True
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(partial_text)
        except Exception as e:
            self.logger.warning(f"Failed to save partial output: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for tracking."""
        try:
            stat = file_path.stat()
            identifier = (
                f"{file_path.absolute()}:{stat.st_mtime}:{stat.st_size}"
            )
            return hashlib.md5(identifier.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    
    def _format_as_markdown(
        self,
        pdf_path: Path,
        extracted_texts: Dict[str, str],
        quality_scores: Dict[str, float],
        total_pages: int,
        include_metadata: bool = True
    ) -> str:
        """Format extracted text as markdown with proper structure."""
        lines = []
        
        # Add document header
        if include_metadata:
            lines.append(f"# {pdf_path.stem}")
            lines.append("")
            lines.append("## Document Information")
            lines.append("")
            lines.append(f"- **Source File**: `{pdf_path.name}`")
            lines.append(f"- **Total Pages**: {total_pages}")
            lines.append(f"- **Extracted Pages**: {len(extracted_texts)}/{total_pages}")
            lines.append(f"- **Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if quality_scores:
                avg_quality = sum(quality_scores.values()) / len(quality_scores)
                lines.append(f"- **Average Quality Score**: {avg_quality:.2f}")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        # Add extracted content page by page
        for page_num in range(1, total_pages + 1):
            page_key = str(page_num)
            
            if page_key in extracted_texts:
                # Page header
                lines.append(f"## Page {page_num}")
                lines.append("")
                
                # Optional quality indicator
                if include_metadata and page_key in quality_scores:
                    quality = quality_scores[page_key]
                    quality_indicator = "✓" if quality >= 0.6 else "⚠"
                    lines.append(f"*Quality: {quality_indicator} {quality:.2f}*")
                    lines.append("")
                
                # Page content
                text = extracted_texts[page_key]
                lines.append(text)
                lines.append("")
                lines.append("---")
                lines.append("")
            else:
                # Placeholder for unprocessed pages
                lines.append(f"## Page {page_num}")
                lines.append("")
                lines.append("*[Not yet processed]*")
                lines.append("")
                lines.append("---")
                lines.append("")
        
        return '\n'.join(lines)
    
    def batch_extract(
        self, input_dir: str, file_pattern: str = "*.pdf",
        resume: bool = True
    ) -> Dict[str, bool]:
        """Batch extract text from multiple PDFs."""
        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {input_path}")
            return {}
        
        pdf_files = list(input_path.glob(file_pattern))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_path}")
            return {}
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Check completed files
        completed_files = set()
        if resume:
            for progress_key, file_data in self.progress.items():
                if file_data.get('completed', False):
                    completed_path = Path(file_data.get('file_path', ''))
                    if completed_path.exists():
                        completed_files.add(completed_path.absolute())
                        self.logger.info(
                            f"Skipping completed: {completed_path.name}"
                        )
        
        results = {}
        for pdf_file in pdf_files:
            if pdf_file.absolute() in completed_files:
                results[pdf_file.name] = True
                continue
            
            self.logger.info(f"\nProcessing: {pdf_file.name}")
            try:
                success = self.extract_from_pdf(str(pdf_file), resume=resume)
                results[pdf_file.name] = success
                
                if success:
                    self.logger.info(
                        f"✓ Successfully processed: {pdf_file.name}"
                    )
                else:
                    self.logger.error(f"✗ Failed: {pdf_file.name}")
            except KeyboardInterrupt:
                self.logger.warning(
                    "Processing interrupted. Progress saved."
                )
                self._save_progress()
                raise
            except Exception as e:
                self.logger.error(f"✗ Error processing {pdf_file.name}: {e}")
                results[pdf_file.name] = False
                self._save_progress()
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        self.logger.info(
            f"\nBatch completed: {successful}/{total} successful"
        )
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agentic PDF text extraction with LangGraph"
    )
    parser.add_argument(
        "pdf_path", nargs='?',
        help="Path to PDF file or directory"
    )
    parser.add_argument("-c", "--config", help="Config JSON file")
    parser.add_argument("-m", "--model", help="Ollama vision model")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-u", "--url", help="Ollama server URL")
    parser.add_argument(
        "--start-page", type=int, default=1, help="Start page"
    )
    parser.add_argument("--end-page", type=int, help="End page")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument(
        "--no-resume", action="store_true", help="Disable resume"
    )
    parser.add_argument("--progress-file", help="Progress file")
    parser.add_argument("--log-file", help="Log file")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else None
    
    extractor = AgenticPDFVisionExtractor(
        config_path=args.config,
        ollama_url=args.url,
        model=args.model,
        output_dir=args.output,
        log_level=log_level,
        progress_file=args.progress_file,
        log_file=args.log_file
    )
    
    # Determine mode
    if args.pdf_path:
        is_batch = args.batch or Path(args.pdf_path).is_dir()
    else:
        config_input = extractor.config.get('processing', {}).get('input_dir')
        if config_input and Path(config_input).exists():
            args.pdf_path = config_input
            is_batch = True
            extractor.logger.info(f"Using config input: {config_input}")
        else:
            parser.error("pdf_path required or input_dir in config")
    
    resume = not args.no_resume
    
    if is_batch:
        try:
            results = extractor.batch_extract(args.pdf_path, resume=resume)
            
            successful = sum(results.values())
            total = len(results)
            print(f"\n{'='*50}")
            print("AGENTIC BATCH PROCESSING SUMMARY")
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
            print("\n\nInterrupted. Progress saved. Resume with same command.")
            sys.exit(0)
    else:
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
                print("\n✓ Agentic extraction completed successfully!")
            else:
                print("\n✗ Agentic extraction failed!")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Progress saved. Resume with same command.")
            sys.exit(0)


if __name__ == "__main__":
    main()

