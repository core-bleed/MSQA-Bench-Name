#!/usr/bin/env python3
"""
Production QA Generator for 40k+ files
Multi-GPU vLLM backend with full crash recovery and resume capability.

Features:
- Robust resume from any crash point (file-level + paragraph-level)
- Exponential backoff with jitter for API errors
- Circuit breaker per file (skip problematic files)
- Atomic writes to prevent corruption
- Memory monitoring and auto-throttling
- Progress tracking with ETA
- JSONL output for streaming/append-friendly format
"""

import os
import sys
import json
import time
import uuid
import signal
import argparse
import logging
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from threading import Lock
import random

from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError

# Optional: GPU/memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False


# ============================================================================
# Data Models
# ============================================================================
class QA(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: List[QA]


@dataclass
class FileProgress:
    file_name: str
    paragraph_index: int
    line_number: int
    completed: bool
    last_updated: str
    qa_count: int = 0
    failure_count: int = 0


@dataclass 
class RunStats:
    run_id: str
    start_time: float
    total_files: int
    completed_files: int = 0
    total_paragraphs: int = 0
    total_qa_pairs: int = 0
    total_failures: int = 0
    skipped_files: int = 0


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CONFIG = {
    "input_dir": "extracted_text/bulk_40k",
    "output_dir": "data/qa_outputs",
    "pdf_source_dir": "data/input",
    "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "base_url": "http://localhost:8000/v1",
    "api_key": "not-needed",
    "max_retries": 5,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "timeout": 300,
    "workers": 8,
    "temperature": 0.3,
    "max_tokens": 1024,
    "min_paragraph_length": 50,
    "max_paragraph_length": 4000,
    "circuit_breaker_threshold": 0.4,
    "circuit_breaker_min_attempts": 10,
    "checkpoint_interval": 25,
    "memory_threshold_pct": 85,
    "log_gpu_interval": 50,
}


# ============================================================================
# Utilities
# ============================================================================
def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("qa_generator")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_json_safe(path: Path, default=None):
    """Load JSON with error handling."""
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(f"Failed to load {path}: {e}")
        return default if default is not None else {}


def save_json_atomic(path: Path, data: dict) -> bool:
    """Atomic JSON save to prevent corruption on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        shutil.move(str(temp_path), str(path))
        return True
    except Exception as e:
        logging.error(f"Failed to save {path}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def get_system_stats() -> Dict:
    """Get system memory and GPU stats."""
    stats = {}
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        stats["ram_used_pct"] = mem.percent
        stats["ram_available_gb"] = mem.available / (1024**3)
    
    if HAS_NVML:
        try:
            gpu_stats = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_stats.append({
                    "gpu": i,
                    "vram_used_gb": mem.used / (1024**3),
                    "vram_total_gb": mem.total / (1024**3),
                    "utilization_pct": util.gpu
                })
            stats["gpus"] = gpu_stats
        except:
            pass
    return stats


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash for file integrity tracking."""
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]


# ============================================================================
# Text Processing
# ============================================================================
EXCLUDE_PREFIXES = [
    "acknowledgments:", "acknowledgements:", "acknowledgment:",
    "funding:", "data availability statement:", "author contributions:",
    "references:", "declarations:", "conflict of interest:",
    "competing interests:", "ethics statement:", "supplementary",
]


def is_valid_paragraph(text: str, min_len: int = 50, max_len: int = 4000) -> bool:
    """Check if paragraph is suitable for QA generation."""
    if not text or len(text) < min_len:
        return False
    if len(text) > max_len:
        return False
    
    first_line = text.split('\n')[0].strip().lower()
    if any(first_line.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    
    # Skip reference-heavy content
    lower_text = text.lower()
    if lower_text.count('doi:') > 2 or lower_text.count('http') > 3:
        return False
    
    # Skip tables/figures
    if text.count('|') > 10 or text.count('\t') > 20:
        return False
    
    # Must have some actual sentences
    if text.count('.') < 2:
        return False
    
    return True


def iter_paragraphs(file_path: Path) -> Iterator[Tuple[str, int, int]]:
    """
    Yield (paragraph_text, start_line, paragraph_index) from file.
    Paragraphs are separated by blank lines.
    """
    buffer = []
    start_line = 1
    para_idx = 0
    
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_num, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                
                if stripped:
                    if not buffer:
                        start_line = line_num
                    buffer.append(stripped)
                elif buffer:
                    para_idx += 1
                    paragraph = " ".join(buffer)
                    # Ensure proper sentence ending
                    if paragraph and not paragraph[-1] in ".?!":
                        paragraph += "."
                    yield paragraph, start_line, para_idx
                    buffer = []
            
            # Don't forget last paragraph if file doesn't end with blank line
            if buffer:
                para_idx += 1
                paragraph = " ".join(buffer)
                if paragraph and not paragraph[-1] in ".?!":
                    paragraph += "."
                yield paragraph, start_line, para_idx
                
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")


def validate_qa_pairs(qa_pairs: List[QA]) -> List[QA]:
    """Filter out low-quality QA pairs."""
    valid = []
    for qa in qa_pairs:
        q = qa.question.strip()
        a = qa.answer.strip()
        
        # Basic length checks
        if len(q) < 15 or len(a) < 20:
            continue
        
        # Skip generic questions
        if q.lower() in ["what?", "why?", "how?", "what is this about?"]:
            continue
        
        # Skip if answer is just repeating the question
        if q.lower() in a.lower() and len(a) < len(q) * 1.5:
            continue
        
        valid.append(QA(question=q, answer=a))
    
    return valid


# ============================================================================
# QA Generator Core
# ============================================================================
class QAGenerator:
    """Thread-safe QA generator with connection pooling."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout=config["timeout"],
            max_retries=0,  # We handle retries ourselves
        )
        self.model = config["model"]
        self.request_count = 0
        self.lock = Lock()
        
    def generate_qa(self, context: str) -> Tuple[List[QA], Dict]:
        """
        Generate QA pairs from context with retry logic.
        Returns (qa_pairs, usage_dict) or ([], {}) on failure.
        """
        # Truncate very long contexts
        context = context[:self.config["max_paragraph_length"]]
        
        prompt = f"""Analyze this text and generate question-answer pairs for training a Q&A system.

TEXT:
{context}

INSTRUCTIONS:
1. Generate 2-5 diverse questions that can be answered from the text
2. Questions should vary: factual (what/who/when), explanatory (why/how), comparative
3. Answers must be accurate and derived from the text
4. Be specific, avoid generic questions like "What is the main topic?"

Return ONLY valid JSON:
{{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}"""

        base_delay = self.config["base_delay"]
        max_delay = self.config["max_delay"]
        
        for attempt in range(1, self.config["max_retries"] + 1):
            try:
                with self.lock:
                    self.request_count += 1
                    req_num = self.request_count
                
                # Log GPU stats periodically
                if req_num % self.config["log_gpu_interval"] == 0:
                    stats = get_system_stats()
                    if "gpus" in stats:
                        gpu_info = " | ".join([
                            f"GPU{g['gpu']}: {g['vram_used_gb']:.1f}GB ({g['utilization_pct']}%)"
                            for g in stats["gpus"]
                        ])
                        self.logger.info(f"[Request #{req_num}] {gpu_info}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise QA dataset generator. Output only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                    response_format={"type": "json_object"},
                )
                
                if not response.choices:
                    return [], {}
                
                content = response.choices[0].message.content.strip()
                usage = response.usage
                usage_dict = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
                
                # Parse and validate
                data = json.loads(content)
                qa_response = QAResponse.model_validate(data)
                valid_pairs = validate_qa_pairs(qa_response.qa_pairs)
                
                return valid_pairs, usage_dict
                
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.warning(f"Parse error (attempt {attempt}): {e}")
                
            except RateLimitError as e:
                wait = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                self.logger.warning(f"Rate limited, waiting {wait:.1f}s")
                time.sleep(wait)
                continue
                
            except (APIConnectionError, APITimeoutError) as e:
                wait = min(base_delay * attempt + random.uniform(0, 1), max_delay)
                self.logger.warning(f"Connection error (attempt {attempt}): {e}, waiting {wait:.1f}s")
                time.sleep(wait)
                continue
                
            except APIStatusError as e:
                if e.status_code >= 500:
                    wait = min(base_delay * (2 ** attempt), max_delay)
                    self.logger.warning(f"Server error {e.status_code}, waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
                else:
                    self.logger.error(f"API error {e.status_code}: {e}")
                    return [], {}
                    
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                
            # Backoff between retries
            if attempt < self.config["max_retries"]:
                wait = min(base_delay * attempt + random.uniform(0, 0.5), max_delay)
                time.sleep(wait)
        
        return [], {}


# ============================================================================
# File Processor
# ============================================================================
class FileProcessor:
    """Process a single file with checkpoint/resume support."""
    
    def __init__(
        self,
        generator: QAGenerator,
        config: Dict,
        output_dir: Path,
        progress_file: Path,
        logger: logging.Logger,
        run_id: str,
    ):
        self.generator = generator
        self.config = config
        self.output_dir = output_dir
        self.progress_file = progress_file
        self.logger = logger
        self.run_id = run_id
        self.progress_lock = Lock()
        
    def load_progress(self) -> Dict[str, FileProgress]:
        """Load progress from checkpoint file."""
        data = load_json_safe(self.progress_file, {})
        progress = {}
        for fname, fp_data in data.items():
            progress[fname] = FileProgress(**fp_data)
        return progress
    
    def save_progress(self, progress: Dict[str, FileProgress]):
        """Save progress atomically."""
        with self.progress_lock:
            data = {k: asdict(v) for k, v in progress.items()}
            save_json_atomic(self.progress_file, data)
    
    def process_file(self, file_path: Path, progress: Dict[str, FileProgress]) -> Dict:
        """Process a single file, resuming from checkpoint if exists."""
        fname = file_path.name
        file_stem = file_path.stem
        
        # Construct PDF source path (original PDF reference)
        pdf_source_dir = Path(self.config.get("pdf_source_dir", "data/input"))
        pdf_path = pdf_source_dir / f"{file_stem}.pdf"
        pdf_path_str = str(pdf_path.resolve()) if pdf_path.exists() else str(pdf_path)
        
        # Setup per-file output
        file_output_dir = self.output_dir / "qa_by_file" / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        qa_file = file_output_dir / "qa.jsonl"
        failed_file = file_output_dir / "failed.jsonl"
        
        # Get resume point
        fp = progress.get(fname, FileProgress(
            file_name=fname,
            paragraph_index=0,
            line_number=0,
            completed=False,
            last_updated=datetime.utcnow().isoformat() + "Z"
        ))
        
        if fp.completed:
            return {"file": fname, "skipped": True, "reason": "already_completed"}
        
        start_para_idx = fp.paragraph_index
        
        stats = {
            "file": fname,
            "attempted": 0,
            "filtered": 0,
            "success": 0,
            "failed": 0,
            "qa_count": fp.qa_count,
            "skipped": False,
        }
        
        try:
            with qa_file.open("a", encoding="utf-8") as qa_out, \
                 failed_file.open("a", encoding="utf-8") as fail_out:
                
                for paragraph, line_num, para_idx in iter_paragraphs(file_path):
                    # Skip already processed paragraphs
                    if para_idx <= start_para_idx:
                        continue
                    
                    # Validate paragraph
                    if not is_valid_paragraph(
                        paragraph,
                        min_len=self.config["min_paragraph_length"],
                        max_len=self.config["max_paragraph_length"]
                    ):
                        stats["filtered"] += 1
                        continue
                    
                    stats["attempted"] += 1
                    
                    # Generate QA
                    qa_pairs, usage = self.generator.generate_qa(paragraph)
                    
                    if qa_pairs:
                        stats["success"] += 1
                        stats["qa_count"] += len(qa_pairs)
                        
                        # Write QA records
                        ts = datetime.utcnow().isoformat() + "Z"
                        for qa in qa_pairs:
                            record = {
                                "id": str(uuid.uuid4()),
                                "run_id": self.run_id,
                                "created_at": ts,
                                "model": self.config["model"],
                                "file_name": fname,
                                "source_pdf": pdf_path_str,
                                "line_number": line_num,
                                "paragraph_index": para_idx,
                                "context": paragraph,
                                "question": qa.question,
                                "answer": qa.answer,
                                "usage": usage,
                            }
                            qa_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        stats["failed"] += 1
                        # Log failure
                        failure = {
                            "file_name": fname,
                            "source_pdf": pdf_path_str,
                            "paragraph_index": para_idx,
                            "line_number": line_num,
                            "context_preview": paragraph[:200],
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                        fail_out.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    
                    # Checkpoint periodically
                    if para_idx % self.config["checkpoint_interval"] == 0:
                        qa_out.flush()
                        fp.paragraph_index = para_idx
                        fp.line_number = line_num
                        fp.qa_count = stats["qa_count"]
                        fp.failure_count = stats["failed"]
                        fp.last_updated = datetime.utcnow().isoformat() + "Z"
                        progress[fname] = fp
                        self.save_progress(progress)
                    
                    # Circuit breaker: skip file if too many failures
                    if stats["attempted"] >= self.config["circuit_breaker_min_attempts"]:
                        fail_rate = stats["failed"] / stats["attempted"]
                        if fail_rate > self.config["circuit_breaker_threshold"]:
                            self.logger.warning(
                                f"Circuit breaker: {fname} has {fail_rate:.0%} failure rate, skipping"
                            )
                            stats["circuit_breaker"] = True
                            break
                    
                    # Memory check
                    if HAS_PSUTIL and stats["attempted"] % 50 == 0:
                        mem = psutil.virtual_memory()
                        if mem.percent > self.config["memory_threshold_pct"]:
                            self.logger.warning(f"Memory high ({mem.percent}%), pausing briefly")
                            time.sleep(2)
                
                # Mark complete
                fp.paragraph_index = para_idx if 'para_idx' in dir() else 0
                fp.completed = True
                fp.qa_count = stats["qa_count"]
                fp.failure_count = stats["failed"]
                fp.last_updated = datetime.utcnow().isoformat() + "Z"
                progress[fname] = fp
                self.save_progress(progress)
                
        except Exception as e:
            self.logger.error(f"Fatal error processing {fname}: {e}", exc_info=True)
            stats["error"] = str(e)
        
        return stats


# ============================================================================
# Main Orchestrator
# ============================================================================
class QAOrchestrator:
    """Main orchestrator for parallel QA generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.shutdown_requested = False
        
        # Setup paths
        self.input_dir = Path(config["input_dir"])
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / "logs" / f"qa_generation_{self.run_id}.log"
        self.logger = setup_logging(log_file)
        
        # Progress tracking
        self.progress_file = self.output_dir / "progress.json"
        self.summary_file = self.output_dir / "run_summary.json"
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
    def run(self, resume: bool = True):
        """Run the QA generation pipeline."""
        self.logger.info("=" * 70)
        self.logger.info(f"🚀 QA Generator Starting")
        self.logger.info(f"   Run ID: {self.run_id}")
        self.logger.info(f"   Input: {self.input_dir}")
        self.logger.info(f"   Output: {self.output_dir}")
        self.logger.info(f"   Model: {self.config['model']}")
        self.logger.info(f"   Workers: {self.config['workers']}")
        self.logger.info(f"   Base URL: {self.config['base_url']}")
        self.logger.info("=" * 70)
        
        # Validate input directory
        if not self.input_dir.exists():
            self.logger.error(f"Input directory not found: {self.input_dir}")
            sys.exit(1)
        
        # Get all files
        all_files = sorted(self.input_dir.glob("*.txt"))
        if not all_files:
            self.logger.error(f"No .txt files found in {self.input_dir}")
            sys.exit(1)
        
        self.logger.info(f"📁 Found {len(all_files)} files to process")
        
        # Initialize generator and processor
        generator = QAGenerator(self.config, self.logger)
        processor = FileProcessor(
            generator=generator,
            config=self.config,
            output_dir=self.output_dir,
            progress_file=self.progress_file,
            logger=self.logger,
            run_id=self.run_id,
        )
        
        # Load existing progress
        progress = processor.load_progress()
        
        # Filter out completed files if resuming
        if resume:
            completed = {k for k, v in progress.items() if v.completed}
            files_to_process = [f for f in all_files if f.name not in completed]
            self.logger.info(f"📌 Resuming: {len(completed)} already completed, {len(files_to_process)} remaining")
        else:
            files_to_process = all_files
            progress = {}
        
        if not files_to_process:
            self.logger.info("✅ All files already processed!")
            return
        
        # Initialize stats
        stats = RunStats(
            run_id=self.run_id,
            start_time=time.time(),
            total_files=len(files_to_process),
        )
        
        # Process files with thread pool
        with tqdm(total=len(files_to_process), desc="Processing", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.config["workers"]) as executor:
                futures = {
                    executor.submit(processor.process_file, f, progress): f
                    for f in files_to_process
                }
                
                for future in as_completed(futures):
                    if self.shutdown_requested:
                        self.logger.warning("Shutdown requested, cancelling remaining tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    file_path = futures[future]
                    try:
                        result = future.result()
                        
                        if result.get("skipped"):
                            stats.skipped_files += 1
                        else:
                            stats.completed_files += 1
                            stats.total_paragraphs += result.get("attempted", 0)
                            stats.total_qa_pairs += result.get("qa_count", 0)
                            stats.total_failures += result.get("failed", 0)
                        
                        pbar.set_postfix({
                            "QA": stats.total_qa_pairs,
                            "Fail": stats.total_failures,
                        })
                        pbar.update(1)
                        
                        self.logger.info(
                            f"✓ {result['file']}: {result.get('qa_count', 0)} QA, "
                            f"{result.get('failed', 0)} fails"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Worker error for {file_path}: {e}")
                        pbar.update(1)
        
        # Save final summary
        duration = time.time() - stats.start_time
        summary = {
            "run_id": stats.run_id,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "duration_seconds": duration,
            "duration_human": str(timedelta(seconds=int(duration))),
            "config": self.config,
            "stats": {
                "total_files": stats.total_files,
                "completed_files": stats.completed_files,
                "skipped_files": stats.skipped_files,
                "total_paragraphs": stats.total_paragraphs,
                "total_qa_pairs": stats.total_qa_pairs,
                "total_failures": stats.total_failures,
            }
        }
        save_json_atomic(self.summary_file, summary)
        
        # Final report
        self.logger.info("=" * 70)
        self.logger.info("🏁 GENERATION COMPLETE")
        self.logger.info(f"   Duration: {timedelta(seconds=int(duration))}")
        self.logger.info(f"   Files processed: {stats.completed_files}/{stats.total_files}")
        self.logger.info(f"   QA pairs generated: {stats.total_qa_pairs:,}")
        self.logger.info(f"   Failures: {stats.total_failures:,}")
        self.logger.info(f"   Output: {self.output_dir}")
        self.logger.info("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Production QA Generator for 40k+ files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/qa_generator.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        help="Override input directory"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Override model name"
    )
    parser.add_argument(
        "--base-url", "-u",
        type=str,
        help="Override vLLM server URL"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Override number of parallel workers"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing progress"
    )
    
    args = parser.parse_args()
    
    # Load config file
    config_path = Path(args.config)
    if config_path.exists():
        config = load_json_safe(config_path, DEFAULT_CONFIG.copy())
        print(f"Loaded config from {config_path}")
    else:
        config = DEFAULT_CONFIG.copy()
        print(f"Config not found at {config_path}, using defaults")
    
    # Apply CLI overrides
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.model:
        config["model"] = args.model
    if args.base_url:
        config["base_url"] = args.base_url
    if args.workers:
        config["workers"] = args.workers
    
    # Run
    orchestrator = QAOrchestrator(config)
    orchestrator.run(resume=not args.no_resume)


if __name__ == "__main__":
    main()
