#!/usr/bin/env python3
"""
QA JSONL Consolidation Script
Merges per-file JSONL outputs into one consolidated file with deduplication.

Features:
- Resume from interruption (tracks processed files)
- Remove duplicate questions (exact + fuzzy matching)
- Progress tracking with checkpoint
- Input directory option
- Quality filtering


# Resume from where it stopped (default)
python scripts/consolidate_qa.py -i dir1 dir2

# Start completely fresh (delete progress + output)
python scripts/consolidate_qa.py -i dir1 dir2 --restart

# Don't resume but keep existing output
python scripts/consolidate_qa.py -i dir1 dir2 --no-resume

# Install optional fuzzy matching
pip install rapidfuzz

# Run consolidation with 2 directories
python scripts/consolidate_qa.py \
    --input-dirs /path/to/qa_dir_1 /path/to/qa_dir_2 \
    --output-file data/qa_outputs/jsonl/consolidated_qa.jsonl \
    --fuzzy-threshold 90
"""


import sys
import json
import argparse
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Iterator
from dataclasses import dataclass, asdict

from tqdm import tqdm

# Optional: fuzzy matching for near-duplicate detection
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("Note: Install 'rapidfuzz' for fuzzy duplicate detection: pip install rapidfuzz")


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CONFIG = {
    "input_dirs": ["data/qa_outputs/qa_by_file"],  # Support multiple directories
    "output_file": "data/qa_outputs/jsonl/consolidated_qa.jsonl",
    "progress_file": "data/qa_outputs/consolidation_progress.json",
    "summary_file": "data/qa_outputs/consolidation_summary.json",
    "min_question_length": 15,
    "min_answer_length": 20,
    "fuzzy_threshold": 90,  # Similarity threshold for fuzzy dedup (0-100)
    "enable_fuzzy_dedup": True,
    "batch_size": 1000,  # Save progress every N files
}


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class ConsolidationProgress:
    """Track consolidation progress for resume capability."""
    run_id: str
    started_at: str
    last_updated: str
    processed_files: List[str]
    total_files_found: int
    total_records_written: int
    duplicates_removed: int
    quality_filtered: int
    completed: bool = False
    input_dirs: List[str] = None
    per_directory_stats: Dict = None
    per_file_stats: Dict = None
    
    def __post_init__(self):
        if self.input_dirs is None:
            self.input_dirs = []
        if self.per_directory_stats is None:
            self.per_directory_stats = {}
        if self.per_file_stats is None:
            self.per_file_stats = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConsolidationProgress":
        # Handle missing fields for backward compatibility
        data.setdefault("input_dirs", [])
        data.setdefault("per_directory_stats", {})
        data.setdefault("per_file_stats", {})
        return cls(**data)


@dataclass
class QARecord:
    """Represents a single QA record."""
    id: str
    question: str
    answer: str
    context: str
    file_name: str
    source_pdf: str = ""
    line_number: int = 0
    paragraph_index: int = 0
    created_at: str = ""
    run_id: str = ""
    model: str = ""
    usage: Dict = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v}


# ============================================================================
# Utilities
# ============================================================================
def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("consolidate_qa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_question_hash(question: str) -> str:
    """Get hash of normalized question for exact dedup."""
    normalized = question.lower().strip()
    # Remove punctuation and extra spaces
    normalized = " ".join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()


def load_progress(progress_file: Path) -> Optional[ConsolidationProgress]:
    """Load progress from checkpoint file."""
    if not progress_file.exists():
        return None
    try:
        with progress_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return ConsolidationProgress.from_dict(data)
    except Exception as e:
        logging.warning(f"Failed to load progress: {e}")
        return None


def save_progress(progress: ConsolidationProgress, progress_file: Path) -> None:
    """Save progress to checkpoint file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress.last_updated = datetime.utcnow().isoformat() + "Z"
    
    temp_file = progress_file.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2)
        temp_file.replace(progress_file)
    except Exception as e:
        logging.error(f"Failed to save progress: {e}")
        if temp_file.exists():
            temp_file.unlink()


def find_jsonl_files(input_dir: Path) -> List[Path]:
    """Find all JSONL files in a single input directory (recursive)."""
    jsonl_files = []
    seen = set()
    
    # Check for qa_by_file structure (subdirectories with qa.jsonl)
    if input_dir.is_dir():
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                qa_file = subdir / "qa.jsonl"
                if qa_file.exists() and str(qa_file) not in seen:
                    jsonl_files.append(qa_file)
                    seen.add(str(qa_file))
        
        # Also check for direct JSONL files in the directory
        for jsonl_file in input_dir.glob("*.jsonl"):
            if str(jsonl_file) not in seen:
                jsonl_files.append(jsonl_file)
                seen.add(str(jsonl_file))
        
        # Recursive search for any qa.jsonl files
        for qa_file in input_dir.rglob("qa.jsonl"):
            if str(qa_file) not in seen:
                jsonl_files.append(qa_file)
                seen.add(str(qa_file))
        
        # Recursive search for any .jsonl files
        for jsonl_file in input_dir.rglob("*.jsonl"):
            if str(jsonl_file) not in seen:
                jsonl_files.append(jsonl_file)
                seen.add(str(jsonl_file))
    
    return sorted(jsonl_files)


def find_jsonl_files_multi(input_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Find all JSONL files across multiple directories.
    
    Returns:
        Dict mapping directory path to list of JSONL files found in it
    """
    result = {}
    for input_dir in input_dirs:
        if input_dir.exists():
            files = find_jsonl_files(input_dir)
            result[str(input_dir)] = files
        else:
            logging.warning(f"Input directory not found: {input_dir}")
            result[str(input_dir)] = []
    return result


def iter_jsonl_records(file_path: Path) -> Iterator[Dict]:
    """Iterate over records in a JSONL file."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON parse error in {file_path}:{line_num}: {e}")
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")


def is_quality_record(record: Dict, min_q_len: int, min_a_len: int) -> bool:
    """Check if record passes quality filters."""
    question = record.get("question", "").strip()
    answer = record.get("answer", "").strip()
    
    if len(question) < min_q_len:
        return False
    if len(answer) < min_a_len:
        return False
    
    # Skip generic questions
    generic_questions = [
        "what?", "why?", "how?", "what is this?",
        "what is the main topic?", "what is this about?"
    ]
    if question.lower() in generic_questions:
        return False
    
    return True


class FuzzyDeduplicator:
    """Handles fuzzy duplicate detection using similarity matching."""
    
    def __init__(self, threshold: int = 90):
        self.threshold = threshold
        self.questions: List[str] = []
        self.question_hashes: Set[str] = set()
    
    def is_duplicate(self, question: str) -> bool:
        """Check if question is a duplicate (exact or fuzzy)."""
        # Exact match check first (fast)
        q_hash = get_question_hash(question)
        if q_hash in self.question_hashes:
            return True
        
        # Fuzzy match check (slower, optional)
        if HAS_RAPIDFUZZ and self.questions:
            normalized = question.lower().strip()
            # Only check against recent questions for performance
            check_against = self.questions[-10000:] if len(self.questions) > 10000 else self.questions
            
            for existing in check_against:
                similarity = fuzz.ratio(normalized, existing.lower())
                if similarity >= self.threshold:
                    return True
        
        return False
    
    def add(self, question: str) -> None:
        """Add question to the deduplicator."""
        self.question_hashes.add(get_question_hash(question))
        self.questions.append(question)


# ============================================================================
# Main Consolidator
# ============================================================================
class QAConsolidator:
    """Main consolidation orchestrator."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Support both single dir (backward compat) and multiple dirs
        if "input_dirs" in config:
            self.input_dirs = [Path(d) for d in config["input_dirs"]]
        elif "input_dir" in config:
            self.input_dirs = [Path(config["input_dir"])]
        else:
            self.input_dirs = [Path("data/qa_outputs/qa_by_file")]
        
        self.output_file = Path(config["output_file"])
        self.progress_file = Path(config["progress_file"])
        self.summary_file = Path(config.get("summary_file", "data/qa_outputs/consolidation_summary.json"))
        
        self.logger = setup_logging(
            Path("logs") / "consolidation.log"
        )
        
        self.deduplicator = FuzzyDeduplicator(
            threshold=config.get("fuzzy_threshold", 90)
        ) if config.get("enable_fuzzy_dedup", True) else None
        
        # Track per-file and per-directory stats
        self.per_file_stats: Dict[str, Dict] = {}
        self.per_directory_stats: Dict[str, Dict] = {}
        
    def run(self, resume: bool = True, restart: bool = False) -> Dict:
        """
        Run the consolidation process.
        
        Args:
            resume: Continue from last checkpoint if available
            restart: Start fresh, ignore existing progress
        """
        self.logger.info("=" * 70)
        self.logger.info("🔄 QA Consolidation Starting")
        self.logger.info(f"   Input directories: {len(self.input_dirs)}")
        for d in self.input_dirs:
            self.logger.info(f"      - {d}")
        self.logger.info(f"   Output: {self.output_file}")
        self.logger.info("=" * 70)
        
        # Validate input directories
        valid_dirs = [d for d in self.input_dirs if d.exists()]
        if not valid_dirs:
            self.logger.error("No valid input directories found")
            return {"success": False, "error": "No valid input directories found"}
        
        invalid_dirs = [d for d in self.input_dirs if not d.exists()]
        for d in invalid_dirs:
            self.logger.warning(f"Directory not found (skipping): {d}")
        
        # Find all JSONL files across all directories
        files_by_dir = find_jsonl_files_multi(valid_dirs)
        all_files = []
        for dir_path, files in files_by_dir.items():
            all_files.extend(files)
            self.per_directory_stats[dir_path] = {
                "files_found": len(files),
                "records_written": 0,
                "duplicates": 0,
                "quality_filtered": 0,
            }
            self.logger.info(f"📁 {dir_path}: {len(files)} JSONL files")
        
        if not all_files:
            self.logger.error("No JSONL files found in any input directory")
            return {"success": False, "error": "No JSONL files found"}
        
        self.logger.info(f"📁 Found {len(all_files)} JSONL files")
        
        # Handle restart vs resume
        progress = None
        processed_set: Set[str] = set()
        
        if restart:
            self.logger.info("🔄 Restart requested - starting fresh")
            # Clear output file
            if self.output_file.exists():
                self.output_file.unlink()
            if self.progress_file.exists():
                self.progress_file.unlink()
        elif resume:
            progress = load_progress(self.progress_file)
            if progress and not progress.completed:
                processed_set = set(progress.processed_files)
                self.logger.info(f"📌 Resuming: {len(processed_set)} files already processed")
                
                # Load existing questions for dedup
                if self.output_file.exists() and self.deduplicator:
                    self.logger.info("Loading existing questions for deduplication...")
                    for record in iter_jsonl_records(self.output_file):
                        q = record.get("question", "")
                        if q:
                            self.deduplicator.add(q)
                    self.logger.info(f"Loaded {len(self.deduplicator.questions)} existing questions")
        
        # Initialize progress if new run
        if progress is None or restart:
            progress = ConsolidationProgress(
                run_id=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
                started_at=datetime.utcnow().isoformat() + "Z",
                last_updated=datetime.utcnow().isoformat() + "Z",
                processed_files=[],
                total_files_found=len(all_files),
                total_records_written=0,
                duplicates_removed=0,
                quality_filtered=0,
                completed=False,
                input_dirs=[str(d) for d in self.input_dirs],
                per_directory_stats=self.per_directory_stats,
                per_file_stats={},
            )
        
        # Filter files to process
        files_to_process = [f for f in all_files if str(f) not in processed_set]
        
        if not files_to_process:
            self.logger.info("✅ All files already processed!")
            return {
                "success": True,
                "total_records": progress.total_records_written,
                "duplicates_removed": progress.duplicates_removed,
            }
        
        self.logger.info(f"📝 Processing {len(files_to_process)} files...")
        
        # Open output file in append mode
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "records_written": 0,
            "duplicates": 0,
            "quality_filtered": 0,
            "errors": 0,
        }
        
        try:
            with self.output_file.open("a", encoding="utf-8") as out_f:
                for file_idx, jsonl_file in enumerate(tqdm(files_to_process, desc="Consolidating")):
                    # Track per-file stats
                    file_stats = {
                        "records_written": 0,
                        "duplicates": 0,
                        "quality_filtered": 0,
                        "total_records": 0,
                    }
                    
                    # Determine which directory this file belongs to
                    file_dir = None
                    for dir_path in self.per_directory_stats.keys():
                        if str(jsonl_file).startswith(dir_path):
                            file_dir = dir_path
                            break
                    
                    try:
                        for record in iter_jsonl_records(jsonl_file):
                            file_stats["total_records"] += 1
                            
                            # Quality check
                            if not is_quality_record(
                                record,
                                self.config.get("min_question_length", 15),
                                self.config.get("min_answer_length", 20)
                            ):
                                stats["quality_filtered"] += 1
                                file_stats["quality_filtered"] += 1
                                continue
                            
                            question = record.get("question", "")
                            
                            # Deduplication check
                            if self.deduplicator and self.deduplicator.is_duplicate(question):
                                stats["duplicates"] += 1
                                file_stats["duplicates"] += 1
                                continue
                            
                            # Add to deduplicator
                            if self.deduplicator:
                                self.deduplicator.add(question)
                            
                            # Write record
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stats["records_written"] += 1
                            file_stats["records_written"] += 1
                        
                        # Mark file as processed and save per-file stats
                        progress.processed_files.append(str(jsonl_file))
                        self.per_file_stats[str(jsonl_file)] = file_stats
                        
                        # Update per-directory stats
                        if file_dir and file_dir in self.per_directory_stats:
                            self.per_directory_stats[file_dir]["records_written"] += file_stats["records_written"]
                            self.per_directory_stats[file_dir]["duplicates"] += file_stats["duplicates"]
                            self.per_directory_stats[file_dir]["quality_filtered"] += file_stats["quality_filtered"]
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {jsonl_file}: {e}")
                        stats["errors"] += 1
                    
                    # Checkpoint periodically
                    if (file_idx + 1) % self.config.get("batch_size", 1000) == 0:
                        out_f.flush()
                        progress.total_records_written = sum(
                            fs.get("records_written", 0) for fs in self.per_file_stats.values()
                        )
                        progress.duplicates_removed = stats["duplicates"]
                        progress.quality_filtered = stats["quality_filtered"]
                        progress.per_directory_stats = self.per_directory_stats
                        progress.per_file_stats = self.per_file_stats
                        save_progress(progress, self.progress_file)
                        self.logger.info(
                            f"Checkpoint: {file_idx + 1}/{len(files_to_process)} files, "
                            f"{stats['records_written']} records"
                        )
            
            # Final progress update
            progress.total_records_written = stats["records_written"]
            progress.duplicates_removed = stats["duplicates"]
            progress.quality_filtered = stats["quality_filtered"]
            progress.per_directory_stats = self.per_directory_stats
            progress.per_file_stats = self.per_file_stats
            progress.completed = True
            save_progress(progress, self.progress_file)
            
        except KeyboardInterrupt:
            self.logger.warning("Interrupted! Progress saved, run again to resume.")
            progress.per_directory_stats = self.per_directory_stats
            progress.per_file_stats = self.per_file_stats
            save_progress(progress, self.progress_file)
            return {
                "success": False,
                "interrupted": True,
                "records_written": progress.total_records_written,
            }
        
        # Generate and save summary
        summary = self._generate_summary(progress, stats)
        self._save_summary(summary)
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, progress: ConsolidationProgress, stats: Dict) -> Dict:
        """Generate comprehensive summary of the consolidation."""
        # Calculate totals
        total_input_records = sum(
            fs.get("total_records", 0) for fs in self.per_file_stats.values()
        )
        
        summary = {
            "success": True,
            "run_id": progress.run_id,
            "started_at": progress.started_at,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "output_file": str(self.output_file),
            
            # Overall stats
            "totals": {
                "input_directories": len(self.input_dirs),
                "input_files": len(progress.processed_files),
                "input_records": total_input_records,
                "output_records": progress.total_records_written,
                "duplicates_removed": progress.duplicates_removed,
                "quality_filtered": progress.quality_filtered,
                "errors": stats.get("errors", 0),
            },
            
            # Per-directory breakdown
            "by_directory": {},
            
            # Top files by QA count
            "top_files": [],
        }
        
        # Per-directory stats
        for dir_path, dir_stats in self.per_directory_stats.items():
            files_in_dir = [f for f in progress.processed_files if f.startswith(dir_path)]
            summary["by_directory"][dir_path] = {
                "files_processed": len(files_in_dir),
                "records_written": dir_stats.get("records_written", 0),
                "duplicates": dir_stats.get("duplicates", 0),
                "quality_filtered": dir_stats.get("quality_filtered", 0),
            }
        
        # Top files by QA count
        file_records = [
            (f, fs.get("records_written", 0))
            for f, fs in self.per_file_stats.items()
        ]
        file_records.sort(key=lambda x: x[1], reverse=True)
        summary["top_files"] = [
            {"file": f, "qa_count": count}
            for f, count in file_records[:20]  # Top 20
        ]
        
        # Deduplication effectiveness
        if total_input_records > 0:
            summary["deduplication"] = {
                "exact_and_fuzzy_duplicates": progress.duplicates_removed,
                "duplicate_rate_pct": round(
                    progress.duplicates_removed / total_input_records * 100, 2
                ),
            }
        
        return summary
    
    def _save_summary(self, summary: Dict) -> None:
        """Save summary to JSON file."""
        self.summary_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.summary_file.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Summary saved to: {self.summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
    
    def _print_summary(self, summary: Dict) -> None:
        """Print formatted summary to console."""
        self.logger.info("=" * 70)
        self.logger.info("✅ CONSOLIDATION COMPLETE")
        self.logger.info("=" * 70)
        
        totals = summary.get("totals", {})
        self.logger.info("")
        self.logger.info("📊 OVERALL SUMMARY")
        self.logger.info(f"   Input directories:    {totals.get('input_directories', 0)}")
        self.logger.info(f"   Input files:          {totals.get('input_files', 0):,}")
        self.logger.info(f"   Input QA records:     {totals.get('input_records', 0):,}")
        self.logger.info(f"   Output QA records:    {totals.get('output_records', 0):,}")
        self.logger.info(f"   Duplicates removed:   {totals.get('duplicates_removed', 0):,}")
        self.logger.info(f"   Quality filtered:     {totals.get('quality_filtered', 0):,}")
        
        # Per-directory breakdown
        by_dir = summary.get("by_directory", {})
        if by_dir:
            self.logger.info("")
            self.logger.info("📁 PER-DIRECTORY BREAKDOWN")
            for dir_path, dir_stats in by_dir.items():
                self.logger.info(f"   {dir_path}:")
                self.logger.info(f"      Files: {dir_stats.get('files_processed', 0):,}")
                self.logger.info(f"      QA records: {dir_stats.get('records_written', 0):,}")
        
        # Top files
        top_files = summary.get("top_files", [])
        if top_files:
            self.logger.info("")
            self.logger.info("🏆 TOP 10 FILES BY QA COUNT")
            for i, item in enumerate(top_files[:10], 1):
                fname = Path(item["file"]).name
                self.logger.info(f"   {i:2}. {fname}: {item['qa_count']:,} QA pairs")
        
        # Deduplication stats
        dedup = summary.get("deduplication", {})
        if dedup:
            self.logger.info("")
            self.logger.info("🔄 DEDUPLICATION")
            self.logger.info(f"   Duplicates found:     {dedup.get('exact_and_fuzzy_duplicates', 0):,}")
            self.logger.info(f"   Duplicate rate:       {dedup.get('duplicate_rate_pct', 0)}%")
        
        self.logger.info("")
        self.logger.info(f"📄 Output file: {summary.get('output_file')}")
        self.logger.info(f"📋 Summary file: {self.summary_file}")
        self.logger.info("=" * 70)


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Consolidate QA JSONL files with deduplication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dirs", "-i",
        type=str,
        nargs="+",
        default=DEFAULT_CONFIG["input_dirs"],
        help="Input directories containing JSONL files (supports multiple directories)"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=DEFAULT_CONFIG["output_file"],
        help="Output consolidated JSONL file"
    )
    parser.add_argument(
        "--summary-file", "-s",
        type=str,
        default=DEFAULT_CONFIG["summary_file"],
        help="Output summary JSON file"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start fresh, ignore existing progress and output"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint (but don't delete existing output)"
    )
    parser.add_argument(
        "--no-fuzzy-dedup",
        action="store_true",
        help="Disable fuzzy duplicate detection (faster, only exact dedup)"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=DEFAULT_CONFIG["fuzzy_threshold"],
        help="Similarity threshold for fuzzy dedup (0-100, higher = stricter)"
    )
    parser.add_argument(
        "--min-question-length",
        type=int,
        default=DEFAULT_CONFIG["min_question_length"],
        help="Minimum question length to include"
    )
    parser.add_argument(
        "--min-answer-length",
        type=int,
        default=DEFAULT_CONFIG["min_answer_length"],
        help="Minimum answer length to include"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Save checkpoint every N files"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing output file (no processing)"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config["input_dirs"] = args.input_dirs
    config["output_file"] = args.output_file
    config["summary_file"] = args.summary_file
    config["enable_fuzzy_dedup"] = not args.no_fuzzy_dedup
    config["fuzzy_threshold"] = args.fuzzy_threshold
    config["min_question_length"] = args.min_question_length
    config["min_answer_length"] = args.min_answer_length
    config["batch_size"] = args.batch_size
    
    # Summary only mode
    if args.summary_only:
        print_existing_summary(Path(args.summary_file), Path(args.output_file))
        return
    
    # Run consolidation
    consolidator = QAConsolidator(config)
    result = consolidator.run(
        resume=not args.no_resume,
        restart=args.restart
    )
    
    if result.get("success"):
        totals = result.get("totals", {})
        print(f"\n✅ Success! Output: {result.get('output_file')}")
        print(f"   Total QA records: {totals.get('output_records', 0):,}")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def print_existing_summary(summary_file: Path, output_file: Path) -> None:
    """Print summary from existing summary file or count output file."""
    print("=" * 70)
    print("📊 QA CONSOLIDATION SUMMARY")
    print("=" * 70)
    
    # Try to load existing summary
    if summary_file.exists():
        try:
            with summary_file.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            
            totals = summary.get("totals", {})
            print(f"\n📄 Output file: {summary.get('output_file', output_file)}")
            print(f"   Run ID: {summary.get('run_id', 'N/A')}")
            print(f"   Completed: {summary.get('completed_at', 'N/A')}")
            print()
            print("📊 TOTALS")
            print(f"   Input directories:    {totals.get('input_directories', 0)}")
            print(f"   Input files:          {totals.get('input_files', 0):,}")
            print(f"   Input QA records:     {totals.get('input_records', 0):,}")
            print(f"   Output QA records:    {totals.get('output_records', 0):,}")
            print(f"   Duplicates removed:   {totals.get('duplicates_removed', 0):,}")
            print(f"   Quality filtered:     {totals.get('quality_filtered', 0):,}")
            
            # Per-directory
            by_dir = summary.get("by_directory", {})
            if by_dir:
                print()
                print("📁 PER-DIRECTORY")
                for dir_path, stats in by_dir.items():
                    print(f"   {dir_path}")
                    print(f"      Files: {stats.get('files_processed', 0):,}, QA: {stats.get('records_written', 0):,}")
            
            # Top files
            top_files = summary.get("top_files", [])
            if top_files:
                print()
                print("🏆 TOP 10 FILES")
                for i, item in enumerate(top_files[:10], 1):
                    fname = Path(item["file"]).name
                    print(f"   {i:2}. {fname}: {item['qa_count']:,}")
            
            print("=" * 70)
            return
        except Exception as e:
            print(f"Warning: Could not load summary file: {e}")
    
    # Fallback: count lines in output file
    if output_file.exists():
        line_count = 0
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        print(f"\n📄 Output file: {output_file}")
        print(f"   Total QA records: {line_count:,}")
        print("=" * 70)
    else:
        print(f"\n❌ No output file found at {output_file}")
        print("   Run consolidation first.")
        print("=" * 70)


if __name__ == "__main__":
    main()
