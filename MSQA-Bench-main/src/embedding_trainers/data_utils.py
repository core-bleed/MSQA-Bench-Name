"""
Data utilities for streaming embedding fine-tuning.

Provides memory-efficient data loading for large JSONL files (1M+ records)
with hash-based splitting and answer cleaning.
"""

import hashlib
import json
import re
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset
from sentence_transformers import InputExample


logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and filtering."""
    min_question_length: int = 10
    max_question_length: int = 256
    min_answer_length: int = 20
    max_answer_length: int = 512
    clean_answers: bool = True
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    subset_size: Optional[int] = None


def clean_answer(text: str, max_len: int = 512) -> str:
    """
    Clean noisy PDF-extracted answers.
    
    Removes:
    - Citation markers like [1], [2,3], [1-5]
    - URLs
    - Figure/Table references
    - Excessive whitespace
    - DOI patterns
    
    Args:
        text: Raw answer text
        max_len: Maximum length to truncate to
        
    Returns:
        Cleaned answer text
    """
    if not text:
        return ""
    
    # Remove citation markers like [1], [2,3], [1-5], (1), (2,3)
    text = re.sub(r'\[\d+(?:[,\-–]\s*\d+)*\]', '', text)
    text = re.sub(r'\(\d+(?:[,\-–]\s*\d+)*\)', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove DOI patterns
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'10\.\d{4,}/\S+', '', text)
    
    # Remove figure/table references
    text = re.sub(r'(?:Fig\.|Figure|Table|Supplementary\s+(?:Fig|Table|Material))\s*\d+[A-Za-z]?', 
                  '', text, flags=re.IGNORECASE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Normalize whitespace (collapse multiple spaces, remove leading/trailing)
    text = ' '.join(text.split())
    
    # Truncate to max length at word boundary
    if len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0]
    
    return text.strip()


def clean_question(text: str, max_len: int = 256) -> str:
    """
    Clean question text.
    
    Args:
        text: Raw question text
        max_len: Maximum length
        
    Returns:
        Cleaned question text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Truncate
    if len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0]
        # Ensure it ends with question mark if truncated
        if not text.endswith('?'):
            text = text + '?'
    
    return text.strip()


def get_split(record_id: str, train_ratio: float = 0.85, val_ratio: float = 0.10) -> str:
    """
    Deterministic hash-based split assignment.
    
    Uses MD5 hash of record ID to assign to train/val/test split.
    This is deterministic and doesn't require loading all data.
    
    Args:
        record_id: Unique identifier for the record
        train_ratio: Proportion for training (default 0.85)
        val_ratio: Proportion for validation (default 0.10)
        
    Returns:
        Split name: "train", "val", or "test"
    """
    # Get hash value as integer
    h = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    
    train_threshold = int(train_ratio * 100)
    val_threshold = train_threshold + int(val_ratio * 100)
    
    if h < train_threshold:
        return "train"
    elif h < val_threshold:
        return "val"
    else:
        return "test"


class StreamingQADataset(IterableDataset):
    """
    Memory-efficient streaming dataset for Q&A pairs from JSONL.
    
    Features:
    - Streams data line-by-line (no full file load)
    - Hash-based deterministic train/val/test split
    - Length filtering for questions and answers
    - Optional answer cleaning
    - Support for subset training (first N records)
    - Resume capability via skip_records parameter
    
    Example:
        >>> dataset = StreamingQADataset(
        ...     "consolidated_qa.jsonl",
        ...     split="train",
        ...     config=DataConfig(subset_size=200000)
        ... )
        >>> for example in dataset:
        ...     print(example.texts)
    """
    
    def __init__(
        self,
        jsonl_path: str,
        split: str = "train",
        config: Optional[DataConfig] = None,
        skip_records: int = 0,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            jsonl_path: Path to JSONL file
            split: Which split to use ("train", "val", "test")
            config: Data configuration (filtering, cleaning, etc.)
            skip_records: Number of records to skip (for resume)
        """
        self.jsonl_path = Path(jsonl_path)
        self.split = split
        self.config = config or DataConfig()
        self.skip_records = skip_records
        
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")
        
        # Validate split
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Must be train, val, or test")
        
        logger.info(f"StreamingQADataset initialized: {jsonl_path}, split={split}")
    
    def _passes_filter(self, question: str, answer: str) -> bool:
        """Check if Q&A pair passes length filters."""
        q_len = len(question)
        a_len = len(answer)
        
        return (
            self.config.min_question_length <= q_len <= self.config.max_question_length
            and self.config.min_answer_length <= a_len <= self.config.max_answer_length
        )
    
    def __iter__(self) -> Iterator[InputExample]:
        """Iterate over Q&A pairs in the split."""
        records_yielded = 0
        records_skipped = 0
        records_filtered = 0
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
                
                # Check required fields
                if 'question' not in record or 'answer' not in record:
                    continue
                
                # Get record ID (use line number as fallback)
                record_id = record.get('id', str(line_num))
                
                # Hash-based split
                record_split = get_split(
                    record_id,
                    self.config.train_ratio,
                    self.config.val_ratio
                )
                if record_split != self.split:
                    continue
                
                # Skip records for resume
                if records_skipped < self.skip_records:
                    records_skipped += 1
                    continue
                
                # Get and clean text
                question = record['question'].strip()
                answer = record['answer'].strip()
                
                if self.config.clean_answers:
                    answer = clean_answer(answer, self.config.max_answer_length)
                    question = clean_question(question, self.config.max_question_length)
                
                # Apply length filters
                if not self._passes_filter(question, answer):
                    records_filtered += 1
                    continue
                
                # Yield the example
                yield InputExample(
                    guid=record_id,
                    texts=[question, answer]
                )
                
                records_yielded += 1
                
                # Check subset limit
                if self.config.subset_size and records_yielded >= self.config.subset_size:
                    logger.info(f"Reached subset limit of {self.config.subset_size}")
                    break
        
        logger.info(
            f"StreamingQADataset [{self.split}]: "
            f"yielded={records_yielded}, filtered={records_filtered}, skipped={records_skipped}"
        )


class ResumableStreamingDataset(StreamingQADataset):
    """
    Extended streaming dataset with exact position tracking for resume.
    
    Tracks byte offset in file for exact resume capability.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_byte_offset = 0
        self.current_record_count = 0
    
    def __iter__(self) -> Iterator[Tuple[InputExample, Dict[str, Any]]]:
        """
        Iterate with position tracking.
        
        Yields:
            Tuple of (InputExample, position_info)
            where position_info contains byte_offset and record_count
        """
        records_yielded = 0
        records_skipped = 0
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            # Skip to byte offset if resuming
            if self.skip_records > 0:
                logger.info(f"Skipping {self.skip_records} records for resume...")
            
            for line_num, line in enumerate(f, 1):
                byte_offset = f.tell()
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if 'question' not in record or 'answer' not in record:
                    continue
                
                record_id = record.get('id', str(line_num))
                record_split = get_split(
                    record_id,
                    self.config.train_ratio,
                    self.config.val_ratio
                )
                
                if record_split != self.split:
                    continue
                
                # Skip for resume
                if records_skipped < self.skip_records:
                    records_skipped += 1
                    continue
                
                question = record['question'].strip()
                answer = record['answer'].strip()
                
                if self.config.clean_answers:
                    answer = clean_answer(answer, self.config.max_answer_length)
                    question = clean_question(question, self.config.max_question_length)
                
                if not self._passes_filter(question, answer):
                    continue
                
                records_yielded += 1
                self.current_byte_offset = byte_offset
                self.current_record_count = records_yielded
                
                example = InputExample(guid=record_id, texts=[question, answer])
                position_info = {
                    'byte_offset': byte_offset,
                    'record_count': records_yielded,
                    'line_num': line_num,
                }
                
                yield example, position_info
                
                if self.config.subset_size and records_yielded >= self.config.subset_size:
                    break


def count_records_in_split(jsonl_path: str, split: str, config: Optional[DataConfig] = None) -> int:
    """
    Count total records in a split without loading all data.
    
    This is useful for calculating steps per epoch.
    
    Args:
        jsonl_path: Path to JSONL file
        split: Which split to count
        config: Data configuration
        
    Returns:
        Number of records in the split
    """
    config = config or DataConfig()
    count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if 'question' not in record or 'answer' not in record:
                continue
            
            record_id = record.get('id', str(line_num))
            record_split = get_split(record_id, config.train_ratio, config.val_ratio)
            
            if record_split != split:
                continue
            
            # Apply length filter check
            question = record['question'].strip()
            answer = record['answer'].strip()
            
            if config.clean_answers:
                answer = clean_answer(answer, config.max_answer_length)
                question = clean_question(question, config.max_question_length)
            
            q_len = len(question)
            a_len = len(answer)
            
            if not (config.min_question_length <= q_len <= config.max_question_length):
                continue
            if not (config.min_answer_length <= a_len <= config.max_answer_length):
                continue
            
            count += 1
            
            if config.subset_size and count >= config.subset_size:
                break
    
    return count


def load_split_samples(
    jsonl_path: str,
    split: str,
    sample_size: int,
    config: Optional[DataConfig] = None
) -> Dict[str, Any]:
    """
    Load a sample of records from a split into memory.
    
    Used for building evaluation corpus.
    
    Args:
        jsonl_path: Path to JSONL file
        split: Which split to sample from
        sample_size: Maximum number of records to load
        config: Data configuration
        
    Returns:
        Dict with 'queries', 'corpus', 'relevant_docs' for IR evaluation
    """
    config = config or DataConfig()
    
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if count >= sample_size:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if 'question' not in record or 'answer' not in record:
                continue
            
            record_id = record.get('id', str(line_num))
            record_split = get_split(record_id, config.train_ratio, config.val_ratio)
            
            if record_split != split:
                continue
            
            question = record['question'].strip()
            answer = record['answer'].strip()
            
            if config.clean_answers:
                answer = clean_answer(answer, config.max_answer_length)
                question = clean_question(question, config.max_question_length)
            
            # Check length
            if not (config.min_question_length <= len(question) <= config.max_question_length):
                continue
            if not (config.min_answer_length <= len(answer) <= config.max_answer_length):
                continue
            
            qid = f"q_{record_id}"
            cid = f"c_{record_id}"
            
            queries[qid] = question
            corpus[cid] = answer
            relevant_docs[qid] = {cid}
            
            count += 1
    
    logger.info(f"Loaded {count} samples from {split} split for evaluation")
    
    return {
        'queries': queries,
        'corpus': corpus,
        'relevant_docs': relevant_docs
    }
