"""
Split Generator for MSQA-Bench.

Generates train/val/test splits at the DOCUMENT level to prevent data leakage.
All QA pairs from the same document go into the same split.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splits."""
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    seed: int = 42  # For reproducibility (used in hash)
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class SplitGenerator:
    """
    Generate document-level train/val/test splits.
    
    Uses deterministic hashing so the same document always goes to the same split,
    regardless of when or how many times the split is computed.
    """
    
    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize split generator.
        
        Args:
            config: Split configuration with ratios
        """
        self.config = config or SplitConfig()
    
    def get_split(self, doc_id: str) -> str:
        """
        Determine the split for a document.
        
        Uses MD5 hash of doc_id for deterministic assignment.
        
        Args:
            doc_id: Unique document identifier
            
        Returns:
            Split name: "train", "val", or "test"
        """
        # Create deterministic hash
        hash_input = f"{doc_id}:{self.config.seed}".encode('utf-8')
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16) % 100
        
        train_threshold = int(self.config.train_ratio * 100)
        val_threshold = train_threshold + int(self.config.val_ratio * 100)
        
        if hash_val < train_threshold:
            return "train"
        elif hash_val < val_threshold:
            return "val"
        else:
            return "test"
    
    def assign_splits(
        self,
        records: List[Dict[str, Any]],
        doc_id_field: str = "doc_id",
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Assign splits to a list of records.
        
        Args:
            records: List of QA records
            doc_id_field: Field name containing the document ID
            
        Returns:
            Tuple of (train_records, val_records, test_records)
        """
        train_records = []
        val_records = []
        test_records = []
        
        for record in records:
            # Get doc_id, falling back to file_name stem
            doc_id = record.get(doc_id_field)
            if not doc_id:
                file_name = record.get('file_name', '')
                doc_id = Path(file_name).stem if file_name else record.get('id', '')
            
            split = self.get_split(doc_id)
            record['split'] = split
            
            if split == "train":
                train_records.append(record)
            elif split == "val":
                val_records.append(record)
            else:
                test_records.append(record)
        
        return train_records, val_records, test_records
    
    def get_split_statistics(
        self,
        records: List[Dict[str, Any]],
        doc_id_field: str = "doc_id",
    ) -> Dict[str, Any]:
        """
        Compute statistics for the splits.
        
        Args:
            records: List of QA records
            doc_id_field: Field name containing the document ID
            
        Returns:
            Dictionary with split statistics
        """
        # Group by document
        doc_to_records: Dict[str, List[Dict]] = defaultdict(list)
        for record in records:
            doc_id = record.get(doc_id_field)
            if not doc_id:
                file_name = record.get('file_name', '')
                doc_id = Path(file_name).stem if file_name else record.get('id', '')
            doc_to_records[doc_id].append(record)
        
        # Count documents and records per split
        stats = {
            "train": {"documents": 0, "qa_pairs": 0},
            "val": {"documents": 0, "qa_pairs": 0},
            "test": {"documents": 0, "qa_pairs": 0},
        }
        
        for doc_id, doc_records in doc_to_records.items():
            split = self.get_split(doc_id)
            stats[split]["documents"] += 1
            stats[split]["qa_pairs"] += len(doc_records)
        
        # Add totals and percentages
        total_docs = sum(s["documents"] for s in stats.values())
        total_pairs = sum(s["qa_pairs"] for s in stats.values())
        
        for split_name, split_stats in stats.items():
            split_stats["doc_percent"] = (
                split_stats["documents"] / total_docs * 100 if total_docs > 0 else 0
            )
            split_stats["pair_percent"] = (
                split_stats["qa_pairs"] / total_pairs * 100 if total_pairs > 0 else 0
            )
        
        stats["total"] = {
            "documents": total_docs,
            "qa_pairs": total_pairs,
        }
        
        return stats


def get_document_split(
    doc_id: str,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> str:
    """
    Convenience function to get split for a document.
    
    Args:
        doc_id: Document identifier
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        Split name: "train", "val", or "test"
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    generator = SplitGenerator(config)
    return generator.get_split(doc_id)


def split_qa_file(
    input_file: Path,
    output_dir: Path,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    doc_id_field: str = "doc_id",
) -> Dict[str, Any]:
    """
    Split a QA JSONL file into train/val/test files.
    
    Args:
        input_file: Input JSONL file
        output_dir: Directory for output split files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        doc_id_field: Field containing document ID
        
    Returns:
        Split statistics
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    generator = SplitGenerator(config)
    
    # Load all records
    records = []
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    logger.info(f"Loaded {len(records)} records from {input_file}")
    
    # Assign splits
    train_records, val_records, test_records = generator.assign_splits(
        records, doc_id_field
    )
    
    # Get statistics
    stats = generator.get_split_statistics(records, doc_id_field)
    
    # Write split files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        output_file = output_dir / f"{split_name}.jsonl"
        with output_file.open('w', encoding='utf-8') as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Wrote {len(split_records)} records to {output_file}")
    
    # Write statistics
    stats_file = output_dir / "split_statistics.json"
    with stats_file.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def verify_no_leakage(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    doc_id_field: str = "doc_id",
) -> bool:
    """
    Verify that there is no document overlap between splits.
    
    Args:
        train_file: Training split JSONL
        val_file: Validation split JSONL
        test_file: Test split JSONL
        doc_id_field: Field containing document ID
        
    Returns:
        True if no leakage detected
    """
    def load_doc_ids(file_path: Path) -> set:
        doc_ids = set()
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    doc_id = record.get(doc_id_field)
                    if not doc_id:
                        file_name = record.get('file_name', '')
                        doc_id = Path(file_name).stem if file_name else record.get('id', '')
                    doc_ids.add(doc_id)
        return doc_ids
    
    train_docs = load_doc_ids(train_file)
    val_docs = load_doc_ids(val_file)
    test_docs = load_doc_ids(test_file)
    
    # Check overlaps
    train_val_overlap = train_docs & val_docs
    train_test_overlap = train_docs & test_docs
    val_test_overlap = val_docs & test_docs
    
    has_leakage = False
    
    if train_val_overlap:
        logger.error(f"Train-Val overlap: {len(train_val_overlap)} documents")
        has_leakage = True
    
    if train_test_overlap:
        logger.error(f"Train-Test overlap: {len(train_test_overlap)} documents")
        has_leakage = True
    
    if val_test_overlap:
        logger.error(f"Val-Test overlap: {len(val_test_overlap)} documents")
        has_leakage = True
    
    if not has_leakage:
        logger.info("No data leakage detected!")
        logger.info(f"  Train: {len(train_docs)} documents")
        logger.info(f"  Val: {len(val_docs)} documents")
        logger.info(f"  Test: {len(test_docs)} documents")
    
    return not has_leakage


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate document-level splits")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--verify", action="store_true", help="Verify no leakage")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path(args.output_dir)
    
    if args.verify:
        # Verify existing splits
        verify_no_leakage(
            output_dir / "train.jsonl",
            output_dir / "val.jsonl", 
            output_dir / "test.jsonl",
        )
    else:
        # Generate splits
        stats = split_qa_file(
            Path(args.input),
            output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        
        print("\nSplit Statistics:")
        for split_name in ["train", "val", "test"]:
            s = stats[split_name]
            print(f"  {split_name}: {s['documents']} docs ({s['doc_percent']:.1f}%), "
                  f"{s['qa_pairs']} QA pairs ({s['pair_percent']:.1f}%)")
        print(f"  Total: {stats['total']['documents']} docs, {stats['total']['qa_pairs']} QA pairs")
