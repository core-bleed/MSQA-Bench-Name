"""
Gold Set Sampler for MSQA-Bench Human Annotation.

Creates stratified samples for human annotation with:
1. Balanced question type distribution
2. Quality score diversity
3. Document diversity
4. Export formats for annotation (CSV, JSONL)

Annotation Protocol:
- Answer Correctness: Yes / Partial / No
- Evidence Support: Yes / Partial / No  
- Evidence Quality: Good / Weak / Missing
- Question Clarity: Good / Ambiguous / Bad
"""

import csv
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AnnotationRecord:
    """A record prepared for human annotation."""
    # Identifiers
    annotation_id: str
    original_id: str
    doc_id: str
    
    # Content for annotation
    question: str
    answer: str
    context: str
    
    # Metadata for stratification tracking
    question_type: str
    quality_score: float
    
    # Annotation fields (to be filled by annotators)
    answer_correct: Optional[str] = None      # Yes / Partial / No
    evidence_support: Optional[str] = None    # Yes / Partial / No
    evidence_quality: Optional[str] = None    # Good / Weak / Missing
    question_clarity: Optional[str] = None    # Good / Ambiguous / Bad
    
    # Annotator info
    annotator_id: Optional[str] = None
    annotation_time: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationRecord':
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class AnnotationSummary:
    """Summary of annotation results."""
    total_annotated: int
    annotators: List[str]
    
    # Answer correctness
    answer_correct_yes: int = 0
    answer_correct_partial: int = 0
    answer_correct_no: int = 0
    
    # Evidence support
    evidence_yes: int = 0
    evidence_partial: int = 0
    evidence_no: int = 0
    
    # Evidence quality
    quality_good: int = 0
    quality_weak: int = 0
    quality_missing: int = 0
    
    # Question clarity
    clarity_good: int = 0
    clarity_ambiguous: int = 0
    clarity_bad: int = 0
    
    # By question type
    by_question_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Inter-annotator agreement (if multiple annotators)
    agreement_rate: Optional[float] = None
    cohens_kappa: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_annotated': self.total_annotated,
            'annotators': self.annotators,
            'answer_correctness': {
                'yes': self.answer_correct_yes,
                'partial': self.answer_correct_partial,
                'no': self.answer_correct_no,
                'accuracy': (
                    (self.answer_correct_yes + 0.5 * self.answer_correct_partial) 
                    / self.total_annotated
                ) if self.total_annotated > 0 else 0,
            },
            'evidence_support': {
                'yes': self.evidence_yes,
                'partial': self.evidence_partial,
                'no': self.evidence_no,
            },
            'evidence_quality': {
                'good': self.quality_good,
                'weak': self.quality_weak,
                'missing': self.quality_missing,
            },
            'question_clarity': {
                'good': self.clarity_good,
                'ambiguous': self.clarity_ambiguous,
                'bad': self.clarity_bad,
            },
            'by_question_type': self.by_question_type,
            'inter_annotator_agreement': self.agreement_rate,
            'cohens_kappa': self.cohens_kappa,
        }


class GoldSetSampler:
    """
    Sample QA pairs for human annotation.
    
    Creates stratified samples ensuring:
    - Representation of all question types
    - Range of quality scores
    - Document diversity
    """
    
    def __init__(
        self,
        target_size: int = 500,
        question_type_field: str = "question_type",
        quality_score_field: str = "quality_score",
        doc_id_field: str = "doc_id",
        random_seed: int = 42,
    ):
        """
        Initialize sampler.
        
        Args:
            target_size: Target number of samples
            question_type_field: Field for question type
            quality_score_field: Field for quality score
            doc_id_field: Field for document ID
            random_seed: Random seed for reproducibility
        """
        self.target_size = target_size
        self.question_type_field = question_type_field
        self.quality_score_field = quality_score_field
        self.doc_id_field = doc_id_field
        self.random_seed = random_seed
        
        random.seed(random_seed)
    
    def sample(
        self,
        records: List[Dict[str, Any]],
        stratify_by: List[str] = None,
    ) -> List[AnnotationRecord]:
        """
        Sample records for annotation.
        
        Args:
            records: All QA records
            stratify_by: Fields to stratify by (default: question_type)
            
        Returns:
            List of AnnotationRecord objects
        """
        if stratify_by is None:
            stratify_by = [self.question_type_field]
        
        # Group records by stratification key
        groups: Dict[str, List[Dict]] = defaultdict(list)
        
        for record in records:
            key_parts = [str(record.get(field, 'unknown')) for field in stratify_by]
            key = '|'.join(key_parts)
            groups[key].append(record)
        
        # Calculate samples per group
        num_groups = len(groups)
        if num_groups == 0:
            return []
        
        base_per_group = self.target_size // num_groups
        remainder = self.target_size % num_groups
        
        # Sample from each group
        sampled = []
        annotation_id = 0
        
        for i, (key, group_records) in enumerate(groups.items()):
            # Add one more to first 'remainder' groups
            n_samples = base_per_group + (1 if i < remainder else 0)
            
            # Don't sample more than available
            n_samples = min(n_samples, len(group_records))
            
            # Random sample
            group_sample = random.sample(group_records, n_samples)
            
            # Convert to AnnotationRecord
            for record in group_sample:
                annotation_id += 1
                
                ann_record = AnnotationRecord(
                    annotation_id=f"ann_{annotation_id:05d}",
                    original_id=record.get('id', ''),
                    doc_id=record.get(self.doc_id_field, ''),
                    question=record.get('question', ''),
                    answer=record.get('answer', ''),
                    context=record.get('context', ''),
                    question_type=record.get(self.question_type_field, 'unknown'),
                    quality_score=record.get(self.quality_score_field, 0.0),
                )
                sampled.append(ann_record)
        
        # Shuffle final sample
        random.shuffle(sampled)
        
        logger.info(f"Sampled {len(sampled)} records from {num_groups} groups")
        
        # Log distribution
        type_dist = defaultdict(int)
        for r in sampled:
            type_dist[r.question_type] += 1
        logger.info(f"Question type distribution: {dict(type_dist)}")
        
        return sampled
    
    def export_for_annotation(
        self,
        records: List[AnnotationRecord],
        output_path: Path,
        format: str = "csv",
    ) -> None:
        """
        Export sampled records for annotation.
        
        Args:
            records: List of AnnotationRecord
            output_path: Output file path
            format: "csv" or "jsonl"
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            self._export_csv(records, output_path)
        else:
            self._export_jsonl(records, output_path)
        
        logger.info(f"Exported {len(records)} records to {output_path}")
    
    def _export_csv(self, records: List[AnnotationRecord], path: Path) -> None:
        """Export to CSV format."""
        fieldnames = [
            'annotation_id', 'question', 'answer', 'context',
            'question_type', 'quality_score',
            'answer_correct', 'evidence_support', 'evidence_quality', 
            'question_clarity', 'annotator_id', 'notes',
        ]
        
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                row = {
                    'annotation_id': record.annotation_id,
                    'question': record.question,
                    'answer': record.answer,
                    'context': record.context[:500] + '...' if len(record.context) > 500 else record.context,
                    'question_type': record.question_type,
                    'quality_score': f"{record.quality_score:.2f}",
                    'answer_correct': '',
                    'evidence_support': '',
                    'evidence_quality': '',
                    'question_clarity': '',
                    'annotator_id': '',
                    'notes': '',
                }
                writer.writerow(row)
    
    def _export_jsonl(self, records: List[AnnotationRecord], path: Path) -> None:
        """Export to JSONL format."""
        with path.open('w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
    
    def load_annotations(self, input_path: Path) -> List[AnnotationRecord]:
        """
        Load completed annotations.
        
        Args:
            input_path: Path to annotation file (CSV or JSONL)
            
        Returns:
            List of AnnotationRecord with annotations filled in
        """
        records = []
        
        if input_path.suffix == '.csv':
            with input_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    record = AnnotationRecord(
                        annotation_id=row.get('annotation_id', ''),
                        original_id=row.get('original_id', ''),
                        doc_id=row.get('doc_id', ''),
                        question=row.get('question', ''),
                        answer=row.get('answer', ''),
                        context=row.get('context', ''),
                        question_type=row.get('question_type', ''),
                        quality_score=float(row.get('quality_score', 0)),
                        answer_correct=row.get('answer_correct') or None,
                        evidence_support=row.get('evidence_support') or None,
                        evidence_quality=row.get('evidence_quality') or None,
                        question_clarity=row.get('question_clarity') or None,
                        annotator_id=row.get('annotator_id') or None,
                        notes=row.get('notes') or None,
                    )
                    records.append(record)
        else:
            with input_path.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        records.append(AnnotationRecord.from_dict(data))
        
        logger.info(f"Loaded {len(records)} annotations from {input_path}")
        return records
    
    def compute_summary(
        self,
        annotations: List[AnnotationRecord],
    ) -> AnnotationSummary:
        """
        Compute summary statistics from annotations.
        
        Args:
            annotations: List of completed annotations
            
        Returns:
            AnnotationSummary with statistics
        """
        summary = AnnotationSummary(
            total_annotated=len(annotations),
            annotators=list(set(a.annotator_id for a in annotations if a.annotator_id)),
        )
        
        by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for ann in annotations:
            # Answer correctness
            if ann.answer_correct:
                val = ann.answer_correct.lower()
                if 'yes' in val:
                    summary.answer_correct_yes += 1
                elif 'partial' in val:
                    summary.answer_correct_partial += 1
                elif 'no' in val:
                    summary.answer_correct_no += 1
            
            # Evidence support
            if ann.evidence_support:
                val = ann.evidence_support.lower()
                if 'yes' in val:
                    summary.evidence_yes += 1
                elif 'partial' in val:
                    summary.evidence_partial += 1
                elif 'no' in val:
                    summary.evidence_no += 1
            
            # Evidence quality
            if ann.evidence_quality:
                val = ann.evidence_quality.lower()
                if 'good' in val:
                    summary.quality_good += 1
                elif 'weak' in val:
                    summary.quality_weak += 1
                elif 'missing' in val:
                    summary.quality_missing += 1
            
            # Question clarity
            if ann.question_clarity:
                val = ann.question_clarity.lower()
                if 'good' in val:
                    summary.clarity_good += 1
                elif 'ambiguous' in val:
                    summary.clarity_ambiguous += 1
                elif 'bad' in val:
                    summary.clarity_bad += 1
            
            # By question type
            qtype = ann.question_type
            if ann.answer_correct:
                by_type[qtype][f"answer_{ann.answer_correct.lower()}"] += 1
        
        summary.by_question_type = {k: dict(v) for k, v in by_type.items()}
        
        return summary
    
    def compute_inter_annotator_agreement(
        self,
        annotations_a: List[AnnotationRecord],
        annotations_b: List[AnnotationRecord],
        field: str = "answer_correct",
    ) -> Tuple[float, float]:
        """
        Compute inter-annotator agreement between two annotators.
        
        Args:
            annotations_a: Annotations from annotator A
            annotations_b: Annotations from annotator B
            field: Field to compute agreement on
            
        Returns:
            Tuple of (agreement_rate, cohen's_kappa)
        """
        # Match by annotation_id
        a_by_id = {a.annotation_id: a for a in annotations_a}
        b_by_id = {b.annotation_id: b for b in annotations_b}
        
        common_ids = set(a_by_id.keys()) & set(b_by_id.keys())
        
        if not common_ids:
            return 0.0, 0.0
        
        agreements = 0
        total = len(common_ids)
        
        # For Cohen's Kappa
        labels_a = []
        labels_b = []
        
        for ann_id in common_ids:
            val_a = getattr(a_by_id[ann_id], field, None)
            val_b = getattr(b_by_id[ann_id], field, None)
            
            if val_a and val_b:
                val_a = val_a.lower().strip()
                val_b = val_b.lower().strip()
                
                labels_a.append(val_a)
                labels_b.append(val_b)
                
                if val_a == val_b:
                    agreements += 1
        
        # Simple agreement rate
        agreement_rate = agreements / total if total > 0 else 0.0
        
        # Cohen's Kappa
        kappa = self._compute_cohens_kappa(labels_a, labels_b)
        
        return agreement_rate, kappa
    
    def _compute_cohens_kappa(
        self, 
        labels_a: List[str], 
        labels_b: List[str]
    ) -> float:
        """Compute Cohen's Kappa coefficient."""
        if len(labels_a) != len(labels_b) or not labels_a:
            return 0.0
        
        n = len(labels_a)
        
        # Get all unique labels
        all_labels = list(set(labels_a) | set(labels_b))
        
        # Count agreements
        observed_agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
        
        # Expected agreement by chance
        expected_agreement = 0.0
        for label in all_labels:
            p_a = labels_a.count(label) / n
            p_b = labels_b.count(label) / n
            expected_agreement += p_a * p_b
        
        # Kappa
        if expected_agreement == 1.0:
            return 1.0
        
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa


def sample_for_annotation(
    input_file: Path,
    output_file: Path,
    target_size: int = 500,
    format: str = "csv",
) -> List[AnnotationRecord]:
    """
    Convenience function to sample QA records for annotation.
    
    Args:
        input_file: Input JSONL file with QA records
        output_file: Output file for annotation
        target_size: Number of samples
        format: Output format ("csv" or "jsonl")
        
    Returns:
        List of sampled AnnotationRecord
    """
    # Load records
    records = []
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    logger.info(f"Loaded {len(records)} records from {input_file}")
    
    # Sample
    sampler = GoldSetSampler(target_size=target_size)
    sampled = sampler.sample(records)
    
    # Export
    sampler.export_for_annotation(sampled, output_file, format=format)
    
    return sampled


# Annotation guidelines template
ANNOTATION_GUIDELINES = """
# MSQA-Bench Human Annotation Guidelines

## Task
Evaluate the quality of automatically generated question-answer pairs from mass spectrometry literature.

## Annotation Fields

### 1. Answer Correctness
Is the answer factually correct?
- **Yes**: Answer is fully correct and accurate
- **Partial**: Answer is partially correct but has minor errors or omissions  
- **No**: Answer is incorrect or contains significant errors

### 2. Evidence Support
Is the answer supported by the provided context?
- **Yes**: Answer can be fully derived from the context
- **Partial**: Answer is partially supported, some information may be inferred
- **No**: Answer contains claims not supported by the context

### 3. Evidence Quality
How well does the context support answering the question?
- **Good**: Context contains clear, direct evidence for the answer
- **Weak**: Context contains related information but evidence is indirect
- **Missing**: Context does not contain relevant information

### 4. Question Clarity
Is the question clear and well-formed?
- **Good**: Question is clear, specific, and answerable
- **Ambiguous**: Question is unclear or could be interpreted multiple ways
- **Bad**: Question is poorly formed, too vague, or unanswerable

## Tips
- Focus on the relationship between question, answer, and context
- Consider domain knowledge when evaluating correctness
- Note any issues or observations in the notes field
- If unsure, default to the more conservative rating

## Examples

### Good Example
Question: "What volatile compounds were identified from A. pullulans?"
Answer: "Four VOCs were identified: ethanol, 2-methyl-1-propanol, 3-methyl-1-butanol, and 2-phenylethanol."
Context: [Contains this exact information]
Rating: Answer=Yes, Evidence=Yes, Quality=Good, Clarity=Good

### Partial Example
Question: "How effective is the method?"
Answer: "The method shows 95% accuracy in compound identification."
Context: [Mentions accuracy but doesn't give exact percentage]
Rating: Answer=Partial, Evidence=Partial, Quality=Weak, Clarity=Ambiguous
"""


def generate_annotation_guidelines(output_path: Path) -> None:
    """Generate annotation guidelines document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ANNOTATION_GUIDELINES)
    logger.info(f"Generated annotation guidelines: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample QA for human annotation")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--size", "-n", type=int, default=500, help="Sample size")
    parser.add_argument("--format", "-f", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--guidelines", "-g", help="Output path for guidelines")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    sampled = sample_for_annotation(
        Path(args.input),
        Path(args.output),
        target_size=args.size,
        format=args.format,
    )
    
    print(f"\nSampled {len(sampled)} records for annotation")
    print(f"Output: {args.output}")
    
    if args.guidelines:
        generate_annotation_guidelines(Path(args.guidelines))
        print(f"Guidelines: {args.guidelines}")
