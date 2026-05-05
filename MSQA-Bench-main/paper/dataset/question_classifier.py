"""
Question Type Classifier for MSQA-Bench.

Classifies questions into types:
- factual: What/Who/When questions with entity answers
- method: How questions about procedures and methods
- definition: "What is X" questions with concept explanations
- comparison: Questions comparing entities or concepts
- numeric: Questions expecting numbers/measurements
- causal: Why questions about causes and effects
- unknown: Questions that don't fit other categories
"""

import re
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, cast
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

_CLASSIFIER: Optional["QuestionClassifier"] = None


def _classifier_pool_init() -> None:
    global _CLASSIFIER  # noqa: PLW0603
    _CLASSIFIER = QuestionClassifier()


def _classify_lines_block(lines: List[str]) -> List[str]:
    clf = cast(QuestionClassifier, _CLASSIFIER)
    out: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        question = record.get("question", "")
        answer = record.get("answer")
        result = clf.classify(question, answer)
        record["question_type"] = result.question_type
        record["question_type_confidence"] = result.confidence
        if result.secondary_type:
            record["question_type_secondary"] = result.secondary_type
        out.append(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def _qc_split_lines(lines: List[str], num_parts: int) -> List[List[str]]:
    if not lines or num_parts <= 1:
        return [lines]
    n = len(lines)
    base, extra = divmod(n, num_parts)
    chunks: List[List[str]] = []
    start = 0
    for i in range(num_parts):
        sz = base + (1 if i < extra else 0)
        part = lines[start : start + sz]
        if part:
            chunks.append(part)
        start += sz
    return chunks


@dataclass
class ClassificationResult:
    """Result of question classification."""
    question_type: str
    confidence: float
    secondary_type: Optional[str] = None
    features: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question_type': self.question_type,
            'confidence': self.confidence,
            'secondary_type': self.secondary_type,
            'features': self.features or {},
        }


class QuestionClassifier:
    """
    Rule-based question type classifier.
    
    Designed for scientific/technical QA where questions follow
    predictable patterns.
    """
    
    # Question type patterns (order matters - more specific patterns first)
    TYPE_PATTERNS = {
        'comparison': [
            r'\b(compare|comparison|differ(?:ence|ent)?|versus|vs\.?|contrast)\b',
            r'\b(better|worse|more|less|higher|lower) than\b',
            r'\b(similar|different) (?:to|from)\b',
            r'\b(advantage|disadvantage)s?\b.*\b(over|compared)\b',
            r'\bwhich (?:is|are) (?:better|more|less)\b',
        ],
        'numeric': [
            r'\bhow (?:many|much|often|long|far|high|low)\b',
            r'\bwhat (?:is|are) the (?:number|amount|quantity|rate|percentage|proportion)\b',
            r'\bwhat percentage\b',
            r'\bhow (?:\w+ )?(?:times|percent)\b',
            r'\bquantif(?:y|ied)\b',
            r'\bmeasure(?:d|ment)?\b.*\bvalue\b',
        ],
        'method': [
            r'^how (?:do|does|can|could|should|would|to|is|are)\b',
            r'\b(?:method|procedure|process|technique|approach|protocol|workflow)\b.*\b(?:used|applied|performed)\b',
            r'\bsteps?\b.*\b(?:to|for|in)\b',
            r'\bhow (?:is|are|was|were)\b.*\b(?:done|performed|conducted|carried out|achieved)\b',
            r'\bdescribe the (?:method|process|procedure)\b',
            r'\bwhat (?:is|are) the (?:steps?|procedure|method)\b',
        ],
        'definition': [
            r'^what (?:is|are) (?:a |an |the )?(?!the (?:difference|result|effect|cause|method|step))',
            r'\bdefin(?:e|ition|ed)\b',
            r'\bwhat does\b.*\bmean\b',
            r'\bexplain (?:what|the term|the concept)\b',
            r'^what (?:is|are) meant by\b',
            r'\brefer(?:s|red)? to\b.*\bwhat\b',
        ],
        'causal': [
            r'^why\b',
            r'\bcause[sd]?\b.*\bwhat\b',
            r'\bwhat (?:cause[sd]?|leads? to|results? in)\b',
            r'\breason(?:s)?\b.*\b(?:for|why)\b',
            r'\bexplain why\b',
            r'\bwhat (?:is|are) the (?:cause|reason|effect|consequence|result)\b',
            r'\bhow (?:does|do)\b.*\b(?:affect|influence|impact)\b',
        ],
        'factual': [
            r'^(?:what|which|who|when|where)\b',
            r'\bidentify\b',
            r'\bname\b.*\b(?:the|a|an)\b',
            r'\blist\b',
            r'\bwhat type\b',
            r'\bwhich (?:\w+ )?(?:is|are|was|were)\b',
        ],
    }
    
    # Scientific domain keywords that indicate specific question types
    DOMAIN_INDICATORS = {
        'method': [
            'mass spectrometry', 'chromatography', 'analysis', 'detection',
            'separation', 'ionization', 'fragmentation', 'sequencing',
            'acquisition', 'processing', 'calibration', 'quantification',
            'extraction', 'purification', 'synthesis', 'experiment',
        ],
        'definition': [
            'term', 'concept', 'meaning', 'defined as', 'refers to',
            'known as', 'called', 'acronym',
        ],
        'numeric': [
            'm/z', 'Da', 'kDa', 'ppm', 'concentration', 'abundance',
            'intensity', 'ratio', 'score', 'threshold', 'limit',
            'sensitivity', 'specificity', 'accuracy', 'precision',
        ],
    }
    
    def __init__(self):
        """Initialize the classifier with compiled patterns."""
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for qtype, patterns in self.TYPE_PATTERNS.items():
            self.compiled_patterns[qtype] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def classify(self, question: str, answer: Optional[str] = None) -> ClassificationResult:
        """
        Classify a question into a type.
        
        Args:
            question: The question text
            answer: Optional answer text (can help with classification)
            
        Returns:
            ClassificationResult with type and confidence
        """
        if not question:
            return ClassificationResult(
                question_type='unknown',
                confidence=0.0,
                features={'error': 'empty_question'},
            )
        
        question_clean = question.strip()
        question_lower = question_clean.lower()
        
        # Extract features
        features = self._extract_features(question_clean, answer)
        
        # Score each type
        type_scores: Dict[str, float] = {}
        
        for qtype, patterns in self.compiled_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if pattern.search(question_lower):
                    matches += 1
                    # Earlier patterns in the list are more specific
                    score += 1.0 / (patterns.index(pattern) + 1)
            
            # Add domain indicator bonus
            for indicator in self.DOMAIN_INDICATORS.get(qtype, []):
                if indicator in question_lower:
                    score += 0.3
            
            if matches > 0:
                type_scores[qtype] = score
        
        # Determine primary type
        if not type_scores:
            # Default classification based on question word
            primary_type = self._classify_by_question_word(question_lower)
            confidence = 0.5
        else:
            # Get highest scoring type
            sorted_types = sorted(type_scores.items(), key=lambda x: -x[1])
            primary_type = sorted_types[0][0]
            
            # Calculate confidence based on score difference
            if len(sorted_types) > 1:
                score_diff = sorted_types[0][1] - sorted_types[1][1]
                confidence = min(0.95, 0.6 + score_diff * 0.2)
            else:
                confidence = 0.8
            
            features['type_scores'] = {k: round(v, 3) for k, v in type_scores.items()}
        
        # Determine secondary type
        secondary_type = None
        if len(type_scores) > 1:
            sorted_types = sorted(type_scores.items(), key=lambda x: -x[1])
            if sorted_types[1][1] > 0.5:
                secondary_type = sorted_types[1][0]
        
        return ClassificationResult(
            question_type=primary_type,
            confidence=round(confidence, 3),
            secondary_type=secondary_type,
            features=features,
        )
    
    def _extract_features(
        self, 
        question: str, 
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract features from question and answer."""
        features = {
            'question_length': len(question),
            'word_count': len(question.split()),
        }
        
        question_lower = question.lower()
        
        # Question word
        for word in ['what', 'how', 'why', 'when', 'where', 'which', 'who']:
            if question_lower.startswith(word):
                features['question_word'] = word
                break
        
        # Contains scientific terms
        ms_terms = ['mass spectrometry', 'ms', 'ms/ms', 'proteomics', 'metabolomics',
                   'peptide', 'protein', 'spectrum', 'spectra', 'ion', 'fragment']
        for term in ms_terms:
            if term in question_lower:
                features['has_ms_terms'] = True
                break
        
        # Contains numbers
        if re.search(r'\d+', question):
            features['has_numbers'] = True
        
        # Answer features (if provided)
        if answer:
            features['answer_length'] = len(answer)
            
            # Check if answer is numeric
            answer_clean = re.sub(r'[^\d\.\-]', '', answer[:50])
            if answer_clean and re.match(r'^[\d\.\-]+$', answer_clean):
                features['numeric_answer'] = True
        
        return features
    
    def _classify_by_question_word(self, question_lower: str) -> str:
        """Fallback classification based on question word."""
        if question_lower.startswith('how'):
            # Could be method or numeric
            if any(w in question_lower for w in ['how many', 'how much', 'how often']):
                return 'numeric'
            return 'method'
        elif question_lower.startswith('why'):
            return 'causal'
        elif question_lower.startswith('what is') or question_lower.startswith('what are'):
            return 'definition'
        elif question_lower.startswith(('what', 'which', 'who', 'when', 'where')):
            return 'factual'
        else:
            return 'unknown'
    
    def classify_batch(
        self, 
        records: List[Dict[str, Any]],
        question_field: str = 'question',
        answer_field: str = 'answer',
    ) -> List[Dict[str, Any]]:
        """
        Classify questions in a batch of records.
        
        Args:
            records: List of QA records
            question_field: Field containing question
            answer_field: Field containing answer
            
        Returns:
            Records with added question_type field
        """
        results = []
        
        for record in records:
            question = record.get(question_field, '')
            answer = record.get(answer_field)
            
            classification = self.classify(question, answer)
            
            # Add classification to record
            record['question_type'] = classification.question_type
            record['question_type_confidence'] = classification.confidence
            if classification.secondary_type:
                record['question_type_secondary'] = classification.secondary_type
            
            results.append(record)
        
        return results


def classify_question(
    question: str,
    answer: Optional[str] = None,
) -> str:
    """
    Convenience function to classify a single question.
    
    Args:
        question: Question text
        answer: Optional answer text
        
    Returns:
        Question type string
    """
    classifier = QuestionClassifier()
    result = classifier.classify(question, answer)
    return result.question_type


def classify_qa_file(
    input_file: Path,
    output_file: Path,
    workers: int = 1,
    chunk_size: int = 8000,
) -> Dict[str, int]:
    """
    Classify all questions in a JSONL file.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file with classifications
        workers: If > 1, use parallel worker processes (CPU).
        chunk_size: Lines per chunk when workers > 1.

    Returns:
        Distribution of question types
    """
    classifier = QuestionClassifier()
    type_counts: Counter = Counter()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as f_in, output_file.open(
        "w", encoding="utf-8"
    ) as f_out:
        if workers <= 1:
            for line in f_in:
                if not line.strip():
                    continue
                record = json.loads(line)
                question = record.get("question", "")
                answer = record.get("answer")
                result = classifier.classify(question, answer)
                record["question_type"] = result.question_type
                record["question_type_confidence"] = result.confidence
                if result.secondary_type:
                    record["question_type_secondary"] = result.secondary_type
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                type_counts[result.question_type] += 1
        else:
            buf: List[str] = []
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_classifier_pool_init
            ) as executor:
                for line in f_in:
                    if not line.strip():
                        continue
                    buf.append(line)
                    if len(buf) >= chunk_size:
                        subs = _qc_split_lines(buf, workers)
                        buf = []
                        merged: List[str] = []
                        for part in executor.map(_classify_lines_block, subs):
                            merged.extend(part)
                        for out_line in merged:
                            f_out.write(out_line)
                            rec = json.loads(out_line)
                            type_counts[rec.get("question_type", "unknown")] += 1
                if buf:
                    subs = _qc_split_lines(buf, workers)
                    merged = []
                    for part in executor.map(_classify_lines_block, subs):
                        merged.extend(part)
                    for out_line in merged:
                        f_out.write(out_line)
                        rec = json.loads(out_line)
                        type_counts[rec.get("question_type", "unknown")] += 1

    logger.info("Question type distribution: %s", dict(type_counts))
    return dict(type_counts)


def get_type_distribution(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Get the distribution of question types in records.
    
    Args:
        records: List of records with question_type field
        
    Returns:
        Dictionary mapping type to percentage
    """
    counts = Counter(r.get('question_type', 'unknown') for r in records)
    total = sum(counts.values())
    
    if total == 0:
        return {}
    
    return {qtype: count / total * 100 for qtype, count in counts.items()}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify question types")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Parallel worker processes (CPU).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        help="Lines per chunk when workers > 1.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    type_counts = classify_qa_file(
        Path(args.input),
        Path(args.output),
        workers=max(1, args.workers),
        chunk_size=max(100, args.chunk_size),
    )
    
    total = sum(type_counts.values())
    print("\nQuestion Type Distribution:")
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {qtype}: {count} ({pct:.1f}%)")
