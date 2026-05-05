"""
Schema Enricher for MSQA-Bench.

Extends the QA schema with benchmark-grade fields including:
- Document metadata (DOI, title, year, venue)
- Context offsets and evidence spans
- Question type classification
- Answer style detection
- Quality scores
- Train/val/test split assignment
"""

import json
import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, cast
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .metadata_extractor import MetadataExtractor, DocumentMetadata

logger = logging.getLogger(__name__)

# Set in worker processes (see _schema_enrich_pool_init after SchemaEnricher is defined)
_PARALLEL_ENRICHER: Any = None


@dataclass
class EvidenceSpan:
    """A span of text that supports the answer."""
    start: int  # Character offset in context
    end: int    # Character offset in context
    text: str   # The actual text (for verification)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextOffsets:
    """Character offsets of context within the source document."""
    start_char: int
    end_char: int
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass 
class EnrichedQARecord:
    """
    Benchmark-grade QA record with all required fields.
    
    This is the target schema for MSQA-Bench.
    """
    # Core QA fields (from original)
    id: str
    question: str
    answer: str
    context: str
    
    # Source tracking (from original)
    file_name: str
    source_pdf: str
    paragraph_index: int
    line_number: int
    
    # Document metadata (new)
    doc_id: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    license: str = "unknown"
    
    # Benchmark fields (new)
    context_offsets: Optional[ContextOffsets] = None
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    question_type: str = "unknown"  # factual|method|definition|comparison|numeric|unknown
    answer_style: str = "unknown"   # extractive|abstractive|unknown
    quality_score: float = 0.0
    split: str = "train"  # train|val|test
    
    # Context grouping (new) - for handling one-context-many-questions
    context_id: Optional[str] = None  # Hash of (doc_id, paragraph_index)
    
    # Generation metadata (from original)
    model: Optional[str] = None
    run_id: Optional[str] = None
    created_at: Optional[str] = None
    
    # Enrichment metadata
    enriched_at: Optional[str] = None
    enrichment_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for k, v in asdict(self).items():
            if v is None:
                result[k] = None
            elif isinstance(v, (ContextOffsets, EvidenceSpan)):
                result[k] = v.to_dict()
            elif isinstance(v, list) and v and isinstance(v[0], EvidenceSpan):
                result[k] = [span.to_dict() for span in v]
            else:
                result[k] = v
        return result
    
    def to_public_dict(self) -> Dict[str, Any]:
        """
        Convert to public release format (no full context for licensing).
        
        This format is safe to release publicly as it only contains:
        - Questions and answers (generated, not copyrighted)
        - Citation pointers (DOI, PMID, etc.)
        - Offset information (not the copyrighted text itself)
        """
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "doc_id": self.doc_id,
            "doi": self.doi,
            "pmid": self.pmid,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "license": self.license,
            "paragraph_index": self.paragraph_index,
            "context_offsets": self.context_offsets.to_dict() if self.context_offsets else None,
            "evidence_span_offsets": [
                {"start": s.start, "end": s.end} for s in self.evidence_spans
            ] if self.evidence_spans else [],
            "question_type": self.question_type,
            "answer_style": self.answer_style,
            "quality_score": self.quality_score,
            "split": self.split,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedQARecord':
        """Create from dictionary."""
        # Handle nested objects
        if 'context_offsets' in data and data['context_offsets']:
            if isinstance(data['context_offsets'], dict):
                data['context_offsets'] = ContextOffsets(**data['context_offsets'])
        
        if 'evidence_spans' in data and data['evidence_spans']:
            spans = []
            for span in data['evidence_spans']:
                if isinstance(span, dict):
                    spans.append(EvidenceSpan(**span))
                else:
                    spans.append(span)
            data['evidence_spans'] = spans
        
        # Filter to valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered)


class SchemaEnricher:
    """
    Enriches QA records with benchmark-grade fields.
    
    Pipeline:
    1. Load original QA record
    2. Extract/lookup document metadata
    3. Compute context offsets
    4. Find evidence spans
    5. Classify question type
    6. Determine answer style
    7. Compute quality score
    8. Assign split
    """
    
    def __init__(
        self,
        metadata_cache: Optional[Dict[str, DocumentMetadata]] = None,
        text_dir: Optional[Path] = None,
        use_api: bool = False,
    ):
        """
        Initialize the schema enricher.
        
        Args:
            metadata_cache: Pre-extracted metadata keyed by doc_id
            text_dir: Directory containing extracted text files
            use_api: Whether to use external APIs for metadata
        """
        self.metadata_cache = metadata_cache or {}
        self.text_dir = Path(text_dir) if text_dir else None
        self.metadata_extractor = MetadataExtractor(use_api=use_api)
        
        # Cache for document texts
        self._text_cache: Dict[str, str] = {}
    
    def enrich_record(self, record: Dict[str, Any]) -> EnrichedQARecord:
        """
        Enrich a single QA record with benchmark fields.
        
        Args:
            record: Original QA record dictionary
            
        Returns:
            EnrichedQARecord with all benchmark fields
        """
        # Extract doc_id from file_name
        file_name = record.get('file_name', '')
        doc_id = Path(file_name).stem if file_name else record.get('id', '')[:40]
        
        # Get or extract document metadata
        metadata = self._get_metadata(doc_id, record)
        
        # Generate context_id for grouping
        context_id = hashlib.md5(
            f"{doc_id}:{record.get('paragraph_index', 0)}".encode()
        ).hexdigest()[:16]
        
        # Find evidence spans in context
        context = record.get('context', '')
        answer = record.get('answer', '')
        evidence_spans = self._find_evidence_spans(context, answer)
        
        # Determine answer style
        answer_style = self._determine_answer_style(context, answer, evidence_spans)
        
        # Create enriched record
        enriched = EnrichedQARecord(
            # Original fields
            id=record.get('id', ''),
            question=record.get('question', ''),
            answer=answer,
            context=context,
            file_name=file_name,
            source_pdf=record.get('source_pdf', ''),
            paragraph_index=record.get('paragraph_index', 0),
            line_number=record.get('line_number', 0),
            
            # Metadata fields
            doc_id=doc_id,
            doi=metadata.doi,
            pmid=metadata.pmid,
            arxiv_id=metadata.arxiv_id,
            title=metadata.title,
            year=metadata.year,
            venue=metadata.venue,
            license=metadata.license,
            
            # Benchmark fields
            context_id=context_id,
            evidence_spans=evidence_spans,
            answer_style=answer_style,
            question_type=record.get('question_type', 'unknown'),
            quality_score=record.get('quality_score', 0.0),
            split="train",            # Will be set by split generator
            
            # Generation metadata
            model=record.get('model'),
            run_id=record.get('run_id'),
            created_at=record.get('created_at'),
            
            # Enrichment metadata
            enriched_at=datetime.utcnow().isoformat() + "Z",
        )
        
        return enriched
    
    def _get_metadata(self, doc_id: str, record: Dict[str, Any]) -> DocumentMetadata:
        """Get or extract metadata for a document."""
        # Check cache first
        if doc_id in self.metadata_cache:
            return self.metadata_cache[doc_id]
        
        # Try to extract from context or load from text file
        context = record.get('context', '')
        
        # If we have the text directory, try to load full text
        if self.text_dir:
            text_file = self.text_dir / f"{doc_id}.txt"
            if text_file.exists():
                if doc_id not in self._text_cache:
                    self._text_cache[doc_id] = text_file.read_text(
                        encoding='utf-8', errors='replace'
                    )
                text = self._text_cache[doc_id]
                metadata = self.metadata_extractor.extract_from_text(text, str(text_file))
                self.metadata_cache[doc_id] = metadata
                return metadata
        
        # Fall back to extracting from context
        metadata = self.metadata_extractor.extract_from_text(context, doc_id)
        self.metadata_cache[doc_id] = metadata
        return metadata
    
    def _find_evidence_spans(
        self, 
        context: str, 
        answer: str
    ) -> List[EvidenceSpan]:
        """
        Find spans in context that support the answer.
        
        Uses simple substring matching and sentence overlap.
        """
        evidence_spans = []
        
        if not context or not answer:
            return evidence_spans
        
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # Strategy 1: Direct substring match
        # Find longest matching substrings
        words = answer.split()
        for n in range(min(len(words), 10), 2, -1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n]).lower()
                if len(phrase) > 20:  # Only meaningful phrases
                    start = context_lower.find(phrase)
                    if start != -1:
                        end = start + len(phrase)
                        # Expand to sentence boundaries
                        sent_start = context.rfind('.', 0, start)
                        sent_start = sent_start + 1 if sent_start != -1 else 0
                        sent_end = context.find('.', end)
                        sent_end = sent_end + 1 if sent_end != -1 else len(context)
                        
                        span = EvidenceSpan(
                            start=sent_start,
                            end=sent_end,
                            text=context[sent_start:sent_end].strip()
                        )
                        
                        # Avoid duplicates
                        if not any(
                            abs(s.start - span.start) < 20 
                            for s in evidence_spans
                        ):
                            evidence_spans.append(span)
                        
                        if len(evidence_spans) >= 3:
                            return evidence_spans
        
        return evidence_spans
    
    def _determine_answer_style(
        self,
        context: str,
        answer: str,
        evidence_spans: List[EvidenceSpan],
    ) -> str:
        """
        Determine if the answer is extractive or abstractive.
        
        - Extractive: Answer is mostly verbatim from context
        - Abstractive: Answer paraphrases or synthesizes information
        """
        if not context or not answer:
            return "unknown"
        
        context_lower = context.lower()
        answer_lower = answer.lower()
        
        # Check for high overlap
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())
        
        if not answer_words:
            return "unknown"
        
        overlap = len(answer_words & context_words) / len(answer_words)
        
        # Check for direct substring
        # Normalize whitespace for comparison
        answer_normalized = ' '.join(answer_lower.split())
        context_normalized = ' '.join(context_lower.split())
        
        if answer_normalized in context_normalized:
            return "extractive"
        
        # Check word overlap threshold
        if overlap > 0.8:
            return "extractive"
        elif overlap < 0.5:
            return "abstractive"
        else:
            # Check evidence spans
            if evidence_spans:
                # If we found good evidence spans, likely extractive
                total_evidence_len = sum(len(s.text) for s in evidence_spans)
                if total_evidence_len > len(answer) * 0.8:
                    return "extractive"
            return "abstractive"


# --- Parallel enrichment (worker pool; SchemaEnricher must be defined above) ---


def _schema_enrich_pool_init(text_dir_str: str) -> None:
    global _PARALLEL_ENRICHER  # noqa: PLW0603
    td = Path(text_dir_str) if text_dir_str else None
    _PARALLEL_ENRICHER = SchemaEnricher(metadata_cache={}, text_dir=td, use_api=False)


def _schema_enrich_lines_block(lines: List[str]) -> List[str]:
    enricher = cast(SchemaEnricher, _PARALLEL_ENRICHER)
    out: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        enriched = enricher.enrich_record(record)
        out.append(json.dumps(enriched.to_dict(), ensure_ascii=False) + "\n")
    return out


def _split_lines_for_workers(lines: List[str], num_parts: int) -> List[List[str]]:
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


def enrich_qa_record(
    record: Dict[str, Any],
    metadata_cache: Optional[Dict[str, DocumentMetadata]] = None,
    text_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to enrich a single QA record.
    
    Args:
        record: Original QA record
        metadata_cache: Optional pre-loaded metadata
        text_dir: Optional directory with text files
        
    Returns:
        Enriched record as dictionary
    """
    enricher = SchemaEnricher(
        metadata_cache=metadata_cache,
        text_dir=text_dir,
    )
    enriched = enricher.enrich_record(record)
    return enriched.to_dict()


def enrich_qa_file(
    input_file: Path,
    output_file: Path,
    text_dir: Optional[Path] = None,
    metadata_file: Optional[Path] = None,
    public_output: Optional[Path] = None,
    workers: int = 1,
    chunk_size: int = 4000,
) -> Dict[str, int]:
    """
    Enrich all QA records in a JSONL file.

    Args:
        input_file: Input JSONL file with QA records
        output_file: Output JSONL file for enriched records
        text_dir: Directory with extracted text files
        metadata_file: Optional pre-extracted metadata JSONL
        public_output: Optional output file for public release format
        workers: If > 1, parallel worker processes (CPU). Disabled if metadata_file
                 or public_output is set (single-process only).
        chunk_size: Lines per chunk when workers > 1.

    Returns:
        Statistics dictionary
    """
    # Load metadata cache if provided
    metadata_cache: Dict[str, DocumentMetadata] = {}
    if metadata_file and metadata_file.exists():
        logger.info("Loading metadata from %s", metadata_file)
        with metadata_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    metadata_cache[data["doc_id"]] = DocumentMetadata.from_dict(data)

    if (metadata_file and metadata_file.exists()) or public_output:
        if workers > 1:
            logger.warning(
                "Parallel enrichment disabled when using --metadata or --public; using workers=1."
            )
        workers = 1

    enricher = SchemaEnricher(
        metadata_cache=metadata_cache,
        text_dir=text_dir,
    )

    stats: Dict[str, int] = {
        "total": 0,
        "enriched": 0,
        "with_doi": 0,
        "extractive": 0,
        "abstractive": 0,
        "with_evidence": 0,
    }

    def _update_stats_from_enriched(enriched: EnrichedQARecord) -> None:
        stats["enriched"] += 1
        if enriched.doi:
            stats["with_doi"] += 1
        if enriched.answer_style == "extractive":
            stats["extractive"] += 1
        elif enriched.answer_style == "abstractive":
            stats["abstractive"] += 1
        if enriched.evidence_spans:
            stats["with_evidence"] += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    public_f = None
    if public_output:
        public_output.parent.mkdir(parents=True, exist_ok=True)
        public_f = public_output.open("w", encoding="utf-8")

    try:
        text_dir_str = str(text_dir.resolve()) if text_dir else ""
    except OSError:
        text_dir_str = str(text_dir) if text_dir else ""

    try:
        with input_file.open("r", encoding="utf-8") as f_in, output_file.open(
            "w", encoding="utf-8"
        ) as f_out:
            if workers <= 1:
                for line in f_in:
                    if not line.strip():
                        continue
                    stats["total"] += 1
                    try:
                        record = json.loads(line)
                        enriched = enricher.enrich_record(record)
                        f_out.write(
                            json.dumps(enriched.to_dict(), ensure_ascii=False) + "\n"
                        )
                        if public_f:
                            public_f.write(
                                json.dumps(
                                    enriched.to_public_dict(), ensure_ascii=False
                                )
                                + "\n"
                            )
                        _update_stats_from_enriched(enriched)
                    except Exception as e:
                        logger.error("Error enriching record: %s", e)
            else:
                buf: List[str] = []
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_schema_enrich_pool_init,
                    initargs=(text_dir_str,),
                ) as executor:
                    for line in f_in:
                        if not line.strip():
                            continue
                        buf.append(line)
                        if len(buf) >= chunk_size:
                            subs = _split_lines_for_workers(buf, workers)
                            buf = []
                            merged: List[str] = []
                            for part in executor.map(_schema_enrich_lines_block, subs):
                                merged.extend(part)
                            for out_line in merged:
                                stats["total"] += 1
                                try:
                                    data = json.loads(out_line)
                                    enriched = EnrichedQARecord.from_dict(data)
                                    f_out.write(out_line)
                                    if public_f:
                                        public_f.write(
                                            json.dumps(
                                                enriched.to_public_dict(),
                                                ensure_ascii=False,
                                            )
                                            + "\n"
                                        )
                                    _update_stats_from_enriched(enriched)
                                except Exception as e:
                                    logger.error("Error post-processing record: %s", e)
                    if buf:
                        subs = _split_lines_for_workers(buf, workers)
                        merged = []
                        for part in executor.map(_schema_enrich_lines_block, subs):
                            merged.extend(part)
                        for out_line in merged:
                            stats["total"] += 1
                            try:
                                data = json.loads(out_line)
                                enriched = EnrichedQARecord.from_dict(data)
                                f_out.write(out_line)
                                if public_f:
                                    public_f.write(
                                        json.dumps(
                                            enriched.to_public_dict(), ensure_ascii=False
                                        )
                                        + "\n"
                                    )
                                _update_stats_from_enriched(enriched)
                            except Exception as e:
                                logger.error("Error post-processing record: %s", e)

    finally:
        if public_f:
            public_f.close()

    logger.info("Enrichment complete: %s", stats)
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich QA records with benchmark fields")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--text-dir", "-t", help="Directory with text files")
    parser.add_argument("--metadata", "-m", help="Pre-extracted metadata JSONL")
    parser.add_argument("--public", "-p", help="Output file for public release format")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Parallel worker processes (CPU). Ignored with --metadata or --public.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Lines per chunk when workers > 1.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    stats = enrich_qa_file(
        Path(args.input),
        Path(args.output),
        text_dir=Path(args.text_dir) if args.text_dir else None,
        metadata_file=Path(args.metadata) if args.metadata else None,
        public_output=Path(args.public) if args.public else None,
        workers=max(1, args.workers),
        chunk_size=max(100, args.chunk_size),
    )
    
    print(f"\nEnrichment Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
