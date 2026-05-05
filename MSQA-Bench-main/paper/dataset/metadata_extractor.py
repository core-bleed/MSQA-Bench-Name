"""
Metadata Extractor for MSQA-Bench.

Extracts DOI, PMID, arXiv ID, title, year, and venue from extracted PDF text.
Optionally queries CrossRef/PubMed APIs for enrichment.
"""

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    doc_id: str  # Hash-based unique identifier
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    authors: Optional[List[str]] = None
    license: str = "unknown"  # open_access | unknown | restricted
    source_url: Optional[str] = None
    extraction_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MetadataExtractor:
    """Extract and enrich document metadata from extracted text."""
    
    # DOI patterns
    DOI_PATTERNS = [
        r'(?:doi[:\s]*)?(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s\]\"\'<>]+)',
        r'(?:doi[:\s]*)(10\.\d{4,}/[^\s\]\"\'<>]+)',
        r'DOI[:\s]*(10\.\d{4,}/[^\s\]\"\'<>]+)',
    ]
    
    # PMID patterns
    PMID_PATTERNS = [
        r'PMID[:\s]*(\d{7,8})',
        r'PubMed[:\s]*ID[:\s]*(\d{7,8})',
        r'pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})',
    ]
    
    # arXiv patterns
    ARXIV_PATTERNS = [
        r'arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',
    ]
    
    # Year patterns
    YEAR_PATTERNS = [
        r'\((\d{4})\)',  # (2020)
        r'©\s*(\d{4})',  # © 2020
        r'Copyright\s+(\d{4})',
        r'Published[:\s]+\d{1,2}\s+\w+\s+(\d{4})',
        r'Received[:\s]+\d{1,2}\s+\w+\s+(\d{4})',
    ]
    
    # Known Open Access venues (partial list)
    OPEN_ACCESS_VENUES = {
        'scientific reports', 'nature communications', 'plos one', 'plos biology',
        'plos medicine', 'bmc bioinformatics', 'bmc genomics', 'frontiers in',
        'peerj', 'elife', 'f1000research', 'mdpi', 'arxiv', 'biorxiv', 'medrxiv',
        'journal of proteome research',  # Often OA for MS papers
    }
    
    # Known venue patterns
    VENUE_PATTERNS = [
        (r'Scientific Reports', 'Scientific Reports'),
        (r'Nature Communications', 'Nature Communications'),
        (r'PLOS ONE', 'PLOS ONE'),
        (r'Journal of Proteome Research', 'Journal of Proteome Research'),
        (r'Analytical Chemistry', 'Analytical Chemistry'),
        (r'Journal of the American Society for Mass Spectrometry', 'JASMS'),
        (r'Proteomics', 'Proteomics'),
        (r'Bioinformatics', 'Bioinformatics'),
        (r'BMC Bioinformatics', 'BMC Bioinformatics'),
        (r'Nature Methods', 'Nature Methods'),
        (r'Molecular & Cellular Proteomics', 'MCP'),
        (r'Mass Spectrometry Reviews', 'Mass Spectrometry Reviews'),
    ]
    
    def __init__(self, use_api: bool = False, cache_dir: Optional[Path] = None):
        """
        Initialize the metadata extractor.
        
        Args:
            use_api: Whether to use CrossRef/PubMed APIs for enrichment
            cache_dir: Directory to cache API responses
        """
        self.use_api = use_api and HAS_REQUESTS
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._api_cache: Dict[str, Dict] = {}
    
    def extract_from_text(self, text: str, file_path: Optional[str] = None) -> DocumentMetadata:
        """
        Extract metadata from document text.
        
        Args:
            text: Extracted text content from PDF
            file_path: Optional path to source file (for doc_id generation)
            
        Returns:
            DocumentMetadata object with extracted fields
        """
        # Generate doc_id from file path or text hash
        if file_path:
            doc_id = Path(file_path).stem
        else:
            doc_id = hashlib.sha256(text[:5000].encode()).hexdigest()[:40]
        
        # Extract identifiers
        doi = self._extract_doi(text)
        pmid = self._extract_pmid(text)
        arxiv_id = self._extract_arxiv(text)
        
        # Extract bibliographic info
        title = self._extract_title(text)
        year = self._extract_year(text)
        venue = self._extract_venue(text)
        
        # Determine license
        license_type = self._determine_license(text, venue, doi)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(doi, pmid, arxiv_id, title, year, venue)
        
        metadata = DocumentMetadata(
            doc_id=doc_id,
            doi=doi,
            pmid=pmid,
            arxiv_id=arxiv_id,
            title=title,
            year=year,
            venue=venue,
            license=license_type,
            extraction_confidence=confidence,
        )
        
        # Enrich from API if enabled and we have a DOI
        if self.use_api and doi:
            metadata = self._enrich_from_crossref(metadata)
        
        return metadata
    
    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text."""
        # Search in first 3000 chars (usually in header/footer)
        search_text = text[:3000] + text[-1000:] if len(text) > 4000 else text
        
        for pattern in self.DOI_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                doi = match.group(1)
                # Clean up DOI
                doi = doi.rstrip('.,;:')
                # Validate DOI format
                if re.match(r'^10\.\d{4,}/\S+$', doi):
                    return doi
        return None
    
    def _extract_pmid(self, text: str) -> Optional[str]:
        """Extract PubMed ID from text."""
        search_text = text[:3000] + text[-1000:] if len(text) > 4000 else text
        
        for pattern in self.PMID_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_arxiv(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text."""
        search_text = text[:3000] + text[-1000:] if len(text) > 4000 else text
        
        for pattern in self.ARXIV_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract paper title from text."""
        lines = text[:2000].split('\n')
        
        # Strategy 1: Look for the longest line in the first 20 lines
        # (titles are often the longest early text)
        candidates = []
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if len(line) > 30 and len(line) < 300:
                # Skip lines that look like headers/footers
                if any(skip in line.lower() for skip in [
                    'www.', 'http', '@', 'copyright', 'doi:', 'page ',
                    'received', 'accepted', 'published', 'abstract'
                ]):
                    continue
                # Skip lines with too many numbers (likely references)
                if len(re.findall(r'\d', line)) > len(line) * 0.3:
                    continue
                candidates.append((i, len(line), line))
        
        if candidates:
            # Prefer longer titles earlier in the document
            candidates.sort(key=lambda x: (-x[1] * (20 - x[0]) / 20))
            title = candidates[0][2]
            # Clean up title
            title = re.sub(r'\s+', ' ', title).strip()
            return title
        
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year from text."""
        search_text = text[:3000]
        
        for pattern in self.YEAR_PATTERNS:
            match = re.search(pattern, search_text)
            if match:
                year = int(match.group(1))
                if 1990 <= year <= 2030:  # Reasonable year range
                    return year
        
        # Fallback: look for any 4-digit year in reasonable range
        years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', search_text)
        if years:
            # Return the most common year, or the first one
            from collections import Counter
            year_counts = Counter(years)
            most_common = year_counts.most_common(1)[0][0]
            return int(most_common)
        
        return None
    
    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract publication venue from text."""
        search_text = text[:3000].lower()
        
        for pattern, venue_name in self.VENUE_PATTERNS:
            if re.search(pattern, text[:3000], re.IGNORECASE):
                return venue_name
        
        # Look for journal indicators
        journal_match = re.search(
            r'(?:published in|journal[:\s]+)([A-Z][A-Za-z\s&]+)',
            text[:3000]
        )
        if journal_match:
            return journal_match.group(1).strip()
        
        return None
    
    def _determine_license(
        self, 
        text: str, 
        venue: Optional[str], 
        doi: Optional[str]
    ) -> str:
        """Determine the license type of the document."""
        text_lower = text[:5000].lower()
        
        # Check for explicit CC license
        if re.search(r'creative\s+commons|cc[\s-]by', text_lower):
            return "open_access"
        
        # Check for Open Access indicators
        if re.search(r'open\s+access|freely\s+available', text_lower):
            return "open_access"
        
        # Check venue against known OA journals
        if venue:
            venue_lower = venue.lower()
            for oa_venue in self.OPEN_ACCESS_VENUES:
                if oa_venue in venue_lower:
                    return "open_access"
        
        # Check DOI prefix for known OA publishers
        if doi:
            # Nature Scientific Reports, PLOS, etc.
            oa_prefixes = ['10.1038/s41598', '10.1371/journal.p', '10.3389/']
            for prefix in oa_prefixes:
                if doi.startswith(prefix):
                    return "open_access"
        
        return "unknown"
    
    def _calculate_confidence(
        self,
        doi: Optional[str],
        pmid: Optional[str],
        arxiv_id: Optional[str],
        title: Optional[str],
        year: Optional[int],
        venue: Optional[str],
    ) -> float:
        """Calculate confidence score for extracted metadata."""
        score = 0.0
        
        # Identifiers are most valuable
        if doi:
            score += 0.3
        if pmid:
            score += 0.2
        if arxiv_id:
            score += 0.2
        
        # Bibliographic info
        if title and len(title) > 20:
            score += 0.15
        if year:
            score += 0.1
        if venue:
            score += 0.05
        
        return min(score, 1.0)
    
    def _enrich_from_crossref(self, metadata: DocumentMetadata) -> DocumentMetadata:
        """Enrich metadata using CrossRef API."""
        if not metadata.doi or not HAS_REQUESTS:
            return metadata
        
        # Check cache
        cache_key = f"crossref_{metadata.doi}"
        if cache_key in self._api_cache:
            return self._apply_crossref_data(metadata, self._api_cache[cache_key])
        
        try:
            url = f"https://api.crossref.org/works/{metadata.doi}"
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'MSQA-Bench/1.0 (mailto:research@example.com)'
            })
            
            if response.status_code == 200:
                data = response.json().get('message', {})
                self._api_cache[cache_key] = data
                return self._apply_crossref_data(metadata, data)
                
        except Exception as e:
            logger.warning(f"CrossRef API error for {metadata.doi}: {e}")
        
        return metadata
    
    def _apply_crossref_data(
        self, 
        metadata: DocumentMetadata, 
        data: Dict
    ) -> DocumentMetadata:
        """Apply CrossRef data to metadata object."""
        # Update title if not found
        if not metadata.title and 'title' in data:
            titles = data['title']
            if titles:
                metadata.title = titles[0]
        
        # Update year
        if not metadata.year:
            pub_date = data.get('published-print') or data.get('published-online')
            if pub_date and 'date-parts' in pub_date:
                parts = pub_date['date-parts'][0]
                if parts:
                    metadata.year = parts[0]
        
        # Update venue
        if not metadata.venue and 'container-title' in data:
            containers = data['container-title']
            if containers:
                metadata.venue = containers[0]
        
        # Update authors
        if 'author' in data:
            metadata.authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in data['author']
            ]
        
        # Check license
        if 'license' in data:
            for lic in data['license']:
                url = lic.get('URL', '').lower()
                if 'creativecommons' in url or 'open' in url:
                    metadata.license = "open_access"
                    break
        
        # Update confidence
        metadata.extraction_confidence = min(metadata.extraction_confidence + 0.2, 1.0)
        
        return metadata


def extract_metadata_from_text(
    text: str,
    file_path: Optional[str] = None,
    use_api: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to extract metadata from text.
    
    Args:
        text: Document text
        file_path: Optional file path
        use_api: Whether to use external APIs
        
    Returns:
        Dictionary of metadata fields
    """
    extractor = MetadataExtractor(use_api=use_api)
    metadata = extractor.extract_from_text(text, file_path)
    return metadata.to_dict()


def batch_extract_metadata(
    text_dir: Path,
    output_file: Path,
    use_api: bool = False,
    workers: int = 4,
) -> Dict[str, DocumentMetadata]:
    """
    Extract metadata from all text files in a directory.
    
    Args:
        text_dir: Directory containing extracted text files
        output_file: Path to save metadata JSONL
        use_api: Whether to use external APIs
        workers: Number of parallel workers
        
    Returns:
        Dictionary mapping doc_id to metadata
    """
    extractor = MetadataExtractor(use_api=use_api)
    text_files = list(Path(text_dir).glob("*.txt"))
    
    logger.info(f"Extracting metadata from {len(text_files)} files...")
    
    results: Dict[str, DocumentMetadata] = {}
    
    def process_file(file_path: Path) -> Tuple[str, DocumentMetadata]:
        text = file_path.read_text(encoding='utf-8', errors='replace')
        metadata = extractor.extract_from_text(text, str(file_path))
        return metadata.doc_id, metadata
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f): f for f in text_files}
        
        for future in as_completed(futures):
            try:
                doc_id, metadata = future.result()
                results[doc_id] = metadata
            except Exception as e:
                logger.error(f"Error processing {futures[future]}: {e}")
    
    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        for metadata in results.values():
            f.write(json.dumps(metadata.to_dict(), ensure_ascii=False) + '\n')
    
    logger.info(f"Saved metadata for {len(results)} documents to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract metadata from PDF text files")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with text files")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--use-api", action="store_true", help="Use CrossRef API")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    batch_extract_metadata(
        Path(args.input_dir),
        Path(args.output),
        use_api=args.use_api,
        workers=args.workers,
    )
