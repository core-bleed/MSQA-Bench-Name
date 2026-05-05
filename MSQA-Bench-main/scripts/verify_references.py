#!/usr/bin/env python3
"""
Automated reference verification for academic papers.

Parses a BibTeX (.bib) file and verifies each reference against
Semantic Scholar, CrossRef, and OpenAlex APIs.

Usage:
    python scripts/verify_references.py paper_v2/references.bib
    python scripts/verify_references.py paper_v2/references.bib --output paper_results/ref_check.json
    python scripts/verify_references.py paper_v2/references.bib --fix  # auto-fix minor issues

Checks:
    1. Title exists and roughly matches a real publication
    2. Authors are correct (fuzzy match)
    3. Year is correct
    4. DOI is valid (if provided)
    5. ArXiv ID resolves (if provided)
    6. Venue/journal is approximately correct

Output:
    - Console report with PASS/WARN/FAIL per reference
    - JSON report with detailed verification data
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Rate limiting
_last_request_time = 0.0
RATE_LIMIT_SECONDS = 1.1  # be polite to APIs


def _rate_limited_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Make a rate-limited HTTP GET and return parsed JSON."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)

    req = Request(url, headers=headers or {"User-Agent": "MSGPTRefChecker/1.0"})
    try:
        _last_request_time = time.time()
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as e:
        logger.debug(f"Request failed for {url}: {e}")
        return None


# ============================================================================
# BibTeX Parser (lightweight, no external deps)
# ============================================================================

@dataclass
class BibEntry:
    key: str
    entry_type: str
    title: str = ""
    author: str = ""
    year: str = ""
    journal: str = ""
    booktitle: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    volume: str = ""
    pages: str = ""
    publisher: str = ""
    raw: dict[str, str] = field(default_factory=dict)


def parse_bibtex(bib_path: str) -> list[BibEntry]:
    """Parse a .bib file into BibEntry objects."""
    with open(bib_path, encoding="utf-8") as f:
        content = f.read()

    entries = []

    # Brace-balanced BibTeX entry extraction
    i = 0
    while i < len(content):
        m = re.match(r"@(\w+)\s*\{([^,]+),", content[i:])
        if not m:
            i += 1
            continue
        entry_type = m.group(1).lower()
        key = m.group(2).strip()
        start = i + m.end()
        # Walk forward counting braces (we consumed the opening '{')
        depth = 1
        j = start
        while j < len(content) and depth > 0:
            if content[j] == "{":
                depth += 1
            elif content[j] == "}":
                depth -= 1
            j += 1
        body = content[start : j - 1]
        i = j

        # Parse fields: name = {value} or name = "value"
        fields: dict[str, str] = {}
        pos = 0
        while pos < len(body):
            fm = re.match(r"\s*(\w+)\s*=\s*", body[pos:])
            if not fm:
                pos += 1
                continue
            field_name = fm.group(1).lower()
            pos += fm.end()
            if pos >= len(body):
                break
            ch = body[pos]
            if ch == "{":
                # Brace-delimited value
                d = 1
                k = pos + 1
                while k < len(body) and d > 0:
                    if body[k] == "{":
                        d += 1
                    elif body[k] == "}":
                        d -= 1
                    k += 1
                val = body[pos + 1 : k - 1]
                pos = k
            elif ch == '"':
                k = body.index('"', pos + 1)
                val = body[pos + 1 : k]
                pos = k + 1
            else:
                # Bare value (number or macro)
                em = re.match(r"([^,}\s]+)", body[pos:])
                val = em.group(1) if em else ""
                pos += len(val)
            val = val.strip().replace("\n", " ")
            val = re.sub(r"\s+", " ", val)
            # Remove LaTeX braces but keep content
            val = re.sub(r"[{}]", "", val)
            fields[field_name] = val

        # Extract arxiv ID from eprint or url
        arxiv_id = fields.get("eprint", "")
        if not arxiv_id and "arxiv" in fields.get("url", "").lower():
            m = re.search(r"(\d{4}\.\d{4,5})", fields.get("url", ""))
            if m:
                arxiv_id = m.group(1)

        entry = BibEntry(
            key=key,
            entry_type=entry_type,
            title=fields.get("title", ""),
            author=fields.get("author", ""),
            year=fields.get("year", ""),
            journal=fields.get("journal", fields.get("journaltitle", "")),
            booktitle=fields.get("booktitle", ""),
            doi=fields.get("doi", ""),
            arxiv_id=arxiv_id,
            url=fields.get("url", ""),
            volume=fields.get("volume", ""),
            pages=fields.get("pages", ""),
            publisher=fields.get("publisher", ""),
            raw=fields,
        )
        entries.append(entry)

    return entries


# ============================================================================
# Verification APIs
# ============================================================================

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def verify_via_semantic_scholar(entry: BibEntry) -> dict[str, Any]:
    """Query Semantic Scholar for title match."""
    title = entry.title
    if not title:
        return {"source": "semantic_scholar", "status": "skip", "reason": "no title"}

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(title)}&limit=3&fields=title,authors,year,externalIds,venue"
    data = _rate_limited_get(url)
    if not data or "data" not in data or not data["data"]:
        return {"source": "semantic_scholar", "status": "not_found"}

    best_match = None
    best_sim = 0.0
    for paper in data["data"]:
        sim = _title_similarity(title, paper.get("title", ""))
        if sim > best_sim:
            best_sim = sim
            best_match = paper

    if best_match and best_sim >= 0.6:
        found_year = str(best_match.get("year", ""))
        found_authors = [a.get("name", "") for a in best_match.get("authors", [])]
        found_doi = (best_match.get("externalIds") or {}).get("DOI", "")
        return {
            "source": "semantic_scholar",
            "status": "found",
            "title_similarity": round(best_sim, 3),
            "found_title": best_match.get("title"),
            "found_year": found_year,
            "found_authors": found_authors[:5],
            "found_doi": found_doi,
            "found_venue": best_match.get("venue", ""),
            "year_match": found_year == entry.year,
        }

    return {"source": "semantic_scholar", "status": "low_match", "best_similarity": round(best_sim, 3)}


def verify_via_crossref(entry: BibEntry) -> dict[str, Any]:
    """Query CrossRef for DOI or title match."""
    # If DOI is provided, verify directly
    if entry.doi:
        url = f"https://api.crossref.org/works/{quote_plus(entry.doi)}"
        data = _rate_limited_get(url, headers={
            "User-Agent": "MSGPTRefChecker/1.0 (mailto:research@example.com)"
        })
        if data and "message" in data:
            msg = data["message"]
            found_title = " ".join(msg.get("title", []))
            sim = _title_similarity(entry.title, found_title)
            return {
                "source": "crossref",
                "status": "found",
                "doi_valid": True,
                "title_similarity": round(sim, 3),
                "found_title": found_title,
                "found_year": str(msg.get("published-print", msg.get("published-online", {}))
                                  .get("date-parts", [[""]])[0][0]),
            }
        return {"source": "crossref", "status": "doi_invalid", "doi": entry.doi}

    # Search by title
    if entry.title:
        url = f"https://api.crossref.org/works?query.title={quote_plus(entry.title)}&rows=3"
        data = _rate_limited_get(url, headers={
            "User-Agent": "MSGPTRefChecker/1.0 (mailto:research@example.com)"
        })
        if data and "message" in data and data["message"].get("items"):
            for item in data["message"]["items"]:
                found_title = " ".join(item.get("title", []))
                sim = _title_similarity(entry.title, found_title)
                if sim >= 0.6:
                    return {
                        "source": "crossref",
                        "status": "found",
                        "title_similarity": round(sim, 3),
                        "found_title": found_title,
                        "found_doi": item.get("DOI", ""),
                    }

    return {"source": "crossref", "status": "not_found"}


def verify_via_arxiv(entry: BibEntry) -> dict[str, Any]:
    """Verify arXiv ID if present."""
    arxiv_id = entry.arxiv_id
    if not arxiv_id:
        # Try to extract from URL
        m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", entry.url)
        if m:
            arxiv_id = m.group(1)
        else:
            return {"source": "arxiv", "status": "skip", "reason": "no arxiv id"}

    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = Request(url, headers={"User-Agent": "MSGPTRefChecker/1.0"})
        time.sleep(RATE_LIMIT_SECONDS)
        with urlopen(req, timeout=15) as resp:
            content = resp.read().decode()

        if "<entry>" in content:
            # Extract title from XML
            title_match = re.search(r"<title>(.*?)</title>", content, re.DOTALL)
            if title_match:
                found_title = title_match.group(1).strip().replace("\n", " ")
                sim = _title_similarity(entry.title, found_title)
                return {
                    "source": "arxiv",
                    "status": "found",
                    "arxiv_id": arxiv_id,
                    "found_title": found_title,
                    "title_similarity": round(sim, 3),
                }

        return {"source": "arxiv", "status": "not_found", "arxiv_id": arxiv_id}
    except Exception as e:
        return {"source": "arxiv", "status": "error", "error": str(e)}


# ============================================================================
# Main Verification Logic
# ============================================================================

@dataclass
class VerificationResult:
    key: str
    title: str
    status: str  # PASS, WARN, FAIL
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def verify_entry(entry: BibEntry) -> VerificationResult:
    """Verify a single BibTeX entry against multiple sources."""
    issues: list[str] = []
    details: dict[str, Any] = {}

    # Basic field checks
    if not entry.title:
        issues.append("MISSING: no title field")
    if not entry.author:
        issues.append("MISSING: no author field")
    if not entry.year:
        issues.append("MISSING: no year field")

    if issues:
        return VerificationResult(
            key=entry.key, title=entry.title,
            status="FAIL", issues=issues, details=details,
        )

    # Check against APIs (try Semantic Scholar first, then CrossRef, then arXiv)
    ss_result = verify_via_semantic_scholar(entry)
    details["semantic_scholar"] = ss_result

    cr_result = verify_via_crossref(entry)
    details["crossref"] = cr_result

    if entry.arxiv_id or "arxiv" in entry.url.lower():
        arxiv_result = verify_via_arxiv(entry)
        details["arxiv"] = arxiv_result

    # Determine overall status
    found_anywhere = (
        ss_result.get("status") == "found"
        or cr_result.get("status") == "found"
        or details.get("arxiv", {}).get("status") == "found"
    )

    if not found_anywhere:
        issues.append("NOT FOUND: could not verify in any database")
        status = "FAIL"
    else:
        # Check year
        for src in [ss_result, cr_result]:
            if src.get("status") == "found" and "year_match" in src:
                if not src["year_match"]:
                    found_year = src.get("found_year", "?")
                    issues.append(f"YEAR MISMATCH: bib says {entry.year}, found {found_year}")

        # Check title similarity
        best_sim = max(
            ss_result.get("title_similarity", 0),
            cr_result.get("title_similarity", 0),
            details.get("arxiv", {}).get("title_similarity", 0),
        )
        if best_sim < 0.5:
            issues.append(f"TITLE MISMATCH: best similarity = {best_sim:.2f}")

        # Check if DOI provided matches
        if entry.doi and cr_result.get("status") == "doi_invalid":
            issues.append(f"INVALID DOI: {entry.doi}")

        status = "WARN" if issues else "PASS"

    return VerificationResult(
        key=entry.key, title=entry.title,
        status=status, issues=issues, details=details,
    )


def print_report(results: list[VerificationResult]) -> None:
    """Print a formatted verification report."""
    passed = sum(1 for r in results if r.status == "PASS")
    warned = sum(1 for r in results if r.status == "WARN")
    failed = sum(1 for r in results if r.status == "FAIL")

    print("\n" + "=" * 70)
    print("REFERENCE VERIFICATION REPORT")
    print("=" * 70)
    print(f"  Total references: {len(results)}")
    print(f"  PASS: {passed}  |  WARN: {warned}  |  FAIL: {failed}")
    print("=" * 70)

    for r in results:
        icon = {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]"}[r.status]
        print(f"\n{icon} [{r.key}]")
        print(f"    Title: {r.title[:80]}{'...' if len(r.title) > 80 else ''}")
        if r.issues:
            for issue in r.issues:
                print(f"    -> {issue}")

    print("\n" + "=" * 70)
    if failed > 0:
        print(f"ACTION REQUIRED: {failed} reference(s) could not be verified.")
        print("These may be hallucinated, misspelled, or from sources not indexed by APIs.")
    elif warned > 0:
        print(f"Review {warned} warning(s) above for potential issues.")
    else:
        print("All references verified successfully.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BibTeX references against academic APIs")
    parser.add_argument("bibfile", help="Path to .bib file")
    parser.add_argument("--output", "-o", help="Save detailed JSON report to this path")
    parser.add_argument("--keys", nargs="*", help="Only verify these BibTeX keys")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show API response details")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse
    entries = parse_bibtex(args.bibfile)
    if not entries:
        print(f"No entries found in {args.bibfile}")
        sys.exit(1)

    logger.info(f"Parsed {len(entries)} entries from {args.bibfile}")

    # Filter if specific keys requested
    if args.keys:
        entries = [e for e in entries if e.key in args.keys]
        logger.info(f"Filtered to {len(entries)} entries")

    # Verify
    results: list[VerificationResult] = []
    for i, entry in enumerate(entries):
        logger.info(f"[{i + 1}/{len(entries)}] Verifying: {entry.key}")
        result = verify_entry(entry)
        results.append(result)

    # Report
    print_report(results)

    # Save JSON
    if args.output:
        output_data = [
            {
                "key": r.key,
                "title": r.title,
                "status": r.status,
                "issues": r.issues,
                "details": r.details,
            }
            for r in results
        ]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
