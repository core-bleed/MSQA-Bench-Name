#!/usr/bin/env python3
"""Prepare a two-tier MSQA-Bench release for NeurIPS E&D.

The script partitions split JSONL files into:
  1. redistributable records: text-bearing fields retained when license permits
  2. restricted records: metadata-only records for sources without such rights

Default redistributable criterion is record["license"] == "open_access",
matching the current paper_results schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SPLITS = ("train", "val", "test")

TEXT_FIELDS = {"context", "evidence_spans"}
LOCAL_FIELDS = {"source_pdf"}
OPTIONAL_SOURCE_FIELDS = {"file_name"}
SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENAI_API_KEY]"),
)
TITLE_SUSPICIOUS_PATTERNS = (
    re.compile(r"^\s*\d+\s*(?:department|division|faculty|institute|laboratory|school|state key)", re.I),
    re.compile(r"^\s*(?:abstract|background:|citation:|received:|the author\(s\)|open access|this item was submitted)", re.I),
    re.compile(r"^\s*(?:archive for|skip to|match commun\.|journal of .*submitted)", re.I),
    re.compile(r"\b(?:creative commons|licensed under|university|department of|laboratory of|academy of sciences)\b", re.I),
    re.compile(r"^(?:[A-Z][A-Za-z'’-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z'’-]+)?\s*,\s*){2,}"),
)
TITLE_STOP_PATTERNS = (
    r"\s+(?:PLOS ONE|Scientific Reports|Nature Communications|BMC [A-Za-z ]+|Frontiers in [A-Za-z \n]+)\s*\|.*$",
    r"\s+www\.[^\s]+.*$",
    r"\s+\|\s+www\..*$",
    r"\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}.*$",
    r"\s+\d{4}\s*\|\s*Volume.*$",
)

RESTRICTED_KEEP_FIELDS = (
    "id",
    "doc_id",
    "doi",
    "pmid",
    "arxiv_id",
    "title",
    "year",
    "venue",
    "license",
    "paragraph_index",
    "line_number",
    "context_id",
    "quality_score",
    "question_type",
    "answer_style",
    "split",
    "model",
    "run_id",
    "created_at",
    "enriched_at",
    "enrichment_version",
)


def iter_jsonl(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def is_redistributable(record: Dict[str, Any], allowed_licenses: set[str]) -> bool:
    return str(record.get("license", "unknown")).lower() in allowed_licenses


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        for pattern, replacement in SECRET_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_secrets(item) for key, item in value.items()}
    return value


def normalize_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip(" .;,:|-")
    for pattern in TITLE_STOP_PATTERNS:
        title = re.sub(pattern, "", title, flags=re.I | re.S).strip(" .;,:|-")
    return title


def suspicious_title(title: Any) -> bool:
    if not isinstance(title, str) or not title.strip():
        return True
    clean = normalize_title(title)
    if len(clean) < 8 or len(clean) > 180:
        return True
    if clean.count(",") >= 3 and not re.search(r"[:?]", clean):
        return True
    return any(pattern.search(clean) for pattern in TITLE_SUSPICIOUS_PATTERNS)


def infer_title_from_context(record: Dict[str, Any]) -> Optional[str]:
    context = record.get("context")
    doi = record.get("doi")
    if not isinstance(context, str) or "doi:" not in context.lower():
        return None

    candidates = []
    for match in re.finditer(r"doi:\s*\S+\s+(.{8,260})", context, flags=re.I | re.S):
        candidate = normalize_title(match.group(1))
        if doi and isinstance(doi, str):
            candidate = candidate.replace(doi, "").strip()
        candidates.append(candidate)

    for candidate in reversed(candidates):
        if not candidate:
            continue
        if re.match(r"^(?:figure|table)\s+\d+", candidate, flags=re.I):
            continue
        if re.search(r"\b(?:doi|copyright|creative commons|downloaded from)\b", candidate, flags=re.I):
            continue
        alpha = sum(ch.isalpha() for ch in candidate)
        if 8 <= len(candidate) <= 180 and alpha / max(len(candidate), 1) > 0.45:
            return candidate
    return None


def repair_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    repaired = dict(record)
    if suspicious_title(repaired.get("title")):
        inferred = infer_title_from_context(repaired)
        if inferred:
            repaired["title"] = inferred
            repaired["title_source"] = "context_footer"
        elif repaired.get("title"):
            repaired["title"] = None
            repaired["title_source"] = "unavailable_after_cleanup"
    else:
        repaired["title"] = normalize_title(str(repaired["title"]))
        repaired["title_source"] = repaired.get("title_source", "extracted")
    return repaired


def clean_redistributable(record: Dict[str, Any]) -> Dict[str, Any]:
    out = redact_secrets(repair_metadata(record))
    for field in LOCAL_FIELDS:
        out.pop(field, None)
    out["document_hash"] = out.get("doc_id")
    out["redistribution_status"] = "text_released"
    out["release_license"] = "CC-BY-4.0"
    return out


def clean_restricted(record: Dict[str, Any], keep_abstractive_answers: bool) -> Dict[str, Any]:
    repaired = repair_metadata(record)
    out = redact_secrets({field: repaired.get(field) for field in RESTRICTED_KEEP_FIELDS if field in repaired})
    if "title_source" in repaired:
        out["title_source"] = repaired["title_source"]
    out["document_hash"] = repaired.get("doc_id")
    out["redistribution_status"] = "metadata_only"
    out["release_license"] = "source_terms_apply"

    answer_style = str(record.get("answer_style", "")).lower()
    can_keep_answer = keep_abstractive_answers and answer_style != "extractive"
    if can_keep_answer and record.get("answer"):
        out["answer"] = record["answer"]
        out["answer_released"] = True
    else:
        out["answer_released"] = False

    return out


def audit_row(record: Dict[str, Any], status: str, answer_released: bool) -> Dict[str, Any]:
    repaired = repair_metadata(record) if record else record
    return {
        "record_id": repaired.get("id"),
        "document_hash": repaired.get("doc_id"),
        "doi": repaired.get("doi"),
        "title": repaired.get("title"),
        "title_source": repaired.get("title_source"),
        "year": repaired.get("year"),
        "venue": repaired.get("venue"),
        "license": repaired.get("license", "unknown"),
        "split": repaired.get("split"),
        "question_type": repaired.get("question_type"),
        "answer_style": repaired.get("answer_style"),
        "redistribution_status": status,
        "answer_released": answer_released,
        "source_pdf_present": bool(repaired.get("source_pdf")),
    }


def write_release_readme(path: Path, allowed_licenses: set[str]) -> None:
    path.write_text(
        "---\n"
        "license: cc-by-4.0\n"
        "language:\n"
        "- en\n"
        "pretty_name: MSQA-Bench\n"
        "task_categories:\n"
        "- question-answering\n"
        "- text-retrieval\n"
        "size_categories:\n"
        '- "1M<n<10M"\n'
        "tags:\n"
        "- mass-spectrometry\n"
        "- scientific-literature\n"
        "- retrieval-augmented-generation\n"
        "- croissant\n"
        "configs:\n"
        "- config_name: redistributable\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: redistributable/train.jsonl\n"
        "  - split: validation\n"
        "    path: redistributable/val.jsonl\n"
        "  - split: test\n"
        "    path: redistributable/test.jsonl\n"
        "- config_name: restricted\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: restricted/train.jsonl\n"
        "  - split: validation\n"
        "    path: restricted/val.jsonl\n"
        "  - split: test\n"
        "    path: restricted/test.jsonl\n"
        "---\n\n"
        "# MSQA-Bench Two-Tier Release\n\n"
        "This directory was generated by `scripts/prepare_neurips_release.py`.\n\n"
        "## Layout\n\n"
        "- `redistributable/*.jsonl`: records whose `license` value permits text-bearing release.\n"
        "- `restricted/*.jsonl`: metadata-only records for sources without redistribution-compatible terms.\n"
        "- `license_audit.csv`: per-record release decision log.\n"
        "- `summary.json`: split-level counts.\n\n"
        "Bibliographic fields are extracted automatically from PDFs and DOI/context metadata. "
        "The `title_source` field marks whether a title was taken from extracted metadata, "
        "inferred from a context footer, or left as an unverified extraction.\n\n"
        "## Redistributable Criterion\n\n"
        f"Allowed license values: `{', '.join(sorted(allowed_licenses))}`.\n\n"
        "Restricted records intentionally omit generated QA text, `context`, `evidence_spans`, `source_pdf`, and `answer` fields.\n"
        "Use `scripts/reconstruct_restricted_record.py` with a local PDF collection to reconstruct passages for private use.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("paper_results/dataset/splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allowed-license",
        action="append",
        default=["open_access"],
        help="License value considered redistributable. May be repeated.",
    )
    parser.add_argument(
        "--keep-abstractive-restricted-answers",
        action="store_true",
        help="Keep non-extractive answers in restricted records. Default is metadata-only.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per split.")
    args = parser.parse_args()

    allowed_licenses = {item.lower() for item in args.allowed_license}
    redistributable_dir = args.output_dir / "redistributable"
    restricted_dir = args.output_dir / "restricted"
    redistributable_dir.mkdir(parents=True, exist_ok=True)
    restricted_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"allowed_licenses": sorted(allowed_licenses), "splits": {}}
    audit_path = args.output_dir / "license_audit.csv"

    with audit_path.open("w", newline="", encoding="utf-8") as audit_file:
        fieldnames = list(audit_row({}, "", False).keys())
        writer = csv.DictWriter(audit_file, fieldnames=fieldnames)
        writer.writeheader()

        for split in SPLITS:
            input_path = args.input_dir / f"{split}.jsonl"
            if not input_path.exists():
                raise FileNotFoundError(input_path)

            licenses = Counter()
            redis_count = 0
            restricted_count = 0

            with (redistributable_dir / f"{split}.jsonl").open("w", encoding="utf-8") as redist_file, (
                restricted_dir / f"{split}.jsonl"
            ).open("w", encoding="utf-8") as restricted_file:
                for record in iter_jsonl(input_path, args.limit):
                    licenses[str(record.get("license", "unknown")).lower()] += 1
                    if is_redistributable(record, allowed_licenses):
                        cleaned = clean_redistributable(record)
                        redist_file.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                        redis_count += 1
                        writer.writerow(audit_row(record, "text_released", True))
                    else:
                        cleaned = clean_restricted(record, args.keep_abstractive_restricted_answers)
                        restricted_file.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                        restricted_count += 1
                        writer.writerow(audit_row(record, "metadata_only", cleaned.get("answer_released", False)))
            summary["splits"][split] = {
                "redistributable": redis_count,
                "restricted": restricted_count,
                "total": redis_count + restricted_count,
                "license_counts": dict(licenses),
            }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_release_readme(args.output_dir / "README.md", allowed_licenses)
    write_release_readme(args.output_dir / "README_RELEASE.md", allowed_licenses)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
