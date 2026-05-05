#!/usr/bin/env python3
"""Reconstruct restricted MSQA-Bench passages from a user's local PDFs.

This helper does not download or redistribute restricted source text. It only
uses PDFs already available on the user's machine. PDF files are matched by
`document_hash`/`doc_id` stem, for example `<pdf-dir>/<document_hash>.pdf`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required: pip install PyMuPDF") from exc


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def extract_text(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n\n".join(page.get_text("text") for page in doc)


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_pdf(pdf_dir: Path, record: Dict[str, Any]) -> Optional[Path]:
    for key in ("document_hash", "doc_id"):
        value = record.get(key)
        if value:
            candidate = pdf_dir / f"{value}.pdf"
            if candidate.exists():
                return candidate
    return None


def reconstruct_record(record: Dict[str, Any], pdf_dir: Path) -> Dict[str, Any]:
    out = dict(record)
    pdf_path = find_pdf(pdf_dir, record)
    if not pdf_path:
        out["reconstruction_status"] = "missing_local_pdf"
        return out

    paragraphs = split_paragraphs(extract_text(pdf_path))
    index = record.get("paragraph_index")
    if isinstance(index, int) and 0 <= index < len(paragraphs):
        out["reconstructed_context"] = paragraphs[index]
        out["reconstruction_status"] = "paragraph_index_match"
    else:
        out["reconstruction_status"] = "pdf_found_paragraph_index_unavailable"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restricted-jsonl", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(args.restricted_jsonl):
            handle.write(json.dumps(reconstruct_record(record, args.pdf_dir), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
