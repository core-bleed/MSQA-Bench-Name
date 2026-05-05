#!/usr/bin/env python3
"""
End-to-end artifact smoke test for MSQA-Bench.

The smoke path intentionally stays small:
1. create a tiny PDF and run the shipped PyMuPDF extraction CLI;
2. download a small redistributable MSQA-Bench sample from Hugging Face;
3. run the shipped BM25 retrieval evaluator on that sample;
4. write a compact Markdown result table.

Use --sample-source fixture for offline CI or local verification.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper_results" / "smoke"
DATASET_NAME = "asad00027/MSQA-Bench"
DATASET_CONFIG = "redistributable"
DATASET_SPLIT = "test"


def require_modules(sample_source: str) -> None:
    required = {
        "fitz": "PyMuPDF",
        "pandas": "pandas",
        "unidecode": "unidecode",
        "numpy": "numpy",
        "rank_bm25": "rank-bm25",
    }
    if sample_source == "huggingface":
        required["datasets"] = "datasets"

    missing = []
    for module_name, package_name in required.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        packages = " ".join(sorted(missing))
        raise SystemExit(
            "Missing smoke-test dependencies. Install them with:\n"
            f"  python3 -m pip install -r requirements-smoke.txt\n"
            f"or install the missing packages directly: python3 -m pip install {packages}"
        )


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(as_text(item) for item in value if item is not None)
    return " ".join(json.dumps(value, ensure_ascii=False).split())


def normalize_record(record: dict[str, Any], index: int, source: str) -> dict[str, str] | None:
    question = as_text(record.get("question"))
    answer = as_text(record.get("answer"))
    if not (10 <= len(question) <= 256 and 20 <= len(answer) <= 512):
        return None

    original_id = ""
    for key in ("id", "record_id", "qa_id", "question_id"):
        if record.get(key):
            original_id = str(record[key])
            break

    smoke_id = original_id or f"{source}_{index}"
    return {
        "id": smoke_id,
        "source_id": original_id,
        "question": question,
        "answer": answer,
    }


def fixture_records(sample_size: int) -> list[dict[str, str]]:
    topics = [
        (
            "Which ionization method is used for peptide profiling?",
            "The workflow uses electrospray ionization with liquid chromatography tandem mass spectrometry for peptide profiling.",
        ),
        (
            "What does the precursor mass tolerance control?",
            "The precursor mass tolerance controls how far an observed precursor ion may deviate from the theoretical peptide mass.",
        ),
        (
            "Which metric summarizes top ten retrieval success?",
            "Recall at ten summarizes whether the relevant evidence passage appears anywhere in the top ten retrieved passages.",
        ),
        (
            "Why are metadata-only records separated from text records?",
            "Metadata-only records preserve provenance for restricted sources while avoiding redistribution of source text or derived QA text.",
        ),
        (
            "What role does collision energy play in MS/MS?",
            "Collision energy fragments selected precursor ions so tandem spectra contain product ions useful for molecular identification.",
        ),
    ]

    records = []
    for index in range(sample_size):
        question, answer = topics[index % len(topics)]
        records.append(
            {
                "id": f"fixture_{index}",
                "source_id": f"fixture_{index}",
                "question": question,
                "answer": answer,
            }
        )
    return records


def download_huggingface_sample(sample_size: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    stream = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    records = []
    scanned = 0
    for record in stream:
        scanned += 1
        normalized = normalize_record(record, len(records), "hf_smoke")
        if normalized is not None:
            records.append(normalized)
        if len(records) >= sample_size:
            break
        if scanned >= max(sample_size * 50, 500):
            break

    if not records:
        raise RuntimeError(
            "No usable Hugging Face records were found. Check network access and dataset availability."
        )
    if len(records) < sample_size:
        print(
            f"Warning: requested {sample_size} records but found {len(records)} usable records.",
            file=sys.stderr,
        )
    return records


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_tiny_pdf(pdf_path: Path) -> None:
    import fitz

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    text = (
        "MSQA-Bench smoke extraction sample\n\n"
        "A liquid chromatography tandem mass spectrometry workflow identifies "
        "peptides from precursor and fragment ions. The tiny document exists "
        "only to verify that the released extraction command runs end to end."
    )
    page.insert_textbox(fitz.Rect(72, 72, 540, 220), text, fontsize=11)
    doc.save(pdf_path)
    doc.close()


def run_extraction(output_dir: Path) -> tuple[Path, Path]:
    extraction_dir = output_dir / "extraction"
    pdf_path = extraction_dir / "tiny_ms_sample.pdf"
    out_txt = extraction_dir / "tiny_ms_sample.txt"
    out_csv = extraction_dir / "tiny_ms_sample.csv"
    create_tiny_pdf(pdf_path)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "pdf_processors" / "pymupdf_processor.py"),
        str(pdf_path),
        "--out-txt",
        str(out_txt),
        "--out-csv",
        str(out_csv),
    ]
    subprocess.run(command, check=True)

    extracted = out_txt.read_text(encoding="utf-8")
    if "tandem mass spectrometry" not in extracted:
        raise RuntimeError(f"Extraction output did not contain expected text: {out_txt}")
    return out_txt, out_csv


def load_bm25_evaluator():
    evaluator_path = PROJECT_ROOT / "scripts" / "evaluate_bm25_baseline.py"
    spec = importlib.util.spec_from_file_location("evaluate_bm25_baseline", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_bm25


def run_retrieval_evaluation(sample_path: Path, sample_size: int, output_dir: Path) -> dict[str, Any]:
    evaluate_bm25 = load_bm25_evaluator()
    results = evaluate_bm25(
        str(sample_path),
        sample_size=sample_size,
        output_dir=str(output_dir / "evaluation"),
        split="all",
    )
    if results.get("num_queries", 0) <= 0:
        raise RuntimeError("BM25 smoke evaluation produced zero queries.")
    return results


def write_result_table(results: dict[str, Any], table_path: Path) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Model | Queries | Corpus | R@1 | R@5 | R@10 | MRR@10 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {results['model']} | {results['num_queries']} | {results['num_corpus']} | "
            f"{results['recall@1']:.4f} | {results['recall@5']:.4f} | "
            f"{results['recall@10']:.4f} | {results['mrr@10']:.4f} | "
            f"{results['ndcg@10']:.4f} |"
        ),
        "",
    ]
    table_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MSQA-Bench artifact smoke test.")
    parser.add_argument(
        "--sample-source",
        choices=("huggingface", "fixture"),
        default="huggingface",
        help="Use Hugging Face data by default; fixture mode is offline.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="Number of QA records to evaluate in the smoke retrieval run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for smoke artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")

    require_modules(args.sample_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1/4 Running tiny PDF extraction...")
    out_txt, out_csv = run_extraction(args.output_dir)

    print("2/4 Preparing QA sample...")
    if args.sample_source == "huggingface":
        records = download_huggingface_sample(args.sample_size)
    else:
        records = fixture_records(args.sample_size)
    sample_path = args.output_dir / "msqa_smoke_sample.jsonl"
    write_jsonl(records, sample_path)

    print("3/4 Running BM25 retrieval evaluation...")
    results = run_retrieval_evaluation(sample_path, len(records), args.output_dir)

    print("4/4 Writing smoke result table...")
    table_path = args.output_dir / "smoke_retrieval_table.md"
    write_result_table(results, table_path)

    print("\nMSQA-Bench smoke test completed.")
    print(f"Extraction text: {out_txt}")
    print(f"Extraction CSV:  {out_csv}")
    print(f"QA sample:       {sample_path}")
    print(f"BM25 results:    {args.output_dir / 'evaluation' / 'bm25_baseline_results.json'}")
    print(f"Result table:    {table_path}")
    print("\n" + table_path.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
