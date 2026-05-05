#!/usr/bin/env python3
"""Patch croissant.jsonld to declare all release JSONL fields.

Closes the C4 gap (Croissant 17 fields vs JSONL 30 fields) and the C6 gap
(huggingface-repo placeholder sha256). Idempotent: re-running rewrites the file
to a fully consistent state.

Usage:
  python scripts/fix_croissant_schema.py \
      --in paper_results/neurips_release/croissant.jsonld \
      --out paper_results/neurips_release/croissant.jsonld
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REDIST_FIELDS = [
    ("id", "sc:Text", "Unique QA pair UUID"),
    ("question", "sc:Text", "Natural-language question"),
    ("answer", "sc:Text", "Reference answer (extractive or abstractive)"),
    ("context", "sc:Text", "Source paragraph supporting the answer"),
    ("file_name", "sc:Text", "Source PDF basename"),
    ("paragraph_index", "sc:Integer", "0-based paragraph index in document"),
    ("line_number", "sc:Integer", "0-based line number in extracted text"),
    ("doc_id", "sc:Text", "Document identifier (SHA-256 of source PDF)"),
    ("doi", "sc:Text", "DOI when available"),
    ("pmid", "sc:Text", "PubMed ID when available"),
    ("arxiv_id", "sc:Text", "arXiv ID when available"),
    ("title", "sc:Text", "Source paper title"),
    ("title_source", "sc:Text", "How the title field was obtained: extracted, context_footer, or unverified_extracted"),
    ("year", "sc:Integer", "Publication year"),
    ("venue", "sc:Text", "Publication venue"),
    ("license", "sc:Text", "Inferred source license (open_access | unknown)"),
    ("context_offsets", "sc:Text", "JSON list [start,end] character offsets of context within full document"),
    ("evidence_spans", "sc:Text", "JSON list of [start,end] offsets of evidence within context"),
    ("question_type", "sc:Text", "{factual, definition, method, causal, comparison, numeric, unknown}"),
    ("answer_style", "sc:Text", "{extractive, abstractive}"),
    ("quality_score", "sc:Float", "Composite quality in [0,1]"),
    ("split", "sc:Text", "{train, val, test}"),
    ("context_id", "sc:Text", "SHA-256 of normalized context text"),
    ("model", "sc:Text", "QA generator model identifier"),
    ("run_id", "sc:Text", "Generation run identifier"),
    ("created_at", "sc:Date", "QA pair creation timestamp"),
    ("enriched_at", "sc:Date", "Schema enrichment timestamp"),
    ("enrichment_version", "sc:Text", "Enrichment schema version"),
    ("document_hash", "sc:Text", "SHA-256 of source PDF (alias of doc_id)"),
    ("redistribution_status", "sc:Text", "{text_released, metadata_only}"),
    ("release_license", "sc:Text", "License under which this record is released (CC-BY-4.0 for redistributable)"),
]

# Restricted records omit generated QA text and source text-bearing fields.
RESTRICTED_FIELDS = [
    f for f in REDIST_FIELDS
    if f[0] not in {"question", "answer", "context", "evidence_spans", "context_offsets"}
] + [
    ("answer_released", "sc:Boolean", "True iff the answer text is included in this record"),
]


def build_field(parent_id: str, name: str, dtype: str, desc: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"{parent_id}/{name}",
        "name": name,
        "description": desc,
        "dataType": dtype,
        "source": {
            "fileSet": {"@id": f"{parent_id.split('-records')[0]}-jsonl"},
            "extract": {"jsonPath": f"$.{name}"},
        },
    }


def patch_recordset(rs: dict, fields: list[tuple[str, str, str]]) -> dict:
    rs_id = rs.get("@id", rs.get("name", "records"))
    rs["field"] = [build_field(rs_id, n, t, d) for n, t, d in fields]
    return rs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp_path = Path(args.inp)
    with inp_path.open() as f:
        cr = json.load(f)

    context = cr.setdefault("@context", {})
    context.setdefault("prov", "http://www.w3.org/ns/prov#")

    rs_list = cr.get("recordSet", [])
    redist = next((r for r in rs_list if "redistributable" in r.get("@id", "")), None)
    restricted = next((r for r in rs_list if "restricted" in r.get("@id", "")), None)
    if redist is None or restricted is None:
        raise SystemExit("Could not find redistributable/restricted record sets in @id")

    patch_recordset(redist, REDIST_FIELDS)
    patch_recordset(restricted, RESTRICTED_FIELDS)

    # C6: drop the huggingface-repo FileObject (it's a repo, not a hashable file).
    fo = cr.get("distribution", [])
    base_dir = inp_path.parent
    fo = [d for d in fo if d.get("@id") != "huggingface-repo"]
    for entry in fo:
        if entry.get("@type") == "cr:FileObject":
            entry.pop("containedIn", None)
            content_url = entry.get("contentUrl")
            if isinstance(content_url, str) and not content_url.startswith(("http://", "https://")):
                local_path = base_dir / content_url
                if local_path.exists() and local_path.is_file():
                    entry["sha256"] = sha256_file(local_path)

    # Add cr:FileSet groupings the field sources point at.
    redist_files = [d["@id"] for d in fo if d.get("@id", "").startswith("redistributable-")]
    restricted_files = [d["@id"] for d in fo if d.get("@id", "").startswith("restricted-")]
    fo = [d for d in fo if d.get("@id") not in {"redistributable-jsonl", "restricted-jsonl"}]
    if redist_files:
        fo.append({
            "@type": "cr:FileSet",
            "@id": "redistributable-jsonl",
            "name": "redistributable-jsonl",
            "description": "Redistributable train/val/test JSONL splits",
            "containedIn": [{"@id": fid} for fid in redist_files],
            "encodingFormat": "application/jsonlines",
            "includes": "*.jsonl",
        })
    if restricted_files:
        fo.append({
            "@type": "cr:FileSet",
            "@id": "restricted-jsonl",
            "name": "restricted-jsonl",
            "description": "Restricted (metadata-only) train/val/test JSONL splits",
            "containedIn": [{"@id": fid} for fid in restricted_files],
            "encodingFormat": "application/jsonlines",
            "includes": "*.jsonl",
        })
    cr["distribution"] = fo

    cr["citeAs"] = cr.get("citeAs", "").replace(
        "NeurIPS 2026 Datasets and Benchmarks Track",
        "NeurIPS 2026 Evaluations and Datasets Track",
    )
    cr["rai:dataBiases"] = (
        "Coverage skews toward English-language, peer-reviewed MS literature in "
        "major analytical-chemistry and proteomics venues; instrument types and "
        "methods covered in those venues are over-represented relative to niche "
        "or industrial MS workflows. A small minority of records are non-English "
        "or carry OCR/extraction artifacts retained for completeness. "
        "Question-type distribution is skewed toward factual (47.7%) and "
        "definition (18.3%) classes. QA pairs inherit biases from the "
        "Qwen2.5-14B-Instruct-AWQ generator."
    )
    cr["rai:dataLimitations"] = (
        "MSQA-Bench should not be used as ground truth for clinical, regulatory, "
        "or industrial decisions. Synthetic QA pairs may inherit generator biases "
        "and should not replace expert human verification in safety-critical "
        "settings. The human audit covers a stratified 200-record sample "
        "(191 completed labels), so it is an estimate of quality rather than an "
        "exhaustive expert-gold benchmark."
    )
    cr["rai:annotationsPerItem"] = (
        "Automatic QA generation and automatic metadata labels for all records; "
        "human rubric labels for the released 200-record audit sample."
    )
    cr["rai:annotatorDemographics"] = (
        "Human annotator demographic attributes were not collected for the audit "
        "sample; annotation focused on technical correctness, evidence support, "
        "question clarity, and evidence quality."
    )
    cr["rai:dataSocialImpact"] = (
        "The dataset can improve retrieval and grounded QA tools for mass "
        "spectrometry literature, potentially reducing literature-search burden. "
        "Risks include over-trust in synthetic answers, propagation of extraction "
        "errors, and misuse as clinical or regulatory ground truth."
    )
    cr["rai:hasSyntheticData"] = True
    cr["rai:hasHumanAnnotation"] = True
    cr["rai:dataAnnotationProtocol"] = (
        "A stratified 200-record audit sample was reviewed with four rubric "
        "dimensions: answer correctness, evidence support, question clarity, and "
        "evidence quality. Completed labels are released in "
        "paper_results/annotation/gold_set_annotated.csv."
    )
    cr["prov:wasDerivedFrom"] = {
        "@type": "sc:Dataset",
        "name": "Mass spectrometry publications discovered via the Semantic Scholar API",
        "url": "https://www.semanticscholar.org/product/api",
    }
    cr["prov:wasGeneratedBy"] = {
        "@type": "sc:SoftwareApplication",
        "name": "MSQA-Bench construction pipeline",
        "softwareRequirements": "Python, PyMuPDF, GROBID, vLLM, Qwen2.5-14B-Instruct-AWQ",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cr, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"  redistributable fields: {len(REDIST_FIELDS)}")
    print(f"  restricted fields:      {len(RESTRICTED_FIELDS)}")
    print(f"  distribution entries:   {len(cr['distribution'])}")


if __name__ == "__main__":
    main()
