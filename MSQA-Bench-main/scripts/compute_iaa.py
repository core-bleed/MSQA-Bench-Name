#!/usr/bin/env python3
"""Compute inter-annotator agreement (Cohen's kappa) on the MSQA-Bench audit.

Given two independently annotated copies of the gold-set CSV (same row order and
same `annotation_id` column), this script reports per-rubric Cohen's kappa,
percentage agreement, and a confusion matrix for the four audit dimensions:

  - answer_correct
  - evidence_support
  - evidence_quality
  - question_clarity

It also writes an adjudicated CSV that flags rows where the two annotators
disagree, leaving the final label blank for a third-party adjudicator.

Usage:
    python scripts/compute_iaa.py \
        --annotator-a paper_results/annotation/gold_set_annotated.csv \
        --annotator-b paper_results/annotation/gold_set_annotated_v2.csv \
        --output paper_results/annotation/iaa_report.json \
        --adjudication-csv paper_results/annotation/gold_adjudication.csv

The script is dependency-light (csv + math from stdlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RUBRIC_FIELDS: Tuple[str, ...] = (
    "answer_correct",
    "evidence_support",
    "evidence_quality",
    "question_clarity",
)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aligned_pairs(
    rows_a: Sequence[Dict[str, str]],
    rows_b: Sequence[Dict[str, str]],
    field: str,
) -> List[Tuple[str, str]]:
    """Return (label_a, label_b) pairs aligned by `annotation_id`."""
    by_id_b = {row["annotation_id"]: row for row in rows_b}
    pairs: List[Tuple[str, str]] = []
    for row in rows_a:
        ann_id = row.get("annotation_id")
        if ann_id is None or ann_id not in by_id_b:
            continue
        a = (row.get(field) or "").strip()
        b = (by_id_b[ann_id].get(field) or "").strip()
        if not a or not b:
            continue
        pairs.append((a, b))
    return pairs


def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Cohen's kappa, percentage agreement, and N for two annotators."""
    if not pairs:
        return {"n": 0, "agreement": 0.0, "kappa": 0.0}
    n = len(pairs)
    labels = sorted({l for pair in pairs for l in pair})
    obs = sum(1 for a, b in pairs if a == b) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    expected = sum((counts_a[l] / n) * (counts_b[l] / n) for l in labels)
    kappa = (obs - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": n, "agreement": obs, "kappa": kappa}


def confusion_matrix(
    pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in pairs:
        matrix[a][b] += 1
    return {k: dict(v) for k, v in matrix.items()}


def write_adjudication(
    rows_a: Sequence[Dict[str, str]],
    rows_b: Sequence[Dict[str, str]],
    output: Path,
) -> int:
    by_id_b = {row["annotation_id"]: row for row in rows_b}
    fieldnames = [
        "annotation_id",
        "question_type",
        "question",
        "answer",
        *(f"{f}_a" for f in RUBRIC_FIELDS),
        *(f"{f}_b" for f in RUBRIC_FIELDS),
        *(f"{f}_final" for f in RUBRIC_FIELDS),
    ]
    disagreements = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_a:
            ann_id = row.get("annotation_id")
            row_b = by_id_b.get(ann_id, {})
            out: Dict[str, str] = {
                "annotation_id": ann_id or "",
                "question_type": row.get("question_type", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            }
            disagreed = False
            for field in RUBRIC_FIELDS:
                a = (row.get(field) or "").strip()
                b = (row_b.get(field) or "").strip()
                out[f"{field}_a"] = a
                out[f"{field}_b"] = b
                if a == b and a:
                    out[f"{field}_final"] = a
                else:
                    out[f"{field}_final"] = ""
                    if a and b and a != b:
                        disagreed = True
            if disagreed:
                disagreements += 1
            writer.writerow(out)
    return disagreements


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotator-a", required=True, type=Path)
    p.add_argument("--annotator-b", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--adjudication-csv", type=Path, default=None)
    args = p.parse_args(argv)

    rows_a = _load_csv(args.annotator_a)
    rows_b = _load_csv(args.annotator_b)

    report: Dict[str, Dict[str, object]] = {"per_field": {}, "summary": {}}
    for field in RUBRIC_FIELDS:
        pairs = _aligned_pairs(rows_a, rows_b, field)
        stats = cohen_kappa(pairs)
        report["per_field"][field] = {
            **stats,
            "confusion_matrix": confusion_matrix(pairs),
        }

    n_aligned = len({r["annotation_id"] for r in rows_a} & {r["annotation_id"] for r in rows_b})
    report["summary"] = {
        "annotator_a": str(args.annotator_a),
        "annotator_b": str(args.annotator_b),
        "rows_aligned": n_aligned,
        "mean_kappa": (
            sum(report["per_field"][f]["kappa"] for f in RUBRIC_FIELDS) / len(RUBRIC_FIELDS)
        ),
    }

    if args.adjudication_csv is not None:
        report["summary"]["disagreements_written"] = write_adjudication(
            rows_a, rows_b, args.adjudication_csv
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")
    print(f"Rows aligned: {n_aligned}")
    for f in RUBRIC_FIELDS:
        s = report["per_field"][f]
        print(f"  {f}: kappa={s['kappa']:.3f}, agreement={s['agreement']:.3f} (n={s['n']})")
    print(f"Mean kappa: {report['summary']['mean_kappa']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
