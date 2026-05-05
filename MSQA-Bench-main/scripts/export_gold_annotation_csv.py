#!/usr/bin/env python3
"""Sample N records stratified by question_type and emit a CSV ready for human gold annotation.

Closes the C3 gap (no human-validated audit). Output is a CSV the user (or two
annotators) fills in with rubric labels, then re-imports to compute κ.

Columns emitted:
  id, question_type, question, answer, context, doi,
  ann1_correctness, ann1_evidence, ann1_clarity, ann1_evidence_quality, ann1_notes,
  ann2_correctness, ann2_evidence, ann2_clarity, ann2_evidence_quality, ann2_notes

Each rubric column expects: yes | partial | no  (clarity uses: clear | ambiguous | bad).

Usage:
  python scripts/export_gold_annotation_csv.py \
      --input paper_results/dataset/splits/test.jsonl \
      --n 100 --out paper_results/annotation/gold_human.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

QUESTION_TYPES = ["factual", "definition", "method", "causal",
                  "comparison", "numeric", "unknown"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--n", type=int, default=100, help="Total sample size (stratified)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    by_type: dict[str, list[dict]] = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qt = r.get("question_type", "unknown")
            if qt in QUESTION_TYPES and r.get("context"):
                by_type[qt].append(r)

    per_type = max(1, args.n // len(QUESTION_TYPES))
    log.info("Stratifying %d records (~%d per type) across %d types",
             args.n, per_type, len(QUESTION_TYPES))

    sampled: list[dict] = []
    for qt in QUESTION_TYPES:
        pool = by_type.get(qt, [])
        if not pool:
            log.warning("No records for type=%s", qt)
            continue
        k = min(per_type, len(pool))
        sampled.extend(random.sample(pool, k))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "question_type", "question", "answer", "context", "doi",
            "ann1_correctness", "ann1_evidence", "ann1_clarity",
            "ann1_evidence_quality", "ann1_notes",
            "ann2_correctness", "ann2_evidence", "ann2_clarity",
            "ann2_evidence_quality", "ann2_notes",
        ])
        for r in sampled:
            w.writerow([
                r.get("id", ""),
                r.get("question_type", ""),
                (r.get("question") or "")[:1000],
                (r.get("answer") or "")[:1000],
                (r.get("context") or "")[:2000],
                r.get("doi", ""),
                "", "", "", "", "",
                "", "", "", "", "",
            ])
    log.info("Wrote %d rows to %s", len(sampled), out_path)


if __name__ == "__main__":
    main()
