"""Tests for the inter-annotator-agreement helper."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "compute_iaa.py"

FIELDS = [
    "annotation_id",
    "question",
    "answer",
    "context",
    "question_type",
    "quality_score",
    "answer_correct",
    "evidence_support",
    "evidence_quality",
    "question_clarity",
    "annotator_id",
    "notes",
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _make_rows(n: int, base: dict[str, str]) -> list[dict[str, str]]:
    return [{**base, "annotation_id": str(i), "question": f"q{i}", "answer": f"a{i}"} for i in range(n)]


def _run(a: Path, b: Path, out: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--annotator-a", str(a), "--annotator-b", str(b), "--output", str(out)],
        check=True,
        capture_output=True,
    )
    return json.loads(out.read_text())


def test_perfect_agreement_yields_kappa_one(tmp_path: Path) -> None:
    rows = _make_rows(40, {"answer_correct": "Yes", "evidence_support": "Yes",
                            "evidence_quality": "Good", "question_clarity": "Good"})
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_csv(a, rows); _write_csv(b, rows)
    rep = _run(a, b, tmp_path / "out.json")
    for field in ("answer_correct", "evidence_support", "evidence_quality", "question_clarity"):
        # When all labels are identical, kappa is undefined (denominator 0); the script returns 1.0.
        assert rep["per_field"][field]["kappa"] == pytest.approx(1.0)
        assert rep["per_field"][field]["agreement"] == pytest.approx(1.0)


def test_pure_disagreement_yields_kappa_below_perfect(tmp_path: Path) -> None:
    rows_a = _make_rows(40, {"answer_correct": "Yes"})
    rows_b = []
    for i, row in enumerate(rows_a):
        copy = dict(row)
        # Flip every other row to "No" so we get 50% agreement on a 2-class field.
        if i % 2:
            copy["answer_correct"] = "No"
        rows_b.append(copy)
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_csv(a, rows_a); _write_csv(b, rows_b)
    rep = _run(a, b, tmp_path / "out.json")
    stats = rep["per_field"]["answer_correct"]
    assert stats["agreement"] == pytest.approx(0.5)
    assert stats["kappa"] < 0.6  # well below perfect


def test_unaligned_rows_are_skipped(tmp_path: Path) -> None:
    rows_a = _make_rows(10, {"answer_correct": "Yes"})
    rows_b = _make_rows(5, {"answer_correct": "Yes"})  # only first 5 ids
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    _write_csv(a, rows_a); _write_csv(b, rows_b)
    rep = _run(a, b, tmp_path / "out.json")
    assert rep["per_field"]["answer_correct"]["n"] == 5
