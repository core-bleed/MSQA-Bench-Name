"""Tests for the deterministic document-level split generator."""

import pytest

from paper.dataset.split_generator import SplitConfig, SplitGenerator


@pytest.fixture
def gen() -> SplitGenerator:
    return SplitGenerator(SplitConfig(train_ratio=0.85, val_ratio=0.10, test_ratio=0.05, seed=42))


def test_split_is_deterministic_across_invocations(gen: SplitGenerator) -> None:
    doc_id = "abc123"
    s1 = gen.get_split(doc_id)
    s2 = gen.get_split(doc_id)
    assert s1 == s2


def test_split_only_returns_known_labels(gen: SplitGenerator) -> None:
    seen = {gen.get_split(f"doc_{i}") for i in range(500)}
    assert seen <= {"train", "val", "test"}


def test_split_distribution_matches_target_ratios_within_tolerance(gen: SplitGenerator) -> None:
    n = 5000
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n):
        counts[gen.get_split(f"doc_{i}")] += 1
    # ratios should be close to 0.85 / 0.10 / 0.05 (within 2pp on n=5000)
    assert abs(counts["train"] / n - 0.85) < 0.02
    assert abs(counts["val"] / n - 0.10) < 0.02
    assert abs(counts["test"] / n - 0.05) < 0.02


def test_assign_splits_keeps_same_doc_in_same_split(gen: SplitGenerator) -> None:
    records = [
        {"doc_id": "A", "qid": i} for i in range(50)
    ] + [
        {"doc_id": "B", "qid": i} for i in range(50)
    ] + [
        {"doc_id": "C", "qid": i} for i in range(50)
    ]
    train, val, test = gen.assign_splits(records, doc_id_field="doc_id")
    by_doc: dict[str, set[str]] = {}
    for split_name, split_records in (("train", train), ("val", val), ("test", test)):
        for record in split_records:
            by_doc.setdefault(record["doc_id"], set()).add(split_name)
    # Each document should land in exactly one split.
    for doc, splits in by_doc.items():
        assert len(splits) == 1, f"document {doc} leaked across splits {splits}"


def test_invalid_ratios_rejected() -> None:
    with pytest.raises(ValueError):
        SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_changing_seed_changes_assignment() -> None:
    a = SplitGenerator(SplitConfig(seed=1))
    b = SplitGenerator(SplitConfig(seed=2))
    different = sum(a.get_split(f"d_{i}") != b.get_split(f"d_{i}") for i in range(200))
    # With independent hashes, ~half the docs should differ; assert "non-trivial" change.
    assert different > 30
