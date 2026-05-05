#!/usr/bin/env python3
"""
BM25 baseline evaluation on MSQA-Bench test set.

Replicates the exact data loading and metric computation from
src/embedding_trainers/evaluators.py and data_utils.py, but avoids
importing them (they pull in torch/sentence_transformers).

Usage:
    python scripts/evaluate_bm25_baseline.py
    python scripts/evaluate_bm25_baseline.py --data /path/to/consolidated_qa.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "output_consolidated" / "consolidated_qa.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_results" / "evaluation"


# ---------------------------------------------------------------------------
# Inlined from src/embedding_trainers/data_utils.py to avoid torch import
# ---------------------------------------------------------------------------

def get_split(record_id: str, train_ratio: float = 0.85, val_ratio: float = 0.10) -> str:
    h = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    train_threshold = int(train_ratio * 100)
    val_threshold = train_threshold + int(val_ratio * 100)
    if h < train_threshold:
        return "train"
    elif h < val_threshold:
        return "val"
    return "test"


def clean_answer(text: str, max_len: int = 512) -> str:
    if not text:
        return ""
    text = re.sub(r'\[\d+(?:[,\-–]\s*\d+)*\]', '', text)
    text = re.sub(r'\(\d+(?:[,\-–]\s*\d+)*\)', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'10\.\d{4,}/\S+', '', text)
    text = re.sub(
        r'(?:Fig\.|Figure|Table|Supplementary\s+(?:Fig|Table|Material))\s*\d+[A-Za-z]?',
        '', text, flags=re.IGNORECASE,
    )
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = ' '.join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0]
    return text.strip()


def clean_question(text: str, max_len: int = 256) -> str:
    if not text:
        return ""
    text = ' '.join(text.split())
    if len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0]
        if not text.endswith('?'):
            text = text + '?'
    return text.strip()


def load_split_samples(
    jsonl_path: str,
    split: str,
    sample_size: int,
) -> dict:
    """Load samples identically to data_utils.load_split_samples by default."""
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if count >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'question' not in record or 'answer' not in record:
                continue

            record_id = record.get('id', str(line_num))
            if split != "all" and get_split(record_id) != split:
                continue

            question = clean_question(record['question'].strip())
            answer = clean_answer(record['answer'].strip())

            if not (10 <= len(question) <= 256):
                continue
            if not (20 <= len(answer) <= 512):
                continue

            qid = f"q_{record_id}"
            cid = f"c_{record_id}"
            queries[qid] = question
            corpus[cid] = answer
            relevant_docs[qid] = {cid}
            count += 1

    split_label = "all records" if split == "all" else f"{split} split"
    logger.info(f"Loaded {count} samples from {split_label}")
    return {"queries": queries, "corpus": corpus, "relevant_docs": relevant_docs}


# ---------------------------------------------------------------------------
# BM25 evaluation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def evaluate_bm25(
    jsonl_path: str,
    sample_size: int = 5000,
    output_dir: str | None = None,
    split: str = "test",
) -> dict:
    logger.info("Loading data...")
    data = load_split_samples(jsonl_path, split, sample_size)

    queries = data["queries"]
    corpus = data["corpus"]
    relevant_docs = data["relevant_docs"]
    logger.info(f"Loaded {len(queries)} queries, {len(corpus)} corpus documents")

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    if not query_ids:
        raise ValueError(
            "No valid test queries were loaded. Check the input JSONL fields, "
            "record ids, split assignment, and --sample-size."
        )

    # Build BM25 index
    logger.info("Building BM25 index...")
    t0 = time.time()
    tokenized_corpus = [tokenize(doc) for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built in {time.time() - t0:.1f}s")

    # Pre-build corpus_id -> index map for O(1) lookup
    cid_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}

    # Evaluate
    logger.info("Scoring queries...")
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    recall_at_10 = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for i, qid in enumerate(query_ids):
        if (i + 1) % 1000 == 0:
            logger.info(f"  {i + 1}/{len(query_ids)} queries scored...")

        relevant = relevant_docs[qid]
        scores = bm25.get_scores(tokenize(queries[qid]))
        ranking = np.argsort(-scores)

        relevant_indices = [cid_to_idx[cid] for cid in relevant if cid in cid_to_idx]
        if not relevant_indices:
            continue

        ranks = [int(np.where(ranking == idx)[0][0]) + 1 for idx in relevant_indices]
        min_rank = min(ranks)

        if min_rank <= 1:
            recall_at_1 += 1
        if min_rank <= 5:
            recall_at_5 += 1
        if min_rank <= 10:
            recall_at_10 += 1

        if min_rank <= 10:
            mrr_sum += 1.0 / min_rank

        dcg = sum(1.0 / np.log2(r + 1) for r in ranks if r <= 10)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(relevant_indices), 10)))
        ndcg_sum += dcg / idcg if idcg > 0 else 0

    n = len(query_ids)
    results = {
        "model": "BM25 (Okapi)",
        "recall@1": recall_at_1 / n,
        "recall@5": recall_at_5 / n,
        "recall@10": recall_at_10 / n,
        "mrr@10": mrr_sum / n,
        "ndcg@10": ndcg_sum / n,
        "map@10": mrr_sum / n,
        "num_queries": n,
        "num_corpus": len(corpus),
    }

    logger.info("=== BM25 Baseline Results ===")
    for k in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
        logger.info(f"  {k:12s}: {results[k]:.4f}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / "bm25_baseline_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {out_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 baseline for MSQA-Bench")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Path to consolidated QA JSONL")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of test queries (default: 5000)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"),
                        help="Deterministic split to evaluate, or all records")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output directory for results JSON")
    args = parser.parse_args()
    evaluate_bm25(args.data, args.sample_size, args.output, split=args.split)


if __name__ == "__main__":
    main()
