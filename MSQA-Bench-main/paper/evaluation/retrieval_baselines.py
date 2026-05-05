"""
Retrieval Baselines for MSQA-Bench.

Implements:
1. BM25 sparse retrieval baseline
2. Dense embedding retrieval (base and fine-tuned models)
3. Unified evaluation framework with standard metrics

Metrics:
- Recall@k (k=1, 5, 10, 20)
- MRR@k
- NDCG@k
- MAP@k
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank_bm25 not installed. BM25 baseline disabled.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Embedding retrieval disabled.")


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    query_id: str
    query: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    relevant_ids: List[str]
    
    # Computed metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'query': self.query[:100],
            'num_retrieved': len(self.retrieved_ids),
            'num_relevant': len(self.relevant_ids),
            'recall@1': self.recall_at_1,
            'recall@5': self.recall_at_5,
            'recall@10': self.recall_at_10,
            'recall@20': self.recall_at_20,
            'mrr': self.mrr,
            'ndcg@10': self.ndcg_at_10,
        }


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    num_queries: int
    num_corpus: int
    
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    map_at_10: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_queries': self.num_queries,
            'num_corpus': self.num_corpus,
            'recall@1': round(self.recall_at_1, 4),
            'recall@5': round(self.recall_at_5, 4),
            'recall@10': round(self.recall_at_10, 4),
            'recall@20': round(self.recall_at_20, 4),
            'mrr@10': round(self.mrr_at_10, 4),
            'ndcg@10': round(self.ndcg_at_10, 4),
            'map@10': round(self.map_at_10, 4),
        }
    
    def __str__(self) -> str:
        return (
            f"R@1={self.recall_at_1:.4f}, R@5={self.recall_at_5:.4f}, "
            f"R@10={self.recall_at_10:.4f}, MRR@10={self.mrr_at_10:.4f}, "
            f"NDCG@10={self.ndcg_at_10:.4f}"
        )


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def index(self, corpus: Dict[str, str]) -> None:
        """Index the corpus for retrieval."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query.
        
        Returns:
            List of (doc_id, score) tuples
        """
        pass
    
    def retrieve_batch(
        self, 
        queries: Dict[str, str], 
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve for multiple queries."""
        results = {}
        for qid, query in queries.items():
            results[qid] = self.retrieve(query, top_k)
        return results


class BM25Retriever(BaseRetriever):
    """BM25 sparse retrieval baseline."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        if not HAS_BM25:
            raise ImportError("rank_bm25 is required for BM25Retriever")
        
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.corpus_texts: List[str] = []
    
    def index(self, corpus: Dict[str, str]) -> None:
        """Index the corpus for BM25 retrieval."""
        self.doc_ids = list(corpus.keys())
        self.corpus_texts = list(corpus.values())
        
        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_texts]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 indexed {len(self.doc_ids)} documents")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k documents using BM25."""
        if self.bm25 is None:
            raise ValueError("Corpus not indexed. Call index() first.")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return results


class EmbeddingRetriever(BaseRetriever):
    """Dense embedding retrieval using sentence-transformers."""
    
    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize embedding retriever.
        
        Args:
            model_name_or_path: Model name or path to fine-tuned model
            device: Device for inference
            batch_size: Batch size for encoding
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required for EmbeddingRetriever")
        if not HAS_NUMPY:
            raise ImportError("numpy is required for EmbeddingRetriever")
        
        self.model_name = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name_or_path}")
        self.model = SentenceTransformer(model_name_or_path, device=device)
        
        self.doc_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
    
    def index(self, corpus: Dict[str, str]) -> None:
        """Index the corpus by computing embeddings."""
        self.doc_ids = list(corpus.keys())
        corpus_texts = list(corpus.values())
        
        logger.info(f"Encoding {len(corpus_texts)} documents...")
        
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        
        logger.info(f"Embedding shape: {self.corpus_embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k documents using embedding similarity."""
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not indexed. Call index() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        
        # Compute cosine similarity (embeddings are normalized)
        similarities = np.dot(self.corpus_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def retrieve_batch(
        self, 
        queries: Dict[str, str], 
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch retrieval for efficiency."""
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not indexed. Call index() first.")
        
        query_ids = list(queries.keys())
        query_texts = list(queries.values())
        
        # Batch encode queries
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Compute all similarities at once
        all_similarities = np.dot(query_embeddings, self.corpus_embeddings.T)
        
        results = {}
        for i, qid in enumerate(query_ids):
            top_indices = np.argsort(-all_similarities[i])[:top_k]
            results[qid] = [
                (self.doc_ids[idx], float(all_similarities[i][idx]))
                for idx in top_indices
            ]
        
        return results


class RetrievalEvaluator:
    """Evaluate retrieval performance with standard metrics."""
    
    def __init__(self, retriever: BaseRetriever):
        """
        Initialize evaluator.
        
        Args:
            retriever: Retriever to evaluate
        """
        self.retriever = retriever
    
    def evaluate(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        relevant_docs: Dict[str, List[str]],
        top_k: int = 20,
    ) -> EvaluationMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: Dict mapping query_id to query text
            corpus: Dict mapping doc_id to document text
            relevant_docs: Dict mapping query_id to list of relevant doc_ids
            top_k: Maximum k for evaluation
            
        Returns:
            EvaluationMetrics with aggregated scores
        """
        # Index corpus
        self.retriever.index(corpus)
        
        # Retrieve for all queries
        if hasattr(self.retriever, 'retrieve_batch'):
            all_results = self.retriever.retrieve_batch(queries, top_k)
        else:
            all_results = {}
            for qid, query in queries.items():
                all_results[qid] = self.retriever.retrieve(query, top_k)
        
        # Compute metrics for each query
        recall_at_1_sum = 0.0
        recall_at_5_sum = 0.0
        recall_at_10_sum = 0.0
        recall_at_20_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        map_sum = 0.0
        
        for qid, query in queries.items():
            retrieved = all_results.get(qid, [])
            relevant = set(relevant_docs.get(qid, []))
            
            if not relevant:
                continue
            
            retrieved_ids = [doc_id for doc_id, _ in retrieved]
            
            # Recall@k
            recall_at_1_sum += self._recall_at_k(retrieved_ids, relevant, 1)
            recall_at_5_sum += self._recall_at_k(retrieved_ids, relevant, 5)
            recall_at_10_sum += self._recall_at_k(retrieved_ids, relevant, 10)
            recall_at_20_sum += self._recall_at_k(retrieved_ids, relevant, 20)
            
            # MRR
            mrr_sum += self._mrr(retrieved_ids, relevant, 10)
            
            # NDCG@10
            ndcg_sum += self._ndcg_at_k(retrieved_ids, relevant, 10)
            
            # MAP@10
            map_sum += self._average_precision(retrieved_ids, relevant, 10)
        
        num_queries = len(queries)
        
        return EvaluationMetrics(
            num_queries=num_queries,
            num_corpus=len(corpus),
            recall_at_1=recall_at_1_sum / num_queries if num_queries > 0 else 0,
            recall_at_5=recall_at_5_sum / num_queries if num_queries > 0 else 0,
            recall_at_10=recall_at_10_sum / num_queries if num_queries > 0 else 0,
            recall_at_20=recall_at_20_sum / num_queries if num_queries > 0 else 0,
            mrr_at_10=mrr_sum / num_queries if num_queries > 0 else 0,
            ndcg_at_10=ndcg_sum / num_queries if num_queries > 0 else 0,
            map_at_10=map_sum / num_queries if num_queries > 0 else 0,
        )
    
    def _recall_at_k(
        self, 
        retrieved: List[str], 
        relevant: set, 
        k: int
    ) -> float:
        """Compute Recall@k."""
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k & relevant)
        return hits / len(relevant)
    
    def _mrr(
        self, 
        retrieved: List[str], 
        relevant: set, 
        k: int
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _ndcg_at_k(
        self, 
        retrieved: List[str], 
        relevant: set, 
        k: int
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _average_precision(
        self, 
        retrieved: List[str], 
        relevant: set, 
        k: int
    ) -> float:
        """Compute Average Precision."""
        if not relevant:
            return 0.0
        
        precision_sum = 0.0
        hits = 0
        
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / (i + 1)
        
        return precision_sum / min(len(relevant), k)


def _embedding_device(explicit: Optional[str] = None) -> str:
    """Prefer CUDA when available for sentence-transformers."""
    if explicit and explicit.lower() not in ("auto", ""):
        return explicit
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def evaluate_retrieval(
    qa_file: Path,
    model_name_or_path: str,
    retriever_type: str = "embedding",
    top_k: int = 20,
    sample_size: Optional[int] = None,
    device: Optional[str] = "auto",
    embed_batch_size: int = 64,
) -> EvaluationMetrics:
    """
    Convenience function to evaluate retrieval on a QA file.
    
    Args:
        qa_file: JSONL file with QA records
        model_name_or_path: Model for embedding retrieval or "bm25"
        retriever_type: "bm25" or "embedding"
        top_k: Maximum k for retrieval
        sample_size: Optional sample size for evaluation
        
    Returns:
        EvaluationMetrics
    """
    # Load QA data
    records = []
    with qa_file.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if sample_size and sample_size < len(records):
        import random
        records = random.sample(records, sample_size)
    
    # Build queries, corpus, and relevance judgments
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant_docs: Dict[str, List[str]] = {}
    
    for i, record in enumerate(records):
        qid = record.get('id', str(i))
        question = record.get('question', '')
        answer = record.get('answer', '')
        context = record.get('context', '')
        
        # Query is the question
        queries[qid] = question
        
        # Corpus entry is the context (with a unique ID)
        context_id = record.get('context_id', f"ctx_{i}")
        corpus[context_id] = context
        
        # The relevant document for this query is the context
        relevant_docs[qid] = [context_id]
    
    # Create retriever
    if retriever_type == "bm25":
        if not HAS_BM25:
            raise ImportError(
                "BM25 requires rank-bm25. Install with: pip install rank-bm25"
            )
        retriever = BM25Retriever()
    else:
        retriever = EmbeddingRetriever(
            model_name_or_path,
            device=_embedding_device(device),
            batch_size=embed_batch_size,
        )
    
    # Evaluate
    evaluator = RetrievalEvaluator(retriever)
    metrics = evaluator.evaluate(queries, corpus, relevant_docs, top_k)
    
    return metrics


def compare_retrievers(
    qa_file: Path,
    models: List[Tuple[str, str]],  # List of (name, model_path_or_type)
    top_k: int = 20,
    sample_size: Optional[int] = None,
    output_file: Optional[Path] = None,
    device: Optional[str] = "auto",
    embed_batch_size: int = 64,
) -> Dict[str, EvaluationMetrics]:
    """
    Compare multiple retrievers on the same data.
    
    Args:
        qa_file: JSONL file with QA records
        models: List of (name, model_path) tuples
        top_k: Maximum k for retrieval
        sample_size: Optional sample size
        output_file: Optional file to save results
        
    Returns:
        Dict mapping model name to metrics
    """
    results = {}

    models_to_run: List[Tuple[str, str]] = []
    for name, model_path in models:
        low = model_path.lower()
        if low == "bm25" and not HAS_BM25:
            logger.warning(
                "Skipping %s: rank_bm25 not installed (pip install rank-bm25)",
                name,
            )
            continue
        if low != "bm25" and not HAS_SENTENCE_TRANSFORMERS:
            logger.warning(
                "Skipping %s: sentence-transformers not installed",
                name,
            )
            continue
        models_to_run.append((name, model_path))

    if not models_to_run:
        raise RuntimeError(
            "No retrievers to evaluate. Install: pip install rank-bm25 sentence-transformers"
        )

    for name, model_path in models_to_run:
        logger.info(f"Evaluating {name}...")
        
        if model_path.lower() == "bm25":
            retriever_type = "bm25"
        else:
            retriever_type = "embedding"
        
        metrics = evaluate_retrieval(
            qa_file,
            model_path,
            retriever_type=retriever_type,
            top_k=top_k,
            sample_size=sample_size,
            device=device,
            embed_batch_size=embed_batch_size,
        )
        
        results[name] = metrics
        logger.info(f"  {name}: {metrics}")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("RETRIEVAL COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<30} {'R@1':<10} {'R@5':<10} {'R@10':<10} {'MRR@10':<10} {'NDCG@10':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics.recall_at_1:<10.4f} {metrics.recall_at_5:<10.4f} "
              f"{metrics.recall_at_10:<10.4f} {metrics.mrr_at_10:<10.4f} {metrics.ndcg_at_10:<10.4f}")
    
    print("=" * 80)
    
    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(
                {name: m.to_dict() for name, m in results.items()},
                f,
                indent=2,
            )
        logger.info(f"Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--model", "-m", default="bm25", 
                       help="Model name/path or 'bm25'")
    parser.add_argument("--sample", "-s", type=int, help="Sample size")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple baselines")
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        help="Embedding device: auto (cuda if available), cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for corpus/query encoding (GPU)",
    )

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.compare:
        # Compare standard baselines
        models = [
            ("BM25", "bm25"),
            ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ]
        
        compare_retrievers(
            Path(args.input),
            models,
            sample_size=args.sample,
            output_file=Path(args.output) if args.output else None,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
        )
    else:
        # Single model evaluation
        retriever_type = "bm25" if args.model.lower() == "bm25" else "embedding"
        
        metrics = evaluate_retrieval(
            Path(args.input),
            args.model,
            retriever_type=retriever_type,
            sample_size=args.sample,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
        )
        
        print(f"\nResults: {metrics}")
        print(f"\nFull metrics: {metrics.to_dict()}")
