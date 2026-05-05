"""
Evaluation utilities for embedding fine-tuning.

Provides Information Retrieval evaluation with proper metrics for Q&A retrieval tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from .data_utils import load_split_samples, DataConfig


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    map_at_10: float
    num_queries: int
    num_corpus: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'recall@1': self.recall_at_1,
            'recall@5': self.recall_at_5,
            'recall@10': self.recall_at_10,
            'mrr@10': self.mrr_at_10,
            'ndcg@10': self.ndcg_at_10,
            'map@10': self.map_at_10,
            'num_queries': self.num_queries,
            'num_corpus': self.num_corpus,
        }
    
    def __str__(self) -> str:
        return (
            f"Recall@1={self.recall_at_1:.4f}, Recall@5={self.recall_at_5:.4f}, "
            f"Recall@10={self.recall_at_10:.4f}, MRR@10={self.mrr_at_10:.4f}, "
            f"NDCG@10={self.ndcg_at_10:.4f}"
        )


def create_ir_evaluator(
    jsonl_path: str,
    split: str = "val",
    sample_size: int = 5000,
    config: Optional[DataConfig] = None,
    output_path: Optional[str] = None,
    name: str = "qa_retrieval"
) -> InformationRetrievalEvaluator:
    """
    Create an InformationRetrievalEvaluator from JSONL data.
    
    This evaluator measures:
    - Recall@k: Is the correct answer in top k results?
    - MRR (Mean Reciprocal Rank): Average of 1/rank of correct answer
    - NDCG (Normalized Discounted Cumulative Gain): Ranking quality
    - MAP (Mean Average Precision): Precision at relevant positions
    
    Args:
        jsonl_path: Path to JSONL file
        split: Which split to use for evaluation ("val" or "test")
        sample_size: Number of Q&A pairs to use (limits memory usage)
        config: Data configuration
        output_path: Where to save detailed results
        name: Name for the evaluator
        
    Returns:
        Configured InformationRetrievalEvaluator
    """
    logger.info(f"Creating IR evaluator from {split} split, sample_size={sample_size}")
    
    # Load samples
    data = load_split_samples(jsonl_path, split, sample_size, config)
    
    queries = data['queries']
    corpus = data['corpus']
    relevant_docs = data['relevant_docs']
    
    if len(queries) == 0:
        raise ValueError(f"No queries loaded from {split} split. Check your data.")
    
    logger.info(f"IR Evaluator: {len(queries)} queries, {len(corpus)} corpus documents")
    
    # Create evaluator with proper metrics
    # Parameters: mrr_at_k, ndcg_at_k, accuracy_at_k, precision_recall_at_k, map_at_k
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        mrr_at_k=[1, 5, 10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[1, 5, 10],
        map_at_k=[10],
        show_progress_bar=True,
        batch_size=32,
    )
    
    return evaluator


def evaluate_model(
    model: SentenceTransformer,
    jsonl_path: str,
    split: str = "test",
    sample_size: int = 5000,
    config: Optional[DataConfig] = None,
    output_dir: Optional[str] = None
) -> EvaluationResult:
    """
    Evaluate a model on a held-out split.
    
    Args:
        model: The SentenceTransformer model to evaluate
        jsonl_path: Path to JSONL file
        split: Which split to use
        sample_size: Number of samples
        config: Data configuration
        output_dir: Where to save results
        
    Returns:
        EvaluationResult with all metrics
    """
    # Load data
    data = load_split_samples(jsonl_path, split, sample_size, config)
    
    queries = data['queries']
    corpus = data['corpus']
    relevant_docs = data['relevant_docs']
    
    if len(queries) == 0:
        raise ValueError(f"No data in {split} split")
    
    logger.info(f"Evaluating on {len(queries)} queries...")
    
    # Encode queries and corpus
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    
    query_texts = [queries[qid] for qid in query_ids]
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    
    query_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Compute similarities (query x corpus)
    # Using cosine similarity
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
    similarities = np.dot(query_embeddings, corpus_embeddings.T)
    
    # Calculate metrics
    recall_at_1 = 0.0
    recall_at_5 = 0.0
    recall_at_10 = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    
    for i, qid in enumerate(query_ids):
        relevant = relevant_docs[qid]
        
        # Get ranking
        scores = similarities[i]
        ranking = np.argsort(-scores)  # Descending order
        
        # Find relevant document positions
        relevant_indices = [corpus_ids.index(cid) for cid in relevant if cid in corpus_ids]
        
        if not relevant_indices:
            continue
        
        # Get ranks of relevant documents
        ranks = [np.where(ranking == idx)[0][0] + 1 for idx in relevant_indices]
        min_rank = min(ranks)
        
        # Recall@k
        if min_rank <= 1:
            recall_at_1 += 1
        if min_rank <= 5:
            recall_at_5 += 1
        if min_rank <= 10:
            recall_at_10 += 1
        
        # MRR
        mrr_sum += 1.0 / min_rank
        
        # NDCG@10
        dcg = sum(1.0 / np.log2(r + 1) for r in ranks if r <= 10)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_indices), 10)))
        ndcg_sum += dcg / idcg if idcg > 0 else 0
    
    n = len(query_ids)
    
    result = EvaluationResult(
        recall_at_1=recall_at_1 / n,
        recall_at_5=recall_at_5 / n,
        recall_at_10=recall_at_10 / n,
        mrr_at_10=mrr_sum / n,
        ndcg_at_10=ndcg_sum / n,
        map_at_10=mrr_sum / n,  # For single relevant doc, MAP ≈ MRR
        num_queries=len(queries),
        num_corpus=len(corpus)
    )
    
    logger.info(f"Evaluation results: {result}")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / f"eval_results_{split}.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    return result


def compare_models(
    base_model_path: str,
    finetuned_model_path: str,
    jsonl_path: str,
    sample_size: int = 5000,
    config: Optional[DataConfig] = None,
    output_dir: Optional[str] = None
) -> Dict[str, EvaluationResult]:
    """
    Compare base model vs fine-tuned model on test set.
    
    Args:
        base_model_path: Path/name of base model
        finetuned_model_path: Path to fine-tuned model
        jsonl_path: Path to JSONL file
        sample_size: Number of test samples
        config: Data configuration
        output_dir: Where to save comparison results
        
    Returns:
        Dict with 'base' and 'finetuned' EvaluationResult objects
    """
    logger.info("Loading base model...")
    base_model = SentenceTransformer(base_model_path)
    
    logger.info("Loading fine-tuned model...")
    finetuned_model = SentenceTransformer(finetuned_model_path)
    
    results = {}
    
    logger.info("Evaluating base model...")
    results['base'] = evaluate_model(
        base_model, jsonl_path, split="test",
        sample_size=sample_size, config=config
    )
    
    logger.info("Evaluating fine-tuned model...")
    results['finetuned'] = evaluate_model(
        finetuned_model, jsonl_path, split="test",
        sample_size=sample_size, config=config
    )
    
    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for metric in ['recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr_at_10', 'ndcg_at_10']:
        base_val = getattr(results['base'], metric)
        ft_val = getattr(results['finetuned'], metric)
        improvement = ((ft_val - base_val) / base_val * 100) if base_val > 0 else 0
        
        print(f"{metric:<15} {base_val:<15.4f} {ft_val:<15.4f} {improvement:+.1f}%")
    
    print("=" * 70)
    
    # Save comparison
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        comparison = {
            'base': results['base'].to_dict(),
            'finetuned': results['finetuned'].to_dict(),
            'improvements': {
                metric: {
                    'base': getattr(results['base'], metric),
                    'finetuned': getattr(results['finetuned'], metric),
                    'absolute': getattr(results['finetuned'], metric) - getattr(results['base'], metric),
                    'relative_pct': ((getattr(results['finetuned'], metric) - getattr(results['base'], metric)) 
                                    / getattr(results['base'], metric) * 100) if getattr(results['base'], metric) > 0 else 0
                }
                for metric in ['recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr_at_10', 'ndcg_at_10']
            }
        }
        
        with open(output_path / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {output_path / 'model_comparison.json'}")
    
    return results


class ManualInspectionEvaluator:
    """
    Manual inspection helper for qualitative evaluation.
    
    Shows top-k results for sample queries to allow manual review
    of model behavior before and after fine-tuning.
    """
    
    def __init__(self, model: SentenceTransformer, corpus: List[str]):
        """
        Initialize evaluator.
        
        Args:
            model: SentenceTransformer model
            corpus: List of answer texts
        """
        self.model = model
        self.corpus = corpus
        
        logger.info(f"Encoding corpus of {len(corpus)} documents...")
        self.corpus_embeddings = model.encode(
            corpus,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # Normalize for cosine similarity
        self.corpus_embeddings = self.corpus_embeddings / np.linalg.norm(
            self.corpus_embeddings, axis=1, keepdims=True
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top-k most similar answers.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'rank', 'score', 'text'
        """
        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        scores = np.dot(query_emb, self.corpus_embeddings.T)[0]
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'score': float(scores[idx]),
                'text': self.corpus[idx]
            })
        
        return results
    
    def inspect(self, queries: List[str], top_k: int = 5) -> None:
        """
        Print search results for manual inspection.
        
        Args:
            queries: List of test queries
            top_k: Number of results per query
        """
        for query in queries:
            print(f"\n{'='*70}")
            print(f"QUERY: {query}")
            print(f"{'='*70}")
            
            results = self.search(query, top_k)
            
            for result in results:
                print(f"\n{result['rank']}. [Score: {result['score']:.4f}]")
                text = result['text']
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"   {text}")
