"""
Example script demonstrating how to use fine-tuned embedding models
for semantic search and document retrieval.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util


class SemanticSearchEngine:
    """Simple semantic search engine using fine-tuned embeddings."""
    
    def __init__(self, model_path: str):
        """
        Initialize the search engine with a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        print(f"Loading model from: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.corpus_embeddings = None
        self.documents = []
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def index_documents(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Index a collection of documents.
        
        Args:
            documents: List of document texts to index
            batch_size: Batch size for encoding
        """
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents
        self.corpus_embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        print("Indexing complete!")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            List of (document, score) tuples if return_scores=True,
            otherwise list of documents
        """
        if self.corpus_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarity scores
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get top-k results
        top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
        
        results = []
        for idx in top_results:
            doc = self.documents[idx]
            score = float(cos_scores[idx])
            results.append((doc, score) if return_scores else doc)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        if self.corpus_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Encode all queries at once
        query_embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Compute similarities for all queries
        all_scores = util.cos_sim(query_embeddings, self.corpus_embeddings)
        
        all_results = []
        for scores in all_scores:
            top_indices = np.argsort(-scores.cpu().numpy())[:top_k]
            results = [
                (self.documents[idx], float(scores[idx]))
                for idx in top_indices
            ]
            all_results.append(results)
        
        return all_results


def load_qa_data(jsonl_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load questions and answers from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        Tuple of (questions, answers) lists
    """
    questions = []
    answers = []
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                questions.append(record.get("question", ""))
                answers.append(record.get("answer", ""))
            except json.JSONDecodeError:
                continue
    
    return questions, answers


def example_semantic_search():
    """Example: Basic semantic search."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Semantic Search")
    print("=" * 70)
    
    # Load model
    model_path = "models/fine_tuned_embeddings/final_model"
    search_engine = SemanticSearchEngine(model_path)
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information from images.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning discovers patterns in unlabeled data.",
        "Transfer learning applies knowledge from one task to improve performance on another.",
    ]
    
    # Index documents
    search_engine.index_documents(documents)
    
    # Perform searches
    queries = [
        "How do computers learn from data?",
        "What is neural network architecture?",
        "Understanding human text with AI",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        results = search_engine.search(query, top_k=3)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank}. [{score:.4f}] {doc}")


def example_qa_retrieval():
    """Example: Q&A retrieval from JSONL."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Q&A Retrieval")
    print("=" * 70)
    
    # Check if JSONL file exists
    jsonl_path = Path("data/consolidated_qa.jsonl")
    if not jsonl_path.exists():
        print(f"JSONL file not found: {jsonl_path}")
        print("Skipping this example.")
        return
    
    # Load Q&A data
    print(f"\nLoading Q&A data from: {jsonl_path}")
    questions, answers = load_qa_data(jsonl_path)
    
    if not answers:
        print("No Q&A pairs found.")
        return
    
    print(f"Loaded {len(answers)} answers")
    
    # Load model and index answers
    model_path = "models/fine_tuned_embeddings/final_model"
    search_engine = SemanticSearchEngine(model_path)
    
    # Index only first 1000 answers for demo
    max_docs = min(1000, len(answers))
    search_engine.index_documents(answers[:max_docs])
    
    # Example queries
    test_queries = [
        "What is the main finding of this research?",
        "What methodology was used in the study?",
        "What are the limitations of this approach?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        results = search_engine.search(query, top_k=3)
        for rank, (answer, score) in enumerate(results, 1):
            # Truncate long answers for display
            display_answer = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"{rank}. [{score:.4f}] {display_answer}")


def example_similarity_comparison():
    """Example: Compare similarity between texts."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Text Similarity Comparison")
    print("=" * 70)
    
    # Load model
    model_path = "models/fine_tuned_embeddings/final_model"
    print(f"\nLoading model from: {model_path}")
    model = SentenceTransformer(model_path)
    
    # Texts to compare
    texts = [
        "Climate change is causing global temperatures to rise.",
        "Global warming leads to increased temperatures worldwide.",
        "Machine learning models require large amounts of data.",
        "The stock market experienced significant volatility today.",
    ]
    
    print("\nTexts:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
    
    # Encode all texts
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Compute similarity matrix
    similarity_matrix = util.cos_sim(embeddings, embeddings)
    
    print("\nSimilarity Matrix:")
    print("-" * 70)
    print("     ", end="")
    for i in range(len(texts)):
        print(f"  T{i+1}  ", end="")
    print()
    
    for i in range(len(texts)):
        print(f"T{i+1}  ", end="")
        for j in range(len(texts)):
            score = similarity_matrix[i][j].item()
            print(f" {score:.3f} ", end="")
        print()
    
    print("\nKey observations:")
    print("- T1 and T2 are highly similar (both about climate/temperature)")
    print("- T3 and T4 are dissimilar to T1 and T2 (different topics)")
    print("- Diagonal is 1.0 (each text is identical to itself)")


def example_batch_search():
    """Example: Batch processing multiple queries."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Batch Search")
    print("=" * 70)
    
    # Load model
    model_path = "models/fine_tuned_embeddings/final_model"
    search_engine = SemanticSearchEngine(model_path)
    
    # Index documents
    documents = [
        "Python is a high-level programming language.",
        "JavaScript is commonly used for web development.",
        "SQL is used for database queries.",
        "Machine learning requires statistical knowledge.",
        "Data visualization helps communicate insights.",
        "Cloud computing provides scalable infrastructure.",
        "DevOps practices improve software delivery.",
        "Cybersecurity protects digital assets.",
    ]
    
    search_engine.index_documents(documents)
    
    # Multiple queries
    queries = [
        "Programming languages for web applications",
        "Analyzing and presenting data",
        "Protecting systems from attacks",
    ]
    
    print("\nPerforming batch search...")
    all_results = search_engine.batch_search(queries, top_k=2)
    
    for query, results in zip(queries, all_results):
        print(f"\nQuery: {query}")
        print("-" * 70)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank}. [{score:.4f}] {doc}")


def main():
    """Run all examples."""
    import os
    
    # Check if model exists
    model_path = Path("models/fine_tuned_embeddings/final_model")
    if not model_path.exists():
        print("=" * 70)
        print("ERROR: Fine-tuned model not found!")
        print("=" * 70)
        print(f"\nExpected location: {model_path}")
        print("\nPlease train a model first using:")
        print("  python src/embedding_trainers/embedding_finetuner.py")
        print("\nOr use a pre-trained model:")
        print("  from sentence_transformers import SentenceTransformer")
        print("  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')")
        return
    
    try:
        # Run examples
        example_semantic_search()
        example_qa_retrieval()
        example_similarity_comparison()
        example_batch_search()
        
        print("\n\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
