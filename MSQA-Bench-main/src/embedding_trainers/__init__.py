"""
Embedding Trainers Module.

Provides streaming fine-tuning for embedding models on large Q&A datasets.

Main components:
- StreamingQADataset: Memory-efficient data loading
- StreamingEmbeddingFinetuner: Full training pipeline with checkpointing
- Evaluators: IR evaluation with Recall@k, MRR, NDCG

Usage:
    from src.embedding_trainers.streaming_finetuner import (
        StreamingEmbeddingFinetuner,
        TrainingConfig,
    )
    from src.embedding_trainers.data_utils import (
        StreamingQADataset,
        DataConfig,
        clean_answer,
    )
    from src.embedding_trainers.evaluators import (
        evaluate_model,
        compare_models,
    )
"""

from .data_utils import (
    DataConfig,
    StreamingQADataset,
    ResumableStreamingDataset,
    clean_answer,
    clean_question,
    get_split,
    count_records_in_split,
    load_split_samples,
)

from .evaluators import (
    EvaluationResult,
    create_ir_evaluator,
    evaluate_model,
    compare_models,
    ManualInspectionEvaluator,
)

__all__ = [
    # Data utilities
    'DataConfig',
    'StreamingQADataset',
    'ResumableStreamingDataset',
    'clean_answer',
    'clean_question',
    'get_split',
    'count_records_in_split',
    'load_split_samples',
    # Evaluators
    'EvaluationResult',
    'create_ir_evaluator',
    'evaluate_model',
    'compare_models',
    'ManualInspectionEvaluator',
]
