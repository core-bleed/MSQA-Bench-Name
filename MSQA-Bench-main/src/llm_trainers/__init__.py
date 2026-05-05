"""
LLM Trainers Module.

Provides QLoRA fine-tuning for open-source LLMs on scientific Q&A datasets.

Main components:
- LLMFineTuner: QLoRA training pipeline with checkpoint/resume
- LLMDataConfig: Data loading, chat formatting, mixed RAG/direct QA
- Evaluators: ROUGE, BERTScore, Token F1, Faithfulness, Perplexity
- Model comparison: Cross-model ranking and reporting

Usage:
    from src.llm_trainers.llm_finetuner import LLMFineTuner, LLMTrainingConfig
    from src.llm_trainers.data_utils import LLMDataConfig, create_hf_dataset
    from src.llm_trainers.evaluators import evaluate_model, LLMEvaluationResult
    from src.llm_trainers.model_comparison import compare_models
"""

from .data_utils import (
    LLMDataConfig,
    format_chat_messages,
    load_records_for_split,
    records_to_chat_messages,
    create_hf_dataset,
    count_records_in_split,
)

from .evaluators import (
    LLMEvaluationResult,
    evaluate_model,
    compute_rouge,
    compute_bertscore,
    compute_token_f1,
    compute_exact_match,
    compute_faithfulness,
    compute_perplexity,
    generate_answers,
)

from .model_comparison import (
    compare_models,
    compare_from_directory,
    rank_models,
    load_eval_result,
    discover_results,
)

__all__ = [
    # Data utilities
    "LLMDataConfig",
    "format_chat_messages",
    "load_records_for_split",
    "records_to_chat_messages",
    "create_hf_dataset",
    "count_records_in_split",
    # Fine-tuner (imported lazily to avoid heavy deps at import time)
    # Use: from src.llm_trainers.llm_finetuner import LLMFineTuner, LLMTrainingConfig
    # Evaluators
    "LLMEvaluationResult",
    "evaluate_model",
    "compute_rouge",
    "compute_bertscore",
    "compute_token_f1",
    "compute_exact_match",
    "compute_faithfulness",
    "compute_perplexity",
    "generate_answers",
    # Comparison
    "compare_models",
    "compare_from_directory",
    "rank_models",
    "load_eval_result",
    "discover_results",
]
