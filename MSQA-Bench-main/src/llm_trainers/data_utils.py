"""
Data utilities for LLM fine-tuning on scientific Q&A.

Provides chat-template formatted data loading for instruction tuning,
with hash-based splitting compatible with the embedding trainer pipeline.
Supports mixed training with context-grounded (RAG) and direct QA formats.
"""

import hashlib
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.embedding_trainers.data_utils import (
    clean_answer,
    clean_question,
    get_split,
)


logger = logging.getLogger(__name__)


SYSTEM_PROMPT_RAG = (
    "You are a scientific assistant. Answer the question accurately "
    "based on the provided context."
)
SYSTEM_PROMPT_DIRECT = (
    "You are a scientific assistant with deep expertise in mass spectrometry "
    "and related analytical techniques. Answer the question accurately and concisely."
)


@dataclass
class LLMDataConfig:
    """Configuration for LLM data loading and formatting."""
    min_question_length: int = 10
    max_question_length: int = 512
    min_answer_length: int = 20
    max_answer_length: int = 2048
    clean_answers: bool = True
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    subset_size: Optional[int] = None
    no_context_ratio: float = 0.3
    max_context_length: int = 1536
    system_prompt_rag: str = SYSTEM_PROMPT_RAG
    system_prompt_direct: str = SYSTEM_PROMPT_DIRECT
    seed: int = 42


def _truncate_context(context: str, max_length: int) -> str:
    """Truncate context to max_length at a sentence boundary."""
    if not context or len(context) <= max_length:
        return context
    truncated = context[:max_length]
    last_period = truncated.rfind(". ")
    if last_period > max_length * 0.5:
        return truncated[: last_period + 1]
    return truncated.rsplit(" ", 1)[0]


def format_chat_messages(
    question: str,
    answer: str,
    context: Optional[str] = None,
    system_prompt_rag: str = SYSTEM_PROMPT_RAG,
    system_prompt_direct: str = SYSTEM_PROMPT_DIRECT,
) -> List[Dict[str, str]]:
    """
    Format a QA record as chat messages for instruction tuning.

    Returns a list of {"role": ..., "content": ...} dicts suitable for
    tokenizer.apply_chat_template().
    """
    if context:
        return [
            {"role": "system", "content": system_prompt_rag},
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}",
            },
            {"role": "assistant", "content": answer},
        ]
    return [
        {"role": "system", "content": system_prompt_direct},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _should_drop_context(record_id: str, no_context_ratio: float) -> bool:
    """Deterministic decision on whether to drop context for this record."""
    h = int(hashlib.md5(f"ctx_{record_id}".encode()).hexdigest(), 16) % 1000
    return h < int(no_context_ratio * 1000)


def load_records_for_split(
    jsonl_path: str,
    split: str,
    config: Optional[LLMDataConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Load all records belonging to a split into memory.

    Each returned dict has keys: id, question, answer, context (may be None),
    and 'has_context' bool indicating the training format.
    """
    config = config or LLMDataConfig()
    records: List[Dict[str, Any]] = []
    filtered = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" not in record or "answer" not in record:
                continue

            record_id = record.get("id", str(line_num))
            if get_split(record_id, config.train_ratio, config.val_ratio) != split:
                continue

            question = record["question"].strip()
            answer = record["answer"].strip()

            if config.clean_answers:
                answer = clean_answer(answer, config.max_answer_length)
                question = clean_question(question, config.max_question_length)

            if not (
                config.min_question_length <= len(question) <= config.max_question_length
                and config.min_answer_length <= len(answer) <= config.max_answer_length
            ):
                filtered += 1
                continue

            context_raw = record.get("context", "")
            drop_ctx = _should_drop_context(record_id, config.no_context_ratio)
            if context_raw and not drop_ctx:
                context = _truncate_context(context_raw, config.max_context_length)
            else:
                context = None

            records.append(
                {
                    "id": record_id,
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "has_context": context is not None,
                }
            )

            if config.subset_size and len(records) >= config.subset_size:
                break

    logger.info(
        f"Loaded {len(records)} records from {split} split "
        f"(filtered={filtered})"
    )
    with_ctx = sum(1 for r in records if r["has_context"])
    logger.info(
        f"  With context: {with_ctx}, Without context: {len(records) - with_ctx}"
    )
    return records


def records_to_chat_messages(
    records: List[Dict[str, Any]],
    config: Optional[LLMDataConfig] = None,
) -> List[List[Dict[str, str]]]:
    """Convert loaded records into lists of chat messages."""
    config = config or LLMDataConfig()
    conversations = []
    for rec in records:
        messages = format_chat_messages(
            question=rec["question"],
            answer=rec["answer"],
            context=rec["context"],
            system_prompt_rag=config.system_prompt_rag,
            system_prompt_direct=config.system_prompt_direct,
        )
        conversations.append(messages)
    return conversations


def create_hf_dataset(
    jsonl_path: str,
    split: str,
    config: Optional[LLMDataConfig] = None,
    tokenizer=None,
    max_seq_length: int = 2048,
):
    """
    Create a HuggingFace Dataset formatted for SFTTrainer.

    Returns a datasets.Dataset with a 'text' column containing the
    formatted conversation strings ready for tokenization.
    """
    from datasets import Dataset

    config = config or LLMDataConfig()
    records = load_records_for_split(jsonl_path, split, config)

    if not records:
        raise ValueError(f"No records found for split '{split}'")

    conversations = records_to_chat_messages(records, config)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        texts = []
        for conv in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                text = _fallback_format(conv)
            texts.append(text)
    else:
        texts = [_fallback_format(conv) for conv in conversations]

    dataset = Dataset.from_dict({"text": texts})
    logger.info(
        f"Created HF dataset for '{split}': {len(dataset)} examples, "
        f"max_seq_length={max_seq_length}"
    )
    return dataset


def _fallback_format(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting when no tokenizer chat template is available."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}\n<|end|>")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n<|end|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n<|end|>")
    return "\n".join(parts)


def count_records_in_split(
    jsonl_path: str, split: str, config: Optional[LLMDataConfig] = None
) -> int:
    """Count total records in a split without full loading."""
    config = config or LLMDataConfig()
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" not in record or "answer" not in record:
                continue

            record_id = record.get("id", str(line_num))
            if get_split(record_id, config.train_ratio, config.val_ratio) != split:
                continue

            question = record["question"].strip()
            answer = record["answer"].strip()

            if config.clean_answers:
                answer = clean_answer(answer, config.max_answer_length)
                question = clean_question(question, config.max_question_length)

            if not (
                config.min_question_length <= len(question) <= config.max_question_length
                and config.min_answer_length <= len(answer) <= config.max_answer_length
            ):
                continue

            count += 1
            if config.subset_size and count >= config.subset_size:
                break

    return count
