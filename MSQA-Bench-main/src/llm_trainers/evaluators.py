"""
Evaluation utilities for LLM fine-tuning on scientific Q&A.

Provides comprehensive metrics for generative QA:
- ROUGE-1/2/L (n-gram overlap)
- BERTScore (semantic similarity)
- Token-level F1 (word overlap)
- Exact Match
- Faithfulness (context grounding via NLI)
- Perplexity
- Generation latency
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np

from .data_utils import LLMDataConfig, load_records_for_split, format_chat_messages


logger = logging.getLogger(__name__)


@dataclass
class LLMEvaluationResult:
    """Container for LLM evaluation results."""
    model_name: str
    num_samples: int

    # ROUGE
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0

    # BERTScore
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0

    # Token-level F1
    token_f1: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0

    # Exact match
    exact_match: float = 0.0

    # Faithfulness (for RAG samples only)
    faithfulness_score: float = 0.0
    faithfulness_samples: int = 0

    # Perplexity
    perplexity: float = 0.0

    # Generation stats
    avg_gen_length: float = 0.0
    avg_ref_length: float = 0.0
    avg_tokens_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"=== Evaluation Results: {self.model_name} ({self.num_samples} samples) ===",
            f"ROUGE-1={self.rouge1:.4f}, ROUGE-2={self.rouge2:.4f}, ROUGE-L={self.rougeL:.4f}",
            f"BERTScore F1={self.bertscore_f1:.4f} (P={self.bertscore_precision:.4f}, R={self.bertscore_recall:.4f})",
            f"Token F1={self.token_f1:.4f} (P={self.token_precision:.4f}, R={self.token_recall:.4f})",
            f"Exact Match={self.exact_match:.4f}",
            f"Faithfulness={self.faithfulness_score:.4f} ({self.faithfulness_samples} RAG samples)",
            f"Perplexity={self.perplexity:.2f}",
            f"Avg gen/ref length={self.avg_gen_length:.0f}/{self.avg_ref_length:.0f} chars",
            f"Tokens/sec={self.avg_tokens_per_second:.1f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_bertscore(
    predictions: List[str], references: List[str], device: str = "cpu"
) -> Dict[str, float]:
    """Compute BERTScore (precision, recall, F1)."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        device=device,
        verbose=False,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def compute_token_f1(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1."""
    precisions, recalls, f1s = [], [], []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if not pred_tokens and not ref_tokens:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue

        common = pred_tokens & ref_tokens
        p = len(common) / len(pred_tokens) if pred_tokens else 0.0
        r = len(common) / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "token_precision": float(np.mean(precisions)),
        "token_recall": float(np.mean(recalls)),
        "token_f1": float(np.mean(f1s)),
    }


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match ratio after normalisation."""
    matches = 0
    for pred, ref in zip(predictions, references):
        if pred.strip().lower() == ref.strip().lower():
            matches += 1
    return matches / len(predictions) if predictions else 0.0


def compute_faithfulness(
    predictions: List[str],
    contexts: List[str],
    device: str = "cpu",
) -> float:
    """
    Compute faithfulness: fraction of generated answer sentences that are
    entailed by the provided context, using a lightweight NLI model.
    """
    if not predictions or not contexts:
        return 0.0

    try:
        from transformers import pipeline

        nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=0 if device == "cuda" else -1,
        )
    except Exception as e:
        logger.warning(f"Could not load NLI model for faithfulness: {e}")
        return 0.0

    scores = []
    for pred, ctx in zip(predictions, contexts):
        if not ctx:
            continue
        sentences = [s.strip() for s in pred.split(".") if s.strip()]
        if not sentences:
            scores.append(0.0)
            continue
        entailed = 0
        for sent in sentences:
            try:
                result = nli(f"{ctx} [SEP] {sent}", truncation=True, max_length=512)
                label = result[0]["label"].lower()
                if "entail" in label:
                    entailed += 1
            except Exception:
                pass
        scores.append(entailed / len(sentences))

    return float(np.mean(scores)) if scores else 0.0


def compute_perplexity(
    model, tokenizer, texts: List[str], max_length: int = 2048
) -> float:
    """Compute average perplexity of the model on a list of texts."""
    device = next(model.parameters()).device
    losses = []

    model.eval()
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = model(**encodings, labels=encodings["input_ids"])
            losses.append(outputs.loss.item())

    avg_loss = float(np.mean(losses))
    return float(np.exp(avg_loss))


# ---------------------------------------------------------------------------
# Generation + full evaluation
# ---------------------------------------------------------------------------

def generate_answers(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    data_config: Optional[LLMDataConfig] = None,
) -> Tuple[List[str], List[str], List[Optional[str]], List[float]]:
    """
    Generate answers for a list of QA records.

    Returns (predictions, references, contexts, tokens_per_second_list).
    """
    data_config = data_config or LLMDataConfig()
    device = next(model.parameters()).device
    predictions, references, contexts, tps_list = [], [], [], []

    model.eval()
    for i, rec in enumerate(records):
        question = rec["question"]
        context = rec.get("context")
        reference = rec["answer"]

        messages = format_chat_messages(
            question=question,
            answer="",
            context=context,
            system_prompt_rag=data_config.system_prompt_rag,
            system_prompt_direct=data_config.system_prompt_direct,
        )
        # Remove the empty assistant turn
        messages = [m for m in messages if m["content"]]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            from .data_utils import _fallback_format

            prompt = _fallback_format(messages) + "\n<|assistant|>\n"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = time.perf_counter() - start_time

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        num_new = len(new_tokens)
        tps = num_new / gen_time if gen_time > 0 else 0.0

        predictions.append(generated)
        references.append(reference)
        contexts.append(context)
        tps_list.append(tps)

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {i + 1}/{len(records)} ({tps:.1f} tok/s)")

    return predictions, references, contexts, tps_list


def evaluate_model(
    model,
    tokenizer,
    jsonl_path: str,
    model_name: str,
    split: str = "test",
    sample_size: int = 50,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    config: Optional[LLMDataConfig] = None,
    output_dir: Optional[str] = None,
    compute_perplexity_flag: bool = True,
    compute_faithfulness_flag: bool = True,
    compute_bertscore_flag: bool = True,
) -> LLMEvaluationResult:
    """
    Comprehensive evaluation of a fine-tuned LLM on the test split.

    Generates answers and computes all metrics.
    """
    config = config or LLMDataConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Evaluating model '{model_name}' on {split} split ({sample_size} samples)...")

    records = load_records_for_split(jsonl_path, split, config)
    if len(records) > sample_size:
        records = records[:sample_size]
    if not records:
        raise ValueError(f"No records in {split} split")

    logger.info(f"Loaded {len(records)} records for evaluation")

    # Generate answers
    predictions, references, contexts, tps_list = generate_answers(
        model, tokenizer, records,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        data_config=config,
    )

    # ROUGE
    logger.info("Computing ROUGE...")
    rouge_scores = compute_rouge(predictions, references)

    # Token F1
    logger.info("Computing token F1...")
    token_scores = compute_token_f1(predictions, references)

    # Exact match
    em = compute_exact_match(predictions, references)

    # BERTScore
    bertscore_results = {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    if compute_bertscore_flag:
        logger.info("Computing BERTScore...")
        try:
            bertscore_results = compute_bertscore(predictions, references, device=device)
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")

    # Faithfulness (only on RAG samples)
    faith_score = 0.0
    faith_samples = 0
    if compute_faithfulness_flag:
        rag_preds = [p for p, c in zip(predictions, contexts) if c]
        rag_ctxs = [c for c in contexts if c]
        faith_samples = len(rag_preds)
        if rag_preds:
            logger.info(f"Computing faithfulness on {faith_samples} RAG samples...")
            try:
                faith_score = compute_faithfulness(rag_preds, rag_ctxs, device=device)
            except Exception as e:
                logger.warning(f"Faithfulness computation failed: {e}")

    # Perplexity
    ppl = 0.0
    if compute_perplexity_flag:
        logger.info("Computing perplexity...")
        try:
            eval_texts = [
                f"Q: {r['question']}\nA: {r['answer']}"
                for r in records[:min(20, len(records))]
            ]
            ppl = compute_perplexity(model, tokenizer, eval_texts)
        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")

    # Build result
    result = LLMEvaluationResult(
        model_name=model_name,
        num_samples=len(records),
        rouge1=rouge_scores.get("rouge1", 0.0),
        rouge2=rouge_scores.get("rouge2", 0.0),
        rougeL=rouge_scores.get("rougeL", 0.0),
        bertscore_precision=bertscore_results.get("bertscore_precision", 0.0),
        bertscore_recall=bertscore_results.get("bertscore_recall", 0.0),
        bertscore_f1=bertscore_results.get("bertscore_f1", 0.0),
        token_f1=token_scores.get("token_f1", 0.0),
        token_precision=token_scores.get("token_precision", 0.0),
        token_recall=token_scores.get("token_recall", 0.0),
        exact_match=em,
        faithfulness_score=faith_score,
        faithfulness_samples=faith_samples,
        perplexity=ppl,
        avg_gen_length=float(np.mean([len(p) for p in predictions])) if predictions else 0.0,
        avg_ref_length=float(np.mean([len(r) for r in references])) if references else 0.0,
        avg_tokens_per_second=float(np.mean(tps_list)) if tps_list else 0.0,
    )

    logger.info(f"\n{result}")

    # Save
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / f"eval_results_{split}.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save individual predictions for inspection
        pred_data = []
        for rec, pred, ref in zip(records, predictions, references):
            pred_data.append({
                "id": rec["id"],
                "question": rec["question"],
                "has_context": rec.get("has_context", False),
                "reference": ref,
                "prediction": pred,
            })

        with open(out_path / f"predictions_{split}.jsonl", "w") as f:
            for item in pred_data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Results saved to {out_path}")

    return result
