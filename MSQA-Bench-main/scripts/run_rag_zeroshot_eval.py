#!/usr/bin/env python3
"""Run zero-shot vs. fine-tuned, open-book vs. closed-book LLM evaluation.

Closes the C1 (no RAG eval) and C2 (no zero-shot baseline) gaps in the paper.

For each (model, adapter?, mode) combination this script:
  1. loads the base model (with or without LoRA adapter),
  2. loads the test split records,
  3. strips ``context`` for closed-book mode (or keeps it for open-book/RAG),
  4. runs ``evaluate_model`` and writes a JSON to ``--output``.

Usage:
  # Zero-shot closed-book on Phi-3.5-mini
  python scripts/run_rag_zeroshot_eval.py \
      --base microsoft/Phi-3.5-mini-instruct \
      --mode closed --sample-size 200 \
      --output paper_results/model_results/llms/phi3.5_mini/eval_zeroshot_closed.json

  # Fine-tuned open-book RAG on Phi-3.5-mini (matches Table 5; sanity check)
  python scripts/run_rag_zeroshot_eval.py \
      --base microsoft/Phi-3.5-mini-instruct \
      --adapter models/fine_tuned_llms/phi3.5_mini/final_adapter \
      --mode open --sample-size 200 \
      --output paper_results/model_results/llms/phi3.5_mini/eval_ft_open.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.llm_trainers.data_utils import LLMDataConfig, load_records_for_split
from src.llm_trainers.evaluators import (
    compute_bertscore,
    compute_exact_match,
    compute_faithfulness,
    compute_perplexity,
    compute_rouge,
    compute_token_f1,
    generate_answers,
    LLMEvaluationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="HuggingFace base model id")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter (omit for zero-shot)")
    p.add_argument("--jsonl", default="paper_results/dataset/splits/test.jsonl",
                   help="Test JSONL")
    p.add_argument("--split", default="test")
    p.add_argument("--mode", choices=["open", "closed"], required=True,
                   help="open = pass context (RAG), closed = strip context")
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-faithfulness", action="store_true")
    p.add_argument("--no-perplexity", action="store_true")
    p.add_argument("--load-4bit", action="store_true",
                   help="Load base in 4-bit NF4 (recommended for 7B+ on 24GB GPU)")
    return p.parse_args()


def load_model(base: str, adapter: str | None, load_4bit: bool):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    log.info("Loading base model: %s (4bit=%s)", base, load_4bit)
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)

    if adapter:
        from peft import PeftModel

        log.info("Loading LoRA adapter: %s", adapter)
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, tok


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = LLMDataConfig()
    records = load_records_for_split(args.jsonl, args.split, cfg)
    if len(records) > args.sample_size:
        records = records[: args.sample_size]
    log.info("Loaded %d records for split=%s", len(records), args.split)

    if args.mode == "closed":
        for r in records:
            r["context"] = None
        log.info("Mode=closed: stripped context from %d records", len(records))
    else:
        with_ctx = sum(1 for r in records if r.get("context"))
        log.info("Mode=open: %d/%d records have context", with_ctx, len(records))

    model, tok = load_model(args.base, args.adapter, args.load_4bit)

    predictions, references, contexts, tps_list = generate_answers(
        model, tok, records,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        data_config=cfg,
    )

    log.info("Computing metrics...")
    rouge = compute_rouge(predictions, references)
    token = compute_token_f1(predictions, references)
    em = compute_exact_match(predictions, references)

    bert = {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    if not args.no_bertscore:
        try:
            bert = compute_bertscore(predictions, references, device="cuda")
        except Exception as e:
            log.warning("BERTScore failed: %s", e)

    faith_score, faith_n = 0.0, 0
    if not args.no_faithfulness:
        rag_preds = [p for p, c in zip(predictions, contexts) if c]
        rag_ctxs = [c for c in contexts if c]
        faith_n = len(rag_preds)
        if rag_preds:
            try:
                faith_score = compute_faithfulness(rag_preds, rag_ctxs, device="cuda")
            except Exception as e:
                log.warning("Faithfulness failed: %s", e)

    ppl = 0.0
    if not args.no_perplexity:
        try:
            ppl = compute_perplexity(model, tok, records, device="cuda")
        except Exception as e:
            log.warning("Perplexity failed: %s", e)

    result = LLMEvaluationResult(
        model_name=f"{args.base}{'+adapter' if args.adapter else ''}@{args.mode}",
        num_samples=len(records),
        rouge1=rouge.get("rouge1", 0.0),
        rouge2=rouge.get("rouge2", 0.0),
        rougeL=rouge.get("rougeL", 0.0),
        bertscore_precision=bert["bertscore_precision"],
        bertscore_recall=bert["bertscore_recall"],
        bertscore_f1=bert["bertscore_f1"],
        token_f1=token["token_f1"],
        token_precision=token["token_precision"],
        token_recall=token["token_recall"],
        exact_match=em,
        faithfulness_score=faith_score,
        faithfulness_samples=faith_n,
        perplexity=ppl,
        avg_gen_length=float(sum(len(p) for p in predictions) / max(len(predictions), 1)),
        avg_ref_length=float(sum(len(r) for r in references) / max(len(references), 1)),
        avg_tokens_per_second=float(sum(tps_list) / max(len(tps_list), 1)),
    )

    payload = result.to_dict()
    payload["mode"] = args.mode
    payload["base_model"] = args.base
    payload["adapter"] = args.adapter
    payload["jsonl"] = args.jsonl
    payload["sample_size"] = len(records)

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s", out_path)
    log.info("\n%s", result)


if __name__ == "__main__":
    main()
