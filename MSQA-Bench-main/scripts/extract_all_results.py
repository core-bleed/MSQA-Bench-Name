#!/usr/bin/env python3
"""
Extract and evaluate all fine-tuned model results for paper.

Run this script ON THE SERVER where models are stored:
    python scripts/extract_all_results.py

It will:
1. Discover all fine-tuned embedding models and LLM adapters
2. Evaluate each model on the test split (base vs fine-tuned)
3. Generate comparison tables (JSON + LaTeX)
4. Save everything to paper_results/model_results/

Output structure:
    paper_results/model_results/
    ├── embeddings/
    │   ├── base_vs_finetuned.json       (all embedding comparisons)
    │   ├── <model_name>/
    │   │   ├── eval_results_test.json
    │   │   └── base_eval_results_test.json
    │   └── embedding_comparison.json
    ├── llms/
    │   ├── <model_name>/
    │   │   ├── eval_results_test.json
    │   │   └── predictions_test.jsonl
    │   └── llm_comparison.json
    ├── tables/
    │   ├── table_embedding_results.tex
    │   ├── table_llm_results.tex
    │   └── table_combined_summary.tex
    └── summary.json
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Allow `from src.xxx import ...` when running this file as a script
# (e.g. `python scripts/extract_all_results.py`). Without this, Python
# resolves imports relative to scripts/ and `src` is not visible.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (adjust if your server layout differs)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_JSONL = str(DEFAULT_PROJECT_DIR / "data" / "consolidated_qa.jsonl")
DEFAULT_EMBEDDING_DIR = str(DEFAULT_PROJECT_DIR / "models" / "fine_tuned_embeddings")
DEFAULT_LLM_DIR = str(DEFAULT_PROJECT_DIR / "models" / "fine_tuned_llms")
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_DIR / "paper_results" / "model_results"

# Embedding models config (name -> base HF path)
EMBEDDING_MODELS = {
    "e5_large_v2": "intfloat/e5-large-v2",
    "bge_large_en_v1.5": "BAAI/bge-large-en-v1.5",
    "nomic_embed_v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "e5_base_v2": "intfloat/e5-base-v2",
    "bge_base_en_v1.5": "BAAI/bge-base-en-v1.5",
}

# Embedding models whose BASE checkpoint requires trust_remote_code=True.
# Fine-tuned snapshots saved by sentence-transformers don't need it because
# the custom modules are baked into the on-disk artifact.
EMBEDDINGS_NEED_TRUST_REMOTE = {"nomic_embed_v1.5"}

# LLM models config (name -> base HF path)
LLM_MODELS = {
    "phi3.5_mini": "microsoft/Phi-3.5-mini-instruct",
    "mistral_7b_v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek_r1_distill_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# LLMs where the cached custom modeling code is incompatible with current
# transformers (e.g. Phi-3.5's modeling_phi3.py references
# DynamicCache.seen_tokens which was removed). For these we force the
# in-tree HF implementation by passing trust_remote_code=False.
LLMS_USE_INTREE = {"phi3.5_mini"}


# ============================================================================
# EMBEDDING EVALUATION
# ============================================================================

def evaluate_single_embedding(
    model_name: str,
    base_model_path: str,
    finetuned_model_path: str,
    jsonl_path: str,
    output_dir: Path,
    sample_size: int = 5000,
    skip_existing: bool = True,
) -> dict[str, Any] | None:
    """Evaluate a single embedding model (base vs fine-tuned)."""
    result_file = output_dir / model_name / "eval_results_test.json"

    if skip_existing and result_file.exists():
        logger.info(f"[SKIP] {model_name}: results already exist at {result_file}")
        with open(result_file) as f:
            return json.load(f)

    # Try several known layouts: subdirectory style, prefix style, and
    # nested 'final_model' for sentence-transformers training output.
    candidates = [
        Path(finetuned_model_path) / model_name,
        Path(finetuned_model_path) / model_name / "final_model",
        Path(finetuned_model_path) / f"fine_tuned_embeddings_{model_name}",
        Path(finetuned_model_path) / f"fine_tuned_embeddings_{model_name}" / "final_model",
    ]
    finetuned_path = next((p for p in candidates if p.exists()), None)
    if finetuned_path is None:
        logger.warning(
            f"[SKIP] {model_name}: no fine-tuned model found. Tried: "
            + ", ".join(str(p) for p in candidates)
        )
        return None

    try:
        from sentence_transformers import SentenceTransformer

        from src.embedding_trainers.evaluators import evaluate_model

        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        trust_remote = model_name in EMBEDDINGS_NEED_TRUST_REMOTE

        # Evaluate base model
        logger.info(f"Evaluating BASE model: {base_model_path} (trust_remote_code={trust_remote})")
        base_model = SentenceTransformer(base_model_path, trust_remote_code=trust_remote)
        base_result = evaluate_model(
            base_model, jsonl_path, split="test",
            sample_size=sample_size, output_dir=str(model_output),
        )
        base_dict = base_result.to_dict()
        with open(model_output / "base_eval_results_test.json", "w") as f:
            json.dump(base_dict, f, indent=2)
        del base_model

        # Evaluate fine-tuned model
        logger.info(f"Evaluating FINE-TUNED model: {finetuned_path}")
        ft_model = SentenceTransformer(str(finetuned_path), trust_remote_code=trust_remote)
        ft_result = evaluate_model(
            ft_model, jsonl_path, split="test",
            sample_size=sample_size, output_dir=str(model_output),
        )
        ft_dict = ft_result.to_dict()
        with open(model_output / "eval_results_test.json", "w") as f:
            json.dump(ft_dict, f, indent=2)
        del ft_model

        # Compute improvements
        improvements = {}
        for metric in ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]:
            base_val = base_dict.get(metric, 0)
            ft_val = ft_dict.get(metric, 0)
            abs_imp = ft_val - base_val
            rel_imp = (abs_imp / base_val * 100) if base_val > 0 else 0
            improvements[metric] = {
                "base": round(base_val, 4),
                "finetuned": round(ft_val, 4),
                "absolute": round(abs_imp, 4),
                "relative_pct": round(rel_imp, 1),
            }

        combined = {
            "model_name": model_name,
            "base_model": base_model_path,
            "finetuned_path": str(finetuned_path),
            "base_results": base_dict,
            "finetuned_results": ft_dict,
            "improvements": improvements,
        }
        with open(model_output / "comparison.json", "w") as f:
            json.dump(combined, f, indent=2)

        logger.info(
            f"[OK] {model_name}: Recall@10 {base_dict.get('recall@10', 0):.4f} -> "
            f"{ft_dict.get('recall@10', 0):.4f} "
            f"(+{improvements.get('recall@10', {}).get('relative_pct', 0):.1f}%)"
        )
        return combined

    except Exception as e:
        logger.error(f"[FAIL] {model_name}: {e}", exc_info=True)
        return None

    finally:
        # Free GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def evaluate_all_embeddings(
    jsonl_path: str,
    embedding_dir: str,
    output_dir: Path,
    sample_size: int = 5000,
    skip_existing: bool = True,
    skip_models: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all embedding models."""
    emb_output = output_dir / "embeddings"
    emb_output.mkdir(parents=True, exist_ok=True)
    skip_models = skip_models or set()

    all_results = []
    for model_name, base_path in EMBEDDING_MODELS.items():
        if model_name in skip_models:
            logger.info(f"[SKIP-CLI] {model_name}: skipped via --skip-models")
            continue
        result = evaluate_single_embedding(
            model_name=model_name,
            base_model_path=base_path,
            finetuned_model_path=embedding_dir,
            jsonl_path=jsonl_path,
            output_dir=emb_output,
            sample_size=sample_size,
            skip_existing=skip_existing,
        )
        if result:
            all_results.append(result)

    # Save combined comparison
    with open(emb_output / "embedding_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Embedding evaluation complete: {len(all_results)}/{len(EMBEDDING_MODELS)} models")
    return all_results


# ============================================================================
# LLM EVALUATION
# ============================================================================

def evaluate_single_llm(
    model_name: str,
    base_model_path: str,
    llm_dir: str,
    jsonl_path: str,
    output_dir: Path,
    sample_size: int = 200,
    skip_existing: bool = True,
) -> dict[str, Any] | None:
    """Evaluate a single fine-tuned LLM on the test split."""
    result_file = output_dir / model_name / "eval_results_test.json"

    if skip_existing and result_file.exists():
        logger.info(f"[SKIP] {model_name}: results already exist at {result_file}")
        with open(result_file) as f:
            return json.load(f)

    candidates = [
        Path(llm_dir) / model_name / "final_adapter",
        Path(llm_dir) / f"fine_tuned_llms_{model_name}" / "final_adapter",
        Path(llm_dir) / model_name / "checkpoint-best",
        Path(llm_dir) / model_name,
    ]
    adapter_path = None
    for cand in candidates:
        if cand.exists() and (cand / "adapter_config.json").exists():
            adapter_path = cand
            break
    if adapter_path is None:
        logger.warning(
            f"[SKIP] {model_name}: no adapter found. Tried: "
            + ", ".join(str(c) for c in candidates)
        )
        return None

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from src.llm_trainers.evaluators import evaluate_model

        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        trust_remote = model_name not in LLMS_USE_INTREE
        logger.info(
            f"Loading base model: {base_model_path} (trust_remote_code={trust_remote})"
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=trust_remote
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

        logger.info(f"Evaluating {model_name} on test split ({sample_size} samples)...")
        result = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            jsonl_path=jsonl_path,
            model_name=model_name,
            split="test",
            sample_size=sample_size,
            max_new_tokens=256,
            temperature=0.1,
            output_dir=str(model_output),
            compute_perplexity_flag=True,
            compute_faithfulness_flag=True,
            compute_bertscore_flag=True,
        )

        result_dict = result.to_dict()
        logger.info(
            f"[OK] {model_name}: ROUGE-L={result_dict['rougeL']:.4f}, "
            f"BERTScore-F1={result_dict['bertscore_f1']:.4f}, "
            f"Token-F1={result_dict['token_f1']:.4f}"
        )

        del model, tokenizer
        torch.cuda.empty_cache()

        return result_dict

    except Exception as e:
        logger.error(f"[FAIL] {model_name}: {e}", exc_info=True)
        return None

    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def evaluate_all_llms(
    jsonl_path: str,
    llm_dir: str,
    output_dir: Path,
    sample_size: int = 200,
    skip_existing: bool = True,
    skip_models: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all fine-tuned LLMs."""
    llm_output = output_dir / "llms"
    llm_output.mkdir(parents=True, exist_ok=True)
    skip_models = skip_models or set()

    all_results = []
    for model_name, base_path in LLM_MODELS.items():
        if model_name in skip_models:
            logger.info(f"[SKIP-CLI] {model_name}: skipped via --skip-models")
            continue
        result = evaluate_single_llm(
            model_name=model_name,
            base_model_path=base_path,
            llm_dir=llm_dir,
            jsonl_path=jsonl_path,
            output_dir=llm_output,
            sample_size=sample_size,
            skip_existing=skip_existing,
        )
        if result:
            all_results.append(result)

    # Save combined comparison
    with open(llm_output / "llm_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"LLM evaluation complete: {len(all_results)}/{len(LLM_MODELS)} models")
    return all_results


# ============================================================================
# LaTeX TABLE GENERATION
# ============================================================================

def generate_embedding_table(results: list[dict[str, Any]], output_path: Path) -> str:
    """Generate LaTeX table for embedding models.

    If any entry has ``improvements`` populated, emit the comparison table
    (Base / +FT / Delta). Otherwise emit a simpler fine-tuned-only table.
    """
    has_comparison = any(r.get("improvements") for r in results)

    if has_comparison:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Retrieval performance of base vs.\ domain-adapted embedding models on the MSQA-Bench test set (5,000 queries). "
            r"$\Delta$ denotes absolute improvement after fine-tuning on 1M+ MS-domain QA pairs.}",
            r"\label{tab:embedding_results}",
            r"\small",
            r"\begin{tabular}{@{}l cc cc cc@{}}",
            r"\toprule",
            r"& \multicolumn{2}{c}{\textbf{Recall@10}} & \multicolumn{2}{c}{\textbf{MRR@10}} & \multicolumn{2}{c}{\textbf{NDCG@10}} \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            r"\textbf{Model} & Base & +FT ($\Delta$) & Base & +FT ($\Delta$) & Base & +FT ($\Delta$) \\",
            r"\midrule",
        ]
        for r in results:
            name = r["model_name"].replace("_", r"\_")
            imp = r.get("improvements") or {}
            ft_only = r.get("finetuned_results") or {}

            def fmt(metric: str, imp=imp, ft_only=ft_only) -> str:
                m = imp.get(metric)
                if m:
                    base = m.get("base", 0)
                    ft = m.get("finetuned", 0)
                    delta = m.get("absolute", 0)
                    sign = "+" if delta >= 0 else ""
                    return f"{base:.3f} & {ft:.3f} ({sign}{delta:.3f})"
                # Fallback: only fine-tuned numbers available (from collected JSON).
                ft = ft_only.get(metric, 0)
                return f"--- & {ft:.3f} (---)"

            row = f"  {name} & {fmt('recall@10')} & {fmt('mrr@10')} & {fmt('ndcg@10')} \\\\"
            lines.append(row)
    else:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Retrieval performance of domain-adapted embedding models on the MSQA-Bench test set (5,000 queries) after fine-tuning on 1M+ MS-domain QA pairs.}",
            r"\label{tab:embedding_results}",
            r"\small",
            r"\begin{tabular}{@{}l ccc cc@{}}",
            r"\toprule",
            r"\textbf{Model} & \textbf{R@1} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR@10} & \textbf{NDCG@10} \\",
            r"\midrule",
        ]
        for r in results:
            name = r["model_name"].replace("_", r"\_")
            ft = r.get("finetuned_results") or r
            row = (
                f"  {name} & "
                f"{ft.get('recall@1', 0):.3f} & "
                f"{ft.get('recall@5', 0):.3f} & "
                f"{ft.get('recall@10', 0):.3f} & "
                f"{ft.get('mrr@10', 0):.3f} & "
                f"{ft.get('ndcg@10', 0):.3f} \\\\"
            )
            lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)
    logger.info(f"Embedding LaTeX table saved: {output_path}")
    return table


def generate_llm_table(results: list[dict[str, Any]], output_path: Path) -> str:
    """Generate LaTeX table for LLM comparison."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Generation performance of QLoRA-adapted LLMs on the MSQA-Bench test set. "
        r"Models are fine-tuned with LoRA rank~64 on 1M+ MS-domain QA pairs. "
        r"Faithfulness measures NLI-based entailment of generated answers against source context.}",
        r"\label{tab:llm_results}",
        r"\small",
        r"\begin{tabular}{@{}l ccc cc c@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{ROUGE-1} & \textbf{ROUGE-L} & \textbf{BERTScore} & \textbf{Token F1} & \textbf{Faithful.} & \textbf{PPL} \\",
        r"\midrule",
    ]

    for r in results:
        name = r.get("model_name", "unknown").replace("_", r"\_")
        row = (
            f"  {name} & "
            f"{r.get('rouge1', 0):.3f} & "
            f"{r.get('rougeL', 0):.3f} & "
            f"{r.get('bertscore_f1', 0):.3f} & "
            f"{r.get('token_f1', 0):.3f} & "
            f"{r.get('faithfulness_score', 0):.3f} & "
            f"{r.get('perplexity', 0):.1f} \\\\"
        )
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    table = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table)
    logger.info(f"LLM LaTeX table saved: {output_path}")
    return table


# ============================================================================
# NORMALIZE COLLECTED RESULTS FOR TABLE GENERATORS
# ============================================================================

def _canonical_model_name(raw: str, prefix: str) -> str:
    """Strip the optional ``fine_tuned_*_`` prefix from a folder name."""
    if raw.startswith(prefix):
        return raw[len(prefix):]
    return raw


def _collected_to_table_inputs(
    collected: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reshape ``collect_existing_results`` output for the table generators.

    Returned entries omit base/finetuned comparison fields, so the
    embedding table will fall back to its fine-tuned-only layout.
    """
    emb: list[dict[str, Any]] = []
    for entry in collected.get("embeddings", []):
        raw_dir = entry.get("_model_dir", "")
        canonical = _canonical_model_name(raw_dir, "fine_tuned_embeddings_")
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        emb.append({
            "model_name": canonical or raw_dir or "unknown",
            "finetuned_results": clean,
            "improvements": {},
        })

    # Preserve the order declared in EMBEDDING_MODELS where possible.
    order = list(EMBEDDING_MODELS.keys())
    emb.sort(key=lambda r: order.index(r["model_name"]) if r["model_name"] in order else len(order))

    llm: list[dict[str, Any]] = []
    for entry in collected.get("llms", []):
        raw_dir = entry.get("_model_dir", "")
        canonical = _canonical_model_name(raw_dir, "fine_tuned_llms_")
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        llm.append({"model_name": canonical or raw_dir or "unknown", **clean})

    llm_order = list(LLM_MODELS.keys())
    llm.sort(key=lambda r: llm_order.index(r["model_name"]) if r["model_name"] in llm_order else len(llm_order))

    return emb, llm


# ============================================================================
# COLLECT EXISTING RESULTS (no GPU needed)
# ============================================================================

def collect_existing_results(
    embedding_dir: str,
    llm_dir: str,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Scan model directories for existing eval_results files without running
    any evaluation. Use this if you just want to collect results that
    training scripts already generated.
    """
    summary: dict[str, Any] = {"embeddings": [], "llms": [], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Collect embedding results
    emb_base = Path(embedding_dir)
    if emb_base.exists():
        for model_dir in sorted(emb_base.iterdir()):
            if not model_dir.is_dir():
                continue
            for pattern in ["eval_results_test.json", "eval_results_val.json", "training_summary.json"]:
                result_file = model_dir / pattern
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    data["_source"] = str(result_file)
                    data["_model_dir"] = model_dir.name
                    summary["embeddings"].append(data)
                    logger.info(f"Found embedding result: {result_file}")
                    break

    # Collect LLM results
    llm_base = Path(llm_dir)
    if llm_base.exists():
        for model_dir in sorted(llm_base.iterdir()):
            if not model_dir.is_dir():
                continue
            for pattern in ["eval_results_test.json", "eval_results_val.json"]:
                result_file = model_dir / pattern
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    data["_source"] = str(result_file)
                    data["_model_dir"] = model_dir.name
                    summary["llms"].append(data)
                    logger.info(f"Found LLM result: {result_file}")
                    break

            # Also check for training logs with final metrics
            trainer_state = model_dir / "final_adapter" / "trainer_state.json"
            if trainer_state.exists():
                try:
                    with open(trainer_state) as f:
                        state = json.load(f)
                    log_history = state.get("log_history", [])
                    if log_history:
                        last_entry = log_history[-1]
                        summary.setdefault("training_logs", {})[model_dir.name] = {
                            "final_loss": last_entry.get("loss") or last_entry.get("eval_loss"),
                            "total_steps": state.get("global_step"),
                            "epoch": last_entry.get("epoch"),
                            "log_entries": len(log_history),
                        }
                        logger.info(f"Found trainer state: {trainer_state}")
                except Exception as e:
                    logger.warning(f"Could not read {trainer_state}: {e}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "collected_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Collected results saved to {out_file}")
    logger.info(f"  Embedding results: {len(summary['embeddings'])}")
    logger.info(f"  LLM results:       {len(summary['llms'])}")

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate all fine-tuned model results for paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just collect existing results (no GPU needed):
  python scripts/extract_all_results.py --collect-only

  # Evaluate all embedding models:
  python scripts/extract_all_results.py --embeddings-only

  # Evaluate all LLM models:
  python scripts/extract_all_results.py --llms-only

  # Evaluate everything:
  python scripts/extract_all_results.py --all

  # Skip models that already have results:
  python scripts/extract_all_results.py --all --skip-existing

  # Custom paths:
  python scripts/extract_all_results.py --all \\
      --data /path/to/consolidated_qa.jsonl \\
      --embedding-dir /path/to/fine_tuned_embeddings \\
      --llm-dir /path/to/fine_tuned_llms
        """,
    )
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect existing results (no evaluation)")
    parser.add_argument("--embeddings-only", action="store_true",
                        help="Only evaluate embedding models")
    parser.add_argument("--llms-only", action="store_true",
                        help="Only evaluate LLM models")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate both embeddings and LLMs")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip models that already have results (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if results exist")
    parser.add_argument("--data", default=DEFAULT_DATA_JSONL,
                        help=f"Path to consolidated QA JSONL (default: {DEFAULT_DATA_JSONL})")
    parser.add_argument("--embedding-dir", default=DEFAULT_EMBEDDING_DIR,
                        help=f"Path to fine-tuned embeddings (default: {DEFAULT_EMBEDDING_DIR})")
    parser.add_argument("--llm-dir", default=DEFAULT_LLM_DIR,
                        help=f"Path to fine-tuned LLMs (default: {DEFAULT_LLM_DIR})")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--emb-sample-size", type=int, default=5000,
                        help="Number of test samples for embedding eval (default: 5000)")
    parser.add_argument("--llm-sample-size", type=int, default=200,
                        help="Number of test samples for LLM eval (default: 200)")
    parser.add_argument("--skip-models", default="",
                        help="Comma-separated model names to skip "
                             "(e.g., 'llama3.1_8b,deepseek_r1_distill_7b')")
    parser.add_argument("--gpu", default=None,
                        help="GPU id(s) to use, sets CUDA_VISIBLE_DEVICES "
                             "(e.g., '2' or '0,1'). Must be set before torch "
                             "is imported.")

    args = parser.parse_args()

    # Pin GPU before any torch import inside evaluate_* functions.
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logger.info(
            f"CUDA_VISIBLE_DEVICES={args.gpu} (PCI_BUS_ID order). "
            "torch will see this as cuda:0."
        )

    if not any([args.collect_only, args.embeddings_only, args.llms_only, args.all]):
        parser.print_help()
        print("\nError: specify one of --collect-only, --embeddings-only, --llms-only, or --all")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_existing = not args.force
    skip_models = {m.strip() for m in args.skip_models.split(",") if m.strip()}
    if skip_models:
        logger.info(f"Will skip models: {sorted(skip_models)}")

    # Allow --data to point at either the consolidated_qa.jsonl file directly
    # or the directory that contains it.
    data_path = Path(args.data)
    if data_path.is_dir():
        candidate = data_path / "consolidated_qa.jsonl"
        if candidate.exists():
            logger.info(f"Resolved --data directory to file: {candidate}")
            args.data = str(candidate)
        else:
            logger.warning(
                f"--data is a directory ({data_path}) and contains no "
                "consolidated_qa.jsonl; evaluation steps will likely fail."
            )

    # ------------------------------------------------------------------ #
    # 1. Always collect existing results first                            #
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 1: Collecting existing results")
    logger.info("=" * 70)
    collected = collect_existing_results(
        args.embedding_dir, args.llm_dir, output_dir,
    )
    fallback_emb, fallback_llm = _collected_to_table_inputs(collected)

    # ------------------------------------------------------------------ #
    # 2. Evaluate embeddings                                              #
    # ------------------------------------------------------------------ #
    emb_results: list[dict[str, Any]] = []
    if args.embeddings_only or args.all:
        logger.info("=" * 70)
        logger.info("STEP 2: Evaluating embedding models")
        logger.info("=" * 70)
        emb_results = evaluate_all_embeddings(
            jsonl_path=args.data,
            embedding_dir=args.embedding_dir,
            output_dir=output_dir,
            sample_size=args.emb_sample_size,
            skip_existing=skip_existing,
            skip_models=skip_models,
        )

    # ------------------------------------------------------------------ #
    # 3. Evaluate LLMs                                                    #
    # ------------------------------------------------------------------ #
    llm_results: list[dict[str, Any]] = []
    if args.llms_only or args.all:
        logger.info("=" * 70)
        logger.info("STEP 3: Evaluating LLM models")
        logger.info("=" * 70)
        llm_results = evaluate_all_llms(
            jsonl_path=args.data,
            llm_dir=args.llm_dir,
            output_dir=output_dir,
            sample_size=args.llm_sample_size,
            skip_existing=skip_existing,
            skip_models=skip_models,
        )

    # ------------------------------------------------------------------ #
    # 4. Generate LaTeX tables (use fresh eval, else fall back to        #
    #    metrics collected from existing eval JSONs in step 1)            #
    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("STEP 4: Generating LaTeX tables")
    logger.info("=" * 70)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _merge_per_model(
        fresh: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        """Prefer fresh per model; fill gaps from fallback. Logs each source."""
        fresh_by_name = {r.get("model_name"): r for r in fresh if r.get("model_name")}
        fallback_by_name = {r.get("model_name"): r for r in fallback if r.get("model_name")}
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Keep the order from fallback (which already follows EMBEDDING_MODELS / LLM_MODELS).
        for name in [*fallback_by_name.keys(), *fresh_by_name.keys()]:
            if name in seen:
                continue
            seen.add(name)
            if name in fresh_by_name:
                merged.append(fresh_by_name[name])
            else:
                merged.append(fallback_by_name[name])
                logger.info(f"  {kind}: {name} -> using collected (fresh eval missing/failed)")
        return merged

    emb_for_table = _merge_per_model(emb_results, fallback_emb, "embedding")
    if emb_for_table:
        logger.info(
            f"Embedding table: {len(emb_results)} fresh + "
            f"{len(emb_for_table) - len(emb_results)} from collected"
        )
        generate_embedding_table(emb_for_table, tables_dir / "table_embedding_results.tex")
    else:
        logger.warning("No embedding results available — table_embedding_results.tex not written")

    llm_for_table = _merge_per_model(llm_results, fallback_llm, "llm")
    if llm_for_table:
        logger.info(
            f"LLM table: {len(llm_results)} fresh + "
            f"{len(llm_for_table) - len(llm_results)} from collected"
        )
        generate_llm_table(llm_for_table, tables_dir / "table_llm_results.tex")
    else:
        logger.warning("No LLM results available — table_llm_results.tex not written")

    # ------------------------------------------------------------------ #
    # 5. Summary                                                          #
    # ------------------------------------------------------------------ #
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_models_evaluated": len(emb_results),
        "embedding_models_in_table": len(emb_for_table),
        "llm_models_evaluated": len(llm_results),
        "llm_models_in_table": len(llm_for_table),
        "output_dir": str(output_dir),
        "data_source": args.data,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS EXTRACTION COMPLETE")
    print("=" * 70)
    print(
        f"  Embeddings: evaluated {len(emb_results)}/{len(EMBEDDING_MODELS)}, "
        f"{len(emb_for_table)} in table"
    )
    print(
        f"  LLMs:       evaluated {len(llm_results)}/{len(LLM_MODELS)}, "
        f"{len(llm_for_table)} in table"
    )
    print(f"  Output directory:     {output_dir}")
    print(f"  LaTeX tables:         {tables_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
