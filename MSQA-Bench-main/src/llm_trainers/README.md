# LLM Trainers (`src/llm_trainers/`)

This module provides a QLoRA-based fine-tuning pipeline for open‑source LLMs on the scientific Q&A data in `data/consolidated_qa.jsonl`. It mirrors the structure of `src/embedding_trainers/`.

## Components

- `data_utils.py`: Loads Q&A JSONL, applies hash-based train/val/test split, formats samples as chat conversations (RAG + direct QA) and builds HuggingFace datasets.
- `llm_finetuner.py`: Main QLoRA trainer using `transformers`, `peft`, `trl` (SFTTrainer), BitsAndBytes 4‑bit quantization, LoRA adapters, and checkpoint/resume.
- `evaluators.py`: Generates answers and computes ROUGE, BERTScore, token‑level F1, exact match, faithfulness (context grounding), perplexity, and latency.
- `model_comparison.py`: Loads per‑model eval results, ranks models across metrics, and writes a combined comparison report.
- `config/llm_finetuner.json`: Global defaults + `models_to_train` list (one entry per LLM).
- `scripts/train_all_llms.py`: Orchestrates multi‑model training, evaluation, and comparison.

## Quick Start

From the project root:

```bash
# (Optional) install/update deps
pip install -r requirements.txt

# Train all configured models sequentially (QLoRA)
python scripts/train_all_llms.py --config config/llm_finetuner.json -y
```

This will, for each model in `models_to_train`:

1. Create a model‑specific config (overriding base model name, batch size, etc.).
2. Fine‑tune the model with QLoRA adapters saved in:
  - `models/fine_tuned_llms/<model_name>/final_adapter/`
3. Run evaluation on the test split and store metrics in:
  - `models/fine_tuned_llms/<model_name>/eval_results_test.json`

After at least two models are trained, a cross‑model comparison report is written to:

- `models/fine_tuned_llms/comparison_report.json`

## Common Commands

```bash
# Train a single model by name (as defined in config/llm_finetuner.json)
python scripts/train_all_llms.py --model qwen2.5_3b -y

# Smoke‑test the pipeline on a small subset of data
python scripts/train_all_llms.py --subset 50 -y

# Only evaluate already‑trained models (no further training)
python scripts/train_all_llms.py --eval-only

# Only run model comparison on existing eval results
python scripts/train_all_llms.py --compare-only
```

## Direct Single‑Model Usage

You can also call the fine‑tuner module directly:

```bash
python -m src.llm_trainers.llm_finetuner --config config/llm_finetuner.json
```

Key options:

- `--resume`: resume from the latest checkpoint for that model.
- `--input_jsonl`: override the training data path.
- `--output_dir`: override where adapters and logs are written.
- `--base_model`: override the HuggingFace model ID.
- `--epochs`, `--batch_size`, `--subset_size`, `--lora_r`: override core hyperparameters.

## Metrics and Artifacts

For each fine‑tuned LLM, you will find:

- `final_adapter/`: LoRA adapter weights + tokenizer (for inference or export).
- `training_summary.json`: training metadata (hyperparameters, runtime, etc.).
- `eval_results_test.json`: aggregate metrics (ROUGE, BERTScore, F1, EM, faithfulness, perplexity).
- `predictions_test.jsonl`: per‑sample question, reference, and generated answer for inspection.
