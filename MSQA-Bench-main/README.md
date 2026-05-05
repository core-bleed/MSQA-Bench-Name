# MSQA-Bench

Code and reproducibility materials for **MSQA-Bench**, a computational mass
spectrometry question-answering resource. The public text-bearing benchmark
contains 504,181 redistributable QA records; the full constructed resource
contains 1,191,742 generated QA records when the 687,561-record metadata-only
reconstruction tier is included.

The dataset release is hosted separately on Hugging Face:

<https://huggingface.co/datasets/asad00027/MSQA-Bench>

This GitHub repository is intended for the NeurIPS Evaluations & Datasets code artifact. It contains the extraction, dataset-construction, filtering, training, evaluation, and release-preparation code. Large generated artifacts such as PDFs, JSONL splits, model checkpoints, logs, and Hugging Face release files are intentionally excluded.

The paper reports the human audit as an estimate of residual label noise, not
as a fully expert-authored gold guarantee. Retrieval numbers are controlled
5K-query in-pool baselines, and generation numbers are diagnostic fine-tuned
adapter baselines.

> **Author handles.** The GitHub account `AsadKhalil`, the Hugging Face account `asad00027`, and the paper email `masad@hse.ru` all refer to the same first author (Muhammad Asad, HSE Moscow).

## Repository Layout

```text
src/                 Core Python pipeline modules
scripts/             Command-line workflows and utility scripts
config/              JSON configuration files
paper/               Dataset/evaluation/annotation helpers used for the paper
docs/                Setup and usage notes
examples/            Small usage examples
figures/             Paper/result figures
requirements.txt     Python dependencies
requirements-smoke.txt Lightweight dependencies for the artifact smoke test
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some workflows require external services or hardware: GROBID for structured PDF parsing, vLLM/Ollama/OpenAI-compatible generation endpoints for QA generation, and CUDA GPUs for model fine-tuning.

## Artifact Smoke Test

For reviewer sanity checks, the following CPU-only workflow installs the
lightweight dependencies, creates and extracts a tiny PDF, downloads a small
redistributable QA sample from Hugging Face, runs the BM25 retrieval evaluator,
and writes a compact result table.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt
python3 scripts/run_smoke_test.py --sample-size 32
```

Outputs are written under `paper_results/smoke/`, including
`smoke_retrieval_table.md` and `evaluation/bm25_baseline_results.json`.
Use the offline fixture mode only when network access to Hugging Face is not
available:

```bash
python3 scripts/run_smoke_test.py --sample-source fixture --sample-size 8
```

## Quickstart: load the dataset and run the BM25 baseline

This reproduces the controlled BM25 row of Table 3 in the paper (R@10 = 0.742)
on a 5K-query in-pool sample from the redistributable test split.

```python
from datasets import load_dataset

# Two configs: "redistributable" (text-bearing, CC-BY-4.0) and
# "restricted" (metadata-only; reconstruct text locally with the script below).
ds = load_dataset("asad00027/MSQA-Bench", "redistributable", split="test")
print(ds[0]["question"], "->", ds[0]["answer"][:80])
```

```bash
# Five-minute reproduction of the BM25 retrieval baseline on a 5K-query
# in-pool sample.
python3 - <<'PY'
from datasets import load_dataset

ds = load_dataset("asad00027/MSQA-Bench", "redistributable", split="test")
ds.to_json("/tmp/msqa_redistributable_test.jsonl")
PY

python3 scripts/evaluate_bm25_baseline.py \
  --data /tmp/msqa_redistributable_test.jsonl \
  --output paper_results/evaluation \
  --sample-size 5000
```

Expected output: `recall@10 ≈ 0.742, mrr@10 ≈ 0.668, ndcg@10 ≈ 0.686` (matches
Table 3 of the paper).

## Common Commands

Run tests:

```bash
python3 -m pytest
```

Validate embedding training setup:

```bash
python3 scripts/validate_embedding_setup.py
```

Run the paper pipeline:

```bash
python3 scripts/run_paper_pipeline.py --config config/paper_pipeline.json
```

Prepare the two-tier NeurIPS/Hugging Face dataset release from generated splits:

```bash
python3 scripts/prepare_neurips_release.py \
  --input-dir paper_results/dataset/splits \
  --output-dir paper_results/neurips_release
```

Reconstruct a restricted metadata-only record from a local PDF collection:

```bash
python3 scripts/reconstruct_restricted_record.py \
  --restricted-jsonl paper_results/neurips_release/restricted/test.jsonl \
  --pdf-dir /path/to/local/pdfs \
  --output-jsonl reconstructed_test.jsonl
```

Compute Cohen's $\kappa$ inter-annotator agreement on a re-annotated audit subset:

```bash
python3 scripts/compute_iaa.py \
  --annotator-a paper_results/annotation/gold_set_annotated.csv \
  --annotator-b paper_results/annotation/gold_set_annotated_v2.csv \
  --output paper_results/annotation/iaa_report.json
```

## Data And Artifact Policy

Do not commit raw PDFs, generated JSONL splits, Hugging Face release folders, trained model checkpoints, API keys, or private paths. Dataset files belong on Hugging Face; archival mirrors and DOIs belong on Zenodo or another long-term artifact host.

## License

Code is released under the MIT License. Dataset records hosted on Hugging Face follow the two-tier release described in the dataset card.
