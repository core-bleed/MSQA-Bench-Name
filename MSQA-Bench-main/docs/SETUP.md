# Setup Guide

## Quick Start

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For the artifact smoke test only, the lightweight dependency set is enough:
   ```bash
   pip install -r requirements-smoke.txt
   ```

3. **(Optional) Set API keys for LLM-assisted preprocessing**
   ```bash
   # Only needed if you re-run the LLM-based PDF cleaners.
   # Required by src/pdf_processors/llm_pdf_processor.py and page_by_page_processor.py.
   export OPENAI_API_KEY="sk-..."
   ```

4. **(Optional) Start GROBID for structured academic parsing**
   ```bash
   docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 \
       grobid/grobid:0.8.1
   ```

5. **(Optional) Start a vLLM server for QA generation**
   The pipeline expects an OpenAI-compatible endpoint at
   `http://localhost:8000/v1` serving Qwen2.5-14B-Instruct-AWQ.

## Reproducing the released splits

The released benchmark is hosted on Hugging Face:
<https://huggingface.co/datasets/asad00027/MSQA-Bench>.

```python
from datasets import load_dataset

ds = load_dataset("asad00027/MSQA-Bench", "redistributable", split="test")
print(ds[0])
```

## Artifact smoke test

Run this before submitting or reviewing the artifact:

```bash
python3 scripts/run_smoke_test.py --sample-size 32
```

The smoke test creates a tiny PDF, runs the PyMuPDF extraction CLI, downloads a
small redistributable QA sample from Hugging Face, runs BM25 retrieval
evaluation, and writes `paper_results/smoke/smoke_retrieval_table.md`.

If the machine has no network access, use the offline fixture mode:

```bash
python3 scripts/run_smoke_test.py --sample-source fixture --sample-size 8
```

To re-run the construction pipeline on new MS PDFs, place them in
`data/input/` and run:

```bash
python scripts/run_paper_pipeline.py --config config/paper_pipeline.json
```

## Directory layout

| Path                      | Purpose                                       |
|---------------------------|-----------------------------------------------|
| `data/input/`             | Source PDFs                                   |
| `data/extracted_text/`    | Extracted plain-text                          |
| `data/qa_outputs/`        | Per-file generated QA JSONLs                  |
| `data/consolidated_qa.jsonl` | Concatenated QA file (input to pipeline)   |
| `paper_results/`          | Per-stage pipeline outputs                    |
| `models/fine_tuned_*`     | Trained adapter / encoder weights             |

## Troubleshooting

- **CUDA out-of-memory.** Lower `per_device_train_batch_size` and raise
  `gradient_accumulation_steps` in `config/llm_finetuner.json`.
- **GROBID timeout.** Re-run the affected file individually with
  `python src/pdf_processors/grobid_processor.py --input-dir data/input/<file>`.
- **HuggingFace gated model.** `meta-llama/Llama-3.1-8B-Instruct` is
  intentionally `enabled: false` in the shipped config; request access on
  HF and flip the flag when granted.
