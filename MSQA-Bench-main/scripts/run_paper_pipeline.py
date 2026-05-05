#!/usr/bin/env python3
"""
MSQA-Bench Paper Pipeline Orchestrator

Runs the full paper pipeline from consolidated_qa.jsonl.
Uses your existing Q&A (e.g. from Qwen); it does not regenerate answers.

  1. Quality filters → filter low-quality, deduplicate
  2. Question classifier → factual/method/definition/etc.
  3. Schema enricher → metadata, evidence spans, answer style
  4. Split generator → train/val/test (85/10/5)
  5. Retrieval baselines → BM25 + embeddings (retrieval metrics only)
  6. Gold set sampler → for human annotation
  7. Generate tables → LaTeX for paper

Usage:
  python scripts/run_paper_pipeline.py --config config/paper_pipeline.json
  python scripts/run_paper_pipeline.py --config config/paper_pipeline.json --steps dataset
  python scripts/run_paper_pipeline.py --config config/paper_pipeline.json --background

Run in background on server:
  nohup python scripts/run_paper_pipeline.py --config config/paper_pipeline.json --background > paper_results/nohup.out 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prefer venv python for subprocess calls
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON_CMD = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def setup_logging(log_file: Optional[Path], level: int = logging.INFO) -> None:
    """Configure logging to file and console."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load pipeline config."""
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_text_dir(config: Dict[str, Any], qa_file: Path) -> Optional[Path]:
    """Resolve text directory; try config and fallbacks."""
    fallbacks = config.get("input", {}).get("text_dir_fallbacks", [])
    text_dir = config.get("input", {}).get("text_dir")
    if text_dir:
        fallbacks = [text_dir] + [f for f in fallbacks if f != text_dir]
    base = PROJECT_ROOT
    for cand in fallbacks:
        p = base / cand
        if p.is_dir():
            return p
    return None


def run_step(
    name: str,
    cmd: List[str],
    cwd: Path,
    log: logging.Logger,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Run a pipeline step; return True on success."""
    log.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env={**(env or {}), **subprocess.os.environ},
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error("%s failed (exit %d)", name, result.returncode)
            if result.stderr:
                log.error("stderr: %s", result.stderr[:500])
            return False
        log.info("%s completed", name)
        return True
    except Exception as e:
        log.exception("%s failed: %s", name, e)
        return False


def _dataset_parallel_args(
    config: Dict[str, Any], workers_override: Optional[int] = None
) -> tuple[int, int, int, int]:
    """Returns (workers, qf_chunk, qc_chunk, se_chunk)."""
    par = config.get("parallel", {})
    w = workers_override if workers_override is not None else int(par.get("dataset_workers", 1))
    w = max(1, w)
    qf_c = max(100, int(par.get("quality_chunk_size", 8000)))
    qc_c = max(100, int(par.get("classifier_chunk_size", 8000)))
    se_c = max(100, int(par.get("enricher_chunk_size", 4000)))
    return w, qf_c, qc_c, se_c


def run_dataset_steps(
    config: Dict[str, Any],
    out: Path,
    log: logging.Logger,
    workers_override: Optional[int] = None,
) -> bool:
    """Run dataset pipeline: quality_filters → question_classifier → schema_enricher → split_generator."""
    inp = config.get("input", {})
    qa_file = PROJECT_ROOT / inp.get("qa_file", "data/consolidated_qa.jsonl")
    if not qa_file.exists():
        log.error("QA file not found: %s", qa_file)
        return False

    text_dir = resolve_text_dir(config, qa_file)
    dataset_dir = out / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    workers, qf_chunk, qc_chunk, se_chunk = _dataset_parallel_args(config, workers_override)
    if workers > 1:
        log.info(
            "Parallel dataset steps: workers=%s (chunks qf=%s qc=%s se=%s)",
            workers,
            qf_chunk,
            qc_chunk,
            se_chunk,
        )

    # 1. Quality filters
    filtered = dataset_dir / "quality_filtered.jsonl"
    cmd_qf = [
        PYTHON_CMD,
        "-m", "paper.dataset.quality_filters",
        "--input", str(qa_file),
        "--output", str(filtered),
        "--min-quality", str(config.get("dataset", {}).get("quality_filters", {}).get("min_quality", 0.4)),
    ]
    if not config.get("dataset", {}).get("quality_filters", {}).get("deduplicate", True):
        cmd_qf.append("--no-dedup")
    if workers > 1:
        cmd_qf.extend(["--workers", str(workers), "--chunk-size", str(qf_chunk)])
    if not run_step("Quality filters", cmd_qf, PROJECT_ROOT, log):
        return False

    # 2. Question classifier
    classified = dataset_dir / "classified.jsonl"
    cmd_qc = [
        PYTHON_CMD,
        "-m", "paper.dataset.question_classifier",
        "--input", str(filtered),
        "--output", str(classified),
    ]
    if workers > 1:
        cmd_qc.extend(["--workers", str(workers), "--chunk-size", str(qc_chunk)])
    if not run_step("Question classifier", cmd_qc, PROJECT_ROOT, log):
        return False

    # 3. Schema enricher
    enriched = dataset_dir / "enriched.jsonl"
    cmd_se = [
        PYTHON_CMD,
        "-m", "paper.dataset.schema_enricher",
        "--input", str(classified),
        "--output", str(enriched),
    ]
    if text_dir:
        cmd_se.extend(["--text-dir", str(text_dir)])
    if workers > 1:
        cmd_se.extend(["--workers", str(workers), "--chunk-size", str(se_chunk)])
    if not run_step("Schema enricher", cmd_se, PROJECT_ROOT, log):
        return False

    # 4. Split generator
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_cfg = config.get("dataset", {}).get("split", {})
    cmd_sg = [
        PYTHON_CMD,
        "-m", "paper.dataset.split_generator",
        "--input", str(enriched),
        "--output-dir", str(splits_dir),
        "--train-ratio", str(split_cfg.get("train_ratio", 0.85)),
        "--val-ratio", str(split_cfg.get("val_ratio", 0.10)),
        "--test-ratio", str(split_cfg.get("test_ratio", 0.05)),
    ]
    if not run_step("Split generator", cmd_sg, PROJECT_ROOT, log):
        return False

    return True


def run_evaluation_steps(config: Dict[str, Any], out: Path, log: logging.Logger) -> bool:
    """Run retrieval evaluation only (BM25 / embeddings). Does not call an LLM."""
    eval_dir = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out / "dataset"
    enriched = dataset_dir / "enriched.jsonl"
    if not enriched.exists():
        log.warning("Enriched file not found; skipping evaluation")
        return True

    ev_cfg = config.get("evaluation", {})

    if ev_cfg.get("retrieval", {}).get("enabled", True):
        ret_cfg = ev_cfg.get("retrieval", {})
        out_ret = eval_dir / "retrieval_results.json"
        cmd = [
            PYTHON_CMD,
            "-m", "paper.evaluation.retrieval_baselines",
            "--input", str(enriched),
            "--output", str(out_ret),
        ]
        if ret_cfg.get("compare_baselines", True):
            cmd.append("--compare")
        else:
            cmd.extend(["--model", ret_cfg.get("models", ["bm25"])[0]])
        if ret_cfg.get("sample_size"):
            cmd.extend(["--sample", str(ret_cfg["sample_size"])])
        device = ret_cfg.get("device", "auto")
        cmd.extend(["--device", str(device)])
        cmd.extend(["--embed-batch-size", str(int(ret_cfg.get("embed_batch_size", 64)))])
        run_step("Retrieval baselines", cmd, PROJECT_ROOT, log)

    return True


def run_annotation_step(config: Dict[str, Any], out: Path, log: logging.Logger) -> bool:
    """Sample for human annotation."""
    ann_cfg = config.get("annotation", {})
    if not ann_cfg.get("enabled", True):
        return True

    ann_dir = out / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    enriched = out / "dataset" / "enriched.jsonl"
    if not enriched.exists():
        log.warning("Enriched file not found; skipping annotation")
        return True

    out_ann = ann_dir / "gold_set_sample.csv"
    cmd = [
        PYTHON_CMD,
        "-m", "paper.annotation.gold_set_sampler",
        "--input", str(enriched),
        "--output", str(out_ann),
        "--size", str(ann_cfg.get("sample_size", 200)),
        "--format", ann_cfg.get("format", "csv"),
    ]
    return run_step("Gold set sampler", cmd, PROJECT_ROOT, log)


def run_figures_step(config: Dict[str, Any], out: Path, log: logging.Logger) -> bool:
    """Generate paper tables."""
    figs_dir = out / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    enriched = out / "dataset" / "enriched.jsonl"
    if not enriched.exists():
        log.warning("Enriched file not found; skipping figures")
        return True

    cmd = [
        PYTHON_CMD,
        "-m", "paper.figures.generate_tables",
        "--qa-file", str(enriched),
        "--output-dir", str(figs_dir),
    ]
    eval_dir = out / "evaluation"
    ret_file = eval_dir / "retrieval_results.json"
    if ret_file.exists():
        cmd.extend(["--retrieval", str(ret_file)])
    run_step("Generate tables", cmd, PROJECT_ROOT, log)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MSQA-Bench paper pipeline from consolidated_qa.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/paper_pipeline.json",
        help="Path to pipeline config",
    )
    parser.add_argument(
        "--steps",
        choices=["dataset", "evaluation", "annotation", "figures", "all"],
        default="all",
        help="Which steps to run",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background (detached); writes to log file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Override output base directory",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Override config parallel.dataset_workers for quality/classify/enrich steps",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    config = load_config(config_path)
    out_cfg = config.get("output", {})
    base_out = args.output_dir or (PROJECT_ROOT / out_cfg.get("base_dir", "paper_results"))
    log_file = base_out / out_cfg.get("log_file", "pipeline.log")
    if isinstance(log_file, str):
        log_file = base_out / "pipeline.log"

    base_out.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    log = logging.getLogger("paper_pipeline")

    log.info("Paper pipeline started at %s", datetime.now(timezone.utc).isoformat())
    log.info("Output directory: %s", base_out)

    steps = args.steps
    if steps == "all":
        steps_list = ["dataset", "evaluation", "annotation", "figures"]
    else:
        steps_list = [steps]

    success = True
    for s in steps_list:
        if s == "dataset":
            success = run_dataset_steps(config, base_out, log, workers_override=args.workers) and success
        elif s == "evaluation":
            success = run_evaluation_steps(config, base_out, log) and success
        elif s == "annotation":
            success = run_annotation_step(config, base_out, log) and success
        elif s == "figures":
            success = run_figures_step(config, base_out, log) and success

    log.info("Paper pipeline finished at %s", datetime.now(timezone.utc).isoformat())
    log.info("Results saved to %s", base_out)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
