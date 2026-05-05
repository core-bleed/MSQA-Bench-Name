#!/usr/bin/env python3
"""
Train multiple LLM models sequentially using QLoRA fine-tuning.

Reads models_to_train from config/llm_finetuner.json and trains each model,
saving LoRA adapters and evaluation results to separate directories.
After all models are trained, runs comparison across all results.

Usage:
    python scripts/train_all_llms.py --config config/llm_finetuner.json
    python scripts/train_all_llms.py --model qwen2.5_3b
    python scripts/train_all_llms.py --subset 50
    python scripts/train_all_llms.py --eval-only
"""

import argparse
import json
import sys
import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, "r") as f:
        return json.load(f)


def get_model_output_dir(base_output_dir: str, model_name: str) -> str:
    base_path = Path(base_output_dir)
    return str(base_path / model_name)


def create_model_config(base_config: Dict, model_config: Dict, output_dir: str) -> Dict:
    """Create a model-specific config by merging base and per-model overrides."""
    model_specific = base_config.copy()

    model_specific["base_model"] = model_config["model_path"]
    model_specific["output_dir"] = output_dir
    model_specific["log_file"] = str(
        Path(base_config.get("log_file", "logs/llm_finetuning.log")).parent
        / f"llm_finetuning_{model_config['name']}.log"
    )

    override_keys = [
        "max_seq_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "trust_remote_code",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "num_epochs",
        "learning_rate",
    ]
    for key in override_keys:
        if key in model_config:
            model_specific[key] = model_config[key]

    # Remove non-config keys
    for key in ("models_to_train", "_comment", "_training_tips", "_data_split_info"):
        model_specific.pop(key, None)

    return model_specific


def save_temp_config(config: Dict, temp_path: Path) -> None:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w") as f:
        json.dump(config, f, indent=2)


def train_model(
    model_config: Dict,
    base_config: Dict,
    project_dir: Path,
    subset_size: Optional[int] = None,
    gpu_id: Optional[int] = None,
    force_cpu: bool = False,
) -> bool:
    """Train a single LLM using the llm_finetuner module. Returns True on success."""
    model_name = model_config["name"]
    model_path = model_config["model_path"]

    logger.info("=" * 80)
    logger.info(f"Training LLM: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info("=" * 80)

    output_dir = get_model_output_dir(base_config["output_dir"], model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Skip if already trained
    final_adapter_path = Path(output_dir) / "final_adapter"
    if final_adapter_path.exists():
        logger.warning(f"Model already exists at {final_adapter_path}")
        if not sys.stdin.isatty():
            logger.info(f"Skipping {model_name} (already exists)")
            return True
        response = input(f"Skip {model_name}? (y/n): ").strip().lower()
        if response == "y":
            logger.info(f"Skipping {model_name}")
            return True

    model_specific_config = create_model_config(base_config, model_config, output_dir)

    temp_config_path = project_dir / "logs" / f"llm_config_{model_name}.json"
    save_temp_config(model_specific_config, temp_config_path)

    python_cmd = sys.executable
    cmd = [
        python_cmd,
        "-u",
        "-m",
        "src.llm_trainers.llm_finetuner",
        "--config",
        str(temp_config_path),
    ]

    if subset_size:
        cmd.extend(["--subset_size", str(subset_size)])

    env = {}
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Forcing CPU usage")
    elif gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Using GPU: {gpu_id}")

    try:
        logger.info(f"Starting training for {model_name}...")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Command: {' '.join(cmd)}")

        merged_env = os.environ.copy()
        merged_env.update(env)

        subprocess.run(
            cmd, cwd=str(project_dir), env=merged_env, check=True
        )

        logger.info(f"Successfully trained {model_name}")
        logger.info(f"  Adapter saved to: {output_dir}/final_adapter")

        temp_config_path.unlink(missing_ok=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {model_name}: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"\nTraining interrupted for {model_name}")
        logger.info(f"  Config saved at: {temp_config_path}")
        logger.info(
            f"  Resume with: python -m src.llm_trainers.llm_finetuner "
            f"--config {temp_config_path} --resume"
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error training {model_name}: {e}", exc_info=True)
        return False


def evaluate_model(
    model_config: Dict,
    base_config: Dict,
    project_dir: Path,
    gpu_id: Optional[int] = None,
) -> bool:
    """Run evaluation on an already-trained model."""
    model_name = model_config["name"]
    output_dir = get_model_output_dir(base_config["output_dir"], model_name)
    adapter_path = Path(output_dir) / "final_adapter"

    if not adapter_path.exists():
        logger.warning(f"No adapter found for {model_name} at {adapter_path}")
        return False

    logger.info(f"Evaluating {model_name}...")

    python_cmd = sys.executable
    cmd = [
        python_cmd,
        "-u",
        "-c",
        f"""
import torch, json
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.llm_trainers.evaluators import evaluate_model
from src.llm_trainers.data_utils import LLMDataConfig

config_path = "{output_dir}/training_summary.json"
if Path(config_path).exists():
    with open(config_path) as f:
        summary = json.load(f)
    base_model_name = summary.get("base_model", "{model_config["model_path"]}")
else:
    base_model_name = "{model_config["model_path"]}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(model, "{adapter_path}")
tokenizer = AutoTokenizer.from_pretrained("{adapter_path}")

result = evaluate_model(
    model, tokenizer,
    jsonl_path="{base_config["input_jsonl"]}",
    model_name="{model_name}",
    split="test",
    sample_size={base_config.get("eval_sample_size", 50)},
    output_dir="{output_dir}",
)
print(result)
""",
    ]

    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    merged_env = os.environ.copy()
    merged_env.update(env)

    try:
        subprocess.run(cmd, cwd=str(project_dir), env=merged_env, check=True)
        logger.info(f"Evaluation complete for {model_name}")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name}: {e}")
        return False


def run_comparison(base_output_dir: str) -> None:
    """Run cross-model comparison on all available results."""
    from src.llm_trainers.model_comparison import compare_from_directory

    logger.info("\nRunning cross-model comparison...")
    ranking = compare_from_directory(base_output_dir)
    if ranking:
        logger.info(f"Best model: {ranking[0]['model_name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple LLMs sequentially with QLoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/llm_finetuner.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Train only this specific model (by name)"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Train on subset of data (for testing)"
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have final_adapter directory",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate already-trained models (skip training)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run comparison on existing evaluation results",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompts"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Comparison-only mode
    if args.compare_only:
        run_comparison(config.get("output_dir", "models/fine_tuned_llms"))
        return

    models_to_train = config.get("models_to_train", [])
    if not models_to_train:
        logger.error("No 'models_to_train' found in config")
        sys.exit(1)

    if args.model:
        models_to_train = [m for m in models_to_train if m["name"] == args.model]
        if not models_to_train:
            logger.error(f"Model '{args.model}' not found in config")
            sys.exit(1)
    else:
        models_to_train = [m for m in models_to_train if m.get("enabled", True)]

    if not models_to_train:
        logger.warning("No models to train (all disabled or filtered)")
        sys.exit(0)

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Summary
    logger.info("=" * 80)
    logger.info("MULTI-MODEL LLM FINE-TUNING (QLoRA)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Models to {'evaluate' if args.eval_only else 'train'}: {len(models_to_train)}")
    for i, model in enumerate(models_to_train, 1):
        logger.info(f"  {i}. {model['name']} ({model['model_path']})")
    if args.subset:
        logger.info(f"Subset size: {args.subset}")
    logger.info("=" * 80)

    if not args.eval_only and not args.yes:
        if sys.stdin.isatty():
            response = input("\nProceed with training? (y/n): ").strip().lower()
            if response != "y":
                logger.info("Cancelled")
                sys.exit(0)
        else:
            logger.info("Running non-interactively, proceeding automatically...")

    results = {}
    start_time = datetime.now()

    for i, model_config in enumerate(models_to_train, 1):
        model_name = model_config["name"]
        logger.info(f"\n[{i}/{len(models_to_train)}] Processing {model_name}")

        if args.eval_only:
            success = evaluate_model(model_config, config, project_dir, gpu_id=args.gpu)
            results[model_name] = {"status": "evaluated" if success else "eval_failed"}
            continue

        if args.skip_existing:
            output_dir = get_model_output_dir(config["output_dir"], model_name)
            if (Path(output_dir) / "final_adapter").exists():
                logger.info(f"Skipping {model_name} (already exists)")
                results[model_name] = {"status": "skipped", "output_dir": output_dir}
                continue

        try:
            success = train_model(
                model_config,
                config,
                project_dir,
                subset_size=args.subset,
                gpu_id=args.gpu,
                force_cpu=args.cpu,
            )
            results[model_name] = {
                "status": "success" if success else "failed",
                "output_dir": get_model_output_dir(config["output_dir"], model_name),
            }
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            logger.info("\nCompleted models:")
            for name, result in results.items():
                logger.info(f"  {name}: {result['status']}")
            sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    successful = [n for n, r in results.items() if r["status"] == "success"]
    failed = [n for n, r in results.items() if r["status"] == "failed"]
    skipped = [n for n, r in results.items() if r["status"] == "skipped"]

    logger.info(f"Total time: {duration}")
    logger.info(f"Successful: {len(successful)}")
    for name in successful:
        logger.info(f"  {name}: {results[name].get('output_dir', '')}")

    if skipped:
        logger.info(f"Skipped: {len(skipped)}")
    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for name in failed:
            logger.warning(f"  {name}")

    logger.info("=" * 80)

    # Save summary
    summary_path = project_dir / "logs" / "multi_llm_training_summary.json"
    summary = {
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")

    # Run comparison if we have results
    if len(successful) >= 2:
        try:
            run_comparison(config.get("output_dir", "models/fine_tuned_llms"))
        except Exception as e:
            logger.warning(f"Comparison failed: {e}")


if __name__ == "__main__":
    main()
