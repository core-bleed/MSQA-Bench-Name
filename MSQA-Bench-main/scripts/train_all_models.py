#!/usr/bin/env python3
"""
Train multiple embedding models sequentially using streaming_finetuner.

Reads models_to_train from config/embedding_finetuner.json and trains each model,
saving results to separate directories.

Usage:
    python scripts/train_all_models.py --config config/embedding_finetuner.json
    
    # Train only enabled models
    python scripts/train_all_models.py
    
    # Train specific model
    python scripts/train_all_models.py --model e5_large_v2
    
    # Test with subset
    python scripts/train_all_models.py --subset 10000
"""

import argparse
import json
import sys
import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def get_model_output_dir(base_output_dir: str, model_name: str) -> str:
    """Generate output directory for a model."""
    base_path = Path(base_output_dir).parent
    return str(base_path / f"fine_tuned_embeddings_{model_name}")


def create_model_config(base_config: Dict, model_config: Dict, output_dir: str) -> Dict:
    """Create a model-specific config from base config and model settings."""
    # Copy base config
    model_specific_config = base_config.copy()
    
    # Update model-specific settings
    model_specific_config['base_model'] = model_config['model_path']
    model_specific_config['max_seq_length'] = model_config['max_seq_length']
    model_specific_config['output_dir'] = output_dir
    model_specific_config['checkpoint_dir'] = str(Path(output_dir) / "checkpoints")

    # Propagate trust_remote_code to top-level config so the trainer can use it
    if 'trust_remote_code' in model_config:
        model_specific_config['trust_remote_code'] = model_config['trust_remote_code']

    # Optional: per-model training overrides (batch size, eval batch, grad accumulation)
    # If these keys are present on the model entry, they will override the global config
    for key in ["train_batch_size", "eval_batch_size", "gradient_accumulation_steps"]:
        if key in model_config:
            model_specific_config[key] = model_config[key]
    
    # Update log file to be model-specific
    log_file = Path(base_config.get('log_file', 'logs/streaming_finetuning.log'))
    log_dir = log_file.parent
    model_specific_config['log_file'] = str(log_dir / f"streaming_finetuning_{model_config['name']}.log")
    
    # Store instruction prefix info (for future use if needed)
    model_specific_config['_model_info'] = {
        'name': model_config['name'],
        'requires_instruction_prefix': model_config.get('requires_instruction_prefix', False),
        'query_prefix': model_config.get('query_prefix'),
        'document_prefix': model_config.get('document_prefix'),
        'trust_remote_code': model_config.get('trust_remote_code', False)
    }
    
    return model_specific_config


def save_temp_config(config: Dict, temp_path: Path) -> None:
    """Save temporary config file."""
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, 'w') as f:
        json.dump(config, f, indent=2)


def train_model(
    model_config: Dict,
    base_config: Dict,
    project_dir: Path,
    subset_size: Optional[int] = None,
    gpu_id: Optional[int] = None,
    force_cpu: bool = False
) -> bool:
    """
    Train a single model using streaming_finetuner.
    
    Returns:
        True if training succeeded, False otherwise
    """
    model_name = model_config['name']
    model_path = model_config['model_path']
    
    logger.info("=" * 80)
    logger.info(f"Training model: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = get_model_output_dir(base_config['output_dir'], model_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    final_model_path = Path(output_dir) / "final_model"
    if final_model_path.exists():
        logger.warning(f"Model already exists at {final_model_path}")
        # Skip prompt if running non-interactively
        if not sys.stdin.isatty():
            logger.info(f"Skipping {model_name} (already exists)")
            return True
        response = input(f"Skip {model_name}? (y/n): ").strip().lower()
        if response == 'y':
            logger.info(f"Skipping {model_name}")
            return True
    
    # Create model-specific config
    model_specific_config = create_model_config(base_config, model_config, output_dir)
    
    # Save temp config
    temp_config_path = project_dir / "logs" / f"config_{model_name}.json"
    save_temp_config(model_specific_config, temp_config_path)
    
    # Build command
    python_cmd = sys.executable
    cmd = [
        python_cmd,
        "-u",
        "-m",
        "src.embedding_trainers.streaming_finetuner",
        "--config",
        str(temp_config_path)
    ]
    
    if subset_size:
        cmd.extend(["--subset_size", str(subset_size)])
    
    # Set GPU/CPU
    env = {}
    # Make CUDA ordinals match `nvidia-smi` indices (PCI bus order)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if force_cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
        logger.info("Forcing CPU usage")
    elif gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Using GPU: {gpu_id}")
    
    # Run training
    try:
        logger.info(f"Starting training for {model_name}...")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Merge environment variables properly
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        
        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            env=merged_env,
            check=True
        )
        
        logger.info(f"✓ Successfully trained {model_name}")
        logger.info(f"  Model saved to: {output_dir}/final_model")
        
        # Clean up temp config
        temp_config_path.unlink()
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Training failed for {model_name}: {e}")
        logger.error(f"  Check logs at: {model_specific_config['log_file']}")
        return False
    except KeyboardInterrupt:
        logger.warning(f"\nTraining interrupted for {model_name}")
        logger.info(f"  Config saved at: {temp_config_path}")
        logger.info(f"  Resume with: python -m src.embedding_trainers.streaming_finetuner --config {temp_config_path} --resume")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error training {model_name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple embedding models sequentially"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/embedding_finetuner.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Train only this specific model (by name)"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Train on subset of data (for testing)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (useful if GPU has compatibility issues)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have final_model directory"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts (for background execution)"
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Get models to train
    models_to_train = config.get('models_to_train', [])
    if not models_to_train:
        logger.error("No 'models_to_train' found in config")
        logger.info("Add 'models_to_train' array to your config file")
        sys.exit(1)
    
    # Filter models
    if args.model:
        models_to_train = [m for m in models_to_train if m['name'] == args.model]
        if not models_to_train:
            logger.error(f"Model '{args.model}' not found in config")
            sys.exit(1)
    else:
        # Only enabled models
        models_to_train = [m for m in models_to_train if m.get('enabled', True)]
    
    if not models_to_train:
        logger.warning("No models to train (all disabled or filtered)")
        sys.exit(0)
    
    # Get project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Summary
    logger.info("=" * 80)
    logger.info("MULTI-MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Models to train: {len(models_to_train)}")
    for i, model in enumerate(models_to_train, 1):
        logger.info(f"  {i}. {model['name']} ({model['model_path']})")
    if args.subset:
        logger.info(f"Subset size: {args.subset}")
    if args.cpu:
        logger.info("Mode: CPU (forced)")
    elif args.gpu is not None:
        logger.info(f"GPU: {args.gpu}")
    logger.info("=" * 80)
    
    # Confirm (skip if --yes flag or running non-interactively)
    if not args.skip_existing and not args.yes:
        # Check if running in background (no TTY)
        if sys.stdin.isatty():
            response = input("\nProceed with training? (y/n): ").strip().lower()
            if response != 'y':
                logger.info("Cancelled")
                sys.exit(0)
        else:
            logger.info("Running non-interactively, proceeding automatically...")
    
    # Train each model
    results = {}
    start_time = datetime.now()
    
    for i, model_config in enumerate(models_to_train, 1):
        model_name = model_config['name']
        
        logger.info(f"\n[{i}/{len(models_to_train)}] Processing {model_name}")
        
        # Check if exists and skip
        if args.skip_existing:
            output_dir = get_model_output_dir(config['output_dir'], model_name)
            final_model_path = Path(output_dir) / "final_model"
            if final_model_path.exists():
                logger.info(f"Skipping {model_name} (already exists)")
                results[model_name] = {'status': 'skipped', 'output_dir': output_dir}
                continue
        
        # Train
        try:
            success = train_model(
                model_config,
                config,
                project_dir,
                subset_size=args.subset,
                gpu_id=args.gpu,
                force_cpu=args.cpu
            )
            results[model_name] = {
                'status': 'success' if success else 'failed',
                'output_dir': get_model_output_dir(config['output_dir'], model_name)
            }
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user")
            logger.info("\nCompleted models:")
            for name, result in results.items():
                logger.info(f"  {name}: {result['status']}")
            sys.exit(1)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    
    successful = [name for name, r in results.items() if r['status'] == 'success']
    failed = [name for name, r in results.items() if r['status'] == 'failed']
    skipped = [name for name, r in results.items() if r['status'] == 'skipped']
    
    logger.info(f"Total time: {duration}")
    logger.info(f"Successful: {len(successful)}")
    if successful:
        for name in successful:
            logger.info(f"  ✓ {name}: {results[name]['output_dir']}")
    
    if skipped:
        logger.info(f"Skipped: {len(skipped)}")
        for name in skipped:
            logger.info(f"  - {name}")
    
    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for name in failed:
            logger.warning(f"  ✗ {name}")
    
    logger.info("=" * 80)
    
    # Save summary
    summary_path = project_dir / "logs" / "multi_model_training_summary.json"
    summary = {
        'started_at': start_time.isoformat(),
        'completed_at': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'results': results
    }
    
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
