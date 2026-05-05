"""
Streaming Embedding Fine-Tuner for Large-Scale Training.

Features:
- Memory-efficient streaming data loading (handles 1M+ records)
- Hash-based deterministic train/val/test split
- Exact-position checkpoint and resume
- Information Retrieval evaluation (Recall@k, MRR, NDCG)
- Gradient accumulation for larger effective batch sizes
- Dynamic step calculation based on dataset size
- Signal handling for graceful shutdown

Usage:
    # Start training
    python streaming_finetuner.py --config config/embedding_finetuner.json
    
    # Resume from checkpoint
    python streaming_finetuner.py --config config/embedding_finetuner.json --resume
"""

import os
import sys
import json
import signal
import random
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from .data_utils import (
    StreamingQADataset,
    DataConfig,
    count_records_in_split,
    load_split_samples,
)
from .evaluators import create_ir_evaluator, EvaluationResult, evaluate_model


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Paths
    input_jsonl: str = "consolidated_qa.jsonl"
    output_dir: str = "models/fine_tuned_embeddings"
    checkpoint_dir: str = "models/fine_tuned_embeddings/checkpoints"
    log_file: str = "logs/streaming_finetuning.log"
    
    # Model
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 256
    # HuggingFace / SentenceTransformer options
    trust_remote_code: bool = False
    
    # Training
    num_epochs: int = 1
    train_batch_size: int = 64
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 4
    
    # Data
    streaming: bool = True
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    subset_size: Optional[int] = None
    
    # Filtering
    min_question_length: int = 10
    max_question_length: int = 256
    min_answer_length: int = 20
    max_answer_length: int = 512
    clean_answers: bool = True
    
    # Evaluation
    eval_sample_size: int = 5000
    evaluation_steps: Optional[int] = None  # Auto-calculated if None
    
    # Checkpointing
    save_steps: Optional[int] = None  # Auto-calculated if None
    checkpoint_save_limit: int = 3
    
    # Loss
    loss_function: str = "MultipleNegativesRankingLoss"
    
    @classmethod
    def from_json(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Filter out unknown keys and comments
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered)
    
    def to_data_config(self) -> DataConfig:
        """Convert to DataConfig for data utilities."""
        return DataConfig(
            min_question_length=self.min_question_length,
            max_question_length=self.max_question_length,
            min_answer_length=self.min_answer_length,
            max_answer_length=self.max_answer_length,
            clean_answers=self.clean_answers,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            subset_size=self.subset_size,
        )


@dataclass
class CheckpointState:
    """State saved in checkpoints for exact resume."""
    # Training progress
    epoch: int
    global_step: int
    records_processed: int
    best_metric: float
    best_metric_step: int
    
    # Timestamps
    started_at: str
    last_saved_at: str
    
    # Config hash for validation
    config_hash: str
    
    # Metrics history
    metrics_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointState':
        return cls(**data)


class StreamingEmbeddingFinetuner:
    """
    Memory-efficient embedding fine-tuner for large datasets.
    
    Supports:
    - Streaming data loading (no full file load)
    - Checkpoint and exact-position resume
    - Gradient accumulation
    - IR evaluation metrics
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.device = self._get_device()
        
        # State
        self.model: Optional[SentenceTransformer] = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0
        self.records_processed = 0
        self.best_metric = 0.0
        self.best_metric_step = 0
        self.metrics_history: List[Dict[str, Any]] = []
        self.started_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Calculate config hash for checkpoint validation
        self.config_hash = self._compute_config_hash()
        
        logger.info(f"StreamingEmbeddingFinetuner initialized")
        logger.info(f"Config hash: {self.config_hash[:16]}...")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        self.shutdown_requested = True
    
    def _compute_config_hash(self) -> str:
        """Compute hash of config for checkpoint validation."""
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_device(self) -> str:
        """Determine device for training."""
        if torch.cuda.is_available():
            device = "cuda"
            # Get the actual device index (0 when CUDA_VISIBLE_DEVICES is set)
            device_idx = 0
            gpu_name = torch.cuda.get_device_name(device_idx)
            gpu_props = torch.cuda.get_device_properties(device_idx)
            
            # Check compute capability
            compute_cap = gpu_props.major * 10 + gpu_props.minor
            logger.info(f"Using CUDA device {device_idx}: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
            logger.info(f"Compute Capability: {gpu_props.major}.{gpu_props.minor} (sm_{gpu_props.major}{gpu_props.minor})")
            
            # Warn if compute capability might be too low
            if compute_cap < 70:
                logger.warning(
                    f"GPU compute capability {gpu_props.major}.{gpu_props.minor} may not be supported by this PyTorch build. "
                    f"PyTorch typically requires compute capability 7.0+. "
                    f"If you get CUDA errors, install PyTorch compiled for your GPU architecture."
                )
        else:
            device = "cpu"
            logger.warning("No GPU detected, using CPU (training will be slow)")
        return device

    def _create_sentence_transformer(self, model_name_or_path: str) -> SentenceTransformer:
        """
        Create a SentenceTransformer instance, respecting config options.
        
        trust_remote_code is required for some HF repos (e.g., nomic-ai models).
        """
        model_kwargs = {}
        if getattr(self.config, "trust_remote_code", False):
            model_kwargs["trust_remote_code"] = True
        return SentenceTransformer(model_name_or_path, device=self.device, **model_kwargs)
    
    def setup_logging(self) -> None:
        """Configure logging."""
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=[
                logging.FileHandler(str(log_path)),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_directories(self) -> None:
        """Create output directories."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def calculate_training_steps(self, num_train_examples: int) -> Dict[str, int]:
        """
        Calculate dynamic training steps based on dataset size.
        
        Args:
            num_train_examples: Number of training examples
            
        Returns:
            Dict with warmup_steps, evaluation_steps, save_steps
        """
        effective_batch = self.config.train_batch_size * self.config.gradient_accumulation_steps
        steps_per_epoch = num_train_examples // effective_batch
        total_steps = steps_per_epoch * self.config.num_epochs
        
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Evaluation and save steps
        # Eval once per epoch by default
        evaluation_steps = self.config.evaluation_steps or steps_per_epoch
        save_steps = self.config.save_steps or max(evaluation_steps, steps_per_epoch // 2)
        
        logger.info(f"Training steps calculation:")
        logger.info(f"  - Examples: {num_train_examples}")
        logger.info(f"  - Effective batch size: {effective_batch}")
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total steps: {total_steps}")
        logger.info(f"  - Warmup steps: {warmup_steps} ({self.config.warmup_ratio*100:.1f}%)")
        logger.info(f"  - Evaluation steps: {evaluation_steps}")
        logger.info(f"  - Save steps: {save_steps}")
        
        return {
            'steps_per_epoch': steps_per_epoch,
            'total_steps': total_steps,
            'warmup_steps': warmup_steps,
            'evaluation_steps': evaluation_steps,
            'save_steps': save_steps,
        }
    
    def get_loss_function(self, model: SentenceTransformer):
        """Get the training loss function."""
        loss_name = self.config.loss_function.lower()
        
        if loss_name == "multiplenegativesrankingloss":
            return losses.MultipleNegativesRankingLoss(model)
        elif loss_name == "contrastiveloss":
            return losses.ContrastiveLoss(model)
        elif loss_name == "cosinesimilarityloss":
            return losses.CosineSimilarityLoss(model)
        else:
            logger.warning(f"Unknown loss: {self.config.loss_function}, using MultipleNegativesRankingLoss")
            return losses.MultipleNegativesRankingLoss(model)
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save training checkpoint with full state.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{self.global_step}"
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(checkpoint_path / "model"))
        
        # Save optimizer state
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint_path / "optimizer.pt")
        
        # Save RNG states
        rng_state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        torch.save(rng_state, checkpoint_path / "rng_state.pt")
        
        # Save training state
        state = CheckpointState(
            epoch=self.current_epoch,
            global_step=self.global_step,
            records_processed=self.records_processed,
            best_metric=self.best_metric,
            best_metric_step=self.best_metric_step,
            started_at=self.started_at,
            last_saved_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            config_hash=self.config_hash,
            metrics_history=self.metrics_history,
        )
        
        with open(checkpoint_path / "state.json", 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = Path(self.config.output_dir) / "best_model"
            self.model.save(str(best_path))
            logger.info(f"Best model saved: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_step_*"),
            key=lambda p: int(p.name.split("_")[-1])
        )
        
        # Keep only the most recent
        while len(checkpoints) > self.config.checkpoint_save_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint, or None to find latest
            
        Returns:
            True if checkpoint loaded successfully
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoints = sorted(
                checkpoint_dir.glob("checkpoint_step_*"),
                key=lambda p: int(p.name.split("_")[-1])
            )
            
            if not checkpoints:
                logger.info("No checkpoints found")
                return False
            
            checkpoint_path = str(checkpoints[-1])
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load state
        with open(checkpoint_path / "state.json", 'r') as f:
            state = CheckpointState.from_dict(json.load(f))
        
        # Validate config hash
        if state.config_hash != self.config_hash:
            logger.warning("Config hash mismatch! Training config may have changed.")
        
        # Load model
        model_path = checkpoint_path / "model"
        self.model = self._create_sentence_transformer(str(model_path))
        self.model.max_seq_length = self.config.max_seq_length
        
        # Restore state
        self.global_step = state.global_step
        self.current_epoch = state.epoch
        self.records_processed = state.records_processed
        self.best_metric = state.best_metric
        self.best_metric_step = state.best_metric_step
        self.metrics_history = state.metrics_history
        self.started_at = state.started_at
        
        # Load RNG states
        rng_path = checkpoint_path / "rng_state.pt"
        if rng_path.exists():
            rng_state = torch.load(rng_path)
            torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['cuda'])
            np.random.set_state(rng_state['numpy'])
            random.setstate(rng_state['python'])
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        logger.info(f"Records processed: {self.records_processed}")
        logger.info(f"Best metric: {self.best_metric:.4f} at step {self.best_metric_step}")
        
        return True
    
    def create_dataloader(self, split: str, skip_records: int = 0) -> DataLoader:
        """Create streaming dataloader for a split."""
        data_config = self.config.to_data_config()
        
        dataset = StreamingQADataset(
            jsonl_path=self.config.input_jsonl,
            split=split,
            config=data_config,
            skip_records=skip_records,
        )
        
        # Note: shuffle=False is required for streaming
        # In-batch negatives provide implicit "shuffling" effect
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=0,  # Required for IterableDataset
            collate_fn=self._collate_fn,
        )
        
        return dataloader
    
    def _collate_fn(self, batch: List[InputExample]) -> List[InputExample]:
        """Simple pass-through collate function."""
        return batch
    
    def evaluate(self, evaluator: InformationRetrievalEvaluator) -> float:
        """Run evaluation and return primary metric."""
        self.model.eval()
        
        output_path = Path(self.config.output_dir) / "eval_results"
        output_path.mkdir(parents=True, exist_ok=True)
        
        score = evaluator(
            self.model,
            output_path=str(output_path),
            epoch=self.current_epoch,
            steps=self.global_step,
        )
        
        self.model.train()
        
        return score
    
    def train(self, resume: bool = False) -> None:
        """
        Execute the training loop.
        
        Args:
            resume: Whether to resume from checkpoint
        """
        logger.info("=" * 70)
        logger.info("STARTING EMBEDDING FINE-TUNING")
        logger.info("=" * 70)
        
        # Resume from checkpoint if requested
        if resume:
            if not self.load_checkpoint():
                logger.info("No checkpoint found, starting fresh")
                resume = False
        
        # Load model if not loaded from checkpoint
        if self.model is None:
            logger.info(f"Loading base model: {self.config.base_model}")
            self.model = self._create_sentence_transformer(self.config.base_model)
            self.model.max_seq_length = self.config.max_seq_length
        
        # Count training examples (streaming - may take a moment)
        logger.info("Counting training examples...")
        data_config = self.config.to_data_config()
        num_train = count_records_in_split(
            self.config.input_jsonl, "train", data_config
        )
        logger.info(f"Training examples: {num_train}")
        
        if num_train == 0:
            raise ValueError("No training examples found!")
        
        # Calculate steps
        steps_info = self.calculate_training_steps(num_train)
        
        # Create loss function
        train_loss = self.get_loss_function(self.model)
        logger.info(f"Loss function: {type(train_loss).__name__}")
        
        # Create evaluator
        logger.info("Creating IR evaluator...")
        evaluator = create_ir_evaluator(
            jsonl_path=self.config.input_jsonl,
            split="val",
            sample_size=self.config.eval_sample_size,
            config=data_config,
        )
        
        # Evaluate base model if starting fresh
        if not resume:
            logger.info("Evaluating base model before training...")
            base_score = self.evaluate(evaluator)
            # Handle both dict and float return types from evaluator
            if isinstance(base_score, dict):
                score_value = base_score.get('mrr@10', base_score.get('map@10', 0.0))
                score_display = base_score
            else:
                score_value = base_score
                score_display = base_score
            self.metrics_history.append({
                'epoch': 0,
                'step': 0,
                'type': 'base_model',
                'score': score_display,
                'timestamp': datetime.now().isoformat() + "Z",
            })
            logger.info(f"Base model score: {score_value:.4f}" if isinstance(score_value, (int, float)) else f"Base model score: {score_display}")
        
        # Training using sentence-transformers fit() with proper parameters
        # Create dataloader
        skip_records = self.records_processed if resume else 0
        train_dataloader = self.create_dataloader("train", skip_records=skip_records)
        
        # Calculate remaining epochs if resuming
        start_epoch = self.current_epoch if resume else 0
        
        logger.info(f"\nStarting training from epoch {start_epoch}")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch: {self.config.train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        
        try:
            # Use sentence-transformers fit() method with all parameters
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.config.num_epochs - start_epoch,
                evaluator=evaluator,
                evaluation_steps=steps_info['evaluation_steps'],
                warmup_steps=steps_info['warmup_steps'],
                output_path=self.config.output_dir,
                save_best_model=True,
                show_progress_bar=True,
                checkpoint_path=self.config.checkpoint_dir,
                checkpoint_save_steps=steps_info['save_steps'],
                checkpoint_save_total_limit=self.config.checkpoint_save_limit,
                # FIX: Pass learning rate to optimizer
                optimizer_params={'lr': self.config.learning_rate},
                # Scheduler
                scheduler='warmupcosine',
                # Use steps from beginning of training
                use_amp=torch.cuda.is_available(),  # Mixed precision if GPU
            )
            
            # Training completed
            logger.info("\n" + "=" * 70)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            
            # Final evaluation on test set
            logger.info("\nFinal evaluation on test set...")
            test_result = evaluate_model(
                self.model,
                self.config.input_jsonl,
                split="test",
                sample_size=self.config.eval_sample_size,
                config=data_config,
                output_dir=self.config.output_dir,
            )
            
            logger.info(f"\nTest Results:")
            logger.info(f"  {test_result}")
            
            # Save final model
            final_path = Path(self.config.output_dir) / "final_model"
            self.model.save(str(final_path))
            logger.info(f"\nFinal model saved: {final_path}")
            
            # Save training summary
            summary = {
                'completed': True,
                'completed_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'started_at': self.started_at,
                'base_model': self.config.base_model,
                'num_epochs': self.config.num_epochs,
                'train_examples': num_train,
                'test_results': test_result.to_dict(),
                'metrics_history': self.metrics_history,
                'final_model_path': str(final_path),
            }
            
            with open(Path(self.config.output_dir) / "training_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            self._handle_shutdown()
        except Exception as e:
            logger.error(f"\nTraining failed: {e}", exc_info=True)
            self._handle_shutdown()
            raise
    
    def _handle_shutdown(self) -> None:
        """Handle graceful shutdown."""
        logger.info("Performing graceful shutdown...")
        
        if self.model is not None:
            # Save checkpoint
            checkpoint_path = self.save_checkpoint()
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save shutdown state
            shutdown_info = {
                'shutdown_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'global_step': self.global_step,
                'epoch': self.current_epoch,
                'records_processed': self.records_processed,
                'checkpoint_path': checkpoint_path,
            }
            
            with open(Path(self.config.output_dir) / "shutdown_state.json", 'w') as f:
                json.dump(shutdown_info, f, indent=2)
            
            logger.info("Shutdown complete. Use --resume to continue training.")


def main():
    parser = argparse.ArgumentParser(
        description="Streaming Embedding Fine-Tuner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/embedding_finetuner.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        help="Override input JSONL file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Override base model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        help="Train on subset (for testing pipeline)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = TrainingConfig.from_json(str(config_path))
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = TrainingConfig()
    
    # Override with command-line args
    if args.input_jsonl:
        config.input_jsonl = args.input_jsonl
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.base_model:
        config.base_model = args.base_model
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.subset_size:
        config.subset_size = args.subset_size
    
    # Create finetuner and train
    finetuner = StreamingEmbeddingFinetuner(config)
    finetuner.train(resume=args.resume)


if __name__ == "__main__":
    main()
