"""
QLoRA LLM Fine-Tuner for Scientific Q&A.

Fine-tunes open-source LLMs using QLoRA (4-bit quantization + LoRA adapters)
on scientific Q&A data. Optimized for single 24GB GPU training.

Features:
- 4-bit NF4 quantization via bitsandbytes
- LoRA adapter training via peft
- SFTTrainer from trl for instruction tuning
- Checkpoint and resume support
- Signal handling for graceful shutdown
- Config-driven multi-model pipeline support

Usage:
    python -m src.llm_trainers.llm_finetuner --config config/llm_finetuner.json
    python -m src.llm_trainers.llm_finetuner --config config/llm_finetuner.json --resume
"""

import json
import os
import signal
import logging
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch

from .data_utils import LLMDataConfig, create_hf_dataset


logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantizationConfig:
    """BitsAndBytes quantization configuration."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class LLMTrainingConfig:
    """Complete LLM fine-tuning configuration."""
    # Paths
    input_jsonl: str = "data/consolidated_qa.jsonl"
    output_dir: str = "models/fine_tuned_llms"
    log_file: str = "logs/llm_finetuning.log"

    # Model
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = 2048
    trust_remote_code: bool = True

    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3

    # Data
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    no_context_ratio: float = 0.3
    max_context_length: int = 1536
    subset_size: Optional[int] = None

    # Filtering
    min_question_length: int = 10
    max_question_length: int = 512
    min_answer_length: int = 20
    max_answer_length: int = 2048
    clean_answers: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Evaluation & checkpointing
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_strategy: str = "steps"

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.001

    # Generation settings for evaluation
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.1
    eval_sample_size: int = 50

    # Misc
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    @classmethod
    def from_json(cls, path: str) -> "LLMTrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        valid_keys = {fld.name for fld in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_data_config(self) -> LLMDataConfig:
        """Convert to LLMDataConfig for data utilities."""
        return LLMDataConfig(
            min_question_length=self.min_question_length,
            max_question_length=self.max_question_length,
            min_answer_length=self.min_answer_length,
            max_answer_length=self.max_answer_length,
            clean_answers=self.clean_answers,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            subset_size=self.subset_size,
            no_context_ratio=self.no_context_ratio,
            max_context_length=self.max_context_length,
            seed=self.seed,
        )

    def to_lora_config(self) -> LoRAConfig:
        return LoRAConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
        )

    def to_quantization_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
        )


class LLMFineTuner:
    """
    QLoRA-based LLM fine-tuner for scientific Q&A.

    Loads a base LLM in 4-bit quantization, attaches LoRA adapters,
    and trains using HuggingFace SFTTrainer on instruction-formatted data.
    """

    def __init__(self, config: LLMTrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("LLMFineTuner initialized")
        logger.info(f"  Base model: {config.base_model}")
        logger.info(f"  Output dir: {config.output_dir}")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        self.shutdown_requested = True

    def setup_logging(self) -> None:
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=[
                logging.FileHandler(str(log_path)),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def setup_directories(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_compute_dtype(self):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.bnb_4bit_compute_dtype, torch.bfloat16)

    def load_model_and_tokenizer(self):
        """Load quantized base model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading model: {self.config.base_model}")

        if self.config.load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self._get_compute_dtype(),
            )
            logger.info(
                f"  Quantization: 4-bit {self.config.bnb_4bit_quant_type}, "
                f"double_quant={self.config.bnb_4bit_use_double_quant}"
            )
        else:
            bnb_config = None
            if not torch.cuda.is_available():
                logger.warning("No GPU available, loading model without quantization")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=self._get_compute_dtype(),
            attn_implementation="eager",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Model loaded: {total_params / 1e9:.2f}B parameters")

    def apply_lora(self):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        logger.info("Applying LoRA adapters...")

        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=self.config.gradient_checkpointing
        )

        target_modules = self.config.lora_target_modules
        if target_modules is None:
            target_modules = self._detect_target_modules()

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, peft_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"  LoRA applied: {trainable_params / 1e6:.2f}M trainable / "
            f"{total_params / 1e6:.1f}M total ({trainable_params / total_params * 100:.2f}%)"
        )
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Rank: {self.config.lora_r}, Alpha: {self.config.lora_alpha}")

    def _detect_target_modules(self) -> List[str]:
        """Auto-detect linear layer names for LoRA targeting."""
        from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
        import bitsandbytes as bnb

        model_type = getattr(self.model.config, "model_type", "").lower()
        if model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
            logger.info(f"  Using known target modules for {model_type}: {modules}")
            return list(modules)

        linear_classes = (torch.nn.Linear, bnb.nn.Linear4bit)
        module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, linear_classes):
                parts = name.split(".")
                module_names.add(parts[-1])

        module_names.discard("lm_head")
        result = sorted(module_names)
        logger.info(f"  Auto-detected target modules: {result}")
        return result

    def _calculate_steps(self, num_train: int) -> Dict[str, int]:
        """Calculate eval_steps and save_steps from dataset size."""
        effective_batch = (
            self.config.per_device_train_batch_size
            * self.config.gradient_accumulation_steps
        )
        steps_per_epoch = max(1, num_train // effective_batch)
        total_steps = steps_per_epoch * self.config.num_epochs

        eval_steps = self.config.eval_steps or max(1, steps_per_epoch // 2)
        save_steps = self.config.save_steps or eval_steps

        logger.info("Training steps:")
        logger.info(f"  Examples: {num_train}")
        logger.info(f"  Effective batch: {effective_batch}")
        logger.info(f"  Steps/epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Eval every: {eval_steps} steps")
        logger.info(f"  Save every: {save_steps} steps")

        return {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
        }

    def train(self, resume_from_checkpoint: Optional[str] = None) -> str:
        """
        Execute the full training pipeline.

        Returns the path to the final saved adapter.
        """
        from trl import SFTConfig, SFTTrainer

        # Hugging Face login for gated models (e.g. Llama)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Hugging Face authentication: token from HF_TOKEN/HUGGING_FACE_HUB_TOKEN")
        else:
            logger.info(
                "No HF_TOKEN set. For gated models (e.g. Llama), set: export HF_TOKEN=hf_..."
            )

        logger.info("=" * 70)
        logger.info("STARTING LLM FINE-TUNING (QLoRA)")
        logger.info("=" * 70)

        # 1 - Load model + tokenizer
        self.load_model_and_tokenizer()

        # 2 - Apply LoRA
        self.apply_lora()

        # 3 - Prepare datasets
        data_config = self.config.to_data_config()

        logger.info("Preparing training dataset...")
        train_dataset = create_hf_dataset(
            self.config.input_jsonl,
            split="train",
            config=data_config,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        logger.info("Preparing validation dataset...")
        val_dataset = create_hf_dataset(
            self.config.input_jsonl,
            split="val",
            config=data_config,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        # 4 - Calculate steps
        steps_info = self._calculate_steps(len(train_dataset))

        # 5 - Auto-detect resume checkpoint
        if resume_from_checkpoint is True or resume_from_checkpoint == "auto":
            resume_from_checkpoint = self._find_latest_checkpoint()

        # 6 - SFTConfig (training args + max_length for sequence truncation)
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            bf16=self.config.bf16 and torch.cuda.is_available(),
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=steps_info["eval_steps"],
            save_strategy="steps",
            save_steps=steps_info["save_steps"],
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.seed,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            remove_unused_columns=False,
            max_length=self.config.max_seq_length,
        )

        # 7 - SFT Trainer
        callbacks = []
        if self.config.early_stopping_patience:
            from transformers import EarlyStoppingCallback
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )
            logger.info(
                f"  Early stopping: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.early_stopping_threshold}"
            )

        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks if callbacks else None,
        )

        # 8 - Train
        logger.info("\nStarting training...")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(
            f"  Batch size: {self.config.per_device_train_batch_size} x "
            f"{self.config.gradient_accumulation_steps} grad_accum = "
            f"{self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps} effective"
        )
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  LoRA rank: {self.config.lora_r}")

        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
                if isinstance(resume_from_checkpoint, str)
                else None
            )

            logger.info("\n" + "=" * 70)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

            # Log training metrics
            metrics = train_result.metrics
            logger.info(f"Training loss: {metrics.get('train_loss', 'N/A')}")
            logger.info(
                f"Training runtime: {metrics.get('train_runtime', 0):.1f}s"
            )
            logger.info(
                f"Samples/second: {metrics.get('train_samples_per_second', 0):.2f}"
            )

            # Save final adapter
            final_path = Path(self.config.output_dir) / "final_adapter"
            self.trainer.save_model(str(final_path))
            self.tokenizer.save_pretrained(str(final_path))
            logger.info(f"Final adapter saved: {final_path}")

            # Save training summary
            self._save_training_summary(metrics, str(final_path), len(train_dataset))

            return str(final_path)

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            return self._handle_shutdown()
        except Exception as e:
            logger.error(f"\nTraining failed: {e}", exc_info=True)
            self._handle_shutdown()
            raise

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the output directory."""
        output_dir = Path(self.config.output_dir)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            latest = str(checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {latest}")
            return latest
        logger.info("No checkpoints found, starting fresh")
        return None

    def _save_training_summary(
        self, metrics: Dict[str, Any], final_path: str, num_train: int
    ) -> None:
        """Save a JSON summary of the training run."""
        summary = {
            "completed": True,
            "completed_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "started_at": self.started_at,
            "base_model": self.config.base_model,
            "num_epochs": self.config.num_epochs,
            "train_examples": num_train,
            "lora_rank": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "learning_rate": self.config.learning_rate,
            "effective_batch_size": (
                self.config.per_device_train_batch_size
                * self.config.gradient_accumulation_steps
            ),
            "training_metrics": metrics,
            "final_adapter_path": final_path,
        }

        summary_path = Path(self.config.output_dir) / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Training summary saved: {summary_path}")

    def _handle_shutdown(self) -> str:
        """Save checkpoint on graceful shutdown."""
        logger.info("Performing graceful shutdown...")
        if self.trainer is not None:
            checkpoint_path = Path(self.config.output_dir) / "checkpoint-interrupted"
            self.trainer.save_model(str(checkpoint_path))
            if self.tokenizer:
                self.tokenizer.save_pretrained(str(checkpoint_path))
            logger.info(f"Interrupted checkpoint saved: {checkpoint_path}")
            logger.info("Use --resume to continue training.")
            return str(checkpoint_path)
        return ""


def main():
    parser = argparse.ArgumentParser(description="QLoRA LLM Fine-Tuner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/llm_finetuner.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument("--input_jsonl", type=str, help="Override input JSONL path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--base_model", type=str, help="Override base model")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override per-device batch size")
    parser.add_argument(
        "--subset_size", type=int, help="Train on subset (for testing pipeline)"
    )
    parser.add_argument("--lora_r", type=int, help="Override LoRA rank")

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = LLMTrainingConfig.from_json(str(config_path))
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        config = LLMTrainingConfig()

    if args.input_jsonl:
        config.input_jsonl = args.input_jsonl
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.base_model:
        config.base_model = args.base_model
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.subset_size:
        config.subset_size = args.subset_size
    if args.lora_r:
        config.lora_r = args.lora_r

    finetuner = LLMFineTuner(config)
    finetuner.train(resume_from_checkpoint="auto" if args.resume else None)


if __name__ == "__main__":
    main()
