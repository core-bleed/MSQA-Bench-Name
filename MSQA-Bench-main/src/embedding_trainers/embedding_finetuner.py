import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from sentence_transformers.util import batch_to_device


class EmbeddingFineTuner:
    """Fine-tune embedding models on Q&A pairs for improved semantic search."""

    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.device = self._get_device()
        
        # Paths
        self.input_jsonl = Path(config.get("input_jsonl", "data/consolidated_qa.jsonl"))
        self.output_dir = Path(config.get("output_dir", "models/fine_tuned_embeddings"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.base_model = config.get("base_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.max_seq_length = config.get("max_seq_length", 384)
        
        # Training configuration
        self.train_batch_size = config.get("train_batch_size", 16)
        self.eval_batch_size = config.get("eval_batch_size", 16)
        self.num_epochs = config.get("num_epochs", 3)
        self.warmup_steps = config.get("warmup_steps", 500)
        self.evaluation_steps = config.get("evaluation_steps", 1000)
        self.save_steps = config.get("save_steps", 1000)
        self.learning_rate = config.get("learning_rate", 2e-5)
        self.train_split = config.get("train_split", 0.9)
        self.loss_function = config.get("loss_function", "MultipleNegativesRankingLoss")
        
        # Progress tracking
        self.progress_file = self.output_dir / config.get("progress_file", "training_progress.json")
        self.metrics_file = self.output_dir / config.get("metrics_file", "training_metrics.jsonl")
        
        self.model: Optional[SentenceTransformer] = None
        
    def _get_device(self) -> str:
        """Determine the device to use for training."""
        if torch.cuda.is_available():
            device = "cuda"
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logging.info("Using CPU for training")
        return device
    
    def setup_logging(self) -> None:
        """Configure logging for the training process."""
        log_file = Path(self.config.get("log_file", "logs/embedding_finetuning.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_jsonl_data(self) -> List[Dict]:
        """Load Q&A pairs from JSONL file."""
        if not self.input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL file not found: {self.input_jsonl}")
        
        data = []
        with self.input_jsonl.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "question" in record and "answer" in record:
                        data.append(record)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        logging.info(f"Loaded {len(data)} Q&A pairs from {self.input_jsonl}")
        return data
    
    def prepare_training_data(self, data: List[Dict]) -> Tuple[List[InputExample], List[InputExample]]:
        """
        Convert Q&A pairs into InputExample format for training.
        
        For embedding models, we create pairs where:
        - text1: question
        - text2: answer (positive passage)
        """
        examples = []
        for idx, record in enumerate(data):
            question = record.get("question", "").strip()
            answer = record.get("answer", "").strip()
            
            if not question or not answer:
                continue
            
            # Create InputExample with question and answer as positive pair
            example = InputExample(
                guid=record.get("id", str(idx)),
                texts=[question, answer],
                label=1.0  # Positive pair
            )
            examples.append(example)
        
        # Split into train and validation
        split_idx = int(len(examples) * self.train_split)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
        
        logging.info(f"Training examples: {len(train_examples)}")
        logging.info(f"Validation examples: {len(eval_examples)}")
        
        return train_examples, eval_examples
    
    def create_evaluator(self, eval_examples: List[InputExample]) -> evaluation.EmbeddingSimilarityEvaluator:
        """Create an evaluator for the model during training."""
        queries = []
        passages = []
        scores = []
        
        for example in eval_examples:
            queries.append(example.texts[0])
            passages.append(example.texts[1])
            scores.append(1.0)  # All are positive pairs
        
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            queries,
            passages,
            scores,
            name="qa_evaluation",
            batch_size=self.eval_batch_size,
            show_progress_bar=True
        )
        
        return evaluator
    
    def get_loss_function(self, model: SentenceTransformer):
        """Get the appropriate loss function for training."""
        loss_name = self.loss_function.lower()
        
        if loss_name == "multiplenegativesrankingloss":
            return losses.MultipleNegativesRankingLoss(model)
        elif loss_name == "contrastiveloss":
            return losses.ContrastiveLoss(model)
        elif loss_name == "cosinesimilarityloss":
            return losses.CosineSimilarityLoss(model)
        elif loss_name == "onlinecontrastiveloss":
            return losses.OnlineContrastiveLoss(model)
        else:
            logging.warning(f"Unknown loss function: {self.loss_function}, using MultipleNegativesRankingLoss")
            return losses.MultipleNegativesRankingLoss(model)
    
    def save_training_metrics(self, epoch: int, step: int, metrics: Dict) -> None:
        """Save training metrics to JSONL file."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch": epoch,
            "step": step,
            **metrics
        }
        
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def save_progress(self, progress_data: Dict) -> None:
        """Save training progress."""
        with self.progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def load_progress(self) -> Optional[Dict]:
        """Load training progress if exists."""
        if self.progress_file.exists():
            try:
                with self.progress_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load progress: {e}")
        return None
    
    def train(self) -> None:
        """Execute the complete training pipeline."""
        try:
            # Load data
            logging.info("Loading Q&A data...")
            data = self.load_jsonl_data()
            
            if not data:
                raise ValueError("No data loaded for training")
            
            # Prepare training data
            logging.info("Preparing training data...")
            train_examples, eval_examples = self.prepare_training_data(data)
            
            if not train_examples:
                raise ValueError("No valid training examples found")
            
            # Load or create model
            logging.info(f"Loading base model: {self.base_model}")
            self.model = SentenceTransformer(self.base_model, device=self.device)
            self.model.max_seq_length = self.max_seq_length
            
            # Create data loader
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=self.train_batch_size
            )
            
            # Get loss function
            train_loss = self.get_loss_function(self.model)
            logging.info(f"Using loss function: {type(train_loss).__name__}")
            
            # Create evaluator
            evaluator = None
            if eval_examples:
                logging.info("Creating evaluator...")
                evaluator = self.create_evaluator(eval_examples)
            
            # Calculate total steps
            steps_per_epoch = len(train_dataloader)
            total_steps = steps_per_epoch * self.num_epochs
            
            logging.info(f"Training configuration:")
            logging.info(f"  - Epochs: {self.num_epochs}")
            logging.info(f"  - Batch size: {self.train_batch_size}")
            logging.info(f"  - Steps per epoch: {steps_per_epoch}")
            logging.info(f"  - Total steps: {total_steps}")
            logging.info(f"  - Warmup steps: {self.warmup_steps}")
            logging.info(f"  - Learning rate: {self.learning_rate}")
            
            # Training arguments
            train_start = datetime.utcnow()
            
            # Train the model
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.num_epochs,
                evaluator=evaluator,
                evaluation_steps=self.evaluation_steps,
                warmup_steps=self.warmup_steps,
                output_path=str(self.output_dir),
                save_best_model=True,
                show_progress_bar=True,
                checkpoint_path=str(self.checkpoint_dir),
                checkpoint_save_steps=self.save_steps,
                checkpoint_save_total_limit=3,
            )
            
            train_end = datetime.utcnow()
            duration = (train_end - train_start).total_seconds()
            
            # Save final model
            final_model_path = self.output_dir / "final_model"
            self.model.save(str(final_model_path))
            logging.info(f"Final model saved to: {final_model_path}")
            
            # Save training summary
            summary = {
                "training_completed": True,
                "completed_at": train_end.isoformat() + "Z",
                "duration_seconds": duration,
                "base_model": self.base_model,
                "num_epochs": self.num_epochs,
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "final_model_path": str(final_model_path),
            }
            
            self.save_progress(summary)
            
            logging.info("=" * 60)
            logging.info("Training completed successfully!")
            logging.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            logging.info(f"Model saved to: {final_model_path}")
            logging.info("=" * 60)
            
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_progress({"training_completed": False, "interrupted": True})
            sys.exit(0)
        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            self.save_progress({"training_completed": False, "error": str(e)})
            raise
    
    def test_model(self, test_queries: Optional[List[str]] = None) -> None:
        """Test the fine-tuned model with sample queries."""
        if self.model is None:
            model_path = self.output_dir / "final_model"
            if not model_path.exists():
                logging.error("No trained model found")
                return
            self.model = SentenceTransformer(str(model_path))
        
        if test_queries is None:
            test_queries = [
                "What is machine learning?",
                "How does neural network training work?",
                "What are the benefits of deep learning?",
            ]
        
        logging.info("Testing model with sample queries:")
        for query in test_queries:
            embedding = self.model.encode(query)
            logging.info(f"Query: {query}")
            logging.info(f"Embedding shape: {embedding.shape}")
            logging.info(f"Embedding norm: {torch.norm(torch.tensor(embedding)):.4f}")
            logging.info("-" * 40)


def load_config(config_path: Path) -> Dict:
    """Load configuration from JSON file."""
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding models on Q&A pairs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/embedding_finetuner.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        help="Override input JSONL file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory for models"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Override base model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override training batch size"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test the existing model without training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    config = load_config(config_path)
    
    # Override with command-line arguments
    if args.input_jsonl:
        config["input_jsonl"] = args.input_jsonl
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.base_model:
        config["base_model"] = args.base_model
    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["train_batch_size"] = args.batch_size
    
    # Create fine-tuner
    finetuner = EmbeddingFineTuner(config)
    
    if args.test_only:
        logging.info("Testing existing model...")
        finetuner.test_model()
    else:
        # Train the model
        finetuner.train()
        
        # Test the trained model
        finetuner.test_model()


if __name__ == "__main__":
    main()
