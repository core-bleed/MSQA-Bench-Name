#!/usr/bin/env python3
"""
Parse LLM training logs and generate charts/stats.

Usage:
    python scripts/visualize_llm_training.py llm_training_20260313_023904.log
    python scripts/visualize_llm_training.py llm_training_20260313_023904.log --output-dir results/
"""

import argparse
import ast
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


@dataclass
class TrainingMetrics:
    """Metrics for a single model training run."""
    model_name: str
    steps: List[int] = field(default_factory=list)
    epochs: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    mean_token_accuracy: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    num_tokens: List[float] = field(default_factory=list)
    step_from_progress: List[int] = field(default_factory=list)
    total_steps: Optional[int] = None


def parse_log(log_path: str) -> Dict[str, TrainingMetrics]:
    """Parse training log and extract metrics per model."""
    models: Dict[str, TrainingMetrics] = {}
    current_model: Optional[str] = None
    model_start_line = 0
    step_counter = 0  # Global step within current model
    total_steps_seen = None

    # Pattern for metrics dict: {'loss': 1.36, 'grad_norm': 0.41, ...}
    metrics_pattern = re.compile(r"^\s*\{['\"]loss['\"]\s*:.*\}$")
    # Pattern for progress bar: " 50%|████▉ | 15660/31619 [80:38:52<65:48:57, 14.04s/it]"
    progress_pattern = re.compile(r"\s*(\d+)%\s*\|[^|]*\|\s*(\d+)/(\d+)\s+\[")

    with open(log_path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()

            # Detect model boundaries
            if "Training LLM:" in line:
                match = re.search(r"Training LLM:\s*(\S+)", line)
                if match:
                    current_model = match.group(1)
                    if current_model not in models:
                        models[current_model] = TrainingMetrics(model_name=current_model)
                    model_start_line = line_no
                    step_counter = 0

            # Parse metrics dict
            if current_model and metrics_pattern.match(stripped):
                try:
                    data = ast.literal_eval(stripped)
                    metrics = models[current_model]
                    step_counter += 1
                    metrics.steps.append(step_counter * 500)  # logging every 500 steps typically
                    metrics.loss.append(float(data.get("loss", 0)))
                    metrics.learning_rate.append(float(data.get("learning_rate", 0)))
                    metrics.grad_norm.append(float(data.get("grad_norm", 0)))
                    metrics.mean_token_accuracy.append(float(data.get("mean_token_accuracy", 0)))
                    metrics.epochs.append(float(data.get("epoch", 0)))
                    metrics.entropy.append(float(data.get("entropy", 0)))
                    metrics.num_tokens.append(float(data.get("num_tokens", 0)))
                except (ValueError, SyntaxError):
                    pass

            # Parse progress bar for step info
            if current_model and "|" in stripped and re.search(r"\d+/\d+", stripped):
                m = progress_pattern.search(stripped)
                if m:
                    current_step = int(m.group(2))
                    total_steps = int(m.group(3))
                    models[current_model].step_from_progress.append(current_step)
                    models[current_model].total_steps = total_steps

    # Refine step numbers from progress bars (loss logged every ~500 steps typically)
    for name, m in models.items():
        if not m.loss:
            continue
        n = len(m.loss)
        if m.step_from_progress and m.total_steps:
            last_progress = m.step_from_progress[-1]
            steps_per_log = max(1, last_progress // n)
            m.steps = [min((i + 1) * steps_per_log, last_progress) for i in range(n)]
        else:
            m.steps = [(i + 1) * 500 for i in range(n)]

    return models


def plot_training_summary(models: Dict[str, TrainingMetrics], output_dir: Path) -> None:
    """Generate multi-panel training charts."""
    if not models:
        print("No model data to plot.")
        return

    n_models = len(models)
    colors = plt.cm.tab10.colors[:n_models]

    # 1. Loss curves (all models)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, m), color in zip(models.items(), colors):
        if m.loss and m.steps:
            ax.plot(m.steps, m.loss, label=name, color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss by Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'training_loss.png'}")

    # 2. Loss vs Epoch (normalized)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, m), color in zip(models.items(), colors):
        if m.loss and m.epochs:
            ax.plot(m.epochs, m.loss, label=name, color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_vs_epoch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'training_loss_vs_epoch.png'}")

    # 3. Mean Token Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, m), color in zip(models.items(), colors):
        if m.mean_token_accuracy and m.steps:
            ax.plot(m.steps, m.mean_token_accuracy, label=name, color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Token Accuracy")
    ax.set_title("Token Accuracy During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "token_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'token_accuracy.png'}")

    # 4. Learning Rate
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, m), color in zip(models.items(), colors):
        if m.learning_rate and m.steps:
            ax.plot(m.steps, m.learning_rate, label=name, color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'learning_rate.png'}")

    # 5. Gradient Norm
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, m), color in zip(models.items(), colors):
        if m.grad_norm and m.steps:
            ax.plot(m.steps, m.grad_norm, label=name, color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grad_norm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'grad_norm.png'}")

    # 6. Summary dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (name, m), color in zip(models.items(), colors):
        if m.loss and m.steps:
            axes[0, 0].plot(m.steps, m.loss, label=name, color=color)
            axes[0, 1].plot(m.steps, m.mean_token_accuracy, label=name, color=color)
            axes[1, 0].plot(m.steps, m.learning_rate, label=name, color=color)
            axes[1, 1].plot(m.steps, m.grad_norm, label=name, color=color)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_title("Token Accuracy")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_title("Gradient Norm")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Grad Norm")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("LLM Training Dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "training_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'training_dashboard.png'}")


def print_stats(models: Dict[str, TrainingMetrics]) -> None:
    """Print summary statistics per model."""
    print("\n" + "=" * 70)
    print("TRAINING STATISTICS")
    print("=" * 70)
    for name, m in models.items():
        n = len(m.loss)
        if n == 0:
            print(f"\n{name}: No metrics extracted")
            continue
        last_epoch = m.epochs[-1] if m.epochs else 0
        total_steps = m.total_steps or (m.steps[-1] if m.steps else 0)
        progress_pct = (m.steps[-1] / total_steps * 100) if total_steps else 0

        print(f"\n{name}:")
        print(f"  Metric points:     {n}")
        print(f"  Max epoch reached: {last_epoch:.2f}")
        print(f"  Total steps:       {total_steps}")
        print(f"  Progress:          {m.steps[-1] if m.steps else 0} / {total_steps} ({progress_pct:.1f}%)")
        print(f"  Loss (first):      {m.loss[0]:.4f}")
        print(f"  Loss (last):       {m.loss[-1]:.4f}")
        print(f"  Loss (min):        {min(m.loss):.4f}")
        print(f"  Token accuracy:    {m.mean_token_accuracy[-1]:.2%} (last)")
        print(f"  Learning rate:     {m.learning_rate[-1]:.2e} (last)")


def main():
    parser = argparse.ArgumentParser(description="Visualize LLM training logs")
    parser.add_argument("log_file", type=str, help="Path to training log file")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for charts (default: same dir as log)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print statistics, do not generate charts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export parsed metrics to JSON",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent / "training_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {log_path}")
    models = parse_log(str(log_path))

    if not models:
        print("No training metrics found in log.")
        return 1

    print(f"\nExtracted metrics for {len(models)} model(s): {list(models.keys())}")

    print_stats(models)

    if args.json:
        data = {}
        for name, m in models.items():
            data[name] = {
                "model_name": name,
                "n_points": len(m.loss),
                "steps": m.steps[:500],  # Limit for JSON size
                "epochs": m.epochs[:500],
                "loss": m.loss[:500],
                "learning_rate": m.learning_rate[:500],
                "mean_token_accuracy": m.mean_token_accuracy[:500],
                "total_steps": m.total_steps,
                "summary": {
                    "first_loss": m.loss[0] if m.loss else None,
                    "last_loss": m.loss[-1] if m.loss else None,
                    "min_loss": min(m.loss) if m.loss else None,
                    "last_epoch": m.epochs[-1] if m.epochs else None,
                },
            }
        json_path = output_dir / "training_metrics.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nExported JSON: {json_path}")

    if not args.no_plot:
        print("\nGenerating charts...")
        try:
            plot_training_summary(models, output_dir)
            print(f"\nCharts saved to: {output_dir}")
        except ImportError as e:
            print(f"\nMatplotlib not available: {e}")
            print("Install with: pip install matplotlib")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
