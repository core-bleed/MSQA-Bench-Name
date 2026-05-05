"""
Cross-model comparison and reporting for LLM fine-tuning.

Loads evaluation results from multiple fine-tuned models, ranks them
across all metrics, and produces a unified comparison report.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from .evaluators import LLMEvaluationResult


logger = logging.getLogger(__name__)


RANKING_METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_f1",
    "token_f1",
    "exact_match",
    "faithfulness_score",
]

LOWER_IS_BETTER = {"perplexity"}


def load_eval_result(result_path: str) -> LLMEvaluationResult:
    """Load a single evaluation result from JSON."""
    with open(result_path, "r") as f:
        data = json.load(f)
    return LLMEvaluationResult(**data)


def discover_results(base_dir: str, split: str = "test") -> Dict[str, LLMEvaluationResult]:
    """
    Discover evaluation result files under base_dir.

    Expects structure: base_dir/<model_name>/eval_results_<split>.json
    """
    base = Path(base_dir)
    results = {}

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        result_file = model_dir / f"eval_results_{split}.json"
        if result_file.exists():
            try:
                result = load_eval_result(str(result_file))
                results[result.model_name] = result
                logger.info(f"Loaded results for {result.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")

    return results


def rank_models(
    results: Dict[str, LLMEvaluationResult],
    metrics: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Rank models by weighted aggregate score across metrics.

    Default weights equally across RANKING_METRICS. Perplexity is
    inverted so lower is better.

    Returns a sorted list of dicts with model_name, per-metric values,
    per-metric ranks, and the weighted aggregate rank.
    """
    metrics = metrics or RANKING_METRICS
    if weights is None:
        weights = {m: 1.0 for m in metrics}

    model_names = list(results.keys())
    if not model_names:
        return []

    # Build per-metric value table
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
    for name in model_names:
        r = results[name]
        for m in metrics:
            metric_values[m].append(getattr(r, m, 0.0))

    # Rank per metric (1 = best)
    metric_ranks: Dict[str, List[int]] = {}
    for m in metrics:
        vals = np.array(metric_values[m])
        if m in LOWER_IS_BETTER:
            order = np.argsort(vals)  # ascending: lowest value gets rank 1
        else:
            order = np.argsort(-vals)  # descending: highest value gets rank 1
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        metric_ranks[m] = ranks.tolist()

    # Weighted aggregate rank
    total_weight = sum(weights.get(m, 1.0) for m in metrics)
    aggregate_scores = []
    for idx in range(len(model_names)):
        weighted_rank_sum = sum(
            metric_ranks[m][idx] * weights.get(m, 1.0) for m in metrics
        )
        aggregate_scores.append(weighted_rank_sum / total_weight)

    # Build output
    ranking = []
    for idx, name in enumerate(model_names):
        entry = {
            "model_name": name,
            "aggregate_rank_score": aggregate_scores[idx],
            "metrics": {},
            "ranks": {},
        }
        for m in metrics:
            entry["metrics"][m] = metric_values[m][idx]
            entry["ranks"][m] = metric_ranks[m][idx]

        r = results[name]
        entry["perplexity"] = r.perplexity
        entry["avg_tokens_per_second"] = r.avg_tokens_per_second
        entry["num_samples"] = r.num_samples

        ranking.append(entry)

    ranking.sort(key=lambda x: x["aggregate_rank_score"])

    for i, entry in enumerate(ranking, 1):
        entry["overall_rank"] = i

    return ranking


def format_comparison_table(ranking: List[Dict[str, Any]]) -> str:
    """Format the ranking as a human-readable console table."""
    if not ranking:
        return "No results to compare."

    metrics = list(ranking[0]["metrics"].keys())
    col_width = 14

    header = f"{'Rank':<5} {'Model':<30}"
    for m in metrics:
        header += f" {m:<{col_width}}"
    header += f" {'Perplexity':<{col_width}} {'Tok/s':<{col_width}}"

    separator = "-" * len(header)
    lines = [separator, header, separator]

    for entry in ranking:
        row = f"{entry['overall_rank']:<5} {entry['model_name']:<30}"
        for m in metrics:
            val = entry["metrics"][m]
            rank = entry["ranks"][m]
            row += f" {val:.4f} (#{rank})"
            padding = col_width - len(f"{val:.4f} (#{rank})")
            row += " " * max(0, padding)
        row += f" {entry['perplexity']:<{col_width}.2f}"
        row += f" {entry['avg_tokens_per_second']:<{col_width}.1f}"
        lines.append(row)

    lines.append(separator)
    return "\n".join(lines)


def compare_models(
    results: Dict[str, LLMEvaluationResult],
    output_dir: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compare multiple models, print ranking, and optionally save report.

    Args:
        results: Mapping of model_name -> LLMEvaluationResult
        output_dir: Where to save comparison_report.json
        metrics: Metrics to rank on (default: RANKING_METRICS)
        weights: Per-metric weights (default: equal)

    Returns:
        Sorted ranking list.
    """
    if len(results) < 2:
        logger.warning("Need at least 2 models to compare")

    ranking = rank_models(results, metrics=metrics, weights=weights)

    # Print table
    table = format_comparison_table(ranking)
    print("\n" + "=" * 80)
    print("LLM MODEL COMPARISON")
    print("=" * 80)
    print(table)

    if ranking:
        best = ranking[0]
        print(f"\nBest model: {best['model_name']} (aggregate rank score: {best['aggregate_rank_score']:.2f})")
    print("=" * 80 + "\n")

    # Save report
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        report = {
            "ranking": ranking,
            "full_results": {name: r.to_dict() for name, r in results.items()},
        }

        report_path = out_path / "comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Comparison report saved: {report_path}")

    return ranking


def compare_from_directory(
    base_dir: str,
    split: str = "test",
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience: discover results under base_dir, compare, and report.
    """
    results = discover_results(base_dir, split=split)
    if not results:
        logger.error(f"No evaluation results found under {base_dir}")
        return []
    return compare_models(results, output_dir=output_dir or base_dir)
