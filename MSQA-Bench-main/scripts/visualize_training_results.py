#!/usr/bin/env python3
"""
Visualization script for embedding fine-tuning results.

Generates comprehensive charts comparing base model vs fine-tuned model performance.

Usage:
    python scripts/visualize_training_results.py \
        --summary models/fine_tuned_embeddings/training_summary.json \
        --output paper/figures/
    
    # Or use defaults
    python scripts/visualize_training_results.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_summary(summary_path: str) -> Dict[str, Any]:
    """Load training summary JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_metrics(data: Dict[str, Any]) -> tuple:
    """Extract base and fine-tuned metrics."""
    base_metrics = data['metrics_history'][0]['score']  # Base model
    ft_metrics = data['test_results']  # Fine-tuned model
    
    return base_metrics, ft_metrics


def plot_metric_comparison_bar(base_metrics: Dict, ft_metrics: Dict, output_dir: Path, base_model_name: str = None):
    """Bar chart comparing key metrics before/after with clear labels."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Extract metrics
    metrics_to_plot = {
        'Recall@1': ('qa_retrieval_cosine_recall@1', 'recall@1'),
        'Recall@5': ('qa_retrieval_cosine_recall@5', 'recall@5'),
        'Recall@10': ('qa_retrieval_cosine_recall@10', 'recall@10'),
        'MRR@10': ('qa_retrieval_cosine_mrr@10', 'mrr@10'),
        'NDCG@10': ('qa_retrieval_cosine_ndcg@10', 'ndcg@10'),
        'MAP@10': ('qa_retrieval_cosine_map@10', 'map@10'),
    }
    
    labels = list(metrics_to_plot.keys())
    base_values = [base_metrics.get(metrics_to_plot[k][0], 0) for k in labels]
    ft_values = [ft_metrics.get(metrics_to_plot[k][1], 0) for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    base_label = f'Base Model\n({base_model_name or "all-MiniLM-L6-v2"})'
    ft_label = 'Fine-tuned Model\n(After Training)'
    
    bars1 = ax.bar(x - width/2, base_values, width, label=base_label, 
                   color='#3498db', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
    bars2 = ax.bar(x + width/2, ft_values, width, label=ft_label,
                   color='#2ecc71', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison: Base vs Fine-tuned Embedding Model', 
                fontweight='bold', pad=25, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add metric definitions as text box
    metric_defs = (
        "Metric Definitions:\n"
        "• Recall@k: Proportion of queries where correct answer is in top-k results\n"
        "• MRR@10: Mean Reciprocal Rank (average of 1/rank of correct answer)\n"
        "• NDCG@10: Normalized Discounted Cumulative Gain (ranking quality)\n"
        "• MAP@10: Mean Average Precision (precision at relevant positions)"
    )
    ax.text(0.02, 0.98, metric_defs, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison_bar.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'metric_comparison_bar.png'}")


def plot_improvement_percentage(base_metrics: Dict, ft_metrics: Dict, output_dir: Path):
    """Histogram showing percentage improvement."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics_to_plot = {
        'Recall@1': ('qa_retrieval_cosine_recall@1', 'recall@1'),
        'Recall@5': ('qa_retrieval_cosine_recall@5', 'recall@5'),
        'Recall@10': ('qa_retrieval_cosine_recall@10', 'recall@10'),
        'MRR@10': ('qa_retrieval_cosine_mrr@10', 'mrr@10'),
        'NDCG@10': ('qa_retrieval_cosine_ndcg@10', 'ndcg@10'),
        'MAP@10': ('qa_retrieval_cosine_map@10', 'map@10'),
    }
    
    labels = list(metrics_to_plot.keys())
    improvements = []
    base_vals = []
    ft_vals = []
    
    for k in labels:
        base_key, ft_key = metrics_to_plot[k]
        base_val = base_metrics.get(base_key, 0)
        ft_val = ft_metrics.get(ft_key, 0)
        base_vals.append(base_val)
        ft_vals.append(ft_val)
        if base_val > 0:
            improvement = ((ft_val - base_val) / base_val) * 100
        else:
            improvement = 0
        improvements.append(improvement)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.barh(labels, improvements, color=colors, alpha=0.8, edgecolor='darkgreen' if improvements[0] > 0 else 'darkred', linewidth=2)
    
    # Add value labels with absolute values
    for i, (bar, val, base, ft) in enumerate(zip(bars, improvements, base_vals, ft_vals)):
        label_text = f'{val:+.1f}%'
        if abs(val) > 1:
            label_text += f'\n({base:.3f} → {ft:.3f})'
        ax.text(val, i, label_text,
               ha='left' if val > 0 else 'right', va='center', fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Improvement (%)', fontweight='bold', fontsize=12)
    ax.set_title('Performance Improvement: Fine-tuned vs Base Model', fontweight='bold', pad=25, fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Improvement')
    red_patch = mpatches.Patch(color='#e74c3c', label='Degradation')
    ax.legend(handles=[green_patch, red_patch], loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_percentage.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'improvement_percentage.png'}")


def plot_recall_at_k_comparison(base_metrics: Dict, ft_metrics: Dict, output_dir: Path, base_model_name: str = None):
    """Line chart showing Recall@k for different k values."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    k_values = [1, 5, 10]
    base_recall = [
        base_metrics.get('qa_retrieval_cosine_recall@1', 0),
        base_metrics.get('qa_retrieval_cosine_recall@5', 0),
        base_metrics.get('qa_retrieval_cosine_recall@10', 0),
    ]
    ft_recall = [
        ft_metrics.get('recall@1', 0),
        ft_metrics.get('recall@5', 0),
        ft_metrics.get('recall@10', 0),
    ]
    
    base_label = f'Base Model ({base_model_name or "all-MiniLM-L6-v2"})'
    ft_label = 'Fine-tuned Model'
    
    ax.plot(k_values, base_recall, marker='o', linewidth=3, markersize=10,
           label=base_label, color='#3498db', linestyle='--', alpha=0.8)
    ax.plot(k_values, ft_recall, marker='s', linewidth=3, markersize=10,
           label=ft_label, color='#2ecc71', linestyle='-', alpha=0.8)
    
    # Add value annotations
    for k, base, ft in zip(k_values, base_recall, ft_recall):
        ax.annotate(f'{base:.3f}', (k, base), textcoords="offset points",
                   xytext=(0,12), ha='center', fontsize=10, color='#3498db', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax.annotate(f'{ft:.3f}', (k, ft), textcoords="offset points",
                   xytext=(0,-18), ha='center', fontsize=10, color='#2ecc71', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('k (Top-k Results)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Recall@k Score', fontweight='bold', fontsize=12)
    ax.set_title('Recall@k Comparison: Base vs Fine-tuned Embedding Model', 
                fontweight='bold', pad=25, fontsize=14)
    ax.set_xticks(k_values)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add definition
    ax.text(0.02, 0.98, 
           'Recall@k: Proportion of queries where\nthe correct answer appears in top-k results',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_at_k_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'recall_at_k_comparison.png'}")


def plot_similarity_comparison(base_metrics: Dict, ft_metrics: Dict, output_dir: Path, base_model_name: str = None):
    """Similarity comparison chart showing how models compare on similarity metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Cosine similarity metrics
    ax1 = axes[0]
    
    similarity_metrics = {
        'Precision@1': ('qa_retrieval_cosine_precision@1', None),
        'Precision@5': ('qa_retrieval_cosine_precision@5', None),
        'Precision@10': ('qa_retrieval_cosine_precision@10', None),
        'Accuracy@1': ('qa_retrieval_cosine_accuracy@1', None),
        'Accuracy@5': ('qa_retrieval_cosine_accuracy@5', None),
        'Accuracy@10': ('qa_retrieval_cosine_accuracy@10', None),
    }
    
    labels = list(similarity_metrics.keys())
    base_vals = [base_metrics.get(similarity_metrics[k][0], 0) for k in labels]
    ft_vals = [ft_metrics.get(similarity_metrics[k][1], 0) if similarity_metrics[k][1] else 0 for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    base_label = f'Base Model\n({base_model_name or "all-MiniLM-L6-v2"})'
    ft_label = 'Fine-tuned Model'
    
    bars1 = ax1.bar(x - width/2, base_vals, width, label=base_label,
                    color='#3498db', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, ft_vals, width, label=ft_label,
                    color='#2ecc71', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax1.set_title('Precision & Accuracy Comparison', fontweight='bold', fontsize=13, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylim([0, max(max(base_vals + ft_vals) * 1.1, 0.1)])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right: Ranking quality metrics
    ax2 = axes[1]
    
    ranking_metrics = {
        'MRR@10': ('qa_retrieval_cosine_mrr@10', 'mrr@10'),
        'NDCG@10': ('qa_retrieval_cosine_ndcg@10', 'ndcg@10'),
        'MAP@10': ('qa_retrieval_cosine_map@10', 'map@10'),
    }
    
    labels2 = list(ranking_metrics.keys())
    base_vals2 = [base_metrics.get(ranking_metrics[k][0], 0) for k in labels2]
    ft_vals2 = [ft_metrics.get(ranking_metrics[k][1], 0) for k in labels2]
    
    x2 = np.arange(len(labels2))
    
    bars3 = ax2.bar(x2 - width/2, base_vals2, width, label=base_label,
                    color='#3498db', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
    bars4 = ax2.bar(x2 + width/2, ft_vals2, width, label=ft_label,
                    color='#2ecc71', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax2.set_title('Ranking Quality Metrics', fontweight='bold', fontsize=13, pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add metric definitions
    metric_info = (
        "Metric Definitions:\n"
        "• MRR@10: Mean Reciprocal Rank - average of 1/rank of correct answer\n"
        "• NDCG@10: Normalized Discounted Cumulative Gain - ranking quality\n"
        "• MAP@10: Mean Average Precision - precision at relevant positions"
    )
    ax2.text(0.02, 0.98, metric_info, transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Similarity & Ranking Comparison: Before vs After Fine-tuning', 
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'similarity_comparison.png'}")


def plot_radar_chart(base_metrics: Dict, ft_metrics: Dict, output_dir: Path, base_model_name: str = None):
    """Radar/spider chart comparing multiple metrics."""
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    categories = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10', 'NDCG@10', 'MAP@10']
    base_values = [
        base_metrics.get('qa_retrieval_cosine_recall@1', 0),
        base_metrics.get('qa_retrieval_cosine_recall@5', 0),
        base_metrics.get('qa_retrieval_cosine_recall@10', 0),
        base_metrics.get('qa_retrieval_cosine_mrr@10', 0),
        base_metrics.get('qa_retrieval_cosine_ndcg@10', 0),
        base_metrics.get('qa_retrieval_cosine_map@10', 0),
    ]
    ft_values = [
        ft_metrics.get('recall@1', 0),
        ft_metrics.get('recall@5', 0),
        ft_metrics.get('recall@10', 0),
        ft_metrics.get('mrr@10', 0),
        ft_metrics.get('ndcg@10', 0),
        ft_metrics.get('map@10', 0),
    ]
    
    # Close the plot
    categories += [categories[0]]
    base_values += [base_values[0]]
    ft_values += [ft_values[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True).tolist()
    
    base_label = f'Base Model ({base_model_name or "all-MiniLM-L6-v2"})'
    ft_label = 'Fine-tuned Model'
    
    ax.plot(angles, base_values, 'o-', linewidth=3, label=base_label, color='#3498db', markersize=8)
    ax.fill(angles, base_values, alpha=0.2, color='#3498db')
    
    ax.plot(angles, ft_values, 's-', linewidth=3, label=ft_label, color='#2ecc71', markersize=8)
    ax.fill(angles, ft_values, alpha=0.2, color='#2ecc71')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Comprehensive Performance Comparison\nBase vs Fine-tuned Embedding Model', 
                fontweight='bold', pad=40, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'radar_chart.png'}")


def plot_histogram_distribution(base_metrics: Dict, ft_metrics: Dict, output_dir: Path, base_model_name: str = None):
    """Histogram showing distribution of metric values with definitions."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()
    
    metrics_config = [
        ('Recall@1', 'qa_retrieval_cosine_recall@1', 'recall@1', 
         'Proportion where correct answer\nis ranked #1'),
        ('Recall@5', 'qa_retrieval_cosine_recall@5', 'recall@5',
         'Proportion where correct answer\nis in top-5 results'),
        ('Recall@10', 'qa_retrieval_cosine_recall@10', 'recall@10',
         'Proportion where correct answer\nis in top-10 results'),
        ('MRR@10', 'qa_retrieval_cosine_mrr@10', 'mrr@10',
         'Mean Reciprocal Rank:\naverage of 1/rank of correct answer'),
        ('NDCG@10', 'qa_retrieval_cosine_ndcg@10', 'ndcg@10',
         'Normalized Discounted\nCumulative Gain (ranking quality)'),
        ('MAP@10', 'qa_retrieval_cosine_map@10', 'map@10',
         'Mean Average Precision:\nprecision at relevant positions'),
    ]
    
    for idx, (title, base_key, ft_key, definition) in enumerate(metrics_config):
        ax = axes[idx]
        
        base_val = base_metrics.get(base_key, 0)
        ft_val = ft_metrics.get(ft_key, 0)
        
        bars = ax.bar(['Base Model\n(' + (base_model_name or 'all-MiniLM-L6-v2')[:15] + '...)', 
                       'Fine-tuned Model\n(After Training)'], 
                     [base_val, ft_val],
                     color=['#3498db', '#2ecc71'], alpha=0.8, width=0.6,
                     edgecolor=['darkblue', 'darkgreen'], linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax.set_ylabel('Score', fontweight='bold', fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add definition as text
        ax.text(0.5, 0.95, definition, transform=ax.transAxes,
               ha='center', va='top', fontsize=8, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Performance Metrics: Base Model vs Fine-tuned Model', 
                fontweight='bold', fontsize=17, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'histogram_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'histogram_distribution.png'}")


def plot_heatmap_comparison(base_metrics: Dict, ft_metrics: Dict, output_dir: Path, base_model_name: str = None):
    """Heatmap showing metric values side-by-side."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10', 'NDCG@10', 'MAP@10']
    base_keys = ['qa_retrieval_cosine_recall@1', 'qa_retrieval_cosine_recall@5',
                'qa_retrieval_cosine_recall@10', 'qa_retrieval_cosine_mrr@10',
                'qa_retrieval_cosine_ndcg@10', 'qa_retrieval_cosine_map@10']
    ft_keys = ['recall@1', 'recall@5', 'recall@10', 'mrr@10', 'ndcg@10', 'map@10']
    
    base_values = [base_metrics.get(k, 0) for k in base_keys]
    ft_values = [ft_metrics.get(k, 0) for k in ft_keys]
    
    data = np.array([base_values, ft_values])
    
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
    
    base_label = f'Base Model\n({base_model_name or "all-MiniLM-L6-v2"})'
    ax.set_yticklabels([base_label, 'Fine-tuned Model\n(After Training)'], fontsize=11)
    
    # Add text annotations
    for i in range(2):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    ax.set_title('Performance Heatmap: Base vs Fine-tuned Embedding Model', 
                fontweight='bold', pad=25, fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=25, fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'heatmap_comparison.png'}")


def plot_training_summary(data: Dict, output_dir: Path):
    """Summary statistics visualization - improved design."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training duration (improved)
    ax1 = fig.add_subplot(gs[0, 0])
    started = data['started_at']
    completed = data['completed_at']
    duration_seconds = (np.datetime64(completed) - np.datetime64(started)) / np.timedelta64(1, 's')
    duration_hours = duration_seconds / 3600
    duration_mins = (duration_seconds % 3600) / 60
    
    bars1 = ax1.bar(['Training\nDuration'], [duration_hours], color='#9b59b6', alpha=0.8, 
                   edgecolor='darkviolet', linewidth=2)
    ax1.text(0, duration_hours, f'{duration_hours:.1f}h\n({duration_mins:.0f}m)', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Hours', fontweight='bold', fontsize=11)
    ax1.set_title('Training Duration', fontweight='bold', fontsize=12, pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, duration_hours * 1.2])
    
    # 2. Dataset size (improved)
    ax2 = fig.add_subplot(gs[0, 1])
    train_examples = data['train_examples']
    train_millions = train_examples / 1e6
    
    bars2 = ax2.bar(['Training\nExamples'], [train_millions], color='#e67e22', alpha=0.8,
                   edgecolor='darkorange', linewidth=2)
    ax2.text(0, train_millions, f'{train_millions:.2f}M\n({train_examples:,})', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Millions', fontweight='bold', fontsize=11)
    ax2.set_title('Dataset Size', fontweight='bold', fontsize=12, pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, train_millions * 1.2])
    
    # 3. Epochs (improved)
    ax3 = fig.add_subplot(gs[0, 2])
    epochs = data['num_epochs']
    
    bars3 = ax3.bar(['Training\nEpochs'], [epochs], color='#16a085', alpha=0.8,
                   edgecolor='darkcyan', linewidth=2)
    ax3.text(0, epochs, str(epochs), ha='center', va='bottom', 
            fontweight='bold', fontsize=14)
    ax3.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax3.set_title('Number of Epochs', fontweight='bold', fontsize=12, pad=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim([0, max(epochs * 1.5, 2)])
    
    # 4. Test set size (improved)
    ax4 = fig.add_subplot(gs[1, 0])
    test_size = data['test_results']['num_queries']
    
    bars4 = ax4.bar(['Test\nQueries'], [test_size], color='#c0392b', alpha=0.8,
                   edgecolor='darkred', linewidth=2)
    ax4.text(0, test_size, f'{test_size:,}', ha='center', va='bottom',
            fontweight='bold', fontsize=11)
    ax4.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax4.set_title('Evaluation Set Size', fontweight='bold', fontsize=12, pad=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim([0, test_size * 1.2])
    
    # 5. Base model info
    ax5 = fig.add_subplot(gs[1, 1])
    base_model = data.get('base_model', 'all-MiniLM-L6-v2')
    model_name_short = base_model.split('/')[-1] if '/' in base_model else base_model
    
    ax5.axis('off')
    ax5.text(0.5, 0.7, 'Base Model', ha='center', va='center', 
            fontsize=14, fontweight='bold', transform=ax5.transAxes)
    ax5.text(0.5, 0.4, model_name_short, ha='center', va='center',
            fontsize=12, style='italic', transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax5.set_title('Model Configuration', fontweight='bold', fontsize=12, pad=10)
    
    # 6. Training status
    ax6 = fig.add_subplot(gs[1, 2])
    completed_status = data.get('completed', False)
    status_text = 'Completed ✓' if completed_status else 'In Progress'
    status_color = '#2ecc71' if completed_status else '#f39c12'
    
    ax6.axis('off')
    ax6.text(0.5, 0.5, status_text, ha='center', va='center',
            fontsize=16, fontweight='bold', color=status_color,
            transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=status_color, 
                     linewidth=3, alpha=0.8))
    ax6.set_title('Training Status', fontweight='bold', fontsize=12, pad=10)
    
    plt.suptitle('Training Summary Statistics', fontweight='bold', fontsize=18, y=0.98)
    plt.savefig(output_dir / 'training_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: {output_dir / 'training_summary.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization charts for embedding fine-tuning results"
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="models/fine_tuned_embeddings/training_summary.json",
        help="Path to training summary JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures",
        help="Output directory for figures (default: paper/figures)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading training summary from: {args.summary}")
    data = load_summary(args.summary)
    
    # Extract metrics
    base_metrics, ft_metrics = extract_metrics(data)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print("=" * 60)
    
    # Get base model name
    base_model_name = data.get('base_model', 'all-MiniLM-L6-v2')
    
    # Generate all charts
    plot_metric_comparison_bar(base_metrics, ft_metrics, output_dir, base_model_name)
    plot_improvement_percentage(base_metrics, ft_metrics, output_dir)
    plot_recall_at_k_comparison(base_metrics, ft_metrics, output_dir, base_model_name)
    plot_radar_chart(base_metrics, ft_metrics, output_dir, base_model_name)
    plot_histogram_distribution(base_metrics, ft_metrics, output_dir, base_model_name)
    plot_heatmap_comparison(base_metrics, ft_metrics, output_dir, base_model_name)
    plot_similarity_comparison(base_metrics, ft_metrics, output_dir, base_model_name)
    plot_training_summary(data, output_dir)
    
    print("=" * 60)
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\nGenerated charts:")
    print("  1. metric_comparison_bar.png - Side-by-side bar chart with metric definitions")
    print("  2. improvement_percentage.png - Percentage improvement histogram")
    print("  3. recall_at_k_comparison.png - Recall@k line chart with definitions")
    print("  4. radar_chart.png - Radar/spider chart")
    print("  5. histogram_distribution.png - Individual metric histograms with definitions")
    print("  6. heatmap_comparison.png - Heatmap visualization")
    print("  7. similarity_comparison.png - Similarity & ranking comparison (NEW)")
    print("  8. training_summary.png - Improved training statistics")


if __name__ == "__main__":
    main()
