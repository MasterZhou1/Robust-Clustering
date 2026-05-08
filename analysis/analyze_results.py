#!/usr/bin/env python3
"""
Analysis and visualization for clustering benchmark results.

Generates:
- Comprehensive markdown report with tables
- Performance comparison plots
- Scalability analysis
- Statistical analysis with mean ± std
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Try to import plotting libraries
HAS_MATPLOTLIB = False
HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for SLURM
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import rcParams
    HAS_MATPLOTLIB = True
    
    # Set up publication-quality defaults
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 13
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 9
    rcParams['figure.titlesize'] = 14
    rcParams['axes.linewidth'] = 1.2
    rcParams['lines.linewidth'] = 2
    rcParams['lines.markersize'] = 8
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    
except (ImportError, AttributeError, ValueError) as e:
    print(f"Warning: matplotlib not available ({type(e).__name__}). Plotting disabled.")

try:
    import pandas as pd
    HAS_PANDAS = True
except (ImportError, ValueError) as e:
    print(f"Warning: pandas not available ({type(e).__name__}). Using basic tables.")


# Colorblind-friendly palette
ALGO_COLORS = {
    "TIKMeans": "#0077BB",      # Blue
    "IKMeans": "#33BBEE",       # Cyan
    "RobustKmeans++": "#009988", # Teal
    "NKMeans": "#EE7733",       # Orange
    "OKMeans": "#CC3311",       # Red
    "OKMeans2": "#EE3377",      # Magenta
    "OKMeansFAISS": "#AA4499",  # Purple
    "OKMeans2FAISS": "#BBBBBB", # Gray
    "KMeans++": "#44BB99",      # Mint
    "KMeans--": "#882255",      # Wine
}

ALGO_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']


def _save_figure(fig, output_path: str):
    """Save figure in both PDF and PNG formats."""
    # Save PDF (vector, publication quality)
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")
    
    # Also save PNG for quick viewing
    fig.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    
    plt.close(fig)


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import RESULTS_DIR, ANALYSIS_DIR


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def aggregate_results(results: List[Dict], exp_config: Optional[Dict] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Aggregate results by dataset and algorithm.
    
    Returns nested dict: {dataset: {algorithm: {metric: value}}}
    
    If exp_config is provided, includes all dataset-algorithm pairs from the config,
    using "NA" for those that were skipped/timed out/errored.
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Track all dataset-algorithm pairs and their statuses
    status_tracker = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        ds = r["dataset"]
        algo = r["algorithm"]
        status = r.get("status", "unknown")
        status_tracker[ds][algo].append(status)
        
        if status == "completed":
            aggregated[ds][algo]["cost"].append(r["cost"])
            aggregated[ds][algo]["recall"].append(r["recall"])
            aggregated[ds][algo]["precision"].append(r.get("precision", 0))
            aggregated[ds][algo]["f1"].append(r.get("f1", 0))
            aggregated[ds][algo]["ari"].append(
                np.nan if r.get("ari") is None else float(r["ari"])
            )
            aggregated[ds][algo]["nmi"].append(
                np.nan if r.get("nmi") is None else float(r["nmi"])
            )
            aggregated[ds][algo]["time_s"].append(r["time_s"])
            # Track individual runs for best cost computation
            aggregated[ds][algo]["runs"].append({
                "cost": r["cost"],
                "recall": r["recall"],
                "time_s": r["time_s"]
            })
    
    # Compute statistics
    stats = {}
    
    # Determine the datasets and algorithms to include
    if exp_config:
        datasets = exp_config.get("datasets", [])
        algorithms = exp_config.get("algorithms", [])
    else:
        datasets = list(aggregated.keys())
        algorithms = set()
        for ds in aggregated:
            algorithms.update(aggregated[ds].keys())
        algorithms = list(algorithms)
    
    for ds in datasets:
        stats[ds] = {}
        for algo in algorithms:
            # Check if we have any completed results
            if ds in aggregated and algo in aggregated[ds] and aggregated[ds][algo]["cost"]:
                metrics = aggregated[ds][algo]
                
                # Find the run with the best (lowest) cost
                runs = metrics["runs"]
                best_run = min(runs, key=lambda x: x["cost"])
                
                stats[ds][algo] = {
                    "cost_mean": np.mean(metrics["cost"]),
                    "cost_std": np.std(metrics["cost"]),
                    "recall_mean": np.mean(metrics["recall"]),
                    "recall_std": np.std(metrics["recall"]),
                    "precision_mean": np.mean(metrics["precision"]),
                    "precision_std": np.std(metrics["precision"]),
                    "f1_mean": np.mean(metrics["f1"]),
                    "f1_std": np.std(metrics["f1"]),
                    "ari_mean": np.nanmean(metrics["ari"]),
                    "ari_std": np.nanstd(metrics["ari"]),
                    "nmi_mean": np.nanmean(metrics["nmi"]),
                    "nmi_std": np.nanstd(metrics["nmi"]),
                    "time_mean": np.mean(metrics["time_s"]),
                    "time_std": np.std(metrics["time_s"]),
                    "n_runs": len(metrics["cost"]),
                    "status": "completed",
                    # Best cost run metrics
                    "best_cost": best_run["cost"],
                    "best_cost_recall": best_run["recall"],
                    "best_cost_time": best_run["time_s"],
                }
            else:
                # Determine the status for this dataset-algorithm pair
                statuses = status_tracker.get(ds, {}).get(algo, [])
                if "timeout" in statuses:
                    status = "timeout"
                elif "skipped_timeout" in statuses:
                    status = "skipped"
                elif "error" in statuses:
                    status = "error"
                elif statuses:
                    status = statuses[0]
                else:
                    status = "not_run"
                
                stats[ds][algo] = {
                    "cost_mean": np.nan,
                    "cost_std": np.nan,
                    "recall_mean": np.nan,
                    "recall_std": np.nan,
                    "precision_mean": np.nan,
                    "precision_std": np.nan,
                    "f1_mean": np.nan,
                    "f1_std": np.nan,
                    "ari_mean": np.nan,
                    "ari_std": np.nan,
                    "nmi_mean": np.nan,
                    "nmi_std": np.nan,
                    "time_mean": np.nan,
                    "time_std": np.nan,
                    "n_runs": 0,
                    "status": status,
                    # Best cost run metrics (NA for incomplete)
                    "best_cost": np.nan,
                    "best_cost_recall": np.nan,
                    "best_cost_time": np.nan,
                }
    
    return stats


def format_value(value: float, precision: int = 2, scientific: bool = False) -> str:
    """Format a numeric value for display."""
    if value is None or np.isnan(value):
        return "NA"
    if scientific and value > 10000:
        return f"{value:.2e}"
    return f"{value:.{precision}f}"


def format_with_std(mean: float, std: float, precision: int = 2) -> str:
    """Format mean ± std."""
    if mean is None or np.isnan(mean):
        return "NA"
    if mean > 10000:
        return f"{mean:.2e} ± {std:.2e}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def get_status_indicator(s: Dict) -> str:
    """Get a status indicator for display."""
    status = s.get("status", "completed")
    if status == "completed":
        return ""
    elif status == "timeout":
        return " [TIMEOUT]"
    elif status == "skipped":
        return " [SKIPPED]"
    elif status == "error":
        return " [ERROR]"
    else:
        return f" [{status.upper()}]"


def generate_table_text(stats: Dict, include_std: bool = True) -> str:
    """Generate a text-based comparison table similar to the paper."""
    
    lines = []
    lines.append("=" * 120)
    lines.append("Comparison results of k-means with outliers approximation algorithms")
    lines.append("Results show mean ± std over multiple runs. NA indicates timeout/skipped/error.")
    lines.append("=" * 120)
    lines.append("")
    
    header = f"{'Dataset':<15} {'Method':<18} {'Cost':>20} {'Recall':>14} {'Time(s)':>14} {'Status':>12}"
    lines.append(header)
    lines.append("-" * 120)
    
    # Sort datasets for consistent ordering
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    datasets = [d for d in dataset_order if d in stats]
    
    # Sort algorithms for consistent ordering
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    for ds in datasets:
        first_row = True
        algos = [a for a in algo_order if a in stats[ds]]
        
        for algo in algos:
            s = stats[ds][algo]
            status = s.get("status", "completed")
            n_runs = s.get("n_runs", 0)
            
            if include_std:
                cost_str = format_with_std(s["cost_mean"], s["cost_std"], 2)
                recall_str = format_with_std(s["recall_mean"], s["recall_std"], 4)
                time_str = format_with_std(s["time_mean"], s["time_std"], 2)
            else:
                cost_str = format_value(s["cost_mean"], precision=2)
                recall_str = format_value(s["recall_mean"], precision=4)
                time_str = format_value(s["time_mean"], precision=2)
            
            # Status column
            if status == "completed":
                status_str = f"n={n_runs}"
            else:
                status_str = status.upper()
            
            ds_name = ds if first_row else ""
            lines.append(f"{ds_name:<15} {algo:<18} {cost_str:>20} {recall_str:>14} {time_str:>14} {status_str:>12}")
            first_row = False
        
        lines.append("-" * 120)
    
    return "\n".join(lines)


def format_csv_value(value):
    """Format value for CSV, replacing NaN with 'NA'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return value


def generate_csv(stats: Dict, output_path: str):
    """Generate CSV file with results."""
    
    # Define consistent ordering
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    rows = []
    for ds in dataset_order:
        if ds not in stats:
            continue
        for algo in algo_order:
            if algo not in stats[ds]:
                continue
            s = stats[ds][algo]
            rows.append({
                "dataset": ds,
                "algorithm": algo,
                "cost_mean": format_csv_value(s["cost_mean"]),
                "cost_std": format_csv_value(s["cost_std"]),
                "recall_mean": format_csv_value(s["recall_mean"]),
                "recall_std": format_csv_value(s["recall_std"]),
                "precision_mean": format_csv_value(s["precision_mean"]),
                "precision_std": format_csv_value(s["precision_std"]),
                "f1_mean": format_csv_value(s["f1_mean"]),
                "f1_std": format_csv_value(s["f1_std"]),
                "ari_mean": format_csv_value(s.get("ari_mean")),
                "ari_std": format_csv_value(s.get("ari_std")),
                "nmi_mean": format_csv_value(s.get("nmi_mean")),
                "nmi_std": format_csv_value(s.get("nmi_std")),
                "time_mean": format_csv_value(s["time_mean"]),
                "time_std": format_csv_value(s["time_std"]),
                "best_cost": format_csv_value(s.get("best_cost")),
                "best_cost_recall": format_csv_value(s.get("best_cost_recall")),
                "best_cost_time": format_csv_value(s.get("best_cost_time")),
                "n_runs": s["n_runs"],
                "status": s.get("status", "completed"),
            })
    
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    else:
        # Manual CSV writing
        with open(output_path, "w") as f:
            if rows:
                header = ",".join(rows[0].keys())
                f.write(header + "\n")
                for row in rows:
                    values = ",".join(str(v) for v in row.values())
                    f.write(values + "\n")


def plot_cost_comparison(stats: Dict, output_path: str):
    """Create bar chart comparing costs across datasets and algorithms."""
    if not HAS_MATPLOTLIB:
        print("Skipping cost comparison plot (matplotlib not available)")
        return
    
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    datasets = [d for d in dataset_order if d in stats]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ds in enumerate(datasets):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        algos = [a for a in algo_order if a in stats[ds] and not np.isnan(stats[ds][a]["cost_mean"])]
        if not algos:
            ax.set_title(f"{ds} (no data)", fontweight='bold')
            ax.text(0.5, 0.5, "No completed experiments", ha='center', va='center', transform=ax.transAxes)
            continue
            
        costs = np.array([stats[ds][a]["cost_mean"] for a in algos])
        stds = np.array([stats[ds][a]["cost_std"] for a in algos])
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]
        
        x = np.arange(len(algos))
        bars = ax.bar(x, costs, yerr=stds, capsize=3, color=colors, 
                     edgecolor='white', linewidth=0.5)
        
        # Auto-adjust y-axis: start from reasonable baseline (not 0)
        y_min = max(0, min(costs) - max(stds) * 1.5)
        y_max = max(costs) + max(stds) * 1.5
        # Start from ~70% of min to show differences better
        y_min = max(0, min(costs) * 0.7)
        ax.set_ylim(y_min, y_max)
        
        ax.set_title(ds, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_ylabel('Cost')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Highlight best
        if len(costs) > 0:
            min_idx = np.argmin(costs)
            bars[min_idx].set_edgecolor('black')
            bars[min_idx].set_linewidth(2)
    
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Clustering Cost Comparison', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_recall_comparison(stats: Dict, output_path: str):
    """Create bar chart comparing recall across datasets and algorithms."""
    if not HAS_MATPLOTLIB:
        print("Skipping recall comparison plot (matplotlib not available)")
        return
    
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    datasets = [d for d in dataset_order if d in stats]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ds in enumerate(datasets):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        algos = [a for a in algo_order if a in stats[ds] and not np.isnan(stats[ds][a]["recall_mean"])]
        if not algos:
            ax.set_title(f"{ds} (no data)", fontweight='bold')
            ax.text(0.5, 0.5, "No completed experiments", ha='center', va='center', transform=ax.transAxes)
            continue
            
        recalls = np.array([stats[ds][a]["recall_mean"] for a in algos])
        stds = np.array([stats[ds][a]["recall_std"] for a in algos])
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]
        
        x = np.arange(len(algos))
        bars = ax.bar(x, recalls, yerr=stds, capsize=3, color=colors,
                     edgecolor='white', linewidth=0.5)
        
        # Auto-adjust y-axis to zoom into data range
        y_min = max(0, min(recalls) - max(stds) * 1.5 - 0.05)
        y_max = min(1.05, max(recalls) + max(stds) * 1.5 + 0.05)
        # For datasets with high recall, start from a reasonable baseline
        if min(recalls) > 0.5:
            y_min = max(0, min(recalls) * 0.85)
        ax.set_ylim(y_min, y_max)
        
        ax.set_title(ds, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_ylabel('Recall')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Highlight best
        if len(recalls) > 0:
            max_idx = np.argmax(recalls)
            bars[max_idx].set_edgecolor('black')
            bars[max_idx].set_linewidth(2)
    
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Outlier Detection Recall Comparison', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def _plot_metric_comparison(stats: Dict, metric: str, ylabel: str, title: str, output_path: str):
    """Bar chart comparing one metric across datasets and algorithms."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping {metric} comparison plot (matplotlib not available)")
        return

    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans",
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]

    datasets = [d for d in dataset_order if d in stats]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        algos = [a for a in algo_order
                 if a in stats[ds] and not np.isnan(stats[ds][a].get(f"{metric}_mean", np.nan))]
        if not algos:
            ax.set_title(f"{ds} (no data)", fontweight='bold')
            ax.text(0.5, 0.5, "No completed experiments", ha='center', va='center', transform=ax.transAxes)
            continue

        means = np.array([stats[ds][a][f"{metric}_mean"] for a in algos])
        stds  = np.array([stats[ds][a][f"{metric}_std"]  for a in algos])
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]

        x = np.arange(len(algos))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                      edgecolor='white', linewidth=0.5)

        if len(means) > 0 and not np.all(np.isnan(means)):
            y_min = max(0, np.nanmin(means) * 0.85)
            y_max = np.nanmax(means) + np.nanmax(stds) * 1.5 + 0.01
            ax.set_ylim(y_min, y_max)
            max_idx = int(np.nanargmax(means))
            bars[max_idx].set_edgecolor('black')
            bars[max_idx].set_linewidth(2)

        ax.set_title(ds, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_ari_comparison(stats: Dict, output_path: str):
    """Bar chart comparing ARI across datasets and algorithms."""
    _plot_metric_comparison(stats, "ari", "ARI", "Clustering Quality — ARI Comparison", output_path)


def plot_nmi_comparison(stats: Dict, output_path: str):
    """Bar chart comparing NMI across datasets and algorithms."""
    _plot_metric_comparison(stats, "nmi", "NMI", "Clustering Quality — NMI Comparison", output_path)


def plot_time_comparison(stats: Dict, output_path: str):
    """Create bar chart comparing runtime across datasets and algorithms."""
    if not HAS_MATPLOTLIB:
        print("Skipping time comparison plot (matplotlib not available)")
        return
    
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    datasets = [d for d in dataset_order if d in stats]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ds in enumerate(datasets):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        algos = [a for a in algo_order if a in stats[ds] and not np.isnan(stats[ds][a]["time_mean"])]
        if not algos:
            ax.set_title(f"{ds} (no data)", fontweight='bold')
            ax.text(0.5, 0.5, "No completed experiments", ha='center', va='center', transform=ax.transAxes)
            continue
            
        times = [stats[ds][a]["time_mean"] for a in algos]
        stds = [stats[ds][a]["time_std"] for a in algos]
        colors = [ALGO_COLORS.get(a, '#888888') for a in algos]
        
        x = np.arange(len(algos))
        bars = ax.bar(x, times, yerr=stds, capsize=3, color=colors,
                     edgecolor='white', linewidth=0.5)
        
        ax.set_title(ds, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right')
        ax.set_ylabel('Time (seconds)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Highlight best
        if times:
            min_idx = np.argmin(times)
            bars[min_idx].set_edgecolor('black')
            bars[min_idx].set_linewidth(2)
    
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Runtime Comparison (log scale)', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_pareto_front(stats: Dict, output_path: str):
    """Create Pareto front plot showing cost vs recall trade-off."""
    if not HAS_MATPLOTLIB:
        print("Skipping Pareto plot (matplotlib not available)")
        return
    
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    datasets = [d for d in dataset_order if d in stats]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ds in enumerate(datasets):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        has_data = False
        all_recalls = []
        all_recall_stds = []
        
        for i, algo in enumerate(algo_order):
            if algo not in stats[ds]:
                continue
            s = stats[ds][algo]
            if np.isnan(s["cost_mean"]) or np.isnan(s["recall_mean"]):
                continue
            has_data = True
            all_recalls.append(s["recall_mean"])
            all_recall_stds.append(s["recall_std"])
            
            color = ALGO_COLORS.get(algo, '#888888')
            marker = ALGO_MARKERS[i % len(ALGO_MARKERS)]
            
            ax.scatter(s["cost_mean"], s["recall_mean"], 
                      marker=marker, c=color, s=120, 
                      label=algo, zorder=5, edgecolors='white', linewidths=0.5)
            ax.errorbar(s["cost_mean"], s["recall_mean"],
                       xerr=s["cost_std"], yerr=s["recall_std"],
                       fmt='none', c=color, alpha=0.4, capsize=0, linewidth=1.5)
        
        # Auto-adjust y-axis to zoom into data range
        if all_recalls:
            y_min = max(0, min(all_recalls) - max(all_recall_stds) - 0.05)
            y_max = min(1.05, max(all_recalls) + max(all_recall_stds) + 0.05)
            # For high recall datasets, zoom in
            if min(all_recalls) > 0.5:
                y_min = max(0, min(all_recalls) * 0.9 - 0.02)
            ax.set_ylim(y_min, y_max)
        
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('Cost')
        ax.set_ylabel('Recall')
        ax.grid(True, alpha=0.3, linestyle='--')
        if not has_data:
            ax.text(0.5, 0.5, "No completed experiments", ha='center', va='center', transform=ax.transAxes)
        elif idx == 0:
            ax.legend(loc='lower right', ncol=2, frameon=True, fancybox=False, edgecolor='gray')
    
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Cost vs Recall Trade-off', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_scalability(results: List[Dict], output_path: str):
    """Plot runtime scaling with dataset size using shaded error bands."""
    if not HAS_MATPLOTLIB:
        print("Skipping scalability plot (matplotlib not available)")
        return
    
    # Collect data
    algo_data = defaultdict(lambda: {"sizes": [], "times": [], "times_std": []})
    
    # Group by algorithm and dataset size
    by_algo_size = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["status"] != "completed":
            continue
        by_algo_size[r["algorithm"]][r["dataset_size"]].append(r["time_s"])
    
    for algo in by_algo_size:
        for size in by_algo_size[algo]:
            times = by_algo_size[algo][size]
            algo_data[algo]["sizes"].append(size)
            algo_data[algo]["times"].append(np.mean(times))
            algo_data[algo]["times_std"].append(np.std(times))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    for i, algo in enumerate(algo_order):
        if algo not in algo_data:
            continue
        data = algo_data[algo]
        # Sort by size
        sorted_idx = np.argsort(data["sizes"])
        sizes = np.array(data["sizes"])[sorted_idx]
        times = np.array(data["times"])[sorted_idx]
        times_std = np.array(data["times_std"])[sorted_idx]
        
        color = ALGO_COLORS.get(algo, '#888888')
        marker = ALGO_MARKERS[i % len(ALGO_MARKERS)]
        
        # Plot line with markers
        ax.plot(sizes, times, marker=marker, color=color, label=algo,
               linewidth=2, markersize=8, zorder=3)
        
        # Shaded error band (for log scale, use multiplicative error)
        ax.fill_between(sizes, 
                       np.maximum(times - times_std, times * 0.1),  # Avoid negative on log
                       times + times_std,
                       color=color, alpha=0.2, zorder=2)
    
    ax.set_xlabel('Dataset Size (n)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Scalability', fontweight='bold')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    _save_figure(fig, output_path)


def generate_markdown_report(data: Dict, stats: Dict, output_path: str):
    """Generate a comprehensive markdown report."""
    
    results = data["results"]
    system_info = data.get("system_info", {})
    exp_config = data.get("experiment_config", {})
    
    lines = []
    lines.append("# Robust Clustering Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {system_info.get('timestamp', 'N/A')}")
    lines.append("")
    
    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("1. [System Information](#system-information)")
    lines.append("2. [Experiment Configuration](#experiment-configuration)")
    lines.append("3. [Results Summary](#results-summary)")
    lines.append("4. [Detailed Results by Dataset](#detailed-results-by-dataset)")
    lines.append("5. [Algorithm Comparison](#algorithm-comparison)")
    lines.append("6. [Key Findings](#key-findings)")
    lines.append("")
    
    # System Information
    lines.append("## System Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    for key, value in system_info.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    
    # Experiment Configuration
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append(f"- **Datasets:** {', '.join(exp_config.get('datasets', []))}")
    lines.append(f"- **Algorithms:** {', '.join(exp_config.get('algorithms', []))}")
    lines.append(f"- **Number of seeds:** {exp_config.get('num_seeds', len(exp_config.get('seeds', [])))}")
    lines.append(f"- **Seeds:** {exp_config.get('seeds', [])}")
    lines.append(f"- **Metric:** {exp_config.get('metric', 'L2')}")
    lines.append(f"- **Timeout:** {exp_config.get('timeout', 'None')} seconds")
    lines.append("")
    
    # Results Summary
    lines.append("## Results Summary")
    lines.append("")
    completed = sum(1 for r in results if r["status"] == "completed")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    lines.append(f"- **Total experiments:** {len(results)}")
    lines.append(f"- **Completed:** {completed}")
    lines.append(f"- **Timeout:** {timeout}")
    lines.append(f"- **Errors:** {errors}")
    lines.append(f"- **Skipped:** {skipped}")
    lines.append("")
    
    # Main Results Table
    lines.append("### Main Results Table")
    lines.append("")
    lines.append("*Results show mean ± std over multiple runs. Bold indicates best per dataset. NA indicates timeout/skipped/error.*")
    lines.append("")
    
    # Sort datasets and algorithms
    dataset_order = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]
    algo_order = ["TIKMeans", "IKMeans", "RobustKmeans++", "NKMeans", 
                  "OKMeans", "OKMeans2", "OKMeansFAISS", "OKMeans2FAISS",
                  "KMeans++", "KMeans--"]
    
    datasets = [d for d in dataset_order if d in stats]
    
    lines.append("| Dataset | Algorithm | Cost | Recall | Time (s) | Status |")
    lines.append("|---------|-----------|------|--------|----------|--------|")
    
    for ds in datasets:
        algos = [a for a in algo_order if a in stats[ds]]
        
        # Find best for this dataset (only among completed results)
        completed_algos = [a for a in algos if stats[ds][a].get("status") == "completed"]
        best_cost = min((stats[ds][a]["cost_mean"] for a in completed_algos), default=float('inf'))
        best_recall = max((stats[ds][a]["recall_mean"] for a in completed_algos), default=0)
        best_time = min((stats[ds][a]["time_mean"] for a in completed_algos), default=float('inf'))
        
        for i, algo in enumerate(algos):
            s = stats[ds][algo]
            status = s.get("status", "completed")
            n_runs = s.get("n_runs", 0)
            
            # Format with bold for best (only if completed)
            cost_str = format_with_std(s["cost_mean"], s["cost_std"], 2)
            if status == "completed" and not np.isnan(s["cost_mean"]) and abs(s["cost_mean"] - best_cost) < 0.01 * best_cost:
                cost_str = f"**{cost_str}**"
            
            recall_str = format_with_std(s["recall_mean"], s["recall_std"], 4)
            if status == "completed" and not np.isnan(s["recall_mean"]) and abs(s["recall_mean"] - best_recall) < 0.01:
                recall_str = f"**{recall_str}**"
            
            time_str = format_with_std(s["time_mean"], s["time_std"], 2)
            if status == "completed" and not np.isnan(s["time_mean"]) and abs(s["time_mean"] - best_time) < 0.1 * best_time:
                time_str = f"**{time_str}**"
            
            # Status column
            if status == "completed":
                status_str = f"n={n_runs}"
            else:
                status_str = status.upper()
            
            ds_name = ds if i == 0 else ""
            lines.append(f"| {ds_name} | {algo} | {cost_str} | {recall_str} | {time_str} | {status_str} |")
    
    lines.append("")
    
    # Detailed Results by Dataset
    lines.append("## Detailed Results by Dataset")
    lines.append("")
    
    for ds in datasets:
        lines.append(f"### {ds}")
        lines.append("")
        
        # Get dataset info from first result (any status)
        ds_results = [r for r in results if r["dataset"] == ds]
        if ds_results:
            r = ds_results[0]
            size = r.get('dataset_size', 'N/A')
            size_str = f"{size:,}" if isinstance(size, int) else str(size)
            outliers = r.get('true_outliers', 'N/A')
            outliers_str = f"{outliers:,}" if isinstance(outliers, int) else str(outliers)
            lines.append(f"- **Size:** {size_str} points")
            lines.append(f"- **Dimensions:** {r.get('dataset_dim', 'N/A')}")
            lines.append(f"- **True outliers:** {outliers_str}")
            lines.append(f"- **k (clusters):** {r.get('k', 'N/A')}")
            lines.append("")
        
        lines.append("| Algorithm | Cost (mean±std) | Recall (mean±std) | Precision (mean±std) | F1 (mean±std) | Time (mean±std) | Status |")
        lines.append("|-----------|-----------------|-------------------|----------------------|---------------|-----------------|--------|")
        
        algos = [a for a in algo_order if a in stats[ds]]
        for algo in algos:
            s = stats[ds][algo]
            status = s.get("status", "completed")
            n_runs = s.get("n_runs", 0)
            
            # Status string
            if status == "completed":
                status_str = f"n={n_runs}"
            else:
                status_str = status.upper()
            
            lines.append(f"| {algo} | "
                        f"{format_with_std(s['cost_mean'], s['cost_std'], 2)} | "
                        f"{format_with_std(s['recall_mean'], s['recall_std'], 4)} | "
                        f"{format_with_std(s['precision_mean'], s['precision_std'], 4)} | "
                        f"{format_with_std(s['f1_mean'], s['f1_std'], 4)} | "
                        f"{format_with_std(s['time_mean'], s['time_std'], 2)}s | "
                        f"{status_str} |")
        
        lines.append("")
    
    # Algorithm Comparison
    lines.append("## Algorithm Comparison")
    lines.append("")
    lines.append("### Algorithms Included")
    lines.append("")
    lines.append("| Algorithm | Description |")
    lines.append("|-----------|-------------|")
    lines.append("| TIKMeans | Fast-Sampling + Center-Reduction |")
    lines.append("| IKMeans | Fast-Sampling + Weighted k-means++ |")
    lines.append("| RobustKmeans++ | Robust k-means++ with Lloyd refinement |")
    lines.append("| NKMeans | Neighborhood-based outlier detection |")
    lines.append("| OKMeans | KNN-based outlier detection (2z-neighbor radius) |")
    lines.append("| OKMeans2 | KNN-based outlier detection (neighbor distance sum) |")
    lines.append("| OKMeansFAISS | OKMeans with FAISS-accelerated KNN |")
    lines.append("| OKMeans2FAISS | OKMeans2 with FAISS-accelerated KNN |")
    lines.append("| KMeans++ | Standard k-means++ (baseline, no outlier handling) |")
    lines.append("| KMeans-- | k-means with outlier-aware Lloyd iterations |")
    lines.append("")
    
    # Key Findings
    lines.append("## Key Findings")
    lines.append("")
    
    # Best algorithms per dataset
    lines.append("### Best Performing Algorithms per Dataset")
    lines.append("")
    
    for ds in datasets:
        # Only consider completed algorithms with valid data
        algos = [a for a in algo_order if a in stats[ds] and not np.isnan(stats[ds][a]["cost_mean"])]
        if not algos:
            lines.append(f"**{ds}:** No completed experiments")
            lines.append("")
            continue
            
        best_cost_algo = min(algos, key=lambda a: stats[ds][a]["cost_mean"])
        best_recall_algo = max(algos, key=lambda a: stats[ds][a]["recall_mean"])
        best_time_algo = min(algos, key=lambda a: stats[ds][a]["time_mean"])
        
        lines.append(f"**{ds}:**")
        lines.append(f"- Best Cost: **{best_cost_algo}** ({stats[ds][best_cost_algo]['cost_mean']:.2f} ± {stats[ds][best_cost_algo]['cost_std']:.2f})")
        lines.append(f"- Best Recall: **{best_recall_algo}** ({stats[ds][best_recall_algo]['recall_mean']:.4f} ± {stats[ds][best_recall_algo]['recall_std']:.4f})")
        lines.append(f"- Fastest: **{best_time_algo}** ({stats[ds][best_time_algo]['time_mean']:.2f}s ± {stats[ds][best_time_algo]['time_std']:.2f}s)")
        lines.append("")
    
    # Overall rankings
    lines.append("### Overall Algorithm Rankings")
    lines.append("")
    
    # Compute average rankings
    algo_rankings = defaultdict(lambda: {"cost_ranks": [], "recall_ranks": [], "time_ranks": []})
    
    for ds in datasets:
        # Only consider completed algorithms with valid data
        algos = [a for a in algo_order if a in stats[ds] and not np.isnan(stats[ds][a]["cost_mean"])]
        if not algos:
            continue
        
        # Rank by cost (lower is better)
        cost_sorted = sorted(algos, key=lambda a: stats[ds][a]["cost_mean"])
        for rank, algo in enumerate(cost_sorted, 1):
            algo_rankings[algo]["cost_ranks"].append(rank)
        
        # Rank by recall (higher is better)
        recall_sorted = sorted(algos, key=lambda a: -stats[ds][a]["recall_mean"])
        for rank, algo in enumerate(recall_sorted, 1):
            algo_rankings[algo]["recall_ranks"].append(rank)
        
        # Rank by time (lower is better)
        time_sorted = sorted(algos, key=lambda a: stats[ds][a]["time_mean"])
        for rank, algo in enumerate(time_sorted, 1):
            algo_rankings[algo]["time_ranks"].append(rank)
    
    lines.append("| Algorithm | Avg Cost Rank | Avg Recall Rank | Avg Time Rank |")
    lines.append("|-----------|---------------|-----------------|---------------|")
    
    for algo in algo_order:
        if algo not in algo_rankings:
            continue
        r = algo_rankings[algo]
        avg_cost = np.mean(r["cost_ranks"]) if r["cost_ranks"] else float('inf')
        avg_recall = np.mean(r["recall_ranks"]) if r["recall_ranks"] else float('inf')
        avg_time = np.mean(r["time_ranks"]) if r["time_ranks"] else float('inf')
        lines.append(f"| {algo} | {avg_cost:.2f} | {avg_recall:.2f} | {avg_time:.2f} |")
    
    lines.append("")
    lines.append("*Lower rank is better. Rankings computed across all datasets.*")
    lines.append("")
    
    # Errors and timeouts
    error_results = [r for r in results if r["status"] == "error"]
    timeout_results = [r for r in results if r["status"] == "timeout"]
    
    if error_results or timeout_results:
        lines.append("### Errors and Timeouts")
        lines.append("")
        
        if timeout_results:
            lines.append(f"**Timeouts ({len(timeout_results)}):**")
            for r in timeout_results[:10]:  # Show first 10
                lines.append(f"- {r['dataset']} / {r['algorithm']} (seed {r['seed']})")
            if len(timeout_results) > 10:
                lines.append(f"- ... and {len(timeout_results) - 10} more")
            lines.append("")
        
        if error_results:
            lines.append(f"**Errors ({len(error_results)}):**")
            for r in error_results[:10]:
                lines.append(f"- {r['dataset']} / {r['algorithm']} (seed {r['seed']}): {r.get('error', 'Unknown')}")
            if len(error_results) > 10:
                lines.append(f"- ... and {len(error_results) - 10} more")
            lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze clustering benchmark results"
    )
    parser.add_argument(
        "--input", "-i", type=str,
        required=True,
        help="Input JSON file with benchmark results"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=None,
        help="Output directory for analysis files"
    )
    parser.add_argument(
        "--generate-table", action="store_true",
        help="Generate text table"
    )
    parser.add_argument(
        "--generate-plots", action="store_true",
        help="Generate comparison plots"
    )
    parser.add_argument(
        "--generate-report", action="store_true",
        help="Generate markdown report"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate all outputs"
    )
    
    args = parser.parse_args()
    
    # Load results
    data = load_results(args.input)
    results = data["results"]
    exp_config = data.get("experiment_config", None)
    stats = aggregate_results(results, exp_config)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    if args.generate_table or args.all:
        table = generate_table_text(stats, include_std=True)
        print("\n" + table + "\n")
        
        table_path = output_dir / "results_table.txt"
        with open(table_path, "w") as f:
            f.write(table)
        print(f"Saved table: {table_path}")
    
    if args.generate_plots or args.all:
        plot_cost_comparison(stats, str(output_dir / "cost_comparison.png"))
        plot_recall_comparison(stats, str(output_dir / "recall_comparison.png"))
        plot_ari_comparison(stats, str(output_dir / "ari_comparison.png"))
        plot_nmi_comparison(stats, str(output_dir / "nmi_comparison.png"))
        plot_time_comparison(stats, str(output_dir / "time_comparison.png"))
        plot_pareto_front(stats, str(output_dir / "pareto_front.png"))
        plot_scalability(results, str(output_dir / "scalability.png"))
    
    if args.generate_report or args.all:
        generate_markdown_report(data, stats, str(output_dir / "report.md"))
    
    # Always generate CSV
    generate_csv(stats, str(output_dir / "results.csv"))
    print(f"Saved CSV: {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
