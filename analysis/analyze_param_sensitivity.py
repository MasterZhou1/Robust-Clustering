#!/usr/bin/env python3
"""
Analysis and visualization for parameter sensitivity experiments.

Generates:
- Parameter sensitivity plots with shaded error bands (PDF)
- CSV with results
- Summary tables

Usage:
    python analyze_param_sensitivity.py -i results/param_sensitivity_*.json --all
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Try to import plotting libraries
HAS_MATPLOTLIB = False
HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for SLURM
    import matplotlib.pyplot as plt
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
    rcParams['legend.fontsize'] = 10
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


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import ANALYSIS_DIR


# Dataset order for plots: Row 1: SKIN-5, SKIN-10, SHUTTLE; Row 2: SUSY-5, SUSY-10, KDDFULL
DATASET_ORDER = ["SKIN-5", "SKIN-10", "SHUTTLE", "SUSY-5", "SUSY-10", "KDDFULL"]

# Color palette - colorblind-friendly
COLORS = {
    "OKMeansFAISS": "#0077BB",    # Blue
    "OKMeans2FAISS": "#EE7733",   # Orange
}
MARKERS = {
    "OKMeansFAISS": "o",
    "OKMeans2FAISS": "s",
}
LABELS = {
    "OKMeansFAISS": "OKMeansFAISS",
    "OKMeans2FAISS": "OKMeans2FAISS",
}


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def aggregate_param_sensitivity_results(results: List[Dict]) -> Dict:
    """
    Aggregate parameter sensitivity results by dataset, algorithm, and c value.
    
    Returns nested dict: {dataset: {algorithm: {c: {metric: value}}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    for r in results:
        if r.get("status") != "completed":
            continue
        
        ds = r["dataset"]
        algo = r["algorithm"]
        c = r["c"]
        
        aggregated[ds][algo][c]["cost"].append(r["cost"])
        aggregated[ds][algo][c]["recall"].append(r["recall"])
        aggregated[ds][algo][c]["precision"].append(r.get("precision", 0))
        aggregated[ds][algo][c]["f1"].append(r.get("f1", 0))
        aggregated[ds][algo][c]["ari"].append(
            np.nan if r.get("ari") is None else float(r["ari"])
        )
        aggregated[ds][algo][c]["nmi"].append(
            np.nan if r.get("nmi") is None else float(r["nmi"])
        )
        aggregated[ds][algo][c]["time_s"].append(r["time_s"])
        aggregated[ds][algo][c]["param_value"].append(r.get("param_value", c))
    
    # Compute statistics
    stats = {}
    for ds in aggregated:
        stats[ds] = {}
        for algo in aggregated[ds]:
            stats[ds][algo] = {}
            for c in aggregated[ds][algo]:
                metrics = aggregated[ds][algo][c]
                stats[ds][algo][c] = {
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
                    "param_value": np.mean([v for v in metrics["param_value"] if v is not None]) if any(v is not None for v in metrics["param_value"]) else None,
                    "n_runs": len(metrics["cost"]),
                }
    
    return stats


def format_csv_value(value):
    """Format value for CSV, replacing NaN with 'NA'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return value


def generate_csv(stats: Dict, output_path: str):
    """Generate CSV file with parameter sensitivity results."""
    
    rows = []
    for ds in sorted(stats.keys()):
        for algo in sorted(stats[ds].keys()):
            for c in sorted(stats[ds][algo].keys()):
                s = stats[ds][algo][c]
                rows.append({
                    "dataset": ds,
                    "algorithm": algo,
                    "c": c,
                    "param_value": s["param_value"],
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
                    "n_runs": s["n_runs"],
                })
    
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    else:
        with open(output_path, "w") as f:
            if rows:
                header = ",".join(rows[0].keys())
                f.write(header + "\n")
                for row in rows:
                    values = ",".join(str(v) for v in row.values())
                    f.write(values + "\n")
    
    print(f"Saved CSV: {output_path}")


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


def plot_param_sensitivity_cost(stats: Dict, output_path: str):
    """Create line plots with shaded error bands showing cost vs c parameter."""
    if not HAS_MATPLOTLIB:
        print("Skipping cost sensitivity plot (matplotlib not available)")
        return
    
    datasets = [d for d in DATASET_ORDER if d in stats]
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        print("No data for cost sensitivity plot")
        return
    
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        
        all_costs = []
        all_stds = []
        
        for algo in sorted(stats[ds].keys()):
            c_values = np.array(sorted(stats[ds][algo].keys()))
            costs = np.array([stats[ds][algo][c]["cost_mean"] for c in c_values])
            stds = np.array([stats[ds][algo][c]["cost_std"] for c in c_values])
            
            all_costs.extend(costs)
            all_stds.extend(stds)
            
            color = COLORS.get(algo, "#333333")
            marker = MARKERS.get(algo, "o")
            label = LABELS.get(algo, algo)
            
            ax.plot(c_values, costs, marker=marker, color=color, label=label,
                   linewidth=2, markersize=8, zorder=3)
            ax.fill_between(c_values, costs - stds, costs + stds,
                           color=color, alpha=0.2, zorder=2)
        
        # Auto-adjust y-axis to zoom into data range (with 10% padding)
        if all_costs:
            y_min = min(all_costs) - max(all_stds)
            y_max = max(all_costs) + max(all_stds)
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
        
        ax.set_xlabel('c parameter')
        ax.set_ylabel('Cost')
        ax.set_title(ds, fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.set_xticks(c_values)
    
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Clustering Cost vs c Parameter', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_param_sensitivity_recall(stats: Dict, output_path: str):
    """Create line plots with shaded error bands showing recall vs c parameter."""
    if not HAS_MATPLOTLIB:
        print("Skipping recall sensitivity plot (matplotlib not available)")
        return
    
    datasets = [d for d in DATASET_ORDER if d in stats]
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        print("No data for recall sensitivity plot")
        return
    
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        
        all_recalls = []
        all_stds = []
        
        for algo in sorted(stats[ds].keys()):
            c_values = np.array(sorted(stats[ds][algo].keys()))
            recalls = np.array([stats[ds][algo][c]["recall_mean"] for c in c_values])
            stds = np.array([stats[ds][algo][c]["recall_std"] for c in c_values])
            
            all_recalls.extend(recalls)
            all_stds.extend(stds)
            
            color = COLORS.get(algo, "#333333")
            marker = MARKERS.get(algo, "o")
            label = LABELS.get(algo, algo)
            
            ax.plot(c_values, recalls, marker=marker, color=color, label=label,
                   linewidth=2, markersize=8, zorder=3)
            ax.fill_between(c_values, recalls - stds, recalls + stds,
                           color=color, alpha=0.2, zorder=2)
        
        # Auto-adjust y-axis to zoom into data range
        if all_recalls:
            y_min = max(0, min(all_recalls) - max(all_stds) - 0.05)
            y_max = min(1.0, max(all_recalls) + max(all_stds) + 0.05)
            # Ensure at least 0.1 range for visibility
            if y_max - y_min < 0.1:
                mid = (y_max + y_min) / 2
                y_min, y_max = mid - 0.05, mid + 0.05
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('c parameter')
        ax.set_ylabel('Recall')
        ax.set_title(ds, fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(c_values)
    
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Outlier Detection Recall vs c Parameter', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_param_sensitivity_time(stats: Dict, output_path: str):
    """Create line plots with shaded error bands showing time vs c parameter."""
    if not HAS_MATPLOTLIB:
        print("Skipping time sensitivity plot (matplotlib not available)")
        return
    
    datasets = [d for d in DATASET_ORDER if d in stats]
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        print("No data for time sensitivity plot")
        return
    
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        
        all_times = []
        
        for algo in sorted(stats[ds].keys()):
            c_values = np.array(sorted(stats[ds][algo].keys()))
            times = np.array([stats[ds][algo][c]["time_mean"] for c in c_values])
            stds = np.array([stats[ds][algo][c]["time_std"] for c in c_values])
            
            all_times.extend(times)
            
            color = COLORS.get(algo, "#333333")
            marker = MARKERS.get(algo, "o")
            label = LABELS.get(algo, algo)
            
            ax.plot(c_values, times, marker=marker, color=color, label=label,
                   linewidth=2, markersize=8, zorder=3)
            ax.fill_between(c_values, np.maximum(0, times - stds), times + stds,
                           color=color, alpha=0.2, zorder=2)
        
        # Start y-axis from 0 for time plots
        if all_times:
            ax.set_ylim(0, max(all_times) * 1.15)
        
        ax.set_xlabel('c parameter')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(ds, fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(c_values)
    
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Runtime vs c Parameter', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def _plot_param_sensitivity_metric(stats: Dict, metric: str, ylabel: str, title: str, output_path: str):
    """Line plots with shaded error bands showing one metric vs c parameter."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping {metric} sensitivity plot (matplotlib not available)")
        return

    datasets = [d for d in DATASET_ORDER if d in stats]
    n_datasets = len(datasets)
    if n_datasets == 0:
        print(f"No data for {metric} sensitivity plot")
        return

    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        all_vals, all_stds = [], []
        last_c_values = None

        for algo in sorted(stats[ds].keys()):
            c_values = np.array(sorted(stats[ds][algo].keys()))
            means = np.array([stats[ds][algo][c].get(f"{metric}_mean", np.nan) for c in c_values])
            stds  = np.array([stats[ds][algo][c].get(f"{metric}_std",  np.nan) for c in c_values])
            if np.all(np.isnan(means)):
                continue

            last_c_values = c_values
            all_vals.extend(means[~np.isnan(means)])
            all_stds.extend(stds[~np.isnan(stds)])

            color  = COLORS.get(algo, "#333333")
            marker = MARKERS.get(algo, "o")
            label  = LABELS.get(algo, algo)

            ax.plot(c_values, means, marker=marker, color=color, label=label,
                    linewidth=2, markersize=8, zorder=3)
            ax.fill_between(c_values, means - stds, means + stds,
                            color=color, alpha=0.2, zorder=2)

        if all_vals:
            y_min = max(0, min(all_vals) - max(all_stds) - 0.01)
            y_max = min(1.0, max(all_vals) + max(all_stds) + 0.01)
            if y_max - y_min < 0.05:
                mid = (y_max + y_min) / 2
                y_min, y_max = mid - 0.025, mid + 0.025
            ax.set_ylim(y_min, y_max)

        ax.set_xlabel('c parameter')
        ax.set_ylabel(ylabel)
        ax.set_title(ds, fontweight='bold')
        ax.legend(frameon=True, fancybox=False, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        if last_c_values is not None:
            ax.set_xticks(last_c_values)

    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_param_sensitivity_ari(stats: Dict, output_path: str):
    """Line plots showing ARI vs c parameter."""
    _plot_param_sensitivity_metric(stats, "ari", "ARI", "ARI vs c Parameter", output_path)


def plot_param_sensitivity_nmi(stats: Dict, output_path: str):
    """Line plots showing NMI vs c parameter."""
    _plot_param_sensitivity_metric(stats, "nmi", "NMI", "NMI vs c Parameter", output_path)


def plot_param_sensitivity_combined(stats: Dict, output_path: str):
    """Create a combined 1x3 grid showing cost, recall, time vs c (aggregated across datasets)."""
    if not HAS_MATPLOTLIB:
        print("Skipping combined sensitivity plot (matplotlib not available)")
        return
    
    # Aggregate across datasets for each algorithm and c value
    algo_c_data = defaultdict(lambda: defaultdict(lambda: {
        "cost": [], "recall": [], "time": [], "cost_std": [], "recall_std": [], "time_std": []
    }))
    
    for ds in stats:
        for algo in stats[ds]:
            for c in stats[ds][algo]:
                s = stats[ds][algo][c]
                algo_c_data[algo][c]["cost"].append(s["cost_mean"])
                algo_c_data[algo][c]["recall"].append(s["recall_mean"])
                algo_c_data[algo][c]["time"].append(s["time_mean"])
                algo_c_data[algo][c]["cost_std"].append(s["cost_std"])
                algo_c_data[algo][c]["recall_std"].append(s["recall_std"])
                algo_c_data[algo][c]["time_std"].append(s["time_std"])
    
    # Normalize costs by dataset (for fair comparison across datasets)
    algo_c_normalized = defaultdict(lambda: defaultdict(lambda: {
        "cost": [], "recall": [], "time": []
    }))
    
    for ds in stats:
        all_costs = []
        for algo in stats[ds]:
            for c in stats[ds][algo]:
                all_costs.append(stats[ds][algo][c]["cost_mean"])
        min_cost = min(all_costs) if all_costs else 1.0
        
        for algo in stats[ds]:
            for c in stats[ds][algo]:
                s = stats[ds][algo][c]
                algo_c_normalized[algo][c]["cost"].append(s["cost_mean"] / min_cost)
                algo_c_normalized[algo][c]["recall"].append(s["recall_mean"])
                algo_c_normalized[algo][c]["time"].append(s["time_mean"])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Plot 1: Normalized Cost vs c
    ax = axes[0]
    for algo in sorted(algo_c_normalized.keys()):
        c_values = np.array(sorted(algo_c_normalized[algo].keys()))
        costs = np.array([np.mean(algo_c_normalized[algo][c]["cost"]) for c in c_values])
        stds = np.array([np.std(algo_c_normalized[algo][c]["cost"]) for c in c_values])
        
        color = COLORS.get(algo, "#333333")
        marker = MARKERS.get(algo, "o")
        label = LABELS.get(algo, algo)
        
        ax.plot(c_values, costs, marker=marker, color=color, label=label,
               linewidth=2, markersize=8, zorder=3)
        ax.fill_between(c_values, costs - stds, costs + stds,
                       color=color, alpha=0.2, zorder=2)
    
    ax.set_xlabel('c parameter')
    ax.set_ylabel('Relative Cost')
    ax.set_title('Cost Sensitivity', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(c_values)
    
    # Plot 2: Recall vs c
    ax = axes[1]
    all_recalls = []
    all_stds_recall = []
    for algo in sorted(algo_c_data.keys()):
        c_values = np.array(sorted(algo_c_data[algo].keys()))
        recalls = np.array([np.mean(algo_c_data[algo][c]["recall"]) for c in c_values])
        stds = np.array([np.std(algo_c_data[algo][c]["recall"]) for c in c_values])
        
        all_recalls.extend(recalls)
        all_stds_recall.extend(stds)
        
        color = COLORS.get(algo, "#333333")
        marker = MARKERS.get(algo, "o")
        label = LABELS.get(algo, algo)
        
        ax.plot(c_values, recalls, marker=marker, color=color, label=label,
               linewidth=2, markersize=8, zorder=3)
        ax.fill_between(c_values, recalls - stds, recalls + stds,
                       color=color, alpha=0.2, zorder=2)
    
    # Auto-adjust y-axis to zoom into data range
    if all_recalls:
        y_min = max(0, min(all_recalls) - max(all_stds_recall) - 0.05)
        y_max = min(1.0, max(all_recalls) + max(all_stds_recall) + 0.05)
        if y_max - y_min < 0.1:
            mid = (y_max + y_min) / 2
            y_min, y_max = mid - 0.05, mid + 0.05
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('c parameter')
    ax.set_ylabel('Recall')
    ax.set_title('Recall Sensitivity', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(c_values)
    
    # Plot 3: Time vs c
    ax = axes[2]
    all_times = []
    for algo in sorted(algo_c_data.keys()):
        c_values = np.array(sorted(algo_c_data[algo].keys()))
        times = np.array([np.mean(algo_c_data[algo][c]["time"]) for c in c_values])
        stds = np.array([np.std(algo_c_data[algo][c]["time"]) for c in c_values])
        
        all_times.extend(times)
        
        color = COLORS.get(algo, "#333333")
        marker = MARKERS.get(algo, "o")
        label = LABELS.get(algo, algo)
        
        ax.plot(c_values, times, marker=marker, color=color, label=label,
               linewidth=2, markersize=8, zorder=3)
        ax.fill_between(c_values, np.maximum(0, times - stds), times + stds,
                       color=color, alpha=0.2, zorder=2)
    
    # Start y-axis from 0 for time
    if all_times:
        ax.set_ylim(0, max(all_times) * 1.15)
    
    ax.set_xlabel('c parameter')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime Sensitivity', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(c_values)
    
    fig.suptitle('Parameter Sensitivity Analysis (aggregated across datasets)', 
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


def generate_table_text(stats: Dict) -> str:
    """Generate a text-based table for parameter sensitivity results."""
    lines = []
    lines.append("=" * 120)
    lines.append("Parameter Sensitivity Results for OKMeans Algorithms")
    lines.append("Results show mean +/- std over multiple runs.")
    lines.append("=" * 120)
    lines.append("")
    
    header = f"{'Dataset':<15} {'Algorithm':<15} {'c':>5} {'param_val':>10} {'Cost':>18} {'Recall':>14} {'Time(s)':>12}"
    lines.append(header)
    lines.append("-" * 120)
    
    for ds in sorted(stats.keys()):
        first_row = True
        for algo in sorted(stats[ds].keys()):
            for c in sorted(stats[ds][algo].keys()):
                s = stats[ds][algo][c]
                
                cost_str = f"{s['cost_mean']:.2f} +/- {s['cost_std']:.2f}"
                recall_str = f"{s['recall_mean']:.4f} +/- {s['recall_std']:.4f}"
                time_str = f"{s['time_mean']:.2f} +/- {s['time_std']:.2f}"
                param_str = f"{s['param_value']:.2f}" if s['param_value'] is not None else "N/A"
                
                ds_name = ds if first_row else ""
                lines.append(f"{ds_name:<15} {algo:<15} {c:>5} {param_str:>10} {cost_str:>18} {recall_str:>14} {time_str:>12}")
                first_row = False
        
        lines.append("-" * 120)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter sensitivity benchmark results"
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
        "--all", action="store_true",
        help="Generate all outputs"
    )
    
    args = parser.parse_args()
    
    # Load results
    data = load_results(args.input)
    results = data["results"]
    stats = aggregate_param_sensitivity_results(results)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Always generate CSV
    generate_csv(stats, str(output_dir / "param_sensitivity_results.csv"))
    
    # Generate outputs
    if args.generate_table or args.all:
        table = generate_table_text(stats)
        print("\n" + table + "\n")
        
        table_path = output_dir / "param_sensitivity_table.txt"
        with open(table_path, "w") as f:
            f.write(table)
        print(f"Saved table: {table_path}")
    
    if args.generate_plots or args.all:
        plot_param_sensitivity_cost(stats, str(output_dir / "param_sensitivity_cost.png"))
        plot_param_sensitivity_recall(stats, str(output_dir / "param_sensitivity_recall.png"))
        plot_param_sensitivity_ari(stats, str(output_dir / "param_sensitivity_ari.png"))
        plot_param_sensitivity_nmi(stats, str(output_dir / "param_sensitivity_nmi.png"))
        plot_param_sensitivity_time(stats, str(output_dir / "param_sensitivity_time.png"))
        plot_param_sensitivity_combined(stats, str(output_dir / "param_sensitivity_combined.png"))


if __name__ == "__main__":
    main()
