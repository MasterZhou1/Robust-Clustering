#!/usr/bin/env python3
"""
Preprocess clustering datasets with outlier injection/identification.

- Normalize each dataset so every dimension has mean 0 and std 1.
- Skin/SUSY: inject synthetic outliers uniformly in [-xi, xi]^d.
- Shuttle: treat the two smallest classes as outliers.
- KDDFULL: treat the three largest classes as inliers and the rest as outliers.

Each preprocessed dataset is saved as a `.npz` containing:
    'data'         - normalized feature matrix (n x d)
    'labels'       - original class labels
    'outlier_mask' - boolean array, True = outlier
    'k'            - number of clusters for inliers
"""

import numpy as np
from pathlib import Path
from collections import Counter
import argparse


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize features to mean=0, std=1 per dimension."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1.0
    return (X - mean) / std


def inject_synthetic_outliers(X: np.ndarray, outlier_ratio: float, xi: float, 
                               seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic outliers uniformly sampled from [-xi, xi]^d.
    
    Args:
        X: Original data (n × d)
        outlier_ratio: Fraction of outliers to inject (e.g., 0.01 for 1%)
        xi: Bound for hypercube sampling
        seed: Random seed
    
    Returns:
        X_augmented: Data with outliers appended (n' × d)
        outlier_mask: Boolean mask (True = outlier)
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_outliers = int(n * outlier_ratio)
    
    # Sample outliers uniformly from [-xi, xi]^d
    outliers = rng.uniform(-xi, xi, size=(n_outliers, d))
    
    # Combine original data with outliers
    X_augmented = np.vstack([X, outliers])
    
    # Create outlier mask
    outlier_mask = np.zeros(len(X_augmented), dtype=bool)
    outlier_mask[n:] = True
    
    return X_augmented, outlier_mask


def process_skin(data_dir: Path, output_dir: Path) -> dict:
    """Process the Skin dataset: inject synthetic outliers for xi in {5, 10}, k=10."""
    print("\n" + "=" * 60)
    print("[SKIN] Processing Skin_NonSkin dataset")
    print("=" * 60)
    
    # Load data
    data = np.loadtxt(data_dir / "Skin_NonSkin.txt")
    X = data[:, :-1].astype(np.float64)  # Features (B, G, R)
    y = data[:, -1].astype(int)  # Labels
    
    print(f"  Original: {X.shape[0]:,} points × {X.shape[1]} features")
    print(f"  Labels: {dict(Counter(y))}")
    
    # Normalize
    X_norm = normalize_features(X)
    print(f"  Normalized: mean={X_norm.mean(axis=0)}, std={X_norm.std(axis=0)}")
    
    results = {"original": {"data": X_norm, "labels": y}}
    
    # Inject outliers for ξ = 5 and ξ = 10
    for xi in [5, 10]:
        X_aug, mask = inject_synthetic_outliers(X_norm, outlier_ratio=0.01, xi=xi)
        n_outliers = mask.sum()
        
        # Extend labels for outliers (assign label -1)
        y_aug = np.concatenate([y, -np.ones(n_outliers, dtype=int)])
        
        name = f"SKIN-{xi}"
        results[name] = {
            "data": X_aug,
            "labels": y_aug,
            "outlier_mask": mask,
            "k": 10,
            "n_outliers": n_outliers
        }
        
        # Save
        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path, 
                           data=X_aug, 
                           labels=y_aug, 
                           outlier_mask=mask,
                           k=10)
        
        print(f"\n  {name}:")
        print(f"    Total: {len(X_aug):,} points")
        print(f"    Inliers: {(~mask).sum():,} ({(~mask).sum()/len(X_aug)*100:.2f}%)")
        print(f"    Outliers: {n_outliers:,} ({n_outliers/len(X_aug)*100:.2f}%)")
        print(f"    Saved: {out_path}")
    
    return results


def process_susy(data_dir: Path, output_dir: Path) -> dict:
    """Process the SUSY dataset: inject synthetic outliers for xi in {5, 10}, k=10."""
    print("\n" + "=" * 60)
    print("[SUSY] Processing SUSY dataset")
    print("=" * 60)
    
    # Load data using pandas for faster parsing (first column is label)
    print("  Loading SUSY.csv (this may take a moment)...")
    try:
        import pandas as pd
        data = pd.read_csv(data_dir / "SUSY.csv", header=None).values
    except ImportError:
        data = np.loadtxt(data_dir / "SUSY.csv", delimiter=',')
    y = data[:, 0].astype(int)  # First column is label
    X = data[:, 1:].astype(np.float64)  # Rest are features
    
    print(f"  Original: {X.shape[0]:,} points × {X.shape[1]} features")
    print(f"  Labels: {dict(Counter(y))}")
    
    # Normalize
    X_norm = normalize_features(X)
    print(f"  Normalized: mean~{X_norm.mean():.4f}, std~{X_norm.std():.4f}")
    
    results = {"original": {"data": X_norm, "labels": y}}
    
    # Inject outliers for ξ = 5 and ξ = 10
    for xi in [5, 10]:
        X_aug, mask = inject_synthetic_outliers(X_norm, outlier_ratio=0.01, xi=xi)
        n_outliers = mask.sum()
        
        # Extend labels for outliers
        y_aug = np.concatenate([y, -np.ones(n_outliers, dtype=int)])
        
        name = f"SUSY-{xi}"
        results[name] = {
            "data": X_aug,
            "labels": y_aug,
            "outlier_mask": mask,
            "k": 10,
            "n_outliers": n_outliers
        }
        
        # Save
        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(out_path,
                           data=X_aug,
                           labels=y_aug,
                           outlier_mask=mask,
                           k=10)
        
        print(f"\n  {name}:")
        print(f"    Total: {len(X_aug):,} points")
        print(f"    Inliers: {(~mask).sum():,} ({(~mask).sum()/len(X_aug)*100:.2f}%)")
        print(f"    Outliers: {n_outliers:,} ({n_outliers/len(X_aug)*100:.2f}%)")
        print(f"    Saved: {out_path}")
    
    return results


def process_shuttle(data_dir: Path, output_dir: Path) -> dict:
    """Process the Shuttle dataset: two smallest classes are outliers, k=10."""
    print("\n" + "=" * 60)
    print("[SHUTTLE] Processing Shuttle dataset")
    print("=" * 60)
    
    # Load data
    data = np.loadtxt(data_dir / "shuttle.trn")
    X = data[:, :-1].astype(np.float64)  # Features
    y = data[:, -1].astype(int)  # Labels
    
    print(f"  Original: {X.shape[0]:,} points × {X.shape[1]} features")
    
    # Analyze class distribution
    label_counts = Counter(y)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Class distribution:")
    for label, count in sorted_labels:
        print(f"    Class {label}: {count:,} ({count/len(y)*100:.2f}%)")
    
    # Identify two smallest classes as outliers
    smallest_classes = [label for label, _ in sorted_labels[-2:]]
    print(f"\n  Two smallest classes (outliers): {smallest_classes}")
    
    # Normalize
    X_norm = normalize_features(X)
    
    # Create outlier mask
    outlier_mask = np.isin(y, smallest_classes)
    n_outliers = outlier_mask.sum()
    
    print(f"\n  SHUTTLE:")
    print(f"    Total: {len(X_norm):,} points")
    print(f"    Inliers: {(~outlier_mask).sum():,} ({(~outlier_mask).sum()/len(X_norm)*100:.2f}%)")
    print(f"    Outliers: {n_outliers:,} ({n_outliers/len(X_norm)*100:.4f}%)")
    
    # Four largest classes percentage
    four_largest = sum(c for _, c in sorted_labels[:4])
    print(f"    Four largest classes: {four_largest:,} ({four_largest/len(y)*100:.2f}%)")
    
    # Save
    out_path = output_dir / "SHUTTLE.npz"
    np.savez_compressed(out_path,
                       data=X_norm,
                       labels=y,
                       outlier_mask=outlier_mask,
                       k=10)
    print(f"    Saved: {out_path}")
    
    return {
        "SHUTTLE": {
            "data": X_norm,
            "labels": y,
            "outlier_mask": outlier_mask,
            "k": 10,
            "n_outliers": n_outliers
        }
    }


def process_kddfull(data_dir: Path, output_dir: Path) -> dict:
    """Process the KDDFULL dataset: three largest classes are inliers, k=3.

    KDD has mixed categorical/numeric features; only numeric columns are used
    (columns 0, 4-11, 12-20, 22-40).
    """
    print("\n" + "=" * 60)
    print("[KDDFULL] Processing KDD Cup 1999 dataset")
    print("=" * 60)
    
    # KDD Cup 1999 column info:
    # 0: duration (numeric)
    # 1: protocol_type (categorical: tcp, udp, icmp)
    # 2: service (categorical: many values)
    # 3: flag (categorical: many values)
    # 4-11: numeric features
    # 12-20: numeric features
    # 21: is_host_login (binary)
    # 22-40: numeric features
    # 41: label (attack type)
    
    print("  Loading kddcup.data using pandas (faster)...")
    
    # Use pandas for much faster CSV parsing
    try:
        import pandas as pd
        # Define column names
        col_names = [f'col{i}' for i in range(42)]
        # Read with pandas (much faster than line-by-line)
        df = pd.read_csv(data_dir / "kddcup.data", header=None, names=col_names, 
                        low_memory=False)
        
        # Clean the label column (remove trailing '.')
        df['col41'] = df['col41'].str.rstrip('.')
        
        # Numeric columns indices
        numeric_cols = [0] + list(range(4, 12)) + list(range(12, 21)) + list(range(22, 41))
        numeric_col_names = [f'col{i}' for i in numeric_cols]
        
        # Extract features and labels
        X = df[numeric_col_names].values.astype(np.float64)
        y = df['col41'].values
        
    except ImportError:
        print("  pandas not available, using slower numpy parsing...")
        # Fallback to numpy genfromtxt (still faster than line-by-line)
        data_rows = []
        labels = []
        
        with open(data_dir / "kddcup.data") as f:
            for line in f:
                parts = line.strip().rstrip('.').split(',')
                if len(parts) >= 42:
                    labels.append(parts[-1])
                    numeric_cols = [0] + list(range(4, 12)) + list(range(12, 21)) + list(range(22, 41))
                    row = []
                    for i in numeric_cols:
                        try:
                            row.append(float(parts[i]))
                        except (ValueError, IndexError):
                            row.append(0.0)
                    data_rows.append(row)
        
        X = np.array(data_rows, dtype=np.float64)
        y = np.array(labels)
    
    print(f"  Original: {X.shape[0]:,} points × {X.shape[1]} numeric features")
    
    # Analyze class distribution
    label_counts = Counter(y)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Total unique classes: {len(label_counts)}")
    print(f"\n  Top classes:")
    for i, (label, count) in enumerate(sorted_labels[:5]):
        print(f"    {i+1}. {label}: {count:,} ({count/len(y)*100:.2f}%)")
    
    # Three largest classes are inliers
    inlier_classes = [label for label, _ in sorted_labels[:3]]
    outlier_classes = [label for label, _ in sorted_labels[3:]]
    
    print(f"\n  Inlier classes (3 largest): {inlier_classes}")
    print(f"  Outlier classes ({len(outlier_classes)} remaining): {outlier_classes[:5]}...")
    
    # Normalize
    X_norm = normalize_features(X)
    
    # Create outlier mask
    outlier_mask = ~np.isin(y, inlier_classes)
    n_outliers = outlier_mask.sum()
    n_inliers = (~outlier_mask).sum()
    
    # Convert string labels to integers for storage
    label_to_int = {label: i for i, label in enumerate(sorted(label_counts.keys()))}
    y_int = np.array([label_to_int[label] for label in y])
    
    print(f"\n  KDDFULL:")
    print(f"    Total: {len(X_norm):,} points")
    print(f"    Inliers: {n_inliers:,} ({n_inliers/len(X_norm)*100:.2f}%)")
    print(f"    Outliers: {n_outliers:,} ({n_outliers/len(X_norm)*100:.2f}%)")
    
    # Save
    out_path = output_dir / "KDDFULL.npz"
    np.savez_compressed(out_path,
                       data=X_norm,
                       labels=y_int,
                       label_names=np.array(list(label_to_int.keys())),
                       outlier_mask=outlier_mask,
                       k=3)
    print(f"    Saved: {out_path}")
    
    return {
        "KDDFULL": {
            "data": X_norm,
            "labels": y_int,
            "outlier_mask": outlier_mask,
            "k": 3,
            "n_outliers": n_outliers
        }
    }


def print_summary(results: dict):
    """Print summary of all processed datasets."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Points':>12} {'Features':>10} {'Outliers':>12} {'Outlier%':>10} {'k':>4}")
    print("-" * 70)
    
    for name, info in results.items():
        if "data" in info:
            n_points = len(info["data"])
            n_features = info["data"].shape[1]
            n_outliers = info.get("n_outliers", info.get("outlier_mask", np.array([])).sum())
            outlier_pct = n_outliers / n_points * 100 if n_points > 0 else 0
            k = info.get("k", "?")
            print(f"{name:<12} {n_points:>12,} {n_features:>10} {n_outliers:>12,} {outlier_pct:>9.2f}% {k:>4}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess clustering datasets with outliers")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent,
                       help="Directory containing raw datasets")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: data-dir/processed)")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["skin", "susy", "shuttle", "kddfull", "all"],
                       default=["all"],
                       help="Datasets to process")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir or (data_dir / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    datasets_to_process = args.datasets
    if "all" in datasets_to_process:
        datasets_to_process = ["skin", "susy", "shuttle", "kddfull"]
    
    all_results = {}
    
    if "skin" in datasets_to_process:
        results = process_skin(data_dir, output_dir)
        all_results.update(results)
    
    if "susy" in datasets_to_process:
        results = process_susy(data_dir, output_dir)
        all_results.update(results)
    
    if "shuttle" in datasets_to_process:
        results = process_shuttle(data_dir, output_dir)
        all_results.update(results)
    
    if "kddfull" in datasets_to_process:
        results = process_kddfull(data_dir, output_dir)
        all_results.update(results)
    
    # Print summary
    summary_results = {k: v for k, v in all_results.items() if k != "original"}
    print_summary(summary_results)
    
    print(f"\nAll processed datasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
