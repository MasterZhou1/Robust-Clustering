"""
Coreset construction for scalable robust clustering.

Builds a weighted subset of points that approximates the full dataset
for efficient clustering with outliers.

Two methods available:
1. SAMPLECORESET from Im et al. 2020 - probability-based sampling
2. Simple uniform sampling with fixed coreset size - used for FAISS algorithms
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import math

from ..defs import Point, Distance, INF
from ..instance import Instance
from ..solution import Solution
from ..algorithm import AlgorithmContext
from .base import BaseAlgorithm, _get_data_matrix
from .kmeanspp import KMeanspp


# Default coreset sizes for each dataset (uniform sampling)
# Outliers are proportionally reduced based on sampling ratio
DATASET_CORESET_SIZES: Dict[str, int] = {
    "SKIN": 24000,
    "SKIN-5": 24000,
    "SKIN-10": 24000,
    "SUSY": 70000,
    "SUSY-5": 70000,
    "SUSY-10": 70000,
    "KDDFULL": 70000,
    "SHUTTLE": 10000,
}

# Default coreset size if dataset not in the map
DEFAULT_CORESET_SIZE = 20000


@dataclass
class CoresetConfig:
    """Configuration for coreset construction."""
    seed: int = 42
    iters: int = -1  # -1 for automatic
    k_mult: int = 32  # Multiplier for center count


def uniform_sample_coreset(instance: Instance, coreset_size: int = None, 
                           seed: int = 42) -> Tuple[Instance, float]:
    """
    Simple uniform sampling to create a coreset.
    
    Samples points uniformly at random to create a smaller representative set.
    Outlier count is proportionally reduced based on the sampling ratio.
    
    Args:
        instance: Original clustering instance
        coreset_size: Target coreset size (if None, uses DATASET_CORESET_SIZES)
        seed: Random seed
    
    Returns:
        Tuple of (coreset_instance, sampling_ratio)
        - coreset_instance: Instance with sampled points
        - sampling_ratio: Fraction of points sampled (coreset_size / n)
    """
    n = len(instance.data_points)
    k = instance.K
    
    # Determine coreset size
    if coreset_size is None:
        # Try to find in dataset-specific sizes
        dataset_name = getattr(instance.data, 'name', '') or ''
        for key, size in DATASET_CORESET_SIZES.items():
            if key.upper() in dataset_name.upper():
                coreset_size = size
                break
        if coreset_size is None:
            coreset_size = DEFAULT_CORESET_SIZE
    
    # If dataset is smaller than coreset size, use all points
    if n <= coreset_size:
        coreset_instance = Instance(
            instance.data, instance.metric,
            K=k,
            Z=instance.Z
        )
        coreset_instance.data_points = instance.data_points.copy()
        return coreset_instance, 1.0
    
    # Compute sampling ratio
    sampling_ratio = coreset_size / n
    
    # Sample points uniformly
    rng = np.random.default_rng(seed)
    sampled_indices_idx = rng.choice(n, size=coreset_size, replace=False)
    sampled_indices = [instance.data_points[i] for i in sampled_indices_idx]
    
    # Proportionally reduce outlier count
    original_z = instance.Z
    coreset_z = max(1, int(round(original_z * sampling_ratio)))
    
    # Create coreset instance
    coreset_instance = Instance(
        instance.data, instance.metric,
        K=k,
        Z=coreset_z
    )
    coreset_instance.data_points = sampled_indices
    
    # Also store Z_perc if available
    if instance.Z_perc is not None:
        coreset_instance.Z_perc = instance.Z_perc
    
    return coreset_instance, sampling_ratio


def sample_coreset(instance: Instance, seed: int = 42, 
                   max_coreset_size: int = 50000) -> Tuple[Instance, float]:
    """
    Construct a sample coreset for k-means with outliers.
    
    Uses the practical implementation from Im et al. 2020 (Section 5):
    - Sample each point independently with probability p = min(2.5k*log(n)/z, 1)
    - Run k-means++ on the sample to choose k + pz centers
    - Resulting coreset is of size k + pz
    
    Additionally, if n > max_coreset_size and p >= 1, we force sampling
    to ensure the coreset is at most max_coreset_size points.
    
    Args:
        instance: Original clustering instance
        seed: Random seed
        max_coreset_size: Maximum coreset size (forces sampling if exceeded)
    
    Returns:
        Tuple of (coreset_instance, sampling_probability)
        - coreset_instance: Instance with sampled points (or original if p >= 1)
        - sampling_probability: The probability p used for sampling
    """
    n = len(instance.data_points)
    k = instance.K
    z = max(instance.Z, 1)  # Avoid division by zero
    
    # Practical sampling probability (Section 5 of Im et al. 2020)
    # p = min(2.5k * log(n) / z, 1)
    p = min(2.5 * k * math.log(max(n, 2)) / z, 1.0)
    
    # Force sampling if dataset is too large even when p >= 1
    if n > max_coreset_size and p >= 1:
        p = max_coreset_size / n  # Force sample to max_coreset_size
    
    rng = np.random.default_rng(seed)
    
    if p >= 1:
        # Use all points - no sampling needed
        # For coreset k-means++, use k + pz = k + z centers (since p=1)
        coreset_k = min(k + z, n, 500)
        # Use Z_perc if available to maintain outlier fraction, matching kclustering-with-fair-outliers
        if instance.Z_perc is not None:
            coreset_instance = Instance(
                instance.data, instance.metric, 
                K=coreset_k,
                Z_perc=instance.Z_perc  # Maintain same outlier fraction
            )
        else:
            coreset_instance = Instance(
                instance.data, instance.metric, 
                K=coreset_k,
                Z=z
            )
        coreset_instance.data_points = instance.data_points.copy()
        # Adjust Z based on actual coreset size (data_points), not full dataset size
        if instance.Z_perc is not None:
            coreset_instance.Z = max(1, int(math.ceil(instance.Z_perc * len(coreset_instance.data_points))))
        return coreset_instance, p
    else:
        # Sample points with probability p
        sampled_mask = rng.random(n) <= p
        sampled_indices = [instance.data_points[i] for i in range(n) if sampled_mask[i]]
        
        # Ensure at least some points are sampled
        if len(sampled_indices) < max(k, 10):
            # Force sample at least k points
            extra_indices = rng.choice(n, size=max(k, 10), replace=False)
            for idx in extra_indices:
                pt = instance.data_points[idx]
                if pt not in sampled_indices:
                    sampled_indices.append(pt)
        
        coreset_size = len(sampled_indices)
        
        # Practical implementation: k + pz centers (Section 5 of paper)
        coreset_pz = int(math.ceil(p * z))
        coreset_k = min(k + coreset_pz, coreset_size, 500)
        
        # Use Z_perc if available to maintain outlier fraction (matching kclustering-with-fair-outliers),
        # otherwise scale absolute Z by sampling probability
        if instance.Z_perc is not None:
            # Maintain same outlier fraction as original instance
            coreset_instance = Instance(
                instance.data, instance.metric,
                K=coreset_k,
                Z_perc=instance.Z_perc  # Maintain same outlier fraction
            )
            coreset_instance.data_points = sampled_indices
            # Adjust Z based on actual coreset size (data_points), not full dataset size
            coreset_instance.Z = max(1, int(math.ceil(instance.Z_perc * coreset_size)))
        else:
            # Scale absolute Z by sampling probability (fallback if Z_perc not available)
            coreset_z = max(1, int(math.ceil(p * z)))
            coreset_instance = Instance(
                instance.data, instance.metric,
                K=coreset_k,
                Z=min(coreset_z, coreset_size - 1)
            )
            coreset_instance.data_points = sampled_indices
        
        return coreset_instance, p


def run_on_coreset(instance: Instance, algorithm, seed: int = 42, 
                   coreset_size: int = None, use_uniform: bool = True) -> Solution:
    """
    Run an algorithm on a coreset for faster execution.
    
    1. Build coreset using uniform sampling (or sample_coreset)
    2. Run algorithm on coreset to get initial centers
    3. Use those centers on the full dataset
    
    Args:
        instance: Original clustering instance
        algorithm: Algorithm to run on coreset
        seed: Random seed
        coreset_size: Target coreset size (for uniform sampling)
        use_uniform: If True, use simple uniform sampling; else use sample_coreset
    
    Returns:
        Solution for the original instance
    """
    # Build coreset
    if use_uniform:
        coreset_inst, p = uniform_sample_coreset(instance, coreset_size, seed)
    else:
        coreset_inst, p = sample_coreset(instance, seed)
    
    # Run algorithm on coreset
    context = AlgorithmContext(seed)
    coreset_sol = Solution(coreset_inst)
    
    # Get centers from coreset - try different methods
    try:
        # First try get_solution which handles the full workflow
        algorithm.get_solution(coreset_inst, coreset_sol, context)
        centers_pt = coreset_sol.centers_pt if coreset_sol.centers_pt else []
        
        # If no centers_pt, try to construct from centers
        if not centers_pt and coreset_sol.centers:
            centers_pt = []
            for c_idx in coreset_sol.centers:
                pt = Point()
                pt.x = coreset_inst.data.points[c_idx].x.copy()
                centers_pt.append(pt)
    except NotImplementedError:
        # Fall back to get_centers_pt if available
        if hasattr(algorithm, 'get_centers_pt'):
            centers_pt = algorithm.get_centers_pt(coreset_inst, coreset_sol, context)
        else:
            raise
    
    # Now cluster the full dataset using these centers
    # Select top-k centers if we have more
    if len(centers_pt) > instance.K:
        # Use k-means++ style selection on centers
        from .vectorized import d2_sampling_vectorized
        centers_X = np.array([c.x for c in centers_pt])
        rng = np.random.default_rng(seed)
        selected_idx = d2_sampling_vectorized(centers_X, instance.K, rng)
        centers_pt = [centers_pt[i] for i in selected_idx]
    
    # Create solution for original instance
    solution = Solution(instance)
    solution.centers_pt = centers_pt
    solution.extra["coreset_size"] = str(len(coreset_inst.data_points))
    solution.extra["sampling_prob"] = f"{p:.4f}"
    # Note: solution.centers (indices) not set; clustering uses centers_pt directly
    
    # Cluster full dataset
    BaseAlgorithm().cluster_colorless(instance, solution)
    
    return solution


class CoresetNKMeans(BaseAlgorithm):
    """
    NKMeans running on a sample coreset for faster execution on large datasets.
    
    Uses Algorithm 2 (SAMPLECORESET) to reduce dataset size before running NKMeans.
    """
    
    def __init__(self, iters: int = 3, guess_base: float = 2.0,
                 run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.iters = iters
        self.guess_base = guess_base
    
    def codename(self) -> str:
        return f"coreset-nkmeans({self.iters},{self.guess_base},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetNKMeans(iters={self.iters},guess_base={self.guess_base},seed={self.seed})"
    
    def runnable(self, instance: Instance) -> bool:
        return instance.Z > 0
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        """Run NKMeans on coreset, then cluster full dataset."""
        from .nkmeans import NKMeans
        
        # Create NKMeans with same parameters
        nkmeans = NKMeans(iters=self.iters, guess_base=self.guess_base, 
                         run_count=1, seed=self.seed)
        
        # Run on coreset
        result = run_on_coreset(instance, nkmeans, self.seed)
        
        # Copy results
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-nkmeans"


class CoresetOKMeans(BaseAlgorithm):
    """
    OKMeans running on a sample coreset for faster execution on large datasets.
    """
    
    def __init__(self, neighbor_mult: float = 2, run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.neighbor_mult = neighbor_mult
    
    def codename(self) -> str:
        return f"coreset-okmeans({self.neighbor_mult},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetOKMeans(neighbor_mult={self.neighbor_mult},seed={self.seed})"
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        """Run OKMeans on coreset, then cluster full dataset."""
        from .knn_outlier import OKMeans
        
        okmeans = OKMeans(neighbor_mult=self.neighbor_mult, run_count=1, seed=self.seed)
        result = run_on_coreset(instance, okmeans, self.seed)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-okmeans"


class CoresetOKMeans2(BaseAlgorithm):
    """
    OKMeans2 running on a sample coreset for faster execution on large datasets.
    """
    
    def __init__(self, c: float = 2, run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.c = c
    
    def codename(self) -> str:
        return f"coreset-okmeans2({self.c},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetOKMeans2(c={self.c},seed={self.seed})"
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        """Run OKMeans2 on coreset, then cluster full dataset."""
        from .knn_outlier import OKMeans2
        
        okmeans2 = OKMeans2(c=self.c, run_count=1, seed=self.seed)
        result = run_on_coreset(instance, okmeans2, self.seed)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-okmeans2"


class CoresetKMeanspp(BaseAlgorithm):
    """
    KMeans++ running on a sample coreset for faster execution on large datasets.
    
    Uses uniform sampling to build a coreset, runs KMeans++ on it to get initial
    centers, then clusters the full dataset using those centers.
    """
    
    def __init__(self, run_count: int = 1, seed: int = 42, coreset_size: int = None):
        super().__init__(run_count, seed)
        self.coreset_size = coreset_size
    
    def codename(self) -> str:
        return f"coreset-kmeanspp({self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetKMeanspp(seed={self.seed})"
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        """Run KMeans++ on coreset, then cluster full dataset."""
        kmeanspp = KMeanspp(run_count=1, seed=self.seed)
        result = run_on_coreset(instance, kmeanspp, self.seed, 
                               coreset_size=self.coreset_size, use_uniform=True)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-kmeanspp"


# =============================================================================
# FAISS-based algorithms with Coreset support
# =============================================================================

class CoresetOKMeansFAISS(BaseAlgorithm):
    """OKMeans with FAISS on a uniform sample coreset."""
    
    def __init__(self, neighbor_mult: float = 2, run_count: int = 1, seed: int = 42,
                 coreset_size: int = None):
        super().__init__(run_count, seed)
        self.neighbor_mult = neighbor_mult
        self.coreset_size = coreset_size
    
    def codename(self) -> str:
        return f"coreset-okmeans-faiss({self.neighbor_mult},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetOKMeansFAISS(neighbor_mult={self.neighbor_mult},seed={self.seed})"
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        from .knn_outlier import OKMeansFAISS
        
        okmeans_faiss = OKMeansFAISS(
            neighbor_mult=self.neighbor_mult, run_count=1, seed=self.seed
        )
        result = run_on_coreset(instance, okmeans_faiss, self.seed, 
                               coreset_size=self.coreset_size, use_uniform=True)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-okmeans-faiss"


class CoresetOKMeans2FAISS(BaseAlgorithm):
    """OKMeans2 with FAISS on a uniform sample coreset."""
    
    def __init__(self, c: float = 2, run_count: int = 1, seed: int = 42,
                 coreset_size: int = None):
        super().__init__(run_count, seed)
        self.c = c
        self.coreset_size = coreset_size
    
    def codename(self) -> str:
        return f"coreset-okmeans2-faiss({self.c},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetOKMeans2FAISS(c={self.c},seed={self.seed})"
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        from .knn_outlier import OKMeans2FAISS
        
        okmeans2_faiss = OKMeans2FAISS(c=self.c, run_count=1, seed=self.seed)
        result = run_on_coreset(instance, okmeans2_faiss, self.seed,
                               coreset_size=self.coreset_size, use_uniform=True)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-okmeans2-faiss"


class CoresetOKMeansBadFAISS(BaseAlgorithm):
    """OKMeansBad (fixed n_neighbors) with FAISS on a uniform sample coreset.
    
    Unlike OKMeans which scales n_neighbors with z, this algorithm uses a fixed
    n_neighbors value, allowing comparison of different fixed neighbor counts.
    """
    
    def __init__(self, n_neighbors: int = 10, run_count: int = 1, seed: int = 42, 
                 coreset_size: int = None):
        super().__init__(run_count, seed)
        self.n_neighbors = n_neighbors
        self.coreset_size = coreset_size
    
    def codename(self) -> str:
        return f"coreset-okmeans-bad-faiss({self.n_neighbors},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetOKMeansBadFAISS(n_neighbors={self.n_neighbors},seed={self.seed})"
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        from .knn_outlier import OKMeansBadFAISS
        
        okmeans_bad_faiss = OKMeansBadFAISS(
            n_neighbors=self.n_neighbors, run_count=1, seed=self.seed
        )
        result = run_on_coreset(instance, okmeans_bad_faiss, self.seed,
                               coreset_size=self.coreset_size, use_uniform=True)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-okmeans-bad-faiss"


class CoresetNKMeansFAISS(BaseAlgorithm):
    """NKMeans with FAISS on a uniform sample coreset."""
    
    def __init__(self, iters: int = 3, guess_base: float = 2.0,
                 run_count: int = 1, seed: int = 42, coreset_size: int = None):
        super().__init__(run_count, seed)
        self.iters = iters
        self.guess_base = guess_base
        self.coreset_size = coreset_size
    
    def codename(self) -> str:
        return f"coreset-nkmeans-faiss({self.iters},{self.guess_base},{self.seed})"
    
    def fullname(self) -> str:
        return f"KMeans::CoresetNKMeansFAISS(iters={self.iters},guess_base={self.guess_base},seed={self.seed})"
    
    def runnable(self, instance: Instance) -> bool:
        from .vectorized import FAISS_AVAILABLE
        return FAISS_AVAILABLE and instance.Z > 0
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        from .nkmeans import NKMeansFAISS
        
        nkmeans_faiss = NKMeansFAISS(
            iters=self.iters, guess_base=self.guess_base, run_count=1, seed=self.seed
        )
        result = run_on_coreset(instance, nkmeans_faiss, self.seed,
                               coreset_size=self.coreset_size, use_uniform=True)
        
        solution.centers = result.centers
        solution.centers_pt = result.centers_pt
        solution.clusters = result.clusters
        solution.outliers = result.outliers
        solution.cost = result.cost
        solution.outliers_cost = result.outliers_cost
        solution.extra = result.extra
        solution.extra["algorithm"] = "coreset-nkmeans-faiss"
