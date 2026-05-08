"""
Configuration for benchmarking experiments.

All experiment configuration is centralized here to avoid duplication
and make it easy to extend with new datasets/algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "processed"
RESULTS_DIR = BASE_DIR / "experiments" / "results"
ANALYSIS_DIR = BASE_DIR / "analysis"


# ============================================================================
# Defaults - Single source of truth for common values
# ============================================================================

DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4200, 5318]
DEFAULT_METRIC = "L2"
DEFAULT_K = 10
DEFAULT_OUTLIER_PCT = 0.01


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    filename: str
    k: int = DEFAULT_K
    outlier_percentage: float = DEFAULT_OUTLIER_PCT
    # Resource estimates
    memory_gb: float = 1.0  # Base memory estimate
    base_time_min: float = 0.5  # Base time estimate for fast algorithms

    @property
    def name(self) -> str:
        """Dataset name derived from filename."""
        return self.filename.replace(".npz", "")


def _dataset(filename: str, k: int = DEFAULT_K, outlier_pct: float = DEFAULT_OUTLIER_PCT,
             memory_gb: float = 1.0, base_time: float = 0.5) -> DatasetConfig:
    """Factory function for cleaner dataset definitions."""
    return DatasetConfig(filename, k, outlier_pct, memory_gb, base_time)


# Dataset configurations - add new datasets here
DATASETS: Dict[str, DatasetConfig] = {
    "SKIN-5": _dataset("SKIN-5.npz", k=10, memory_gb=0.5, base_time=0.5),
    "SKIN-10": _dataset("SKIN-10.npz", k=10, memory_gb=0.5, base_time=0.5),
    "SUSY-5": _dataset("SUSY-5.npz", k=10, memory_gb=4.0, base_time=2.0),
    "SUSY-10": _dataset("SUSY-10.npz", k=10, memory_gb=4.0, base_time=2.0),
    "SHUTTLE": _dataset("SHUTTLE.npz", k=10, outlier_pct=0.0004, memory_gb=0.1, base_time=0.1),
    "KDDFULL": _dataset("KDDFULL.npz", k=3, outlier_pct=0.0093, memory_gb=8.0, base_time=3.0),
}


# ============================================================================
# Algorithm Configuration
# ============================================================================

@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm."""
    class_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = DEFAULT_TIMEOUT
    max_n: Optional[int] = None  # Max dataset size (None = no limit)
    # Resource estimates
    memory_mult: float = 1.5  # Memory multiplier relative to dataset
    complexity: float = 2.0  # Relative time complexity

    @property
    def name(self) -> str:
        """Algorithm name (same as class_name for display)."""
        return self.class_name


def _algo(class_name: str, memory_mult: float = 1.5, complexity: float = 2.0,
          timeout: float = DEFAULT_TIMEOUT, **params) -> AlgorithmConfig:
    """Factory function for cleaner algorithm definitions."""
    # Always include run_count=1 for benchmarking
    params.setdefault("run_count", 1)
    return AlgorithmConfig(
        class_name=class_name,
        params=params,
        timeout_seconds=timeout,
        memory_mult=memory_mult,
        complexity=complexity,
    )


# Algorithm configurations - add new algorithms here
# Memory multipliers: 1.0=minimal, 2.0=moderate, 3.0=high (KNN structures)
# Complexity: 1.0=fast, 2.0=moderate, 5.0=slow
ALGORITHMS: Dict[str, AlgorithmConfig] = {
    # Fast-Sampling algorithms (Huang et al. ICML 2024)
    "TIKMeans": _algo("TIKMeans", memory_mult=1.2, complexity=1.0,
                      epsilon=0.5, eta=0.5, beta=1.5),
    "IKMeans": _algo("IKMeans", memory_mult=1.2, complexity=1.5,
                     epsilon=0.5, eta=0.5, beta=1.5, samples_per_iter=5),
    
    # Robust k-means++
    "RobustKmeans++": _algo("RobustKMeanspp", memory_mult=1.5, complexity=2.0,
                            beta=0.1, delta=0.2),
    
    # KNN-based outlier detection (use coreset for scalability)
    "OKMeans": _algo("CoresetOKMeans", memory_mult=2.5, complexity=4.0,
                         neighbor_mult=2),
    "OKMeans2": _algo("CoresetOKMeans2", memory_mult=2.5, complexity=4.0,
                          c=3),
    # FAISS-accelerated KNN outlier detection
    "OKMeansFAISS": _algo("CoresetOKMeansFAISS", memory_mult=2.5, complexity=4.0,
                              neighbor_mult=2),
    # FAISS-accelerated KNN outlier detection
    "OKMeans2FAISS": _algo("CoresetOKMeans2FAISS", memory_mult=2.5, complexity=4.0,
                              c=3),
    
    # NKMeans with coreset + FAISS
    "NKMeans": _algo("CoresetNKMeansFAISS", memory_mult=2.0, complexity=5.0,
                     iters=3, guess_base=2.0),
    
    # K-Means variants
    "KMeans++": _algo("CoresetKMeanspp", memory_mult=1.1, complexity=1.0),
    "KMeans--": _algo("KMeansmm", memory_mult=1.5, complexity=2.0),
}


# ============================================================================
# Algorithm Registry - Dynamic class lookup
# ============================================================================

_ALGORITHM_CLASSES: Dict[str, Type] = None


def get_algorithm_classes() -> Dict[str, Type]:
    """
    Get mapping of class names to algorithm classes.
    
    Lazy-loaded to avoid import issues. All algorithm classes are
    imported here once, eliminating duplication in benchmark files.
    """
    global _ALGORITHM_CLASSES
    if _ALGORITHM_CLASSES is not None:
        return _ALGORITHM_CLASSES
    
    from algorithms.kmeans import (
        TIKMeans, IKMeans, RobustKMeanspp, NKMeans, KMeanspp,
        KMeansmm, OKMeans, OKMeans2,
        CoresetNKMeans, CoresetOKMeans, CoresetOKMeans2, CoresetKMeanspp,
        OKMeansFAISS, OKMeans2FAISS, NKMeansFAISS,
        CoresetOKMeansFAISS, CoresetOKMeans2FAISS, CoresetNKMeansFAISS,
        OKMeansBad, OKMeansBadFAISS, CoresetOKMeansBadFAISS,
    )
    
    _ALGORITHM_CLASSES = {
        # Fast-Sampling algorithms
        "TIKMeans": TIKMeans,
        "IKMeans": IKMeans,
        # Robust variants
        "RobustKMeanspp": RobustKMeanspp,
        # Base algorithms
        "NKMeans": NKMeans,
        "KMeanspp": KMeanspp,
        "KMeansmm": KMeansmm,
        "OKMeans": OKMeans,
        "OKMeans2": OKMeans2,
        # Coreset-based versions
        "CoresetNKMeans": CoresetNKMeans,
        "CoresetOKMeans": CoresetOKMeans,
        "CoresetOKMeans2": CoresetOKMeans2,
        "CoresetKMeanspp": CoresetKMeanspp,
        # FAISS-based implementations
        "OKMeansFAISS": OKMeansFAISS,
        "OKMeans2FAISS": OKMeans2FAISS,
        "NKMeansFAISS": NKMeansFAISS,
        # FAISS + Coreset
        "CoresetOKMeansFAISS": CoresetOKMeansFAISS,
        "CoresetOKMeans2FAISS": CoresetOKMeans2FAISS,
        "CoresetNKMeansFAISS": CoresetNKMeansFAISS,
        # Bad baseline (fixed n_neighbors=10)
        "OKMeansBad": OKMeansBad,
        "OKMeansBadFAISS": OKMeansBadFAISS,
        "CoresetOKMeansBadFAISS": CoresetOKMeansBadFAISS,
    }
    return _ALGORITHM_CLASSES


def get_algorithm(config: AlgorithmConfig, seed: int):
    """
    Create an algorithm instance from configuration.
    
    This is the single place where algorithms are instantiated,
    eliminating duplication between benchmark.py and parallel_benchmark.py.
    """
    classes = get_algorithm_classes()
    cls = classes.get(config.class_name)
    if cls is None:
        raise ValueError(f"Unknown algorithm class: {config.class_name}")
    
    params = config.params.copy()
    params["seed"] = seed
    return cls(**params)


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    timeout: float = DEFAULT_TIMEOUT
    metric: str = DEFAULT_METRIC
    save_intermediate: bool = True
    skip_on_timeout: bool = True


# ============================================================================
# Resource Estimation
# ============================================================================

def estimate_memory_gb(dataset_name: str, algorithm_name: str) -> float:
    """Estimate memory needed for a (dataset, algorithm) pair."""
    dataset = DATASETS.get(dataset_name)
    algorithm = ALGORITHMS.get(algorithm_name)
    
    base_mem = dataset.memory_gb if dataset else 4.0
    multiplier = algorithm.memory_mult if algorithm else 1.5
    overhead = 2.0  # Python/NumPy overhead
    
    return base_mem * multiplier + overhead


def estimate_time_minutes(dataset_name: str, algorithm_name: str) -> float:
    """Estimate time for a single run in minutes."""
    dataset = DATASETS.get(dataset_name)
    algorithm = ALGORITHMS.get(algorithm_name)
    
    base_time = dataset.base_time_min if dataset else 1.0
    complexity = algorithm.complexity if algorithm else 2.0
    
    return base_time * complexity


def calculate_optimal_resources(
    datasets: List[str] = None,
    algorithms: List[str] = None,
    num_seeds: int = None,
    target_parallelism: int = 8
) -> Dict[str, Any]:
    """Calculate optimal SLURM resource allocation."""
    if datasets is None:
        datasets = list(DATASETS.keys())
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    if num_seeds is None:
        num_seeds = len(DEFAULT_SEEDS)
    
    total_tasks = len(datasets) * len(algorithms) * num_seeds
    
    # Find max memory needed
    max_memory = max(
        estimate_memory_gb(d, a)
        for d in datasets
        for a in algorithms
    )
    
    # Estimate total time
    total_time_minutes = sum(
        estimate_time_minutes(d, a) * num_seeds
        for d in datasets
        for a in algorithms
    )
    
    # Calculate resources
    total_memory = max_memory * target_parallelism
    cpus = target_parallelism * 2  # 2 threads per worker
    parallel_time_minutes = total_time_minutes / target_parallelism
    wall_time_minutes = parallel_time_minutes * 1.5  # 50% buffer
    
    # Job array configuration
    tasks_per_job = 50
    num_jobs = (total_tasks + tasks_per_job - 1) // tasks_per_job
    
    return {
        "total_tasks": total_tasks,
        "recommended": {
            "cpus_per_task": cpus,
            "memory_gb": int(total_memory + 8),
            "wall_time_minutes": int(wall_time_minutes),
            "num_workers": target_parallelism,
        },
        "job_array": {
            "num_jobs": num_jobs,
            "tasks_per_job": tasks_per_job,
            "array_spec": f"0-{num_jobs - 1}",
        },
        "estimates": {
            "max_memory_per_task_gb": max_memory,
            "total_time_sequential_minutes": total_time_minutes,
            "total_time_parallel_minutes": parallel_time_minutes,
        },
    }


def print_resource_recommendation(
    datasets: List[str] = None,
    algorithms: List[str] = None,
    num_seeds: int = None,
    parallelism: int = 8
):
    """Print recommended SLURM resources."""
    rec = calculate_optimal_resources(datasets, algorithms, num_seeds, parallelism)
    
    print("=" * 60)
    print("RECOMMENDED SLURM RESOURCES")
    print("=" * 60)
    print(f"Total tasks: {rec['total_tasks']}")
    print()
    print("Single Job:")
    r = rec['recommended']
    print(f"  #SBATCH --cpus-per-task={r['cpus_per_task']}")
    print(f"  #SBATCH --mem={r['memory_gb']}G")
    print(f"  #SBATCH --time={r['wall_time_minutes']//60}:{r['wall_time_minutes']%60:02d}:00")
    print(f"  Workers: {r['num_workers']}")
    print()
    print("Job Array:")
    j = rec['job_array']
    print(f"  #SBATCH --array={j['array_spec']}")
    print(f"  Tasks per job: {j['tasks_per_job']}")
    print()
    print("Estimates:")
    e = rec['estimates']
    print(f"  Max memory/task: {e['max_memory_per_task_gb']:.1f} GB")
    print(f"  Sequential: {e['total_time_sequential_minutes']:.0f} min")
    print(f"  Parallel ({parallelism}x): {e['total_time_parallel_minutes']:.0f} min")
    print("=" * 60)
