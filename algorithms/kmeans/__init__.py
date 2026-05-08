"""
K-Means clustering algorithms with outlier handling.

This module contains implementations of:
- KMeans++ (Lloyd's algorithm with D2 initialization)
- KMeans-- (robust k-means with outliers)
- NKMeans (neighborhood-based outlier detection)
- Coreset construction for scalable clustering
- IKMeans / TIKMeans (Fast-Sampling algorithms from Huang et al. ICML 2024)
- OKMeans / OKMeans2 (KNN-based outlier detection)
- RobustKMeanspp (robust k-means++ with outlier handling)

OPTIMIZED: All algorithms use vectorized numpy operations for speed.
"""

from .base import (
    BaseAlgorithm,
    InitialCentersUAR,
    InitialCentersD2,
    Random,
    outlier_lloyd,
)

# Vectorized utilities
from .vectorized import (
    compute_distances_to_centers_vectorized,
    compute_pairwise_distances_vectorized,
    assign_to_clusters_vectorized,
    d2_sampling_vectorized,
    lloyd_iteration_vectorized,
    outlier_lloyd_vectorized,
    knn_distances_vectorized,
    knn_distances_faiss,
    FAISS_AVAILABLE,
)
from .kmeanspp import KMeanspp
from .kmeansmm import KMeansmm
from .nkmeans import NKMeans, BallGrid, NKMeansFAISS
from .coreset import (
    CoresetConfig,
    sample_coreset,
    uniform_sample_coreset,
    run_on_coreset,
    CoresetNKMeans,
    CoresetOKMeans,
    CoresetOKMeans2,
    CoresetKMeanspp,
    # Coreset + FAISS algorithms
    CoresetOKMeansFAISS,
    CoresetOKMeans2FAISS,
    CoresetNKMeansFAISS,
    CoresetOKMeansBadFAISS,
    DATASET_CORESET_SIZES,
)
from .robust_kmeanspp import RobustKMeanspp
from .fast_sampling import (
    IKMeans,
    TIKMeans,
    fast_sampling,
    ose,
    center_reduction,
)
from .knn_outlier import (
    OKMeans,
    OKMeans2,
    OKMeansFAISS,
    OKMeans2FAISS,
    OKMeansBad,
    OKMeansBadFAISS,
)

__all__ = [
    # Base classes and utilities
    'BaseAlgorithm',
    'InitialCentersUAR',
    'InitialCentersD2',
    'Random',
    'outlier_lloyd',
    # Algorithms
    'KMeanspp',
    'KMeansmm',
    'NKMeans',
    'BallGrid',
    'CoresetConfig',
    'sample_coreset',
    'uniform_sample_coreset',
    'run_on_coreset',
    'CoresetNKMeans',
    'CoresetOKMeans',
    'CoresetOKMeans2',
    'CoresetKMeanspp',
    'CoresetOKMeansFAISS',
    'CoresetOKMeans2FAISS',
    'CoresetNKMeansFAISS',
    'CoresetOKMeansBadFAISS',
    'DATASET_CORESET_SIZES',
    'RobustKMeanspp',
    # Fast-Sampling algorithms (Huang et al. ICML 2024) - BLAS-optimized
    'IKMeans',
    'TIKMeans',
    'fast_sampling',
    'ose',
    'center_reduction',
    # KNN-based outlier detection
    'OKMeans',
    'OKMeans2',
    'OKMeansFAISS',
    'OKMeans2FAISS',
    'OKMeansBad',
    'OKMeansBadFAISS',
    # FAISS utilities
    'knn_distances_faiss',
    'FAISS_AVAILABLE',
    'NKMeansFAISS',
]
