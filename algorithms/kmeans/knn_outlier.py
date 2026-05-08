"""
KNN-based outlier detection algorithms for k-means with outliers.

Algorithm 1 (OKMeans): Outliers = points with largest 2z-neighbor radius
Algorithm 2 (OKMeans2): Outliers = points with largest sum of neighbor distances
"""

from typing import Set
import numpy as np

from ..instance import Instance
from ..solution import Solution
from ..algorithm import AlgorithmContext
from .base import BaseAlgorithm, _get_data_matrix
from .kmeanspp import KMeanspp
from .vectorized import knn_distances_vectorized, knn_distances_faiss, FAISS_AVAILABLE


class OKMeans(BaseAlgorithm):
    """Algorithm 1: OKMeans - KNN-based outlier detection using 2z-neighbor radius."""
    
    def __init__(self, neighbor_mult: float = 2, run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.neighbor_mult = neighbor_mult
    
    def codename(self) -> str:
        return f"okmeans({self.neighbor_mult},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::OKMeans(neighbor_mult={self.neighbor_mult},seed={self.seed})"
    
    def _compute_2z_radius(self, instance: Instance) -> np.ndarray:
        """Compute 2z-neighbor radius for all points (vectorized)."""
        X = _get_data_matrix(instance)
        n = X.shape[0]
        z = instance.Z
        n_neighbors = int(min(self.neighbor_mult * z, n - 1))
        if n_neighbors < 1:
            n_neighbors = min(n - 1, 1)
        
        knn_dists, _ = knn_distances_vectorized(X, n_neighbors)
        return knn_dists[:, -1]
    
    def _identify_outliers(self, instance: Instance) -> Set[int]:
        """Identify outlier indices."""
        z = instance.Z
        n = len(instance.data_points)
        if z <= 0 or z >= n:
            return set()
        r_x = self._compute_2z_radius(instance)
        outlier_indices = np.argsort(-r_x)[:z]
        return set(outlier_indices.tolist())
    
    def get_centers(self, instance: Instance, solution: Solution, context: AlgorithmContext):
        outlier_indices = self._identify_outliers(instance)
        outlier_pts = [instance.data_points[i] for i in outlier_indices]
        solution.original_outliers = outlier_pts
        
        restricted = Instance(instance.data, instance.metric, instance.K, Z=0)
        restricted.data_points = [instance.data_points[i] for i in range(len(instance.data_points)) if i not in outlier_indices]
        
        kpp = KMeanspp(run_count=self.run_count, seed=self.seed)
        restricted_sol = Solution(restricted)
        kpp.get_solution(restricted, restricted_sol, context)
        
        solution.centers = restricted_sol.centers
        solution.centers_pt = restricted_sol.centers_pt
        solution.extra["outlier_method"] = "2z_radius"
        solution.extra["num_outliers_detected"] = str(len(outlier_indices))
    
    def get_solution(self, instance: Instance, solution: Solution, context: AlgorithmContext):
        self.get_centers(instance, solution, context)
        self.cluster_colorless(instance, solution)

class OKMeansBad(BaseAlgorithm):
    """OKMeans with fixed n_neighbors (doesn't scale with z like OKMeans does)."""
    
    def __init__(self, n_neighbors: int = 10, run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.n_neighbors = n_neighbors
    
    def codename(self) -> str:
        return f"okmeans-bad({self.n_neighbors},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::OKMeansBad(n_neighbors={self.n_neighbors},seed={self.seed})"
    
    def _compute_2z_radius(self, instance: Instance) -> np.ndarray:
        """Compute 2z-neighbor radius with fixed n_neighbors (ignores z)."""
        X = _get_data_matrix(instance)
        n = X.shape[0]
        # Fixed n_neighbors - doesn't scale with z (unlike OKMeans)
        n_neighbors = min(self.n_neighbors, n - 1)
        if n_neighbors < 1:
            n_neighbors = 1
        
        knn_dists, _ = knn_distances_vectorized(X, n_neighbors)
        return knn_dists[:, -1]
    
    def _identify_outliers(self, instance: Instance) -> Set[int]:
        """Identify outlier indices."""
        z = instance.Z
        n = len(instance.data_points)
        if z <= 0 or z >= n:
            return set()
        r_x = self._compute_2z_radius(instance)
        outlier_indices = np.argsort(-r_x)[:z]
        return set(outlier_indices.tolist())
    
    def get_centers(self, instance: Instance, solution: Solution, context: AlgorithmContext):
        outlier_indices = self._identify_outliers(instance)
        outlier_pts = [instance.data_points[i] for i in outlier_indices]
        solution.original_outliers = outlier_pts
        
        restricted = Instance(instance.data, instance.metric, instance.K, Z=0)
        restricted.data_points = [instance.data_points[i] for i in range(len(instance.data_points)) if i not in outlier_indices]
        
        kpp = KMeanspp(run_count=self.run_count, seed=self.seed)
        restricted_sol = Solution(restricted)
        kpp.get_solution(restricted, restricted_sol, context)
        
        solution.centers = restricted_sol.centers
        solution.centers_pt = restricted_sol.centers_pt
        solution.extra["outlier_method"] = f"fixed_{self.n_neighbors}_neighbors"
        solution.extra["num_outliers_detected"] = str(len(outlier_indices))
    
    def get_solution(self, instance: Instance, solution: Solution, context: AlgorithmContext):
        self.get_centers(instance, solution, context)
        self.cluster_colorless(instance, solution)


class OKMeansBadFAISS(OKMeansBad):
    """OKMeansBad using FAISS for fast KNN computation."""
    
    def __init__(self, n_neighbors: int = 10, run_count: int = 1, seed: int = 42):
        super().__init__(n_neighbors, run_count, seed)
    
    def codename(self) -> str:
        return f"okmeans-bad-faiss({self.n_neighbors},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::OKMeansBadFAISS(n_neighbors={self.n_neighbors},seed={self.seed})"
    
    def runnable(self, instance: Instance) -> bool:
        return FAISS_AVAILABLE
    
    def _compute_2z_radius(self, instance: Instance) -> np.ndarray:
        """Compute 2z-neighbor radius with fixed n_neighbors using FAISS."""
        X = _get_data_matrix(instance)
        n = X.shape[0]
        # Fixed n_neighbors - doesn't scale with z (unlike OKMeans)
        n_neighbors = min(self.n_neighbors, n - 1)
        if n_neighbors < 1:
            n_neighbors = 1
        
        knn_dists, _ = knn_distances_faiss(X, n_neighbors)
        return knn_dists[:, -1]
        
        
class OKMeans2(BaseAlgorithm):
    """Algorithm 2: OKMeans2 - KNN-based outlier detection using neighbor distance sum."""
    
    def __init__(self, c: float = 2, run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.c = c
    
    def codename(self) -> str:
        return f"okmeans2({self.c},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::OKMeans2(c={self.c},seed={self.seed})"
    
    def _compute_neighbor_distance_sum(self, instance: Instance) -> np.ndarray:
        """Compute neighbor distance sum for all points (vectorized)."""
        X = _get_data_matrix(instance)
        n = X.shape[0]
        z = instance.Z
        n_neighbors = int(min(self.c * z, n - 1))
        if n_neighbors < 1:
            n_neighbors = min(n - 1, 1)
        
        knn_dists, _ = knn_distances_vectorized(X, n_neighbors)
        
        start_idx = min(z, knn_dists.shape[1] - 1)
        end_idx = int(min(self.c * z, knn_dists.shape[1]))
        
        if start_idx >= end_idx:
            return knn_dists.sum(axis=1)
        return knn_dists[:, start_idx:end_idx].sum(axis=1)
    
    def _identify_outliers(self, instance: Instance) -> Set[int]:
        z = instance.Z
        n = len(instance.data_points)
        if z <= 0 or z >= n:
            return set()
        s_x = self._compute_neighbor_distance_sum(instance)
        outlier_indices = np.argsort(-s_x)[:z]
        return set(outlier_indices.tolist())
    
    def get_centers(self, instance: Instance, solution: Solution, context: AlgorithmContext):
        outlier_indices = self._identify_outliers(instance)
        outlier_pts = [instance.data_points[i] for i in outlier_indices]
        solution.original_outliers = outlier_pts
        
        restricted = Instance(instance.data, instance.metric, instance.K, Z=0)
        restricted.data_points = [instance.data_points[i] for i in range(len(instance.data_points)) if i not in outlier_indices]
        
        kpp = KMeanspp(run_count=self.run_count, seed=self.seed)
        restricted_sol = Solution(restricted)
        kpp.get_solution(restricted, restricted_sol, context)
        
        solution.centers = restricted_sol.centers
        solution.centers_pt = restricted_sol.centers_pt
        solution.extra["outlier_method"] = "neighbor_distance_sum"
        solution.extra["num_outliers_detected"] = str(len(outlier_indices))
    
    def get_solution(self, instance: Instance, solution: Solution, context: AlgorithmContext):
        self.get_centers(instance, solution, context)
        self.cluster_colorless(instance, solution)


class OKMeansFAISS(OKMeans):
    """OKMeans using FAISS for fast KNN computation."""
    
    def __init__(self, neighbor_mult: float = 2, run_count: int = 1, seed: int = 42):
        super().__init__(neighbor_mult, run_count, seed)
    
    def codename(self) -> str:
        return f"okmeans-faiss({self.neighbor_mult},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::OKMeansFAISS(neighbor_mult={self.neighbor_mult},seed={self.seed})"
    
    def runnable(self, instance: Instance) -> bool:
        return FAISS_AVAILABLE
    
    def _compute_2z_radius(self, instance: Instance) -> np.ndarray:
        """Compute 2z-neighbor radius for all points using FAISS."""
        X = _get_data_matrix(instance)
        n = X.shape[0]
        z = instance.Z
        n_neighbors = int(min(self.neighbor_mult * z, n - 1))
        if n_neighbors < 1:
            n_neighbors = min(n - 1, 1)
        
        knn_dists, _ = knn_distances_faiss(X, n_neighbors)
        return knn_dists[:, -1]


class OKMeans2FAISS(OKMeans2):
    """OKMeans2 using FAISS for fast KNN computation."""
    
    def __init__(self, c: float = 2, run_count: int = 1, seed: int = 42):
        super().__init__(c, run_count, seed)
    
    def codename(self) -> str:
        return f"okmeans2-faiss({self.c},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::OKMeans2FAISS(c={self.c},seed={self.seed})"
    
    def runnable(self, instance: Instance) -> bool:
        return FAISS_AVAILABLE
    
    def _compute_neighbor_distance_sum(self, instance: Instance) -> np.ndarray:
        """Compute neighbor distance sum for all points using FAISS."""
        X = _get_data_matrix(instance)
        n = X.shape[0]
        z = instance.Z
        n_neighbors = int(min(self.c * z, n - 1))
        if n_neighbors < 1:
            n_neighbors = min(n - 1, 1)
        
        knn_dists, _ = knn_distances_faiss(X, n_neighbors)
        
        start_idx = min(z, knn_dists.shape[1] - 1)
        end_idx = int(min(self.c * z, knn_dists.shape[1]))
        
        if start_idx >= end_idx:
            return knn_dists.sum(axis=1)
        return knn_dists[:, start_idx:end_idx].sum(axis=1)
