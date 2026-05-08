"""
Base classes and utilities for K-Means algorithms.

OPTIMIZED: Uses vectorized numpy operations for speed.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import heapq

from ..defs import Point, Distance, INF
from ..instance import Instance
from ..solution import Solution
from ..algorithm import Algorithm, AlgorithmContext
from .vectorized import (
    compute_distances_to_centers_vectorized,
    assign_to_clusters_vectorized,
    d2_sampling_vectorized,
    lloyd_iteration_vectorized,
    outlier_lloyd_vectorized,
)


def _get_data_matrix(instance: Instance) -> np.ndarray:
    """Extract data matrix from instance."""
    return np.array([instance.data.points[idx].x for idx in instance.data_points])


class InitialCentersUAR:
    """Uniform At Random center initialization."""
    
    name = "UAR"
    
    @staticmethod
    def centers(instance: Instance, context: AlgorithmContext) -> List[int]:
        """Select K random points as initial centers."""
        n = len(instance.data_points)
        k = min(instance.K, n)
        
        indices = context.generator.choice(n, size=k, replace=False)
        return [instance.data_points[i] for i in indices]


class InitialCentersD2:
    """D-squared (k-means++) center initialization - Vectorized."""
    
    name = "D2"
    
    @staticmethod
    def centers(instance: Instance, context: AlgorithmContext) -> List[int]:
        """Select K centers using D2 sampling (kmeans++ initialization)."""
        X = _get_data_matrix(instance)
        n = len(instance.data_points)
        k = min(instance.K, n)
        
        # Use vectorized D2 sampling
        selected_indices = d2_sampling_vectorized(X, k, context.generator)
        
        return [instance.data_points[i] for i in selected_indices]


class BaseAlgorithm(Algorithm):
    """
    Base class for K-Means variants.
    
    Provides common functionality with vectorized operations.
    """
    
    def problem(self) -> str:
        return "kmeans"
    
    def codename(self) -> str:
        return "BaseAlgorithm"
    
    def fullname(self) -> str:
        return "BaseAlgorithm"
    
    def get_centers_pt(self, instance: Instance, solution: Solution, 
                       context: AlgorithmContext) -> List[Point]:
        """Compute centers as Point objects. Override in subclasses."""
        raise NotImplementedError("Subclass must implement get_centers_pt")
    
    def get_centers(self, instance: Instance, solution: Solution, 
                    context: AlgorithmContext):
        """Compute centers as Point objects."""
        centers_pt = self.get_centers_pt(instance, solution, context)
        solution.centers_pt = centers_pt
        # Note: solution.centers (indices) not set here; clustering uses centers_pt directly
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        """Compute full clustering solution."""
        self.get_centers(instance, solution, context)
        self.cluster_colorless(instance, solution)
    
    def cluster_colorless(self, instance: Instance, solution: Solution):
        """
        Assign points to clusters (colorless/standard mode) - Vectorized.
        
        The Z furthest points become outliers.
        """
        X = _get_data_matrix(instance)
        n = len(instance.data_points)
        
        # Get centers as array
        if solution.centers_pt:
            centers_X = np.array([c.x for c in solution.centers_pt])
        else:
            centers_X = np.array([instance.data.points[c].x for c in solution.centers])
        
        # Compute assignments and distances (vectorized)
        assignments, dists_sq = assign_to_clusters_vectorized(X, centers_X)
        
        # Get distances (not squared) for sorting
        dists = np.sqrt(dists_sq)
        
        # Find Z furthest points as outliers
        z = instance.Z
        sorted_indices = np.argsort(-dists)  # Descending
        outlier_indices = set(sorted_indices[:z].tolist())
        
        # Build clusters and compute cost
        n_centers = len(centers_X)
        solution.clusters = [[] for _ in range(n_centers)]
        solution.cost = 0
        solution.outliers_cost = 0
        solution.outliers = []
        
        for i in range(n):
            pt_idx = instance.data_points[i]
            c_idx = assignments[i]
            d = dists[i]
            
            if i in outlier_indices:
                solution.outliers.append(pt_idx)
                solution.outliers_cost += d ** 2
            else:
                solution.clusters[c_idx].append(pt_idx)
                solution.cost += d ** 2
    
    def cost_colorless(self, instance: Instance, centers, 
                       pts: Optional[List[int]] = None, Z: int = 0) -> Distance:
        """Compute clustering cost (colorless mode) - Vectorized."""
        if pts is None:
            pts = instance.data_points
        
        # Get data matrix for specified points
        pt_to_idx = {pt: i for i, pt in enumerate(instance.data_points)}
        indices = [pt_to_idx[pt] for pt in pts]
        
        X = _get_data_matrix(instance)
        X_subset = X[indices]
        
        # Get centers as array
        if isinstance(centers[0], Point):
            centers_X = np.array([c.x for c in centers])
        else:
            centers_X = np.array([instance.data.points[c].x for c in centers])
        
        # Compute distances
        dists_sq = compute_distances_to_centers_vectorized(X_subset, centers_X, squared=True)
        
        # Sort and exclude Z furthest
        sorted_dists = np.sort(dists_sq)
        return float(sorted_dists[:len(pts) - Z].sum())
    
    def cost_auto(self, instance: Instance, centers) -> Distance:
        """Compute cost for clustering."""
        return self.cost_colorless(instance, centers, Z=instance.Z)


class Random(BaseAlgorithm):
    """Random center selection algorithm."""
    
    def codename(self) -> str:
        return f"Rnd({self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::Random(seed={self.seed},run_count={self.run_count})"
    
    def get_centers(self, instance: Instance, solution: Solution, 
                    context: AlgorithmContext):
        """Select random centers."""
        solution.centers = InitialCentersUAR.centers(instance, context)


def outlier_lloyd(instance: Instance, centers: List[Point], 
                  max_iters: int = -1) -> Tuple[List[Point], int]:
    """
    Lloyd's algorithm with outlier handling (k-means-- style) - Vectorized.
    """
    X = _get_data_matrix(instance)
    centers_X = np.array([c.x for c in centers])
    z = instance.Z
    
    if max_iters == -1:
        max_iters = 100
    
    final_centers, outlier_mask, cost = outlier_lloyd_vectorized(
        X, centers_X, z, max_iters
    )
    
    # Convert back to Point objects
    result_centers = []
    for c_coords in final_centers:
        pt = Point()
        pt.x = c_coords
        result_centers.append(pt)
    
    return result_centers, max_iters
