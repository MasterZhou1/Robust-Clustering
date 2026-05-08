"""
K-Means-- algorithm implementation.

OPTIMIZED: Uses vectorized numpy operations for speed.
"""

from typing import List
import numpy as np

from ..defs import Point
from ..instance import Instance
from ..solution import Solution
from ..algorithm import AlgorithmContext
from .base import BaseAlgorithm, InitialCentersD2, _get_data_matrix
from .vectorized import outlier_lloyd_vectorized


class KMeansmm(BaseAlgorithm):
    """
    K-Means-- algorithm (robust k-means with outliers) - Vectorized.
    
    Uses Lloyd's algorithm that excludes the Z furthest points as outliers.
    """
    
    def __init__(self, initial_centers=None, run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.initial_centers = initial_centers or InitialCentersD2
    
    def codename(self) -> str:
        return f"kmeans--[{self.initial_centers.name}]({self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::Kmeans--[{self.initial_centers.name}](seed={self.seed},run_count={self.run_count})"
    
    def get_centers_pt(self, instance: Instance, solution: Solution, 
                       context: AlgorithmContext) -> List[Point]:
        """Compute cluster centers using k-means-- (vectorized)."""
        X = _get_data_matrix(instance)
        
        # Get initial centers
        center_indices = self.initial_centers.centers(instance, context)
        centers_X = np.array([instance.data.points[idx].x for idx in center_indices])
        
        # Run outlier-aware Lloyd's algorithm (vectorized)
        final_centers, outlier_mask, cost = outlier_lloyd_vectorized(
            X, centers_X, instance.Z, max_iters=100
        )
        
        # Convert to Point objects
        result = []
        for c_coords in final_centers:
            pt = Point()
            pt.x = c_coords
            result.append(pt)
        
        return result
