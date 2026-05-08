"""
Instance class representing a clustering problem instance.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import math

from .defs import Point, Distance, Metric, INF, METRICS
from .dataset import Dataset


class Instance:
    """
    Represents a clustering problem instance.
    
    Contains:
    - Reference to the dataset
    - Number of clusters (K)
    - Number of outliers (Z) or outlier percentage
    - Distance metric
    """
    
    def __init__(self, data: Dataset, metric: str, K: int, 
                 Z: Optional[int] = None, Z_perc: Optional[float] = None):
        self.data = data
        self.K = K
        self.metric = metric
        self.metric_fn = METRICS[metric]
        
        self.data_points: List[int] = list(range(data.size))
        
        self.Z_perc = Z_perc
        if Z_perc is not None:
            self.Z = int(math.ceil(Z_perc * data.size))
        else:
            self.Z = Z if Z is not None else 0
        
        z_str = f"{Z_perc:.4f}" if Z_perc is not None else str(self.Z)
        self.codename = f"{data.name_fout}_k{K}_z{z_str}_m{metric}"
    
    def init_data_points(self):
        """Reset data_points to full dataset."""
        self.data_points = list(range(self.data.size))
    
    def minmax_dists(self, max_iters: int = 10000) -> Tuple[Distance, Distance]:
        """Get min/max distances for the dataset."""
        return self.data.minmax_dists(self.metric, self.metric_fn, max_iters)
    
    def opt_guesses(self, p: float = 1, base: float = 2, max_guesses: int = 30) -> List[Distance]:
        """Generate simple geometric sequence of guesses for optimal cost value.
        
        Returns: [base, base^2, base^3, ..., base^max_guesses]
        """
        return [base ** i for i in range(10, max_guesses + 1)]
    
    def dist(self, a: Union[int, Point], b: Union[int, Point, List, np.ndarray]) -> Distance:
        """Compute distance between points."""
        pt_a = self.data.points[a] if isinstance(a, int) else a
        
        if isinstance(b, int):
            return self.metric_fn(pt_a, self.data.points[b])
        if isinstance(b, Point):
            return self.metric_fn(pt_a, b)
        
        d = INF
        for item in b:
            if isinstance(item, int):
                d = min(d, self.metric_fn(pt_a, self.data.points[item]))
            else:
                d = min(d, self.metric_fn(pt_a, item))
        return d
    
    def dist_center(self, a: int, centers: List[Union[int, Point]]) -> int:
        """Find the index of the closest center."""
        pt_a = self.data.points[a]
        best_idx = -1
        best_dist = INF
        
        for i, center in enumerate(centers):
            if isinstance(center, int):
                d = self.metric_fn(pt_a, self.data.points[center])
            else:
                d = self.metric_fn(pt_a, center)
            
            if d < best_dist:
                best_dist = d
                best_idx = i
        
        return best_idx
    
    def to_dict(self) -> dict:
        """Convert instance to dictionary."""
        result = {
            "dataset": self.data.to_dict(),
            "N": len(self.data_points),
            "N_orig": len(self.data.points),
            "K": self.K,
            "Z": self.Z,
            "metric": self.metric,
            "codename": self.codename,
        }
        
        if self.Z_perc is not None:
            result["Z_perc"] = self.Z_perc
        
        return result
