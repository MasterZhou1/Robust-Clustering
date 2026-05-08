"""
Dataset class for loading and managing point data.
"""

from typing import Dict, List, Tuple
import numpy as np

from .defs import Point, Distance, PointID, Metric, INF


class Dataset:
    """
    Represents a dataset of points for clustering.
    
    Supports:
    - Creating from a list of Point objects
    - Min/max distance computation
    """
    
    def __init__(self, points: List[Point]):
        """Initialize dataset from a list of points."""
        assert len(points) > 0, "Points list cannot be empty"
        
        self.name = ""
        self.name_fout = ""
        self.points = points
        self.size = len(points)
        self.dim = len(points[0].x)
        
        self.pt_index: Dict[PointID, int] = {}
        for i, pt in enumerate(points):
            self.pt_index[pt.id] = i
        
        self._minmax_dists: Dict[str, Tuple[Distance, Distance]] = {}
    
    def info(self):
        """Print dataset information."""
        print(f"# {self.name} ({self.name_fout})")
        print(f"Size: {self.size}, dim: {self.dim}")
    
    def prune(self, new_size: int) -> 'Dataset':
        """Reduce dataset to first new_size points."""
        self.size = new_size
        self.points = self.points[:new_size]
        return self
    
    def get_point_id(self, point_index: int) -> PointID:
        """Get point ID from index."""
        return self.points[point_index].id
    
    def get_point_index(self, point_id: PointID) -> int:
        """Get point index from ID."""
        return self.pt_index[point_id]
    
    def minmax_dists(self, metric: str, dist_fn: Metric, max_iters: int) -> Tuple[Distance, Distance]:
        """Compute min and max distances in the dataset."""
        if metric in self._minmax_dists:
            return self._minmax_dists[metric]
        
        assert max_iters != -1, "max_iters must be specified for new computation"
        
        d_min, d_max = INF, -INF
        rng = np.random.default_rng(42)
        
        n_samples = min(self.size, max_iters)
        for _ in range(n_samples):
            u = rng.integers(self.size)
            for v in range(self.size):
                duv = dist_fn(self.points[u], self.points[v])
                if duv == 0:
                    continue
                d_min = min(d_min, duv)
                d_max = max(d_max, duv)
        
        self._minmax_dists[metric] = (d_min, d_max)
        return d_min, d_max
    
    def to_dict(self) -> dict:
        """Convert dataset metadata to dictionary."""
        return {
            "name": self.name,
            "name_fout": self.name_fout,
            "size": self.size,
            "dim": self.dim,
        }
