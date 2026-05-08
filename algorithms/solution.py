"""
Solution class representing clustering results.
"""

from typing import List, Dict, Optional
import json
import numpy as np

from .defs import Point, Distance, INF


class Solution:
    """
    Represents a clustering solution.
    
    Contains:
    - Center points (as indices or Point objects)
    - Cluster assignments
    - Outliers
    - Cost metrics
    """
    
    def __init__(self, instance=None):
        """Initialize an empty solution."""
        self.instance = instance
        self.problem = ""
        self.algo_codename = ""
        self.algo_fullname = ""
        
        self.centers: List[int] = []
        self.centers_pt: List[Point] = []
        
        self.outliers: List[int] = []
        self.original_outliers: List[int] = []
        
        self.cost: Distance = INF
        self.outliers_cost: Distance = 0
        
        self.clusters: List[List[int]] = []
        
        self.extra: Dict[str, str] = {}
        
        self.elapsed_ms: float = -1
    
    def codename(self) -> str:
        """Generate solution codename."""
        if self.instance:
            return f"{self.instance.codename}_{self.algo_codename}"
        return self.algo_codename
    
    def info(self):
        """Print solution information."""
        print(f"{self.codename()} @ {self.algo_codename}[{self.algo_fullname}]")
        if self.instance:
            print(f"\tn={len(self.instance.data.points)},k={self.instance.K},z={self.instance.Z}")
        print(f"\tCost: {self.cost}(outlier cost:{self.outliers_cost}), "
              f"centers:{len(self.centers)}, centers_pt:{len(self.centers_pt)}, "
              f"outliers:{len(self.outliers)} (original outliers: {len(self.original_outliers)})")
        print(f"\tElapsedMs: {self.elapsed_ms}")
    
    def get_centers_pt(self) -> List[Point]:
        """Get centers as Point objects."""
        if self.centers_pt:
            return self.centers_pt
        if self.instance:
            return [self.instance.data.points[c] for c in self.centers]
        return []
    
    def to_dict(self) -> dict:
        """Convert solution to dictionary for serialization."""
        result = {
            "problem": self.problem,
            "algo_codename": self.algo_codename,
            "algo_fullname": self.algo_fullname,
            "elapsed_ms": self.elapsed_ms,
            "cost": self.cost,
            "outliers_cost": self.outliers_cost,
            "extra": self.extra,
        }
        
        if self.instance:
            result["instance"] = self.instance.to_dict()
            result["centers"] = [self.instance.data.get_point_id(c) for c in self.centers]
            result["outliers"] = [self.instance.data.get_point_id(o) for o in self.outliers]
            result["original_outliers"] = [self.instance.data.get_point_id(o) 
                                           for o in self.original_outliers]
        else:
            result["centers"] = self.centers
            result["outliers"] = self.outliers
            result["original_outliers"] = self.original_outliers
        
        centers_pt_list = []
        for pt in self.centers_pt:
            if isinstance(pt.x, np.ndarray):
                centers_pt_list.append(pt.x.tolist())
            else:
                centers_pt_list.append(list(pt.x))
        result["centers_pt"] = centers_pt_list
        result["clusters"] = self.clusters
        
        return result
    
    def save(self, path: str):
        """Save solution to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str, instance=None) -> 'Solution':
        """Load solution from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        sol = cls(instance)
        sol.problem = data.get("problem", "")
        sol.algo_codename = data.get("algo_codename", "")
        sol.algo_fullname = data.get("algo_fullname", "")
        sol.elapsed_ms = data.get("elapsed_ms", -1)
        sol.cost = data.get("cost", INF)
        sol.outliers_cost = data.get("outliers_cost", 0)
        sol.extra = data.get("extra", {})
        
        if instance:
            sol.centers = [instance.data.get_point_index(c) for c in data.get("centers", [])]
            sol.outliers = [instance.data.get_point_index(o) for o in data.get("outliers", [])]
            sol.original_outliers = [instance.data.get_point_index(o) 
                                     for o in data.get("original_outliers", [])]
        else:
            sol.centers = data.get("centers", [])
            sol.outliers = data.get("outliers", [])
            sol.original_outliers = data.get("original_outliers", [])
        
        for pt_coords in data.get("centers_pt", []):
            pt = Point()
            pt.x = np.array(pt_coords)
            sol.centers_pt.append(pt)
        
        sol.clusters = data.get("clusters", [])
        
        return sol
