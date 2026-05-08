"""
Basic type definitions and metrics for clustering algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Callable
import numpy as np

# Type aliases
Distance = float
PointCoord = float
PointID = int

# Infinity constant
INF = 1e30


@dataclass
class Point:
    """Represents a point in the dataset."""
    id: PointID = 0
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        if isinstance(self.x, list):
            self.x = np.array(self.x, dtype=np.float64)


# Type for metric function
Metric = Callable[[Point, Point], Distance]


class Metrics:
    """Distance metrics for clustering."""
    
    @staticmethod
    def lp(a: Point, b: Point, p: int) -> Distance:
        """Lp norm distance between two points."""
        return np.sum(np.abs(a.x - b.x) ** p) ** (1.0 / p)
    
    @staticmethod
    def l1(a: Point, b: Point) -> Distance:
        """L1 (Manhattan) distance."""
        return np.sum(np.abs(a.x - b.x))
    
    @staticmethod
    def l2(a: Point, b: Point) -> Distance:
        """L2 (Euclidean) distance."""
        return np.sqrt(np.sum((a.x - b.x) ** 2))
    
    # String constants for metric names
    L1 = "L1"
    L2 = "L2"


# Metric registry
METRICS = {
    Metrics.L1: Metrics.l1,
    Metrics.L2: Metrics.l2,
}
