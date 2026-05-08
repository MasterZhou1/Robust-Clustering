"""
NKMeans algorithm implementations.

OPTIMIZED: Uses vectorized numpy operations for speed.
"""

from typing import List, Set, Dict
from collections import defaultdict
import numpy as np
import math

from ..defs import Point, Distance, INF
from ..instance import Instance
from ..solution import Solution
from ..algorithm import AlgorithmContext
from .base import BaseAlgorithm, _get_data_matrix
from .kmeanspp import KMeanspp
from .vectorized import FAISS_AVAILABLE

# FAISS is imported inside NKMeansFAISS class methods to avoid import errors
# when FAISS is not available or has compatibility issues


MAX_BALL_DIM = 10


class BallGrid:
    """Spatial index for efficient ball queries (optimized)."""
    
    def __init__(self, instance: Instance, r: Distance):
        self.instance = instance
        self.r = r
        
        self.dim = min(MAX_BALL_DIM, instance.data.dim)
        self.size = 0
        self.grid: Dict[tuple, List[int]] = defaultdict(list)
        
        # Get data matrix for faster access
        self.X = _get_data_matrix(instance)
        
        for i, pt_idx in enumerate(instance.data_points):
            self.size += 1
            bucket = self._bucket(i)
            self.grid[bucket].append(i)
    
    def _bucket(self, data_idx: int) -> tuple:
        pt_x = self.X[data_idx]
        bucket = []
        for i in range(min(self.dim, len(pt_x))):
            bucket.append(int(math.ceil(pt_x[i] / self.r)))
        return tuple(bucket)
    
    def query(self, data_idx: int, threshold: int) -> int:
        """Count points within radius r of query point."""
        ball_count = 0
        query_x = self.X[data_idx]
        bucket_orig = list(self._bucket(data_idx))
        
        # Generate all neighboring buckets
        def generate_neighbors(dim_idx, current_bucket):
            if dim_idx == self.dim:
                yield tuple(current_bucket)
                return
            
            for delta in [0, -1, 1]:
                current_bucket.append(bucket_orig[dim_idx] + delta)
                yield from generate_neighbors(dim_idx + 1, current_bucket)
                current_bucket.pop()
        
        # Check all neighboring buckets
        for bucket in generate_neighbors(0, []):
            if bucket not in self.grid:
                continue
            
            for j in self.grid[bucket]:
                dist = np.sqrt(np.sum((query_x - self.X[j]) ** 2))
                if dist <= self.r:
                    ball_count += 1
                
                if ball_count >= threshold:
                    return ball_count
        
        return ball_count


class NKMeans(BaseAlgorithm):
    """NKMeans algorithm (neighborhood-based outlier detection) - Optimized."""
    
    def __init__(self, iters: int = 10, guess_base: float = 2.0,
                 run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.iters = iters
        self.guess_base = guess_base
    
    def codename(self) -> str:
        return f"nkmeans({self.iters},{self.guess_base},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::NKMeans(iters={self.iters},guess_base={self.guess_base},seed={self.seed},run_count={self.run_count})"
    
    def runnable(self, instance: Instance) -> bool:
        return instance.Z > 0
    
    def get_centers_pt(self, instance: Instance, solution: Solution, 
                       context: AlgorithmContext) -> List[Point]:
        """Compute centers using neighborhood-based outlier detection."""
        X = _get_data_matrix(instance)
        n = len(instance.data_points)
        
        opt_guesses = instance.opt_guesses(p=2, base=self.guess_base)
        solution.extra["opt_guesses"] = str(len(opt_guesses))
        
        total_iter = 0
        best_centers: List[Point] = []
        best_outliers: Set[int] = set()
        best_cost = INF
        
        for opt in opt_guesses:
            r = 2 * math.sqrt(opt / instance.Z)
            threshold = 2 * instance.Z
            
            # Find heavy points using grid
            grid = BallGrid(instance, r)
            heavy = []
            
            for i in range(n):
                if grid.query(i, threshold) >= threshold:
                    heavy.append(i)
            
            if not heavy:
                continue
            
            # Points not near any heavy point are outliers (vectorized)
            heavy_X = X[heavy]
            
            # Compute distances from all points to heavy points
            outliers: Set[int] = set()
            inliers = 0
            
            for i in range(n):
                # Check if point is near any heavy point
                dists = np.sqrt(np.sum((X[i] - heavy_X) ** 2, axis=1))
                min_dist = dists.min() if len(dists) > 0 else INF
                
                if min_dist > r:
                    outliers.add(i)
                else:
                    inliers += 1
            
            if inliers == 0:
                continue
            if len(outliers) > (3 * instance.K + 2) * instance.Z:
                continue
            
            # Cluster the inliers
            cost, centers, iter_count = self._cluster(
                instance, solution, context, outliers, self.iters, X
            )
            total_iter += iter_count
            
            if cost < best_cost:
                best_cost = cost
                best_centers = centers
                best_outliers = outliers
            
            break
        
        
        solution.extra["iter"] = str(total_iter)
        solution.original_outliers = [instance.data_points[i] for i in best_outliers]
        return best_centers
    
    def _cluster(self, instance: Instance, solution: Solution, 
                 context: AlgorithmContext, outliers: Set[int], 
                 cluster_iters: int, X: np.ndarray):
        """Cluster inliers."""
        best_cost = INF
        best_centers: List[Point] = []
        
        # Create restricted instance
        restricted = Instance(instance.data, instance.metric, instance.K, Z=0)
        restricted.data_points = [
            instance.data_points[i] for i in range(len(instance.data_points)) 
            if i not in outliers
        ]
        
        # Handle case where all points are outliers
        if len(restricted.data_points) == 0:
            return INF, [], cluster_iters
        
        for _ in range(cluster_iters):
            kpp = KMeanspp()
            centers = kpp.get_centers_pt(restricted, solution, context)
            cost = self.cost_colorless(restricted, centers, Z=0)
            
            if cost < best_cost:
                best_cost = cost
                best_centers = centers
        
        return best_cost, best_centers, cluster_iters


class NKMeansFAISS(NKMeans):
    """NKMeans using FAISS for fast ball/range queries."""
    
    def __init__(self, iters: int = 10, guess_base: float = 2.0,
                 run_count: int = 1, seed: int = 42):
        super().__init__(iters, guess_base, run_count, seed)
    
    def codename(self) -> str:
        return f"nkmeans-faiss({self.iters},{self.guess_base},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::NKMeansFAISS(iters={self.iters},guess_base={self.guess_base},seed={self.seed})"
    
    def runnable(self, instance: Instance) -> bool:
        return FAISS_AVAILABLE and instance.Z > 0
    
    def _build_faiss_index(self, X: np.ndarray):
        """Build FAISS index for the data."""
        import faiss
        
        d = X.shape[1]
        X_f32 = np.ascontiguousarray(X.astype(np.float32))
        
        index = faiss.IndexFlatL2(d)
        index.add(X_f32)
        
        return index, X_f32
    
    def _count_in_ball(self, index, X_f32: np.ndarray, query_idx: int, 
                       radius_sq: float, threshold: int) -> int:
        """Count points within radius of query point using range_search."""
        query = X_f32[query_idx:query_idx+1]
        
        try:
            lims, D, I = index.range_search(query, radius_sq)
            return len(I)
        except AttributeError:
            k = min(threshold * 2, X_f32.shape[0])
            dists_sq, _ = index.search(query, k)
            return int(np.sum(dists_sq[0] <= radius_sq))
    
    def get_centers_pt(self, instance: Instance, solution: Solution, 
                       context: AlgorithmContext) -> List[Point]:
        """Compute centers using neighborhood-based outlier detection with FAISS."""
        import faiss
        
        X = _get_data_matrix(instance)
        n = len(instance.data_points)
        
        opt_guesses = instance.opt_guesses(p=2, base=self.guess_base)
        solution.extra["opt_guesses"] = str(len(opt_guesses))
        
        index, X_f32 = self._build_faiss_index(X)
        
        total_iter = 0
        best_centers: List[Point] = []
        best_outliers: Set[int] = set()
        best_cost = INF
        
        for opt in opt_guesses:
            r = 2 * math.sqrt(opt / instance.Z)
            r_sq = r * r
            threshold = 2 * instance.Z
            
            # Find heavy points using FAISS ball queries
            heavy = [i for i in range(n) 
                     if self._count_in_ball(index, X_f32, i, r_sq, threshold) >= threshold]
            
            if not heavy:
                continue
            
            # Build index for heavy points
            heavy_X = X_f32[heavy]
            heavy_index = faiss.IndexFlatL2(X.shape[1])
            heavy_index.add(heavy_X)
            
            # Query distances to nearest heavy point
            dists_sq, _ = heavy_index.search(X_f32, 1)
            dists = np.sqrt(np.maximum(dists_sq[:, 0], 0))
            
            outliers = {i for i in range(n) if dists[i] > r}
            inliers = n - len(outliers)
            
            if inliers == 0 or len(outliers) > (3 * instance.K + 2) * instance.Z:
                continue
            
            cost, centers, iter_count = self._cluster(
                instance, solution, context, outliers, self.iters, X
            )
            total_iter += iter_count
            
            if cost < best_cost:
                best_cost = cost
                best_centers = centers
                best_outliers = outliers
            
            break
        
        if not best_centers:
            solution.extra["nkmeans_fallback"] = "True"
            kpp = KMeanspp()
            best_centers = kpp.get_centers_pt(instance, solution, context)
        
        solution.extra["iter"] = str(total_iter)
        solution.original_outliers = [instance.data_points[i] for i in best_outliers]
        return best_centers
