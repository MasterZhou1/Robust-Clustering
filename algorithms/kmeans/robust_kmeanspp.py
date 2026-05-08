"""
Robust k-means++ algorithm implementation.

OPTIMIZED: Uses vectorized numpy operations for speed.
"""

from typing import List, Set
import numpy as np

from ..defs import Point, INF
from ..instance import Instance
from ..solution import Solution
from ..algorithm import AlgorithmContext
from .base import BaseAlgorithm, _get_data_matrix


class RobustKMeanspp(BaseAlgorithm):
    """
    Robust k-means++ algorithm (vectorized).
    
    Builds a set of O(k/δ) candidate centers by mixing D² and uniform sampling,
    then reduces to exactly k centers using weighted k-means++.
    """
    
    def __init__(self, beta: float = 0.1, delta: float = 0.1,
                 t_mult: float = 1.0, m_mult: float = 1.0,
                 run_count: int = 1, seed: int = 42):
        super().__init__(run_count, seed)
        self.beta = beta
        self.delta = delta
        self.t_mult = t_mult
        self.m_mult = m_mult
    
    def codename(self) -> str:
        return f"robust-kmeans++({self.beta},{self.delta},{self.seed},{self.run_count})"
    
    def fullname(self) -> str:
        return f"KMeans::RobustKmeans++(beta={self.beta},delta={self.delta},seed={self.seed},run_count={self.run_count})"
    
    def get_centers(self, instance: Instance, solution: Solution,
                    context: AlgorithmContext):
        """Compute centers using Robust k-means++ (vectorized)."""
        X = _get_data_matrix(instance)
        n = len(instance.data_points)
        k = instance.K
        z = instance.Z
        
        t = max(1, int(self.t_mult * k))
        m = max(1, int(self.m_mult / self.delta))
        
        S: Set[int] = set()
        dist_sq = np.full(n, INF)
        total_cost = INF
        
        for i in range(1, t + 1):
            # Compute probabilities once per outer iteration (not m times)
            if i == 1 and len(S) == 0:
                probs = np.ones(n) / n
            else:
                if total_cost > 0 and total_cost < INF:
                    d2_probs = dist_sq / (2 * total_cost)
                else:
                    d2_probs = np.zeros(n)
                
                uniform_probs = np.ones(n) / (2 * n)
                probs = d2_probs + uniform_probs
                probs = probs / probs.sum()
            
            # Sample m points using the same probability distribution
            new_points = [context.generator.choice(n, p=probs) for _ in range(m)]
            
            for idx in new_points:
                if idx not in S:
                    S.add(idx)
                    # Update distances (vectorized)
                    new_dists = np.sum((X - X[idx]) ** 2, axis=1)
                    dist_sq = np.minimum(dist_sq, new_dists)
                    total_cost = dist_sq.sum()
        
        candidates = [instance.data_points[i] for i in S]
        solution.extra["num_candidates"] = str(len(candidates))
        solution.extra["t"] = str(t)
        solution.extra["m"] = str(m)
        
        # Reduce to exactly k centers using weighted k-means++ with Lloyd on candidates
        if len(candidates) > k:
            from .vectorized import compute_distances_to_centers_vectorized, assign_to_clusters_vectorized, weighted_lloyd_iteration
            
            # Compute distances to candidates
            cand_data_idx = list(S)
            cand_X = X[cand_data_idx]
            
            # Distances from all points to nearest candidate
            dists_sq_to_cand = compute_distances_to_centers_vectorized(X, cand_X, squared=True)
            
            # Identify z furthest as outliers for weighting
            sorted_indices = np.argsort(-dists_sq_to_cand)
            outliers = set(sorted_indices[:z].tolist())
            
            # Compute weights (cluster sizes excluding outliers)
            inlier_mask = np.ones(n, dtype=bool)
            inlier_mask[list(outliers)] = False
            inlier_X = X[inlier_mask]
            
            weights = {}
            if inlier_X.shape[0] > 0:
                assignments, _ = assign_to_clusters_vectorized(inlier_X, cand_X)
                for c_idx in assignments:
                    c = candidates[c_idx]
                    weights[c] = weights.get(c, 0) + 1
            else:
                weights = {c: 1 for c in candidates}
            
            # Weighted k-means++ to select exactly k centers from candidates
            selected: List[int] = []
            selected_cand_idx: List[int] = []  # Track indices in cand_X
            cand_weights = np.array([weights.get(c, 1) for c in candidates], dtype=float)
            
            # First center proportional to weight
            probs = cand_weights / cand_weights.sum()
            first_idx = context.generator.choice(len(candidates), p=probs)
            selected.append(candidates[first_idx])
            selected_cand_idx.append(first_idx)
            
            cand_dists_sq = np.sum((cand_X - cand_X[first_idx]) ** 2, axis=1)
            
            while len(selected) < k:
                weighted_dists = cand_dists_sq * cand_weights
                for s_idx in selected_cand_idx:
                    weighted_dists[s_idx] = 0
                
                total = weighted_dists.sum()
                if total <= 0:
                    remaining_idx = [i for i in range(len(candidates)) if i not in selected_cand_idx]
                    if remaining_idx:
                        idx = remaining_idx[context.generator.integers(len(remaining_idx))]
                        selected.append(candidates[idx])
                        selected_cand_idx.append(idx)
                else:
                    probs = weighted_dists / total
                    idx = context.generator.choice(len(candidates), p=probs)
                    if idx not in selected_cand_idx:
                        selected.append(candidates[idx])
                        selected_cand_idx.append(idx)
                        new_dists = np.sum((cand_X - cand_X[idx]) ** 2, axis=1)
                        cand_dists_sq = np.minimum(cand_dists_sq, new_dists)
            
            # Run weighted Lloyd refinement on candidates
            centers_X = cand_X[selected_cand_idx]
            
            refined_centers, _, _ = weighted_lloyd_iteration(
                cand_X, centers_X, cand_weights, max_iters=30
            )
            
            # Store as Point objects
            solution.centers_pt = []
            for c_coords in refined_centers:
                pt = Point()
                pt.x = c_coords
                solution.centers_pt.append(pt)
            
            solution.centers = selected
        else:
            solution.centers = candidates
    
    def get_solution(self, instance: Instance, solution: Solution,
                     context: AlgorithmContext):
        self.get_centers(instance, solution, context)
        self.cluster_colorless(instance, solution)
