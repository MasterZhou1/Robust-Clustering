"""
Vectorized distance computations for fast k-means algorithms.

Uses numpy broadcasting and sklearn for optimized KNN.
"""

import numpy as np
from typing import List, Optional, Tuple
import os

# Try to import sklearn for fast KNN
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import FAISS for fast KNN
try:
    import faiss
    FAISS_AVAILABLE = True
except (ImportError, AttributeError):
    # AttributeError can happen with NumPy version mismatch
    FAISS_AVAILABLE = False

# Number of threads for parallel operations
NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 4))


def compute_distances_to_centers_vectorized(
    X: np.ndarray, 
    centers: np.ndarray,
    squared: bool = True
) -> np.ndarray:
    """
    Compute distances from all points to nearest center using vectorized operations.
    
    Args:
        X: Data points (n, d)
        centers: Center points (k, d)
        squared: If True, return squared distances
    
    Returns:
        Array of shape (n,) with distance to nearest center
    """
    n = X.shape[0]
    k = centers.shape[0]
    
    if k == 0:
        return np.full(n, np.inf)
    
    # For small k, use broadcasting (memory efficient for small k)
    if k <= 50:
        # X: (n, d), centers: (k, d)
        # diff: (n, k, d)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists_sq: (n, k)
        dists_sq = np.sum(diff ** 2, axis=2)
        # min distance to any center
        min_dists_sq = np.min(dists_sq, axis=1)
    else:
        # For large k, compute in chunks to save memory
        min_dists_sq = np.full(n, np.inf)
        chunk_size = 20
        for i in range(0, k, chunk_size):
            c_chunk = centers[i:i+chunk_size]
            diff = X[:, np.newaxis, :] - c_chunk[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            chunk_min = np.min(dists_sq, axis=1)
            min_dists_sq = np.minimum(min_dists_sq, chunk_min)
    
    if squared:
        return min_dists_sq
    return np.sqrt(min_dists_sq)


def compute_pairwise_distances_vectorized(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances.
    
    Args:
        X: First set of points (n, d)
        Y: Second set of points (m, d), or None for X vs X
    
    Returns:
        Distance matrix (n, m) or (n, n)
    """
    if Y is None:
        Y = X
    
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    X_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y_sq = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    
    dists_sq = X_sq + Y_sq - 2 * np.dot(X, Y.T)
    # Ensure non-negative (numerical issues)
    dists_sq = np.maximum(dists_sq, 0)
    
    return dists_sq


def assign_to_clusters_vectorized(
    X: np.ndarray,
    centers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign points to nearest center.
    
    Args:
        X: Data points (n, d)
        centers: Center points (k, d)
    
    Returns:
        Tuple of (assignments, distances_sq)
        - assignments: (n,) array of cluster indices
        - distances_sq: (n,) array of squared distances to assigned center
    """
    n = X.shape[0]
    k = centers.shape[0]
    
    if k == 0:
        return np.zeros(n, dtype=int), np.full(n, np.inf)
    
    # Compute all distances
    if k <= 100:
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        assignments = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(n), assignments]
    else:
        # Large k: compute in chunks
        assignments = np.zeros(n, dtype=int)
        min_dists_sq = np.full(n, np.inf)
        
        chunk_size = 50
        for i in range(0, k, chunk_size):
            c_chunk = centers[i:i+chunk_size]
            diff = X[:, np.newaxis, :] - c_chunk[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            chunk_min = np.min(dists_sq, axis=1)
            chunk_argmin = np.argmin(dists_sq, axis=1) + i
            
            update_mask = chunk_min < min_dists_sq
            assignments[update_mask] = chunk_argmin[update_mask]
            min_dists_sq[update_mask] = chunk_min[update_mask]
    
    return assignments, min_dists_sq


def update_distances_new_center_vectorized(
    X: np.ndarray,
    current_dists_sq: np.ndarray,
    new_center: np.ndarray
) -> np.ndarray:
    """
    Update distances after adding a new center.
    
    Args:
        X: Data points (n, d)
        current_dists_sq: Current min distances (n,)
        new_center: New center point (d,)
    
    Returns:
        Updated min distances (n,)
    """
    # Distance to new center
    new_dists_sq = np.sum((X - new_center) ** 2, axis=1)
    
    # Take minimum
    return np.minimum(current_dists_sq, new_dists_sq)


def compute_cluster_means_vectorized(
    X: np.ndarray,
    assignments: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute cluster means efficiently.
    
    Args:
        X: Data points (n, d)
        assignments: Cluster assignments (n,)
        k: Number of clusters
    
    Returns:
        Cluster means (k, d)
    """
    d = X.shape[1]
    means = np.zeros((k, d))
    counts = np.zeros(k)
    
    # Use bincount for efficient aggregation
    for dim in range(d):
        means[:, dim] = np.bincount(assignments, weights=X[:, dim], minlength=k)
    counts = np.bincount(assignments, minlength=k).astype(float)
    
    # Avoid division by zero
    counts[counts == 0] = 1
    means = means / counts[:, np.newaxis]
    
    return means


def knn_distances_vectorized(
    X: np.ndarray, 
    k: int,
    chunk_size: int = 500,
    sample_size: int = 20000,
    max_k: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor distances for all points.
    
    Uses sklearn's NearestNeighbors when available (much faster with BallTree/KDTree).
    Falls back to chunked numpy computation otherwise.
    
    Args:
        X: Data points (n, d)
        k: Number of neighbors
        chunk_size: Chunk size for memory efficiency (numpy fallback)
        sample_size: Max size before using approximate sampling (numpy fallback)
        max_k: Maximum k to use (caps very large k for efficiency)
    
    Returns:
        Tuple of (knn_distances, knn_indices)
        - knn_distances: (n, k) array of distances to k nearest neighbors
        - knn_indices: (n, k) array of indices of k nearest neighbors
    """
    n = X.shape[0]
    # Cap k for efficiency - very large k is computationally expensive
    k = min(k, n - 1, max_k)
    
    if k <= 0:
        return np.zeros((n, 1)), np.zeros((n, 1), dtype=int)
    
    # Use sklearn if available - much faster with optimized algorithms
    if SKLEARN_AVAILABLE:
        # Choose algorithm based on data size and dimensionality
        # Brute force is fastest for small n but uses O(n²) memory
        # KDTree/BallTree use O(n) memory and are better for large n
        if n < 50000 and k < 100:
            algorithm = 'brute'  # Fast for small data
        elif X.shape[1] <= 15:
            algorithm = 'kd_tree'  # Good for low dimensions
        else:
            algorithm = 'ball_tree'  # Good for high dimensions
        
        # n_jobs=1 is often faster than parallel on shared systems
        nn = NearestNeighbors(
            n_neighbors=k + 1,  # +1 because query point is its own nearest neighbor
            algorithm=algorithm,
            n_jobs=1,  # Single thread often faster on shared systems
            metric='euclidean'
        )
        nn.fit(X)
        
        # Query returns distances and indices (includes self as nearest)
        dists, indices = nn.kneighbors(X)
        
        # Remove self (first column) - sklearn returns sorted by distance
        knn_dists = dists[:, 1:k+1]
        knn_indices = indices[:, 1:k+1]
        
        return knn_dists, knn_indices
    
    # Fallback to numpy-based computation
    knn_dists = np.zeros((n, k))
    knn_indices = np.zeros((n, k), dtype=int)
    
    # For very large datasets or when k is large, use approximate KNN with random landmarks
    use_approximate = n > sample_size or (n > 10000 and k > 100)
    
    if use_approximate:
        # Use landmark-based approximate KNN
        rng = np.random.default_rng(42)
        # Reduce landmark size for very large datasets
        landmark_count = min(sample_size, n, max(k * 10, 5000))
        landmark_indices = rng.choice(n, size=landmark_count, replace=False)
        landmarks = X[landmark_indices]
        
        # Adjust k if we have fewer landmarks
        effective_k = min(k, landmark_count - 1)
        
        # Use smaller chunks for memory efficiency
        effective_chunk = min(chunk_size, 200)
        
        # Process in small chunks
        for i in range(0, n, effective_chunk):
            end_i = min(i + effective_chunk, n)
            X_chunk = X[i:end_i]
            chunk_len = end_i - i
            
            # Compute distances to landmarks only
            dists_sq = compute_pairwise_distances_vectorized(X_chunk, landmarks)
            
            # Find k nearest landmarks
            if effective_k < len(landmarks) // 2:
                part_idx = np.argpartition(dists_sq, effective_k, axis=1)[:, :effective_k]
                for j in range(chunk_len):
                    idx = part_idx[j]
                    d = dists_sq[j, idx]
                    sorted_order = np.argsort(d)
                    knn_indices[i + j, :effective_k] = landmark_indices[idx[sorted_order]]
                    knn_dists[i + j, :effective_k] = np.sqrt(d[sorted_order])
            else:
                sorted_idx = np.argsort(dists_sq, axis=1)[:, :effective_k]
                for j in range(chunk_len):
                    knn_indices[i + j, :effective_k] = landmark_indices[sorted_idx[j]]
                    knn_dists[i + j, :effective_k] = np.sqrt(dists_sq[j, sorted_idx[j]])
    else:
        # Process in chunks for exact KNN
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            X_chunk = X[i:end_i]
            chunk_len = end_i - i
            
            # Compute distances from chunk to all points
            dists_sq = compute_pairwise_distances_vectorized(X_chunk, X)
            
            # Set self-distance to inf
            for j in range(chunk_len):
                dists_sq[j, i + j] = np.inf
            
            # Get k nearest using partition (O(n) instead of O(n log n))
            if k < n // 2:
                part_idx = np.argpartition(dists_sq, k, axis=1)[:, :k]
                for j in range(chunk_len):
                    idx = part_idx[j]
                    d = dists_sq[j, idx]
                    sorted_order = np.argsort(d)
                    knn_indices[i + j] = idx[sorted_order]
                    knn_dists[i + j] = np.sqrt(d[sorted_order])
            else:
                sorted_idx = np.argsort(dists_sq, axis=1)[:, :k]
                for j in range(chunk_len):
                    knn_indices[i + j] = sorted_idx[j]
                    knn_dists[i + j] = np.sqrt(dists_sq[j, sorted_idx[j]])
    
    return knn_dists, knn_indices


def knn_distances_faiss(
    X: np.ndarray, 
    k: int,
    use_gpu: bool = False,
    approximate: bool = False,
    nprobe: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor distances using FAISS (Facebook AI Similarity Search).
    
    FAISS provides highly optimized implementations for similarity search,
    especially beneficial for large datasets.
    
    Args:
        X: Data points (n, d), will be converted to float32
        k: Number of neighbors
        use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        approximate: If True, use IVF index for approximate search (faster for large n)
        nprobe: Number of clusters to visit for IVF index (higher = more accurate but slower)
    
    Returns:
        Tuple of (knn_distances, knn_indices)
        - knn_distances: (n, k) array of distances to k nearest neighbors
        - knn_indices: (n, k) array of indices of k nearest neighbors
    """
    # Import locally to handle compatibility issues at runtime
    try:
        import faiss
    except (ImportError, AttributeError) as e:
        raise ImportError(f"FAISS is not available: {e}. Install with: pip install faiss-cpu or faiss-gpu")
    
    n, d = X.shape
    k = min(k, n - 1)
    
    if k <= 0:
        return np.zeros((n, 1)), np.zeros((n, 1), dtype=int)
    
    # FAISS requires float32
    X_f32 = np.ascontiguousarray(X.astype(np.float32))
    
    # Build index
    if approximate and n > 10000:
        # Use IVF (Inverted File) index for approximate search
        # Number of centroids for the coarse quantizer
        nlist = min(int(np.sqrt(n)), 256)
        
        # Create quantizer
        quantizer = faiss.IndexFlatL2(d)
        
        # Create IVF index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        
        # Train the index (required for IVF)
        index.train(X_f32)
        index.add(X_f32)
        
        # Set search parameters
        index.nprobe = min(nprobe, nlist)
    else:
        # Use exact flat index
        index = faiss.IndexFlatL2(d)
        index.add(X_f32)
    
    # Move to GPU if requested and available
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass  # Fall back to CPU if GPU fails
    
    # Search for k+1 neighbors (includes self)
    dists_sq, indices = index.search(X_f32, k + 1)
    
    # Remove self (first column) and convert squared distances to distances
    knn_dists = np.sqrt(np.maximum(dists_sq[:, 1:k+1], 0))
    knn_indices = indices[:, 1:k+1]
    
    return knn_dists, knn_indices


def d2_sampling_vectorized(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    existing_centers: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    D² sampling (k-means++ initialization) using vectorized operations.
    
    Args:
        X: Data points (n, d)
        k: Number of centers to select
        rng: Random number generator
        existing_centers: Optional existing centers to start from
    
    Returns:
        Selected center indices (k,)
    """
    n = X.shape[0]
    k = min(k, n)
    
    selected_indices = []
    
    if existing_centers is not None and len(existing_centers) > 0:
        centers = existing_centers.copy()
        dists_sq = compute_distances_to_centers_vectorized(X, centers, squared=True)
    else:
        # Pick first center uniformly at random
        first_idx = rng.integers(n)
        selected_indices.append(first_idx)
        dists_sq = np.sum((X - X[first_idx]) ** 2, axis=1)
    
    while len(selected_indices) < k:
        # Sample proportional to squared distance
        total = dists_sq.sum()
        if total <= 0:
            # All points are centers, pick remaining randomly
            remaining = list(set(range(n)) - set(selected_indices))
            if remaining:
                idx = rng.choice(remaining)
                selected_indices.append(idx)
            continue
        
        probs = dists_sq / total
        idx = rng.choice(n, p=probs)
        selected_indices.append(idx)
        
        # Update distances
        new_dists = np.sum((X - X[idx]) ** 2, axis=1)
        dists_sq = np.minimum(dists_sq, new_dists)
    
    return np.array(selected_indices)


def lloyd_iteration_vectorized(
    X: np.ndarray,
    centers: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Lloyd's algorithm using vectorized operations.
    
    Args:
        X: Data points (n, d)
        centers: Initial centers (k, d)
        max_iters: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple of (final_centers, assignments, cost)
    """
    k = centers.shape[0]
    prev_cost = np.inf
    
    for _ in range(max_iters):
        # Assign points to nearest center
        assignments, dists_sq = assign_to_clusters_vectorized(X, centers)
        cost = dists_sq.sum()
        
        # Check convergence
        if abs(prev_cost - cost) < tol * prev_cost:
            break
        prev_cost = cost
        
        # Update centers
        centers = compute_cluster_means_vectorized(X, assignments, k)
    
    return centers, assignments, cost


def compute_weighted_cluster_means(
    X: np.ndarray,
    assignments: np.ndarray,
    point_weights: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute weighted cluster means.
    
    Args:
        X: Data points (n, d)
        assignments: Cluster assignments (n,)
        point_weights: Weight for each point (n,)
        k: Number of clusters
    
    Returns:
        Weighted cluster means (k, d)
    """
    d = X.shape[1]
    means = np.zeros((k, d))
    
    # Compute weighted sums for each dimension
    for dim in range(d):
        weighted_values = X[:, dim] * point_weights
        means[:, dim] = np.bincount(assignments, weights=weighted_values, minlength=k)
    
    # Compute total weights per cluster
    total_weights = np.bincount(assignments, weights=point_weights, minlength=k)
    
    # Avoid division by zero
    total_weights[total_weights == 0] = 1
    means = means / total_weights[:, np.newaxis]
    
    return means


def weighted_lloyd_iteration(
    X: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weighted Lloyd's algorithm using vectorized operations.
    
    Each point has a weight that affects the centroid computation.
    Centroid = sum(weight_i * x_i) / sum(weight_i) for points in cluster.
    
    Args:
        X: Data points (n, d)
        centers: Initial centers (k, d)
        weights: Weight for each point (n,) - e.g., cluster sizes from previous step
        max_iters: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple of (final_centers, assignments, cost)
    """
    k = centers.shape[0]
    prev_cost = np.inf
    
    for _ in range(max_iters):
        # Assign points to nearest center
        assignments, dists_sq = assign_to_clusters_vectorized(X, centers)
        
        # Weighted cost
        cost = (dists_sq * weights).sum()
        
        # Check convergence
        if abs(prev_cost - cost) < tol * max(prev_cost, 1e-10):
            break
        prev_cost = cost
        
        # Update centers using weighted means
        centers = compute_weighted_cluster_means(X, assignments, weights, k)
    
    return centers, assignments, cost


def outlier_lloyd_vectorized(
    X: np.ndarray,
    centers: np.ndarray,
    z: int,
    max_iters: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Outlier-aware Lloyd's algorithm (k-means--).
    
    Excludes z furthest points from cluster updates.
    
    Args:
        X: Data points (n, d)
        centers: Initial centers (k, d)
        z: Number of outliers to exclude
        max_iters: Maximum iterations
    
    Returns:
        Tuple of (final_centers, outlier_mask, cost)
    """
    n = X.shape[0]
    k = centers.shape[0]
    prev_cost = np.inf
    
    for _ in range(max_iters):
        # Assign all points
        assignments, dists_sq = assign_to_clusters_vectorized(X, centers)
        
        # Find z furthest points (outliers)
        outlier_indices = np.argsort(-dists_sq)[:z]
        outlier_mask = np.zeros(n, dtype=bool)
        outlier_mask[outlier_indices] = True
        
        # Compute cost (excluding outliers)
        inlier_cost = dists_sq[~outlier_mask].sum()
        
        # Check convergence
        if inlier_cost >= prev_cost:
            break
        prev_cost = inlier_cost
        
        # Update centers using only inliers
        inlier_assignments = assignments[~outlier_mask]
        inlier_X = X[~outlier_mask]
        
        # Compute new centers
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k)
        
        for c_idx in range(k):
            mask = inlier_assignments == c_idx
            if mask.sum() > 0:
                new_centers[c_idx] = inlier_X[mask].mean(axis=0)
                counts[c_idx] = mask.sum()
            else:
                new_centers[c_idx] = centers[c_idx]
        
        centers = new_centers
    
    return centers, outlier_mask, prev_cost
