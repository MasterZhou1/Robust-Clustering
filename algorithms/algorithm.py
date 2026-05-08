"""
Base Algorithm class for clustering algorithms.
"""

import time
import numpy as np

from .defs import INF
from .instance import Instance
from .solution import Solution


class AlgorithmContext:
    """
    Context for algorithm execution.
    
    Contains:
    - Random number generator state
    - Run information
    """
    
    def __init__(self, seed: int = 42):
        """Initialize context with random seed."""
        self.seed = seed
        self.generator = np.random.default_rng(seed)
        self.run_id = 0


class Algorithm:
    """
    Base class for clustering algorithms.
    
    Subclasses should implement:
    - codename(): Return algorithm identifier
    - fullname(): Return full algorithm name
    - problem(): Return problem type (e.g., "kmeans")
    - get_solution(): Compute clustering solution
    """
    
    def __init__(self, run_count: int = 1, seed: int = 42):
        self.run_count = run_count
        self.seed = seed
    
    def codename(self) -> str:
        """Return algorithm codename (filename-friendly)."""
        raise NotImplementedError
    
    def fullname(self) -> str:
        """Return full algorithm name."""
        raise NotImplementedError
    
    def problem(self) -> str:
        """Return problem type."""
        raise NotImplementedError
    
    def get_solution(self, instance: Instance, solution: Solution, 
                     context: AlgorithmContext):
        """Compute clustering solution."""
        raise NotImplementedError
    
    def init_context(self) -> AlgorithmContext:
        """Create algorithm context."""
        return AlgorithmContext(self.seed)
    
    def with_run_count(self, run_count: int) -> 'Algorithm':
        """Set run count and return self for chaining."""
        self.run_count = run_count
        return self
    
    def with_seed(self, seed: int) -> 'Algorithm':
        """Set seed and return self for chaining."""
        self.seed = seed
        return self
    
    def runnable(self, instance: Instance) -> bool:
        """Check if algorithm can run on this instance."""
        return True
    
    def run(self, instance: Instance) -> Solution:
        """Run algorithm and return best solution."""
        best_sol = Solution(instance)
        best_sol.cost = INF
        
        start_time = time.time()
        context = self.init_context()
        
        for run_id in range(self.run_count):
            context.run_id = run_id
            sol = Solution(instance)
            self.get_solution(instance, sol, context)
            
            if sol.cost < best_sol.cost:
                best_sol = sol
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        best_sol.instance = instance
        best_sol.problem = self.problem()
        best_sol.algo_codename = self.codename()
        best_sol.algo_fullname = self.fullname()
        best_sol.elapsed_ms = elapsed_ms
        
        return best_sol
    
    def save_path(self, root: str, instance: Instance) -> str:
        """Generate save path for solution."""
        sol = Solution(instance)
        sol.algo_codename = self.codename()
        return f"{root}/{sol.codename()}.json"
