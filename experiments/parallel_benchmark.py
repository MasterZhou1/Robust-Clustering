#!/usr/bin/env python3
"""
Parallel benchmarking script for robust clustering algorithms.

Optimized for SLURM clusters with:
- Multiprocessing for parallel experiment execution
- Dataset caching to avoid redundant loads
- Smart thread/process allocation
- Support for job arrays (run subset of experiments)

Usage:
    python parallel_benchmark.py --workers 8
    python parallel_benchmark.py --datasets SKIN-5 --workers 4
    python parallel_benchmark.py --task-range 0 50
"""

# CRITICAL: Configure threading BEFORE any numpy import
# For multiprocessing workers, use 2 threads per worker (set by init_worker)
import os
_THREADS = os.environ.get('OMP_NUM_THREADS', '2')
os.environ['OMP_NUM_THREADS'] = _THREADS
os.environ['MKL_NUM_THREADS'] = _THREADS
os.environ['OPENBLAS_NUM_THREADS'] = _THREADS
os.environ['NUMEXPR_NUM_THREADS'] = _THREADS

import argparse
import json
import platform
import socket
import sys
import time
import signal
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import (
    DATASETS, ALGORITHMS, ExperimentConfig, DATASET_DIR, RESULTS_DIR,
    get_algorithm, DEFAULT_TIMEOUT, DEFAULT_SEEDS
)


# ============================================================================
# Resource Configuration
# ============================================================================

@dataclass
class ResourceConfig:
    """Configuration for parallel execution resources."""
    num_workers: int = 4
    threads_per_worker: int = 4
    memory_per_worker_gb: float = 64.0
    
    @classmethod
    def from_slurm(cls, threads_per_worker: int = None) -> 'ResourceConfig':
        """Create config from SLURM environment variables."""
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
        mem_str = os.environ.get('SLURM_MEM_PER_NODE', '64G')
        
        # Parse memory string (can be in MB or GB)
        if mem_str.endswith('G'):
            mem_gb = int(mem_str[:-1])
        elif mem_str.endswith('M'):
            mem_gb = int(mem_str[:-1]) / 1024
        else:
            mem_gb = int(mem_str) / 1024  # Assume MB
        
        # Get threads from env or use provided value
        threads = threads_per_worker or int(os.environ.get('OMP_NUM_THREADS', 4))
        
        # Calculate workers: use all CPUs, cap by memory (64GB per worker for large datasets)
        max_workers_by_cpu = cpus // threads
        max_workers_by_mem = max(1, int(mem_gb / 64.0))
        num_workers = max(1, min(max_workers_by_cpu, max_workers_by_mem))
        mem_per_worker = mem_gb / num_workers
        
        return cls(num_workers=num_workers, threads_per_worker=threads, 
                   memory_per_worker_gb=mem_per_worker)
    
    @classmethod
    def auto(cls, cpus: int = None, memory_gb: float = None, 
             threads_per_worker: int = None) -> 'ResourceConfig':
        """Auto-configure based on available resources."""
        cpus = cpus or os.cpu_count() or 4
        memory_gb = memory_gb or 64.0
        threads = threads_per_worker or int(os.environ.get('OMP_NUM_THREADS', 4))
        
        max_workers_by_cpu = cpus // threads
        max_workers_by_mem = max(1, int(memory_gb / 64.0))
        num_workers = max(1, min(max_workers_by_cpu, max_workers_by_mem))
        
        return cls(num_workers=num_workers, threads_per_worker=threads,
                   memory_per_worker_gb=memory_gb/num_workers)


def get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try to get more detailed CPU info on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass
    
    # Get memory info
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    mem_kb = int(line.split()[1])
                    info["memory_total_gb"] = round(mem_kb / 1024 / 1024, 2)
                    break
    except FileNotFoundError:
        pass
    
    # Get SLURM job info if available
    slurm_vars = ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK",
                  "SLURM_MEM_PER_NODE", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
                  "SLURM_NODELIST"]
    for var in slurm_vars:
        if var in os.environ:
            info[var.lower()] = os.environ[var]
    
    return info


# ============================================================================
# Dataset Cache
# ============================================================================

_DATASET_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, Any, int]] = {}


def preload_datasets(dataset_names: List[str]) -> None:
    """Preload datasets into memory before forking workers."""
    global _DATASET_CACHE
    
    for name in dataset_names:
        if name in _DATASET_CACHE:
            continue
        
        config = DATASETS[name]
        filepath = DATASET_DIR / config.filename
        
        if filepath.exists():
            npz = np.load(filepath)
            _DATASET_CACHE[name] = (
                npz["data"],
                npz["outlier_mask"],
                npz.get("labels", None),
                int(npz.get("k", config.k))
            )
            print(f"  Preloaded {name}: {_DATASET_CACHE[name][0].shape}")
        else:
            print(f"  Warning: Dataset {name} not found")


def get_cached_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, Any, int]:
    """Get dataset from cache or load if not cached."""
    if name in _DATASET_CACHE:
        return _DATASET_CACHE[name]
    
    config = DATASETS[name]
    filepath = DATASET_DIR / config.filename
    npz = np.load(filepath)
    return (npz["data"], npz["outlier_mask"], npz.get("labels", None), 
            int(npz.get("k", config.k)))


# ============================================================================
# Worker Functions
# ============================================================================

def init_worker(threads: int):
    """Initialize worker process."""
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[var] = str(threads)
    
    try:
        import faiss
        faiss.omp_set_num_threads(threads)
    except ImportError:
        pass
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_experiment_task(task: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Run a single experiment task."""
    from algorithms import Dataset, Instance, Solution, Point
    
    dataset_name = task['dataset']
    algo_name = task['algorithm']
    seed = task['seed']
    
    result = {
        "task_id": task.get('task_id', 0),
        "dataset": dataset_name,
        "algorithm": algo_name,
        "seed": seed,
        "status": "pending",
        "error": None,
        "worker_pid": os.getpid(),
    }
    
    try:
        data, outlier_mask, labels, k = get_cached_dataset(dataset_name)
        
        result.update({
            "dataset_size": len(data),
            "dataset_dim": data.shape[1],
            "true_outliers": int(outlier_mask.sum()),
            "k": k,
        })
        
        # Create instance
        points = [Point() for _ in range(len(data))]
        for i, pt in enumerate(points):
            pt.id = i
            pt.x = data[i].copy()
        
        dataset = Dataset(points=points)
        dataset.name = "benchmark"
        dataset.name_fout = "benchmark"
        
        instance = Instance(data=dataset, metric="L2", K=k, Z=int(outlier_mask.sum()))
        
        # Get algorithm using centralized factory
        algo_config = ALGORITHMS[algo_name]
        algorithm = get_algorithm(algo_config, seed)
        
        if not algorithm.runnable(instance):
            result["status"] = "skipped"
            result["error"] = "Algorithm not runnable"
            return result
        
        # Run with timeout
        signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(int(timeout))
        
        run_start = time.time()
        solution = algorithm.run(instance)
        run_time = time.time() - run_start
        
        signal.alarm(0)
        
        # Metrics (sklearn): shared binary outlier masks + cluster labels for ARI/NMI
        n = len(data)
        y_outlier_true = outlier_mask.astype(np.int64)
        y_outlier_pred = np.zeros(n, dtype=np.int64)
        for o in solution.outliers:
            y_outlier_pred[o] = 1
        precision = float(
            precision_score(y_outlier_true, y_outlier_pred, pos_label=1, zero_division=0.0)
        )
        recall = float(
            recall_score(y_outlier_true, y_outlier_pred, pos_label=1, zero_division=0.0)
        )
        f1 = float(f1_score(y_outlier_true, y_outlier_pred, pos_label=1, zero_division=0.0))

        ari, nmi = None, None
        if labels is not None:
            labels_arr = np.asarray(labels).ravel()
            y_pred_cluster = np.full(n, -1, dtype=np.int64)
            for ci, cl in enumerate(solution.clusters):
                for p in cl:
                    y_pred_cluster[p] = ci
            inlier_mask = y_outlier_pred == 0
            ari = float(adjusted_rand_score(labels_arr[inlier_mask], y_pred_cluster[inlier_mask]))
            nmi = float(
                normalized_mutual_info_score(labels_arr[inlier_mask], y_pred_cluster[inlier_mask])
            )
        
        result.update({
            "status": "completed",
            "cost": solution.cost,
            "outliers_cost": solution.outliers_cost,
            "time_s": run_time,
            "elapsed_ms": solution.elapsed_ms,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "ari": ari,
            "nmi": nmi,
            "detected_outliers": len(solution.outliers),
            "num_centers": len(solution.centers),
            "algo_codename": solution.algo_codename,
            "algo_fullname": solution.algo_fullname,
        })
        
        if solution.extra:
            result["extra"] = solution.extra
            
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Timed out after {timeout}s"
        signal.alarm(0)
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        try:
            signal.alarm(0)
        except Exception:
            pass
    
    return result


# ============================================================================
# Task Generation
# ============================================================================

def generate_tasks(
    datasets: List[str] = None,
    algorithms: List[str] = None,
    seeds: List[int] = None,
) -> List[Dict[str, Any]]:
    """Generate all experiment tasks."""
    datasets = datasets or list(DATASETS.keys())
    algorithms = algorithms or list(ALGORITHMS.keys())
    seeds = seeds or DEFAULT_SEEDS
    
    tasks = []
    for i, (dataset, algo, seed) in enumerate(
        (d, a, s) for d in datasets for a in algorithms for s in seeds
    ):
        tasks.append({
            'task_id': i,
            'dataset': dataset,
            'algorithm': algo,
            'seed': seed,
        })
    return tasks


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_parallel_benchmark(
    datasets: List[str] = None,
    algorithms: List[str] = None,
    seeds: List[int] = None,
    num_workers: int = 4,
    threads_per_worker: int = 2,
    timeout: float = DEFAULT_TIMEOUT,
    output_file: str = None,
    task_range: Tuple[int, int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run benchmark with parallel execution."""
    all_tasks = generate_tasks(datasets, algorithms, seeds)
    
    # Filter by task range if specified
    if task_range:
        start, end = task_range
        tasks = all_tasks[start:end]
        if verbose:
            print(f"Running tasks {start} to {end} of {len(all_tasks)}")
    else:
        tasks = all_tasks
    
    if not tasks:
        print("No tasks to run!")
        return {"results": []}
    
    needed_datasets = list(set(t['dataset'] for t in tasks))
    
    system_info = get_system_info()
    
    if verbose:
        print("=" * 70)
        print("PARALLEL BENCHMARK")
        print("=" * 70)
        print(f"Tasks: {len(tasks)}, Workers: {num_workers}, Timeout: {timeout}s")
        print(f"Datasets: {needed_datasets}")
        print()
        print("Preloading datasets...")
    
    preload_datasets(needed_datasets)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "system_info": system_info,
        "experiment_config": {
            "datasets": needed_datasets,
            "algorithms": list(set(t['algorithm'] for t in tasks)),
            "seeds": list(set(t['seed'] for t in tasks)),
            "num_workers": num_workers,
            "timeout": timeout,
            "task_range": task_range,
            "total_tasks": len(all_tasks),
        },
        "results": [],
    }
    
    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(threads_per_worker,)
    ) as executor:
        futures = {executor.submit(run_experiment_task, t, timeout): t for t in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            completed += 1
            
            try:
                result = future.result()
                all_results["results"].append(result)
                
                if verbose:
                    status = result.get("status", "?")
                    if status == "completed":
                        print(f"[{completed}/{len(tasks)}] {task['dataset']}/{task['algorithm']}/s{task['seed']} "
                              f"-> Cost: {result['cost']:.2f}, Recall: {result['recall']:.4f}, "
                              f"Time: {result['time_s']:.2f}s")
                    else:
                        print(f"[{completed}/{len(tasks)}] {task['dataset']}/{task['algorithm']}/s{task['seed']} "
                              f"-> {status}")
                        
            except Exception as e:
                all_results["results"].append({
                    "task_id": task.get('task_id', 0),
                    "dataset": task['dataset'],
                    "algorithm": task['algorithm'],
                    "seed": task['seed'],
                    "status": "error",
                    "error": str(e),
                })
                if verbose:
                    print(f"[{completed}/{len(tasks)}] {task['dataset']}/{task['algorithm']} -> ERROR: {e}")
            
            if output_file and completed % 10 == 0:
                with open(output_file, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    if verbose:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        counts = {}
        for r in all_results["results"]:
            s = r.get("status", "?")
            counts[s] = counts.get(s, 0) + 1
        
        print(f"Total: {len(all_results['results'])}, Time: {elapsed:.2f}s")
        for status, count in sorted(counts.items()):
            print(f"  {status}: {count}")
        
        total_run_time = sum(r.get('time_s', 0) for r in all_results['results'] if r.get('time_s'))
        if elapsed > 0:
            print(f"Effective parallelism: {total_run_time/elapsed:.1f}x")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run parallel clustering benchmarks")
    parser.add_argument("--datasets", "-d", nargs="+", default=None,
                        help=f"Datasets (default: all). Available: {list(DATASETS.keys())}")
    parser.add_argument("--algorithms", "-a", nargs="+", default=None,
                        help=f"Algorithms (default: all). Available: {list(ALGORITHMS.keys())}")
    parser.add_argument("--seeds", "-s", nargs="+", type=int, default=None,
                        help="Random seeds")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Threads per worker (default: 4)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--timeout", "-t", type=float, default=DEFAULT_TIMEOUT,
                        help=f"Timeout per algorithm (default: {DEFAULT_TIMEOUT}s)")
    parser.add_argument("--task-range", nargs=2, type=int, metavar=('START', 'END'),
                        help="Task range for job arrays")
    parser.add_argument("--quiet", "-q", action="store_true")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.datasets:
        for d in args.datasets:
            if d not in DATASETS and d != "all":
                parser.error(f"Unknown dataset: {d}")
        if "all" in args.datasets:
            args.datasets = None
    
    if args.algorithms:
        for a in args.algorithms:
            if a not in ALGORITHMS and a != "all":
                parser.error(f"Unknown algorithm: {a}")
        if "all" in args.algorithms:
            args.algorithms = None
    
    # Auto-configure workers
    if args.workers is None:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            config = ResourceConfig.from_slurm(threads_per_worker=args.threads)
            args.workers = config.num_workers
            print(f"Auto-configured from SLURM: {args.workers} workers × {args.threads} threads")
        else:
            config = ResourceConfig.auto(threads_per_worker=args.threads)
            args.workers = config.num_workers
            print(f"Auto-configured: {args.workers} workers × {args.threads} threads")
    
    # Default output file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        array_id = os.environ.get('SLURM_ARRAY_TASK_ID', '')
        suffix = f"_{array_id}" if array_id else ""
        args.output = str(RESULTS_DIR / f"benchmark_{job_id}{suffix}_{timestamp}.json")
    
    # Task range from job array
    task_range = tuple(args.task_range) if args.task_range else None
    
    if task_range is None and 'SLURM_ARRAY_TASK_ID' in os.environ:
        array_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        tasks_per_job = int(os.environ.get('TASKS_PER_JOB', 50))
        task_range = (array_id * tasks_per_job, (array_id + 1) * tasks_per_job)
        print(f"Job array task {array_id}: tasks {task_range[0]} to {task_range[1]}")
    
    run_parallel_benchmark(
        datasets=args.datasets,
        algorithms=args.algorithms,
        seeds=args.seeds,
        num_workers=args.workers,
        threads_per_worker=args.threads,
        timeout=args.timeout,
        output_file=args.output,
        task_range=task_range,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
