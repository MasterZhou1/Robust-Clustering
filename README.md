# [Simple KNN-Based Outlier Detection Achieves Robust Clustering](https://arxiv.org/abs/2605.07130)

**Authors:** [**Tianle Jiang**](https://sites.google.com/view/tianle-jiang/), [**Yufa Zhou**](https://masterzhou1.github.io/)

**Duke University**

---

**Abstract**

Being robust to the presence of outliers is crucial for applying clustering algorithms in practice.
In the $\textit{robust k-Means}$ problem (i.e., $k$-Means with outliers), the goal is to remove $z$ outliers and minimize the $k$-Means cost on the remaining points.
Despite the close connection between robust $k$-Means and outlier detection, both theoretical and empirical understanding of the effectiveness of $\textit{classic outlier detection heuristics}$ for robust $k$-Means remains limited.

In this paper, we prove that under a practical assumption on the optimal cluster sizes, simply removing points with large $K$-Nearest-Neighbor distances achieves performance comparable to prior work in terms of approximation guarantees: it yields a constant-factor reduction from robust $k$-Means to standard $k$-Means, without introducing additional centers or discarding extra outliers, as is commonly required by existing approaches.

Empirically, experiments on real-world datasets show that our method outperforms or matches several more sophisticated algorithms in terms of clustering cost and runtime. 
These results demonstrate that simple KNN-based heuristics can be surprisingly effective for robust clustering, highlighting new opportunities to bridge techniques from outlier detection and clustering.

[Paper Link](https://arxiv.org/abs/2605.07130)

---

A Python implementation of robust k-means clustering algorithms that handle outliers, with BLAS-optimized routines and a parallel benchmarking suite.

## Installation

```bash
git clone <your-fork-url>.git
cd Robust-Clustering

conda create -n clustering python=3.10 -y
conda activate clustering
pip install -r requirements.txt

# Recommended: ensure NumPy is built against threaded OpenBLAS
conda install -y "libblas=*=*openblas" -c conda-forge
```

## Datasets

Raw datasets are not bundled with this repo.

```bash
python dataset/download_datasets.py
python dataset/preprocess_datasets.py --datasets all
```

This produces `.npz` files in `dataset/processed/`. See `dataset/preprocess_datasets.py` for the normalization and outlier-generation methodology.

## Usage

### Local

```bash
python experiments/parallel_benchmark.py \
    --datasets SHUTTLE \
    --algorithms TIKMeans IKMeans OKMeans \
    --workers 4 \
    --output experiments/results/run.json

python analysis/analyze_results.py --input experiments/results/run.json --all
```

### SLURM

The provided SLURM scripts resolve paths relative to the repo root. Update `--partition` / `--account` and the environment-activation line to match your cluster, then:

```bash
sbatch experiments/run_parallel.slurm
sbatch experiments/run_param_sensitivity.slurm
```

See `experiments/README.md` for further options.

## Project Structure

```
algorithms/    # Clustering algorithm implementations
dataset/       # Dataset download + preprocessing
experiments/   # Benchmark drivers, configuration, SLURM scripts
analysis/      # Result analysis and plotting
```

## Citation

```
@article{jiang2026simple,
  title={Simple KNN-Based Outlier Detection Achieves Robust Clustering},
  author={Jiang, Tianle and Zhou, Yufa},
  journal={arXiv preprint arXiv:2605.07130},
  year={2026}
}
```
