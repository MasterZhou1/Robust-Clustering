# Experiments

Benchmark drivers for the algorithms in `algorithms/`.

## Layout

```
experiments/
├── parallel_benchmark.py          # Main benchmark
├── param_sensitivity_benchmark.py # Parameter sweep for OKMeans
├── config.py                      # Datasets, algorithms, defaults
├── run_parallel.slurm
├── run_param_sensitivity.slurm
└── results/                       # JSON outputs (gitignored)
```

Algorithms and datasets are configured in `config.py` (`DATASETS`, `ALGORITHMS`, `DEFAULT_SEEDS`, `DEFAULT_TIMEOUT`). Add new entries there.

## Local

```bash
python experiments/parallel_benchmark.py \
    --datasets <NAMES> \
    --algorithms <NAMES> \
    --workers 4 \
    --output experiments/results/run.json
```

```bash
python experiments/param_sensitivity_benchmark.py -d <NAMES> --workers 4
```

The default per-task timeout is `DEFAULT_TIMEOUT` from `config.py`; override with `--timeout`.

## SLURM

The scripts use `set -e` and resolve paths relative to their own location, so they can be submitted from any working directory. Before submitting:

- Set `--partition` (and `--account` if required).
- Replace the environment-activation comment with `conda activate <env>` or `source <venv>/bin/activate`.
- Adjust `--mem`, `--cpus-per-task`, and `--time` for your hardware.

```bash
sbatch experiments/run_parallel.slurm
sbatch experiments/run_parallel.slurm --datasets <NAMES>
sbatch experiments/run_param_sensitivity.slurm
```

## Output & analysis

Results are written to `experiments/results/` as JSON. Run the corresponding analyzer to produce figures and CSV tables (also gitignored):

```bash
python analysis/analyze_results.py            --input experiments/results/<file>.json --all
python analysis/analyze_param_sensitivity.py  --input experiments/results/<file>.json --all
```

## Threading

Distance computations rely on multi-threaded BLAS. The SLURM scripts and `parallel_benchmark.py` set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` before NumPy is imported.
