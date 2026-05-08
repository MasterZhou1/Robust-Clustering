#!/bin/bash
#SBATCH --job-name=rc-parallel
#SBATCH --output=logs/parallel_%j.out
#SBATCH --error=logs/parallel_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=768G
#SBATCH --cpus-per-task=32
# NOTE: Update --partition (and --account if your cluster requires it)
# to match your SLURM environment, e.g.:
#   #SBATCH --partition=<your-partition>
#   #SBATCH --account=<your-account>

# Parallel Benchmark
#
# Threading Strategy:
#   - 8 workers x 4 threads = 32 CPUs (good for large matrix ops)
#   - 768GB / 8 workers = 96GB per worker (handles SUSY, KDDFULL)
#
# Usage (from the repository root):
#   sbatch experiments/run_parallel.slurm
#   sbatch experiments/run_parallel.slurm --datasets SKIN-5 SUSY-5

set -e

# Resolve repo root from this script's location and cd into it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p experiments/logs experiments/results

echo "============================================================"
echo "Parallel Benchmark | Job: ${SLURM_JOB_ID:-local} | Node: ${SLURM_NODELIST:-$(hostname)}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?} | Memory: ${SLURM_MEM_PER_NODE:-?}MB"
echo "Start: $(date)"
echo "============================================================"

# Activate your Python environment here.
# Example (conda): conda activate clustering
# Example (venv):  source .venv/bin/activate

# Threading: 4 threads per worker (32 CPUs / 8 workers = 4)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="experiments/results/parallel_${SLURM_JOB_ID:-local}_${TIMESTAMP}.json"

echo "Workers: 8, Threads/worker: 4, Timeout: 7200s (2h)"
echo "Output: $OUTPUT"
echo ""

python3 -u experiments/parallel_benchmark.py \
    --output "$OUTPUT" \
    --workers 8 \
    --threads 4 \
    --timeout 7200 \
    "$@"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Exit: $EXIT_CODE | End: $(date)"
echo "============================================================"

[ $EXIT_CODE -eq 0 ] && python3 analysis/analyze_results.py --input "$OUTPUT" --all 2>/dev/null || true

exit $EXIT_CODE
