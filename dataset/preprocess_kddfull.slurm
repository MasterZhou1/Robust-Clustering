#!/bin/bash
#SBATCH --job-name=preprocess_kdd
#SBATCH --output=slurm_logs/preprocess_kdd_%j.out
#SBATCH --error=slurm_logs/preprocess_kdd_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
# NOTE: Set --partition / --account to match your SLURM environment.

set -e

# Resolve dataset directory from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p slurm_logs

# Activate your Python environment here.
# Example (conda): conda activate clustering
# Example (venv):  source ../.venv/bin/activate

echo "Starting preprocessing of KDDFULL dataset..."
echo "Time: $(date)"
echo "Python: $(which python)"

python preprocess_datasets.py --datasets kddfull

echo "Completed at: $(date)"
