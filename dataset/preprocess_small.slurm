#!/bin/bash
#SBATCH --job-name=preprocess_small
#SBATCH --output=slurm_logs/preprocess_small_%j.out
#SBATCH --error=slurm_logs/preprocess_small_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# NOTE: Set --partition / --account to match your SLURM environment.

set -e

# Resolve dataset directory from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p slurm_logs

# Activate your Python environment here.
# Example (conda): conda activate clustering
# Example (venv):  source ../.venv/bin/activate

echo "Starting preprocessing of SKIN and SHUTTLE datasets..."
echo "Time: $(date)"
echo "Python: $(which python)"

python preprocess_datasets.py --datasets skin shuttle

echo "Completed at: $(date)"
