#!/usr/bin/env bash
#SBATCH --job-name=lnp_mlp
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gpu_mlp_%j.out
#SBATCH --error=logs/gpu_mlp_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
source /home/sa594/LNP_ML/transfection_binary/environment/activate_lnp_ml.sh

# This first tabular MLP is CPU-backed. Reserve GPU jobs for the later
# PyTorch multimodal fusion, Chemprop, and AttentiveFP stages.
python scripts/02_train.py --models mlp --schemes grouped random
python scripts/03_summarize.py
