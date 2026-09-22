#!/usr/bin/env bash
#SBATCH --job-name=lnp_cpu_models
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/cpu_models_%j.out
#SBATCH --error=logs/cpu_models_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
source /home/sa594/LNP_ML/transfection_binary/environment/activate_lnp_ml.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python scripts/02_train.py \
  --models majority logistic random_forest xgboost lightgbm \
  --schemes grouped random
python scripts/03_summarize.py

