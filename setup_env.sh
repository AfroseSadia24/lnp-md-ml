#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${PWD}/.micromamba/envs/lnp_ml"
mkdir -p "${PWD}/.micromamba/envs"

if command -v micromamba >/dev/null 2>&1; then
  micromamba create -y -p "$ENV_PREFIX" -c conda-forge \
    python=3.11 pandas=2.2 numpy=1.26 scipy=1.13 scikit-learn=1.5 \
    matplotlib=3.9 seaborn=0.13 joblib=1.4 pyyaml=6.0 \
    xgboost=2.1 lightgbm=4.5 pytorch=2.4 cpuonly -c pytorch
else
  echo "micromamba was not found. Load your micromamba module or install it first."
  exit 1
fi

cat > activate_lnp_ml.sh <<EOF
#!/usr/bin/env bash
eval "\$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_PREFIX"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
EOF
chmod +x activate_lnp_ml.sh
echo "Environment created. Run: source activate_lnp_ml.sh"

