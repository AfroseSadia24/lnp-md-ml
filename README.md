# LNP transfection ML workflow for HPC

This package implements the uploaded frozen preprocessing and evaluation plan for the currently available transfection dataset.

## What the current dataset supports

- Binary transfection endpoint: 452 rows, 335 active and 117 inactive.
- Paper-grouped evaluation using normalized DOI groups.
- Random-vs-grouped evaluation gap.
- Majority, logistic regression, random forest, XGBoost, LightGBM, and MLP models.
- Fold-safe imputation, encoding, scaling, variance filtering, and correlation filtering.
- Sensitivity analysis that excludes the curator-assigned ID 803 label.

The uploaded CSV does not contain the organ-tropism endpoint or the 377-column MD table. The code therefore does not invent those analyses. Add those files later using the schemas described under `data/README.md`.

## First HPC commands

```bash
cd /home/sa594/LNP_ML
mkdir -p transfection_binary
cd transfection_binary
# Transfer this package here, then:
bash setup_env.sh
source activate_lnp_ml.sh
mkdir -p data/raw
cp /path/to/LNP_transfection_binary_target_with_803.csv data/raw/
python scripts/00_audit.py --input data/raw/LNP_transfection_binary_target_with_803.csv
python scripts/01_preprocess.py --input data/raw/LNP_transfection_binary_target_with_803.csv
```

Run CPU models interactively for a quick check:

```bash
python scripts/02_train.py --models majority logistic random_forest --schemes grouped random --quick
```

Submit the full CPU benchmark:

```bash
sbatch slurm/run_cpu_models.slurm
```

Submit the GPU MLP after preprocessing finishes:

```bash
sbatch slurm/run_gpu_mlp.slurm
```

Build the combined tables and figures after all jobs finish:

```bash
python scripts/03_summarize.py
```

For your current dataset, the quick smoke test produced a clear split-protocol gap. Logistic regression reached macro-F1 0.649 under random folds but 0.531 under DOI-grouped folds. Treat this only as a pipeline check. The full nested-CV run is the reportable analysis.

## Output map

```text
data/processed/transfection_modelready_v1.csv
data/processed/folds_grouped_v1.csv
data/processed/folds_random_v1.csv
reports/qc_report.json
reports/data_dictionary.csv
results/predictions/*.csv
results/metrics/*.json
results/main_results.csv
figures/split_scheme_gap.png
figures/model_benchmark.png
models/*.joblib
logs/*.out
```

## Scientific rules implemented

1. DOI strings are normalized before grouping.
2. DOI is never used as a model feature.
3. Route and label-source fields are audit fields, not headline features.
4. All fitted preprocessing occurs inside each training fold.
5. SMOTE and target encoding are not used.
6. Grouped folds are the primary result. Random folds are used only to measure optimism.
7. Macro-F1, MCC, and minority-class PR-AUC are the main metrics.
8. ID 803 is retained but flagged. Use `--exclude-id-803` for the sensitivity run.
9. Hyperparameter selection uses inner CV. Outer test folds remain untouched.
10. Per-row predictions are saved. Summary figures are regenerated from those files.

## Recommended run order

```bash
python scripts/00_audit.py --input data/raw/LNP_transfection_binary_target_with_803.csv
python scripts/01_preprocess.py --input data/raw/LNP_transfection_binary_target_with_803.csv
sbatch slurm/run_cpu_models.slurm
sbatch slurm/run_gpu_mlp.slurm
python scripts/03_summarize.py
```

Monitor jobs with:

```bash
squeue -u "$USER"
tail -f logs/cpu_models_<JOBID>.out
tail -f logs/gpu_mlp_<JOBID>.out
```
