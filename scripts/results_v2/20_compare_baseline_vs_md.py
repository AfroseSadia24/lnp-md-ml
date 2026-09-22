from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")
TABLES = ROOT / "results_v2/tables"
METRICS_DIR = ROOT / "results_v2/metrics"
PREDICTIONS_DIR = ROOT / "results_v2/predictions"
MODELS_DIR = ROOT / "results_v2/models"

DATA_PATH = TABLES / "exact_md_subset.csv"
FOLD_PATH = TABLES / "doi_grouped_folds.csv"

TARGET = "transfection_active"


def read_feature_list(filename):
    path = TABLES / filename

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


baseline_numeric = read_feature_list(
    "baseline_numeric_features.txt"
)

baseline_categorical = read_feature_list(
    "baseline_categorical_features.txt"
)

md_features = read_feature_list(
    "md_features.txt"
)

baseline_features = (
    baseline_numeric
    + baseline_categorical
)

enhanced_numeric = (
    baseline_numeric
    + md_features
)

enhanced_features = (
    enhanced_numeric
    + baseline_categorical
)


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

folds = pd.read_csv(
    FOLD_PATH,
    low_memory=False,
)

if len(data) != 13:
    raise RuntimeError(
        f"Expected 13 rows, found {len(data)}."
    )

if len(baseline_features) != 80:
    raise RuntimeError(
        "Expected 80 baseline features."
    )

if len(enhanced_features) != 92:
    raise RuntimeError(
        "Expected 92 enhanced features."
    )

missing_features = [
    feature
    for feature in enhanced_features
    if feature not in data.columns
]

if missing_features:
    raise RuntimeError(
        f"Missing features: {missing_features}"
    )

if folds["analysis_row_id"].duplicated().any():
    raise RuntimeError(
        "Duplicated row IDs in fold table."
    )

data = data.merge(
    folds[
        [
            "analysis_row_id",
            "fold",
        ]
    ],
    on="analysis_row_id",
    how="left",
    validate="one_to_one",
)

if data["fold"].isna().any():
    raise RuntimeError(
        "Some rows do not have a fold assignment."
    )

data["fold"] = data["fold"].astype(int)
data[TARGET] = data[TARGET].astype(int)

for column in enhanced_numeric:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


def build_pipeline(
    numeric_features,
    categorical_features,
    random_state,
):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = ExtraTreesClassifier(
        n_estimators=800,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    y_true,
    prediction,
    probability,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
        "pr_auc_active": float(
            average_precision_score(
                y_true,
                probability,
            )
        ),
        "pr_auc_inactive": float(
            average_precision_score(
                1 - y_true,
                1 - probability,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_active": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                prediction,
            )
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "specificity": float(
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),
        "precision_active": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


feature_sets = {
    "baseline": {
        "numeric": baseline_numeric,
        "categorical": baseline_categorical,
        "all": baseline_features,
    },
    "baseline_plus_md": {
        "numeric": enhanced_numeric,
        "categorical": baseline_categorical,
        "all": enhanced_features,
    },
}


METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


metric_rows = []
prediction_tables = []

for model_number, (
    model_name,
    feature_information,
) in enumerate(feature_sets.items()):

    numeric_features = feature_information[
        "numeric"
    ]

    categorical_features = feature_information[
        "categorical"
    ]

    all_features = feature_information["all"]

    for fold in sorted(data["fold"].unique()):
        training = data[
            data["fold"] != fold
        ].copy()

        testing = data[
            data["fold"] == fold
        ].copy()

        y_train = training[TARGET].to_numpy(
            dtype=int
        )

        y_test = testing[TARGET].to_numpy(
            dtype=int
        )

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                f"Training fold {fold} lacks a class."
            )

        if len(np.unique(y_test)) != 2:
            raise RuntimeError(
                f"Testing fold {fold} lacks a class."
            )

        # Use the same seed for both feature sets.
        # Only the feature block is allowed to change.
        random_state = 1000 + int(fold)

        pipeline = build_pipeline(
            numeric_features,
            categorical_features,
            random_state,
        )

        pipeline.fit(
            training[all_features],
            y_train,
        )

        probability = pipeline.predict_proba(
            testing[all_features]
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        fold_metrics = calculate_metrics(
            y_test,
            prediction,
            probability,
        )

        metric_rows.append(
            {
                "model": model_name,
                "fold": int(fold),
                "train_rows": int(len(training)),
                "test_rows": int(len(testing)),
                "number_numeric_features": int(
                    len(numeric_features)
                ),
                "number_categorical_features": int(
                    len(categorical_features)
                ),
                **fold_metrics,
            }
        )

        prediction_table = testing[
            [
                "analysis_row_id",
                "lnp_id",
                "paper_doi",
                "md_source_lnp_id",
                TARGET,
                "fold",
            ]
        ].copy()

        prediction_table["model"] = model_name
        prediction_table["prob_active"] = probability
        prediction_table["prediction"] = prediction

        prediction_tables.append(
            prediction_table
        )

        model_path = (
            MODELS_DIR
            / f"{model_name}_fold_{fold}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        print(
            f"Finished {model_name}, fold {fold}, "
            f"test rows={len(testing)}",
            flush=True,
        )


fold_metrics = pd.DataFrame(metric_rows)

predictions = pd.concat(
    prediction_tables,
    ignore_index=True,
)

fold_metrics.to_csv(
    METRICS_DIR
    / "baseline_vs_md_by_fold.csv",
    index=False,
)

predictions.to_csv(
    PREDICTIONS_DIR
    / "baseline_vs_md_oof.csv",
    index=False,
)


reported_metrics = [
    "roc_auc",
    "pr_auc_active",
    "pr_auc_inactive",
    "balanced_accuracy",
    "macro_f1",
    "f1_active",
    "mcc",
    "sensitivity",
    "specificity",
    "precision_active",
]

summary_rows = []

for model_name, model_table in (
    fold_metrics.groupby("model")
):
    for metric in reported_metrics:
        summary_rows.append(
            {
                "model": model_name,
                "metric": metric,
                "mean": float(
                    model_table[metric].mean()
                ),
                "standard_deviation": float(
                    model_table[metric].std(
                        ddof=1
                    )
                ),
            }
        )

fold_summary = pd.DataFrame(summary_rows)

fold_summary.to_csv(
    METRICS_DIR
    / "baseline_vs_md_fold_summary.csv",
    index=False,
)


pooled_rows = []

for model_name, model_predictions in (
    predictions.groupby("model")
):
    pooled_metrics = calculate_metrics(
        model_predictions[TARGET].to_numpy(
            dtype=int
        ),
        model_predictions[
            "prediction"
        ].to_numpy(dtype=int),
        model_predictions[
            "prob_active"
        ].to_numpy(dtype=float),
    )

    pooled_rows.append(
        {
            "model": model_name,
            "rows": int(len(model_predictions)),
            **pooled_metrics,
        }
    )

pooled_summary = pd.DataFrame(pooled_rows)

pooled_summary.to_csv(
    METRICS_DIR
    / "baseline_vs_md_oof_summary.csv",
    index=False,
)


wide = fold_metrics.pivot(
    index="fold",
    columns="model",
    values=reported_metrics,
)

delta_rows = []

for fold in sorted(data["fold"].unique()):
    for metric in reported_metrics:
        baseline_value = float(
            wide.loc[
                fold,
                (metric, "baseline"),
            ]
        )

        enhanced_value = float(
            wide.loc[
                fold,
                (
                    metric,
                    "baseline_plus_md",
                ),
            ]
        )

        delta_rows.append(
            {
                "fold": int(fold),
                "metric": metric,
                "baseline": baseline_value,
                "baseline_plus_md": (
                    enhanced_value
                ),
                "delta_md_minus_baseline": (
                    enhanced_value
                    - baseline_value
                ),
            }
        )

paired_deltas = pd.DataFrame(delta_rows)

paired_deltas.to_csv(
    METRICS_DIR
    / "baseline_vs_md_paired_deltas.csv",
    index=False,
)


manifest_hashes = {}

for filename in [
    "baseline_numeric_features.txt",
    "baseline_categorical_features.txt",
    "md_features.txt",
    "doi_grouped_folds.csv",
]:
    file_path = TABLES / filename

    manifest_hashes[filename] = (
        hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()
    )

run_configuration = {
    "algorithm": "ExtraTreesClassifier",
    "n_estimators": 800,
    "class_weight": "balanced",
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "classification_threshold": 0.5,
    "validation": "DOI-grouped four-fold CV",
    "baseline_numeric_features": len(
        baseline_numeric
    ),
    "baseline_categorical_features": len(
        baseline_categorical
    ),
    "md_features": len(md_features),
    "rows": len(data),
    "target": TARGET,
    "file_hashes": manifest_hashes,
}

with open(
    METRICS_DIR
    / "baseline_vs_md_run_config.json",
    "w",
) as handle:
    json.dump(
        run_configuration,
        handle,
        indent=2,
    )


print()
print("===== FOLD-LEVEL RESULTS =====")

display_columns = [
    "model",
    "fold",
    "roc_auc",
    "pr_auc_active",
    "pr_auc_inactive",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
]

print(
    fold_metrics[
        display_columns
    ].to_string(index=False)
)

print()
print("===== POOLED OUT-OF-FOLD RESULTS =====")

print(
    pooled_summary[
        [
            "model",
            "roc_auc",
            "pr_auc_active",
            "pr_auc_inactive",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
        ]
    ].to_string(index=False)
)

print()
print("===== MEAN PAIRED DIFFERENCES =====")

mean_deltas = (
    paired_deltas.groupby("metric")[
        "delta_md_minus_baseline"
    ]
    .mean()
    .reset_index()
)

print(mean_deltas.to_string(index=False))

print()
print(
    "Written: results_v2/metrics/"
    "baseline_vs_md_by_fold.csv"
)
print(
    "Written: results_v2/metrics/"
    "baseline_vs_md_fold_summary.csv"
)
print(
    "Written: results_v2/metrics/"
    "baseline_vs_md_oof_summary.csv"
)
print(
    "Written: results_v2/metrics/"
    "baseline_vs_md_paired_deltas.csv"
)
print(
    "Written: results_v2/predictions/"
    "baseline_vs_md_oof.csv"
)
