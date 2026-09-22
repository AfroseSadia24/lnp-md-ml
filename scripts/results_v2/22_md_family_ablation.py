from pathlib import Path
import json
import os

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
METRICS = ROOT / "results_v2/metrics"
PREDICTIONS = ROOT / "results_v2/predictions"
MODELS = ROOT / "results_v2/models/md_family_ablation"

DATA_PATH = TABLES / "exact_md_subset.csv"

TARGET = "transfection_active"
SOURCE = "md_source_lnp_id"

NUMBER_OF_JOBS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        "1",
    )
)


def read_list(filename):
    return [
        value.strip()
        for value in (
            TABLES / filename
        ).read_text().splitlines()
        if value.strip()
    ]


baseline_numeric = read_list(
    "baseline_numeric_features.txt"
)

baseline_categorical = read_list(
    "baseline_categorical_features.txt"
)

approved_md = read_list(
    "md_features.txt"
)


md_families = {
    "size_shape": [
        "md_LNP_only_Rg_nm_mean_last100ns",
        "md_LNP_core_Rg_nm_mean_last100ns",
        "md_LNP_only_SASA_nm2_mean_last100ns",
        "md_RNA_Rg_nm_mean_last100ns",
        "md_LNP_RNA_Rg_nm_mean_last100ns",
        (
            "md_LNP_only_shape_anisotropy_"
            "kappa2_mean_last100ns"
        ),
        (
            "md_LNP_only_compactness_inv_"
            "Rg_1_per_nm_mean_last100ns"
        ),
    ],
    "rna_association": [
        (
            "md_RNA_LNP_COM_distance_nm_"
            "mean_last100ns"
        ),
        (
            "md_RNA_LNP_mindist_nm_"
            "mean_last100ns"
        ),
        (
            "md_RNA_LNP_contacts_0p6nm_"
            "mean_last100ns"
        ),
        (
            "md_RNA_buried_fraction_contact_"
            "based_mean_last100ns"
        ),
    ],
    "water_penetration": [
        (
            "md_water_inside_LNP_core_RgSphere_"
            "count_fraction_of_total_water_"
            "mean_last100ns"
        ),
    ],
}


family_features = [
    feature
    for family in md_families.values()
    for feature in family
]

if len(family_features) != len(
    set(family_features)
):
    raise RuntimeError(
        "An MD descriptor occurs in multiple families."
    )

if set(family_features) != set(approved_md):
    missing_from_families = sorted(
        set(approved_md) - set(family_features)
    )

    unexpected = sorted(
        set(family_features) - set(approved_md)
    )

    raise RuntimeError(
        "MD family assignment does not match the "
        "approved MD manifest.\n"
        f"Missing: {missing_from_families}\n"
        f"Unexpected: {unexpected}"
    )


feature_sets = {
    "baseline": [],
    "add_size_shape": (
        md_families["size_shape"]
    ),
    "add_rna_association": (
        md_families["rna_association"]
    ),
    "add_water_penetration": (
        md_families["water_penetration"]
    ),
    "all_md": approved_md,
    "all_except_size_shape": [
        feature
        for feature in approved_md
        if feature not in md_families["size_shape"]
    ],
    "all_except_rna_association": [
        feature
        for feature in approved_md
        if feature
        not in md_families["rna_association"]
    ],
    "all_except_water_penetration": [
        feature
        for feature in approved_md
        if feature
        not in md_families["water_penetration"]
    ],
}

feature_set_order = [
    "baseline",
    "add_size_shape",
    "add_rna_association",
    "add_water_penetration",
    "all_except_size_shape",
    "all_except_rna_association",
    "all_except_water_penetration",
    "all_md",
]


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
).reset_index(drop=True)

data[TARGET] = data[TARGET].astype(int)
data[SOURCE] = data[SOURCE].astype(str)

all_numeric = (
    baseline_numeric
    + approved_md
)

for column in all_numeric:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

if len(data) != 13:
    raise RuntimeError(
        f"Expected 13 rows, found {len(data)}."
    )

sources = sorted(
    data[SOURCE].unique()
)

if len(sources) != 6:
    raise RuntimeError(
        f"Expected six MD sources, found {sources}."
    )

missing_columns = [
    feature
    for feature in (
        baseline_numeric
        + baseline_categorical
        + approved_md
    )
    if feature not in data.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing features: {missing_columns}"
    )

source_unique_counts = data.groupby(
    SOURCE
)[approved_md].nunique(dropna=False)

if (source_unique_counts > 1).any().any():
    problem_sources = source_unique_counts.loc[
        (source_unique_counts > 1).any(axis=1)
    ].index.tolist()

    raise RuntimeError(
        "Multiple MD vectors occur within these "
        f"sources: {problem_sources}"
    )


def source_weights(source_values):
    values = pd.Series(
        source_values
    ).reset_index(drop=True)

    sizes = values.groupby(
        values
    ).transform("size")

    return (
        1.0
        / sizes.to_numpy(dtype=float)
    )


def build_pipeline(
    numeric_features,
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

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                baseline_categorical,
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
        n_jobs=NUMBER_OF_JOBS,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    y_true,
    prediction,
    probability,
    weights=None,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
        sample_weight=weights,
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
                sample_weight=weights,
            )
        ),
        "pr_auc_active": float(
            average_precision_score(
                y_true,
                probability,
                sample_weight=weights,
            )
        ),
        "pr_auc_inactive": float(
            average_precision_score(
                1 - y_true,
                1 - probability,
                sample_weight=weights,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
                sample_weight=weights,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                prediction,
                sample_weight=weights,
            )
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                prediction,
                sample_weight=weights,
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
                sample_weight=weights,
                zero_division=0,
            )
        ),
    }


MODELS.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_tables = []


for fold, held_source in enumerate(sources):
    test_mask = data[SOURCE].eq(held_source)

    training = data.loc[
        ~test_mask
    ].copy()

    testing = data.loc[
        test_mask
    ].copy()

    if training[TARGET].nunique() != 2:
        raise RuntimeError(
            f"Training data lacks a class when "
            f"holding out {held_source}."
        )

    training_weights = source_weights(
        training[SOURCE]
    )

    training_weights = (
        training_weights
        * len(training_weights)
        / training_weights.sum()
    )

    for feature_set_name in feature_set_order:
        current_md = feature_sets[
            feature_set_name
        ]

        numeric_features = (
            baseline_numeric
            + current_md
        )

        model_features = (
            numeric_features
            + baseline_categorical
        )

        # Same seed for every feature set in a fold.
        random_state = 2000 + fold

        pipeline = build_pipeline(
            numeric_features,
            random_state,
        )

        pipeline.fit(
            training[model_features],
            training[TARGET].to_numpy(
                dtype=int
            ),
            classifier__sample_weight=(
                training_weights
            ),
        )

        probability = pipeline.predict_proba(
            testing[model_features]
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        output = testing[
            [
                "analysis_row_id",
                "lnp_id",
                "paper_doi",
                SOURCE,
                TARGET,
            ]
        ].copy()

        output["fold"] = fold
        output["held_source"] = held_source
        output["feature_set"] = feature_set_name
        output["number_md_features"] = len(
            current_md
        )
        output["prob_active"] = probability
        output["prediction"] = prediction

        prediction_tables.append(output)

        joblib.dump(
            pipeline,
            MODELS
            / (
                f"{feature_set_name}_"
                f"holdout_{held_source}.joblib"
            ),
        )

        print(
            f"Finished {feature_set_name}: "
            f"held out {held_source}",
            flush=True,
        )


predictions = pd.concat(
    prediction_tables,
    ignore_index=True,
)

predictions.to_csv(
    PREDICTIONS
    / "md_family_ablation_loso_predictions.csv",
    index=False,
)


summary_rows = []

for feature_set_name in feature_set_order:
    model_data = predictions.loc[
        predictions["feature_set"].eq(
            feature_set_name
        )
    ].copy()

    y_true = model_data[TARGET].to_numpy(
        dtype=int
    )

    prediction = model_data[
        "prediction"
    ].to_numpy(dtype=int)

    probability = model_data[
        "prob_active"
    ].to_numpy(dtype=float)

    row_metrics = calculate_metrics(
        y_true,
        prediction,
        probability,
    )

    balanced_weights = source_weights(
        model_data[SOURCE]
    )

    balanced_metrics = calculate_metrics(
        y_true,
        prediction,
        probability,
        weights=balanced_weights,
    )

    for weighting, result in [
        ("row_weighted", row_metrics),
        ("source_balanced", balanced_metrics),
    ]:
        summary_rows.append(
            {
                "feature_set": feature_set_name,
                "weighting": weighting,
                "number_md_features": len(
                    feature_sets[
                        feature_set_name
                    ]
                ),
                **result,
            }
        )


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    METRICS
    / "md_family_ablation_summary.csv",
    index=False,
)


metric_names = [
    "roc_auc",
    "pr_auc_active",
    "pr_auc_inactive",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
]

delta_rows = []

for weighting in [
    "row_weighted",
    "source_balanced",
]:
    weighted = summary.loc[
        summary["weighting"].eq(weighting)
    ].set_index("feature_set")

    for feature_set_name in feature_set_order:
        if feature_set_name == "baseline":
            continue

        for metric in metric_names:
            baseline_value = float(
                weighted.loc[
                    "baseline",
                    metric,
                ]
            )

            model_value = float(
                weighted.loc[
                    feature_set_name,
                    metric,
                ]
            )

            delta_rows.append(
                {
                    "weighting": weighting,
                    "feature_set": feature_set_name,
                    "metric": metric,
                    "baseline": baseline_value,
                    "model_value": model_value,
                    "delta_vs_baseline": (
                        model_value
                        - baseline_value
                    ),
                }
            )


deltas = pd.DataFrame(delta_rows)

deltas.to_csv(
    METRICS
    / "md_family_ablation_deltas.csv",
    index=False,
)


family_effect_rows = []

for weighting in [
    "row_weighted",
    "source_balanced",
]:
    weighted = summary.loc[
        summary["weighting"].eq(weighting)
    ].set_index("feature_set")

    for family_name in md_families:
        add_name = f"add_{family_name}"
        remove_name = (
            f"all_except_{family_name}"
        )

        for metric in metric_names:
            baseline_value = float(
                weighted.loc[
                    "baseline",
                    metric,
                ]
            )

            add_value = float(
                weighted.loc[
                    add_name,
                    metric,
                ]
            )

            full_value = float(
                weighted.loc[
                    "all_md",
                    metric,
                ]
            )

            reduced_value = float(
                weighted.loc[
                    remove_name,
                    metric,
                ]
            )

            family_effect_rows.append(
                {
                    "weighting": weighting,
                    "family": family_name,
                    "number_features": len(
                        md_families[family_name]
                    ),
                    "metric": metric,
                    "add_family_delta_vs_baseline": (
                        add_value
                        - baseline_value
                    ),
                    "leave_family_out_effect": (
                        full_value
                        - reduced_value
                    ),
                }
            )


family_effects = pd.DataFrame(
    family_effect_rows
)

family_effects.to_csv(
    METRICS
    / "md_family_ablation_family_effects.csv",
    index=False,
)


previous_path = (
    METRICS
    / "md_source_loso_summary.csv"
)

if previous_path.exists():
    previous = pd.read_csv(previous_path)

    checks = [
        (
            "baseline",
            "baseline",
        ),
        (
            "all_md",
            "baseline_plus_md",
        ),
    ]

    for current_name, previous_name in checks:
        for weighting in [
            "row_weighted",
            "source_balanced",
        ]:
            current_row = summary.loc[
                summary["feature_set"].eq(
                    current_name
                )
                & summary["weighting"].eq(
                    weighting
                )
            ].iloc[0]

            previous_row = previous.loc[
                previous["model"].eq(
                    previous_name
                )
                & previous["weighting"].eq(
                    weighting
                )
            ].iloc[0]

            for metric in metric_names:
                if not np.isclose(
                    current_row[metric],
                    previous_row[metric],
                    atol=1e-10,
                    rtol=0,
                ):
                    raise RuntimeError(
                        "Ablation result does not "
                        "reproduce the previous LOSO "
                        f"result: {current_name}, "
                        f"{weighting}, {metric}"
                    )


family_manifest_rows = []

for family_name, features in (
    md_families.items()
):
    for feature in features:
        family_manifest_rows.append(
            {
                "family": family_name,
                "feature": feature,
            }
        )

pd.DataFrame(
    family_manifest_rows
).to_csv(
    TABLES
    / "md_family_manifest.csv",
    index=False,
)


config = {
    "analysis": (
        "Exploratory MD descriptor-family ablation"
    ),
    "validation": (
        "leave-one-MD-source-out"
    ),
    "training_weighting": (
        "equal total weight per MD source"
    ),
    "independent_md_sources": len(sources),
    "rows": len(data),
    "algorithm": "ExtraTreesClassifier",
    "n_estimators": 800,
    "class_weight": "balanced",
    "min_samples_leaf": 2,
    "random_state_rule": (
        "2000 plus LOSO fold; identical across "
        "all feature sets"
    ),
    "families": md_families,
}

with open(
    METRICS
    / "md_family_ablation_config.json",
    "w",
) as handle:
    json.dump(
        config,
        handle,
        indent=2,
    )


source_balanced_summary = summary.loc[
    summary["weighting"].eq(
        "source_balanced"
    ),
    [
        "feature_set",
        "number_md_features",
        "roc_auc",
        "pr_auc_active",
        "pr_auc_inactive",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
    ],
].copy()

source_balanced_effects = family_effects.loc[
    family_effects["weighting"].eq(
        "source_balanced"
    )
    & family_effects["metric"].isin(
        [
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "roc_auc",
            "pr_auc_active",
            "pr_auc_inactive",
        ]
    )
].copy()


print()
print(
    "===== SOURCE-BALANCED FAMILY ABLATION ====="
)

print(
    source_balanced_summary.to_string(
        index=False
    )
)

print()
print(
    "===== SOURCE-BALANCED FAMILY EFFECTS ====="
)

print(
    source_balanced_effects.to_string(
        index=False
    )
)

print()
print(
    "PASS: baseline and full-MD results reproduce "
    "the previous source-LOSO analysis."
)

print()
print(
    "Written: results_v2/metrics/"
    "md_family_ablation_summary.csv"
)

print(
    "Written: results_v2/metrics/"
    "md_family_ablation_deltas.csv"
)

print(
    "Written: results_v2/metrics/"
    "md_family_ablation_family_effects.csv"
)

print(
    "Written: results_v2/predictions/"
    "md_family_ablation_loso_predictions.csv"
)

print(
    "Written: results_v2/tables/"
    "md_family_manifest.csv"
)
