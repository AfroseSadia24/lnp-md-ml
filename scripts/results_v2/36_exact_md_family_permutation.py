from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

DATA_PATH = (
    ROOT
    / "results_v2/tables/exact_md_subset.csv"
)

TABLE_DIRECTORY = ROOT / "results_v2/tables"
OUTPUT_DIRECTORY = ROOT / "results_v2/permutation"

NUMBER_OF_WORKERS = int(
    os.environ.get("SLURM_CPUS_PER_TASK", "1")
)

TARGET = "transfection_active"
SOURCE = "md_source_lnp_id"

METRICS = [
    "roc_auc",
    "pr_auc_active",
    "pr_auc_inactive",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
]


def read_list(filename):
    path = TABLE_DIRECTORY / filename

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def normalized_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def source_weights(groups):
    groups = pd.Series(groups)

    sizes = groups.groupby(
        groups
    ).transform("size")

    return 1.0 / sizes.to_numpy(dtype=float)


def calculate_metrics(
    y_true,
    prediction,
    probability,
    groups,
):
    weights = source_weights(groups)

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
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                prediction,
                sample_weight=weights,
            )
        ),
    }


def make_pipeline(
    numeric_features,
    categorical_features,
    random_state,
):
    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        [
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
        sparse_threshold=0,
    )

    classifier = ExtraTreesClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", classifier),
        ]
    )


def evaluate_model(
    working_data,
    added_md_features,
):
    numeric_features = (
        BASELINE_NUMERIC
        + list(added_md_features)
    )

    model_features = (
        numeric_features
        + BASELINE_CATEGORICAL
    )

    y = working_data[
        TARGET
    ].to_numpy(dtype=int)

    groups = working_data[
        SOURCE
    ].to_numpy(dtype=str)

    predictions = np.empty(
        len(working_data),
        dtype=int,
    )

    probabilities = np.empty(
        len(working_data),
        dtype=float,
    )

    for fold, held_source in enumerate(SOURCES):
        testing_mask = groups == held_source
        training_mask = ~testing_mask

        y_train = y[training_mask]

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                "Training fold lacks both classes: "
                f"{held_source}"
            )

        training_groups = groups[
            training_mask
        ]

        training_weights = source_weights(
            training_groups
        )

        pipeline = make_pipeline(
            numeric_features,
            BASELINE_CATEGORICAL,
            random_state=2000 + fold,
        )

        pipeline.fit(
            working_data.loc[
                training_mask,
                model_features,
            ],
            y_train,
            model__sample_weight=training_weights,
        )

        probability = pipeline.predict_proba(
            working_data.loc[
                testing_mask,
                model_features,
            ]
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        probabilities[testing_mask] = probability
        predictions[testing_mask] = prediction

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
        groups,
    )

    return metrics


def evaluate_assignment(
    family,
    permutation_number,
    donors,
):
    features = FAMILY_FEATURES[family]

    permuted_data = DATA.copy()

    for recipient, donor in zip(
        SOURCES,
        donors,
    ):
        recipient_mask = (
            permuted_data[SOURCE] == recipient
        )

        donor_vector = (
            FAMILY_VECTORS[family]
            .loc[donor, features]
            .to_numpy(dtype=float)
        )

        permuted_data.loc[
            recipient_mask,
            features,
        ] = np.tile(
            donor_vector,
            (
                int(recipient_mask.sum()),
                1,
            ),
        )

    family_metrics = evaluate_model(
        permuted_data,
        features,
    )

    is_identity = all(
        recipient == donor
        for recipient, donor in zip(
            SOURCES,
            donors,
        )
    )

    result = {
        "family": family,
        "number_features": len(features),
        "permutation": permutation_number,
        "is_identity": is_identity,
        "assignment": ";".join(
            f"{recipient}<-{donor}"
            for recipient, donor in zip(
                SOURCES,
                donors,
            )
        ),
    }

    for metric in METRICS:
        result[metric] = family_metrics[metric]

        result[
            f"delta_{metric}"
        ] = (
            family_metrics[metric]
            - BASELINE_METRICS[metric]
        )

    return result


def holm_adjust(raw_p_values):
    raw_p_values = np.asarray(
        raw_p_values,
        dtype=float,
    )

    number_tests = len(raw_p_values)
    order = np.argsort(raw_p_values)

    adjusted = np.empty(
        number_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, position in enumerate(order):
        multiplier = number_tests - rank

        candidate = min(
            1.0,
            multiplier
            * raw_p_values[position],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[position] = running_maximum

    return adjusted


DATA = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

BASELINE_NUMERIC = read_list(
    "baseline_numeric_features.txt"
)

BASELINE_CATEGORICAL = read_list(
    "baseline_categorical_features.txt"
)

APPROVED_MD = read_list(
    "md_features.txt"
)

for column in BASELINE_CATEGORICAL:
    DATA[column] = DATA[column].where(
        DATA[column].isna(),
        DATA[column].astype(str),
    )

DATA[SOURCE] = DATA[SOURCE].astype(str)
DATA[TARGET] = DATA[TARGET].astype(int)

SOURCES = sorted(
    DATA[SOURCE].unique()
)

if len(SOURCES) != 6:
    raise RuntimeError(
        f"Expected six MD sources, found {SOURCES}"
    )

manifest_path = (
    TABLE_DIRECTORY
    / "md_family_manifest.csv"
)

manifest = pd.read_csv(manifest_path)

family_column_candidates = [
    "family",
    "md_family",
    "feature_family",
]

feature_column_candidates = [
    "feature",
    "md_feature",
    "column",
    "descriptor",
]

family_column = next(
    (
        column
        for column in family_column_candidates
        if column in manifest.columns
    ),
    None,
)

feature_column = next(
    (
        column
        for column in feature_column_candidates
        if column in manifest.columns
    ),
    None,
)

if family_column is None or feature_column is None:
    raise RuntimeError(
        "Could not identify the family and feature "
        "columns in md_family_manifest.csv. "
        f"Columns found: {manifest.columns.tolist()}"
    )

manifest["normalized_family"] = (
    manifest[family_column].map(
        normalized_name
    )
)

family_aliases = {
    "size_shape": {
        "size_shape",
        "size_and_shape",
    },
    "rna_association": {
        "rna_association",
        "rna_contact",
        "rna_contacts",
    },
    "water_penetration": {
        "water_penetration",
        "water",
    },
}

FAMILY_FEATURES = {}

for standard_name, aliases in family_aliases.items():
    selected = manifest[
        manifest["normalized_family"].isin(
            aliases
        )
    ][feature_column].astype(str).tolist()

    FAMILY_FEATURES[standard_name] = selected

expected_counts = {
    "size_shape": 7,
    "rna_association": 4,
    "water_penetration": 1,
}

for family, expected_count in expected_counts.items():
    found = len(FAMILY_FEATURES[family])

    if found != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} features for "
            f"{family}, found {found}: "
            f"{FAMILY_FEATURES[family]}"
        )

manifest_features = set(
    feature
    for features in FAMILY_FEATURES.values()
    for feature in features
)

if manifest_features != set(APPROVED_MD):
    raise RuntimeError(
        "Family manifest does not exactly match "
        "the approved MD feature list."
    )

missing_columns = sorted(
    set(
        BASELINE_NUMERIC
        + BASELINE_CATEGORICAL
        + APPROVED_MD
        + [TARGET, SOURCE]
    )
    - set(DATA.columns)
)

if missing_columns:
    raise RuntimeError(
        f"Missing dataset columns: {missing_columns}"
    )

if not np.isfinite(
    DATA[APPROVED_MD].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Nonfinite MD values detected."
    )

FAMILY_VECTORS = {}

for family, features in FAMILY_FEATURES.items():
    unique_counts = DATA.groupby(
        SOURCE
    )[features].nunique(dropna=False)

    if (unique_counts > 1).any().any():
        raise RuntimeError(
            f"More than one {family} vector "
            "was found within an MD source."
        )

    FAMILY_VECTORS[family] = (
        DATA.groupby(
            SOURCE,
            sort=True,
        )[features]
        .first()
        .loc[SOURCES]
    )

BASELINE_METRICS = evaluate_model(
    DATA,
    [],
)


def main():
    print("===== FAMILY MANIFEST =====")

    for family, features in FAMILY_FEATURES.items():
        print(
            f"{family}: {len(features)} features"
        )

        for feature in features:
            print(f"  {feature}")

    print()
    print("===== BASELINE SOURCE-BALANCED METRICS =====")

    for metric in METRICS:
        print(
            f"{metric}: "
            f"{BASELINE_METRICS[metric]:.6f}"
        )

    permutations = list(
        itertools.permutations(SOURCES)
    )

    print()
    print(
        "Permutations per family:",
        len(permutations),
    )

    print(
        "Total model assignments:",
        len(permutations)
        * len(FAMILY_FEATURES),
    )

    print(
        "Parallel workers:",
        NUMBER_OF_WORKERS,
        flush=True,
    )

    tasks = []

    for family in FAMILY_FEATURES:
        for permutation_number, donors in enumerate(
            permutations,
            start=1,
        ):
            tasks.append(
                (
                    family,
                    permutation_number,
                    donors,
                )
            )

    results = Parallel(
        n_jobs=NUMBER_OF_WORKERS,
        verbose=10,
    )(
        delayed(evaluate_assignment)(
            family,
            permutation_number,
            donors,
        )
        for (
            family,
            permutation_number,
            donors,
        ) in tasks
    )

    distribution = pd.DataFrame(results)

    for family in FAMILY_FEATURES:
        family_rows = distribution[
            distribution["family"] == family
        ]

        if len(family_rows) != 720:
            raise RuntimeError(
                f"{family} has {len(family_rows)} "
                "permutations instead of 720."
            )

        if family_rows[
            "is_identity"
        ].sum() != 1:
            raise RuntimeError(
                f"{family} does not have exactly "
                "one identity assignment."
            )

    pvalue_rows = []

    for family in FAMILY_FEATURES:
        family_rows = distribution[
            distribution["family"] == family
        ]

        observed_row = family_rows[
            family_rows["is_identity"]
        ].iloc[0]

        for metric in METRICS:
            delta_column = f"delta_{metric}"

            observed_delta = float(
                observed_row[delta_column]
            )

            exact_p = float(
                np.mean(
                    family_rows[delta_column]
                    >= observed_delta - 1e-12
                )
            )

            pvalue_rows.append(
                {
                    "family": family,
                    "number_features": int(
                        observed_row[
                            "number_features"
                        ]
                    ),
                    "metric": metric,
                    "baseline_metric": float(
                        BASELINE_METRICS[metric]
                    ),
                    "observed_family_metric": float(
                        observed_row[metric]
                    ),
                    "observed_delta": observed_delta,
                    "null_mean_delta": float(
                        family_rows[
                            delta_column
                        ].mean()
                    ),
                    "null_median_delta": float(
                        family_rows[
                            delta_column
                        ].median()
                    ),
                    "null_q025": float(
                        family_rows[
                            delta_column
                        ].quantile(0.025)
                    ),
                    "null_q975": float(
                        family_rows[
                            delta_column
                        ].quantile(0.975)
                    ),
                    "exact_one_sided_p": exact_p,
                }
            )

    pvalues = pd.DataFrame(pvalue_rows)

    pvalues[
        "holm_adjusted_p_mcc"
    ] = np.nan

    pvalues[
        "holm_reject_0p05_mcc"
    ] = False

    mcc_mask = pvalues["metric"] == "mcc"

    adjusted_mcc = holm_adjust(
        pvalues.loc[
            mcc_mask,
            "exact_one_sided_p",
        ].to_numpy(dtype=float)
    )

    pvalues.loc[
        mcc_mask,
        "holm_adjusted_p_mcc",
    ] = adjusted_mcc

    pvalues.loc[
        mcc_mask,
        "holm_reject_0p05_mcc",
    ] = adjusted_mcc < 0.05

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_path = (
        OUTPUT_DIRECTORY
        / "md_family_exact_permutation_distribution.csv"
    )

    pvalue_path = (
        OUTPUT_DIRECTORY
        / "md_family_exact_permutation_pvalues.csv"
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "md_family_exact_permutation_summary.json"
    )

    distribution.to_csv(
        distribution_path,
        index=False,
    )

    pvalues.to_csv(
        pvalue_path,
        index=False,
    )

    mcc_results = pvalues[
        pvalues["metric"] == "mcc"
    ].copy()

    summary = {
        "test": (
            "Exact permutation of each MD family "
            "among six independent MD sources"
        ),
        "primary_metric": "source_balanced_mcc",
        "number_of_sources": len(SOURCES),
        "permutations_per_family": 720,
        "number_of_families": len(
            FAMILY_FEATURES
        ),
        "multiple_testing_correction": (
            "Holm correction across the three "
            "family-level MCC tests"
        ),
        "baseline_metrics": BASELINE_METRICS,
        "family_features": FAMILY_FEATURES,
        "mcc_results": (
            mcc_results.to_dict(
                orient="records"
            )
        ),
    }

    with open(summary_path, "w") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print("===== EXACT FAMILY MCC RESULTS =====")

    display_columns = [
        "family",
        "number_features",
        "baseline_metric",
        "observed_family_metric",
        "observed_delta",
        "exact_one_sided_p",
        "holm_adjusted_p_mcc",
        "holm_reject_0p05_mcc",
    ]

    print(
        mcc_results[
            display_columns
        ]
        .sort_values(
            "exact_one_sided_p"
        )
        .to_string(index=False)
    )

    print()
    print("===== ACCEPTANCE CHECKS =====")
    print(
        "Six independent MD sources: PASS"
    )
    print(
        "720 assignments per family: PASS"
    )
    print(
        "One identity assignment per family: PASS"
    )
    print(
        "Source-balanced training weights: PASS"
    )
    print(
        "Holm correction across three MCC tests: PASS"
    )

    print()
    print(f"Written: {distribution_path}")
    print(f"Written: {pvalue_path}")
    print(f"Written: {summary_path}")


if __name__ == "__main__":
    main()
