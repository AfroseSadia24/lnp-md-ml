from pathlib import Path
import itertools
import json
import os

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
TABLES = ROOT / "results_v2/tables"
METRICS = ROOT / "results_v2/metrics"
PREDICTIONS = ROOT / "results_v2/predictions"
OUTPUT = ROOT / "results_v2/permutation"

DATA_PATH = TABLES / "exact_md_subset.csv"
PREVIOUS_PREDICTIONS = (
    PREDICTIONS
    / "md_source_loso_predictions.csv"
)

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
        line.strip()
        for line in (
            TABLES / filename
        ).read_text().splitlines()
        if line.strip()
    ]


baseline_numeric = read_list(
    "baseline_numeric_features.txt"
)

baseline_categorical = read_list(
    "baseline_categorical_features.txt"
)

md_features = read_list(
    "md_features.txt"
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
).reset_index(drop=True)

data[TARGET] = data[TARGET].astype(int)
data[SOURCE] = data[SOURCE].astype(str)

for column in enhanced_numeric:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

sources = sorted(
    data[SOURCE].unique()
)

if len(data) != 13:
    raise RuntimeError(
        f"Expected 13 rows, found {len(data)}."
    )

if len(sources) != 6:
    raise RuntimeError(
        f"Expected six sources, found {sources}."
    )

if len(md_features) != 12:
    raise RuntimeError(
        "Expected 12 MD descriptors."
    )


unique_counts = data.groupby(
    SOURCE
)[md_features].nunique(dropna=False)

if (unique_counts > 1).any().any():
    raise RuntimeError(
        "More than one MD vector exists "
        "within an MD source."
    )


source_vectors = (
    data.groupby(
        SOURCE,
        sort=True,
    )[md_features]
    .first()
    .loc[sources]
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


def calculate_metrics(
    y_true,
    prediction,
    probability,
    weights=None,
):
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
    }


def build_pipeline(random_state):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            )
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
                enhanced_numeric,
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
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("classifier", classifier),
        ]
    )


previous = pd.read_csv(
    PREVIOUS_PREDICTIONS,
    low_memory=False,
)

baseline_predictions = previous.loc[
    previous["model"].eq("baseline"),
    [
        "analysis_row_id",
        "prediction",
        "prob_active",
    ],
].copy()

baseline_predictions = data[
    [
        "analysis_row_id",
        TARGET,
        SOURCE,
    ]
].merge(
    baseline_predictions,
    on="analysis_row_id",
    how="left",
    validate="one_to_one",
)

if baseline_predictions[
    [
        "prediction",
        "prob_active",
    ]
].isna().any().any():
    raise RuntimeError(
        "Baseline LOSO predictions are incomplete."
    )


y = baseline_predictions[
    TARGET
].to_numpy(dtype=int)

baseline_class = baseline_predictions[
    "prediction"
].to_numpy(dtype=int)

baseline_probability = baseline_predictions[
    "prob_active"
].to_numpy(dtype=float)

evaluation_source_weights = source_weights(
    baseline_predictions[SOURCE]
)

baseline_row_metrics = calculate_metrics(
    y,
    baseline_class,
    baseline_probability,
)

baseline_source_metrics = calculate_metrics(
    y,
    baseline_class,
    baseline_probability,
    weights=evaluation_source_weights,
)


def evaluate_permutation(
    permutation_number,
    donor_sources,
):
    recipient_to_donor = dict(
        zip(
            sources,
            donor_sources,
        )
    )

    assigned = data.copy()

    for recipient, donor in (
        recipient_to_donor.items()
    ):
        assigned.loc[
            assigned[SOURCE].eq(recipient),
            md_features,
        ] = source_vectors.loc[
            donor,
            md_features,
        ].to_numpy(dtype=float)

    all_probability = np.empty(
        len(assigned),
        dtype=float,
    )

    all_prediction = np.empty(
        len(assigned),
        dtype=int,
    )

    for fold, held_source in enumerate(sources):
        test_mask = assigned[
            SOURCE
        ].eq(held_source)

        train_mask = ~test_mask

        training = assigned.loc[
            train_mask
        ].copy()

        testing = assigned.loc[
            test_mask
        ].copy()

        y_train = training[
            TARGET
        ].to_numpy(dtype=int)

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                f"Training fold lacks both classes: "
                f"{held_source}"
            )

        training_weights = source_weights(
            training[SOURCE]
        )

        training_weights = (
            training_weights
            * len(training_weights)
            / training_weights.sum()
        )

        pipeline = build_pipeline(
            random_state=2000 + fold
        )

        pipeline.fit(
            training[enhanced_features],
            y_train,
            classifier__sample_weight=(
                training_weights
            ),
        )

        probability = pipeline.predict_proba(
            testing[enhanced_features]
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        test_indices = np.flatnonzero(
            test_mask.to_numpy()
        )

        all_probability[
            test_indices
        ] = probability

        all_prediction[
            test_indices
        ] = prediction

    row_metrics = calculate_metrics(
        data[TARGET].to_numpy(dtype=int),
        all_prediction,
        all_probability,
    )

    source_metrics = calculate_metrics(
        data[TARGET].to_numpy(dtype=int),
        all_prediction,
        all_probability,
        weights=source_weights(data[SOURCE]),
    )

    identity = all(
        recipient == donor
        for recipient, donor in zip(
            sources,
            donor_sources,
        )
    )

    result = {
        "permutation": permutation_number,
        "is_identity": identity,
        "assignment": ";".join(
            f"{recipient}<-{donor}"
            for recipient, donor in zip(
                sources,
                donor_sources,
            )
        ),
    }

    for metric, value in row_metrics.items():
        result[
            f"row_weighted_{metric}"
        ] = value

        result[
            f"row_weighted_delta_{metric}"
        ] = (
            value
            - baseline_row_metrics[metric]
        )

    for metric, value in source_metrics.items():
        result[
            f"source_balanced_{metric}"
        ] = value

        result[
            f"source_balanced_delta_{metric}"
        ] = (
            value
            - baseline_source_metrics[metric]
        )

    return result


all_permutations = list(
    itertools.permutations(sources)
)

print(
    "Running exact permutations:",
    len(all_permutations),
    flush=True,
)

print(
    "Parallel workers:",
    NUMBER_OF_JOBS,
    flush=True,
)

results = Parallel(
    n_jobs=NUMBER_OF_JOBS,
    prefer="processes",
    verbose=10,
)(
    delayed(evaluate_permutation)(
        permutation_number,
        donor_sources,
    )
    for permutation_number, donor_sources in enumerate(
        all_permutations,
        start=1,
    )
)

distribution = pd.DataFrame(results)

identity_rows = distribution.loc[
    distribution["is_identity"]
]

if len(identity_rows) != 1:
    raise RuntimeError(
        "Expected exactly one identity assignment."
    )

observed = identity_rows.iloc[0]

metric_names = [
    "roc_auc",
    "pr_auc_active",
    "pr_auc_inactive",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
]

p_value_rows = []

for weighting in [
    "row_weighted",
    "source_balanced",
]:
    for metric in metric_names:
        column = (
            f"{weighting}_delta_{metric}"
        )

        observed_delta = float(
            observed[column]
        )

        exact_p = float(
            (
                distribution[column]
                >= observed_delta - 1e-12
            ).mean()
        )

        p_value_rows.append(
            {
                "weighting": weighting,
                "metric": metric,
                "observed_delta": observed_delta,
                "null_mean_delta": float(
                    distribution[column].mean()
                ),
                "null_median_delta": float(
                    distribution[column].median()
                ),
                "null_q025": float(
                    distribution[column].quantile(
                        0.025
                    )
                ),
                "null_q975": float(
                    distribution[column].quantile(
                        0.975
                    )
                ),
                "exact_one_sided_p": exact_p,
            }
        )


p_values = pd.DataFrame(p_value_rows)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

distribution.to_csv(
    OUTPUT
    / "incremental_md_exact_permutation_distribution.csv",
    index=False,
)

p_values.to_csv(
    OUTPUT
    / "incremental_md_exact_permutation_pvalues.csv",
    index=False,
)


summary = {
    "test": (
        "Exact permutation of six MD vectors "
        "among six MD-source groups"
    ),
    "number_of_permutations": 720,
    "number_of_sources": 6,
    "number_of_rows": 13,
    "primary_statistic": (
        "source_balanced_delta_mcc"
    ),
    "baseline_row_metrics": (
        baseline_row_metrics
    ),
    "baseline_source_balanced_metrics": (
        baseline_source_metrics
    ),
    "observed_identity_results": {
        column: (
            bool(observed[column])
            if column == "is_identity"
            else (
                int(observed[column])
                if column == "permutation"
                else observed[column]
            )
        )
        for column in distribution.columns
        if column != "assignment"
    },
    "exact_tests": (
        p_values.to_dict(
            orient="records"
        )
    ),
}

with open(
    OUTPUT
    / "incremental_md_exact_permutation_summary.json",
    "w",
) as handle:
    json.dump(
        summary,
        handle,
        indent=2,
    )


print()
print("===== EXACT INCREMENTAL MD TEST =====")

print(
    p_values.to_string(
        index=False
    )
)

primary = p_values.loc[
    p_values["weighting"].eq(
        "source_balanced"
    )
    & p_values["metric"].eq("mcc")
].iloc[0]

print()
print("===== PRIMARY RESULT =====")

print(
    "Source-balanced MCC delta:",
    f"{primary['observed_delta']:.6f}",
)

print(
    "Exact one-sided p-value:",
    f"{primary['exact_one_sided_p']:.6f}",
)

print()
print(
    "Written: results_v2/permutation/"
    "incremental_md_exact_permutation_distribution.csv"
)

print(
    "Written: results_v2/permutation/"
    "incremental_md_exact_permutation_pvalues.csv"
)

print(
    "Written: results_v2/permutation/"
    "incremental_md_exact_permutation_summary.json"
)
