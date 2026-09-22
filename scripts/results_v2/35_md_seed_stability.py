from __future__ import annotations

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
METRIC_DIRECTORY = ROOT / "results_v2/metrics"
PREDICTION_DIRECTORY = ROOT / "results_v2/predictions"

NUMBER_OF_SEEDS = int(
    os.environ.get("N_SEEDS", "200")
)

NUMBER_OF_WORKERS = int(
    os.environ.get("SLURM_CPUS_PER_TASK", "1")
)

METRICS = [
    "roc_auc",
    "pr_auc_active",
    "pr_auc_inactive",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
]


def read_feature_list(filename):
    path = TABLE_DIRECTORY / filename

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def calculate_metrics(
    y_true,
    prediction,
    probability,
    weights,
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


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

if "analysis_row_id" not in data.columns:
    data["analysis_row_id"] = np.arange(
        len(data),
        dtype=int,
    )

target_column = "transfection_active"
source_column = "md_source_lnp_id"

baseline_numeric = read_feature_list(
    "baseline_numeric_features.txt"
)

baseline_categorical = read_feature_list(
    "baseline_categorical_features.txt"
)

md_features = read_feature_list(
    "md_features.txt"
)

required_columns = (
    baseline_numeric
    + baseline_categorical
    + md_features
    + [
        target_column,
        source_column,
        "analysis_row_id",
        "lnp_id",
    ]
)

missing_columns = sorted(
    set(required_columns) - set(data.columns)
)

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )

if set(baseline_numeric) & set(md_features):
    raise RuntimeError(
        "Baseline numeric and MD feature blocks overlap."
    )

data[source_column] = (
    data[source_column].astype(str)
)

data[target_column] = (
    data[target_column].astype(int)
)

sources = sorted(
    data[source_column].unique()
)

if len(sources) != 6:
    raise RuntimeError(
        f"Expected six MD sources, found {sources}"
    )

feature_sets = {
    "baseline": {
        "numeric": baseline_numeric,
        "categorical": baseline_categorical,
    },
    "baseline_plus_md": {
        "numeric": baseline_numeric + md_features,
        "categorical": baseline_categorical,
    },
}


def evaluate_seed(seed):
    seed_predictions = []

    for model_name, feature_specification in (
        feature_sets.items()
    ):
        numeric_features = (
            feature_specification["numeric"]
        )

        categorical_features = (
            feature_specification["categorical"]
        )

        model_features = (
            numeric_features + categorical_features
        )

        for fold, held_source in enumerate(sources):
            testing_mask = (
                data[source_column] == held_source
            )

            training_mask = ~testing_mask

            training_sources = set(
                data.loc[
                    training_mask,
                    source_column,
                ]
            )

            if held_source in training_sources:
                raise RuntimeError(
                    "Held-out source found in training."
                )

            y_train = data.loc[
                training_mask,
                target_column,
            ].to_numpy(dtype=int)

            if len(np.unique(y_train)) != 2:
                raise RuntimeError(
                    "Training fold lacks both classes: "
                    f"{held_source}"
                )

            model_seed = (
                seed + fold
            )

            model = make_pipeline(
                numeric_features,
                categorical_features,
                model_seed,
            )

            # Each training MD source receives equal total weight.
            training_source_sizes = (
                data.loc[training_mask]
                .groupby(source_column)[source_column]
                .transform("size")
                .to_numpy(dtype=float)
            )

            training_weights = (
                1.0 / training_source_sizes
            )

            model.fit(
                data.loc[
                    training_mask,
                    model_features,
                ],
                y_train,
                model__sample_weight=training_weights,
            )

            probability = model.predict_proba(
                data.loc[
                    testing_mask,
                    model_features,
                ]
            )[:, 1]

            prediction = (
                probability >= 0.5
            ).astype(int)

            testing_rows = data.loc[
                testing_mask,
                [
                    "analysis_row_id",
                    "lnp_id",
                    source_column,
                    target_column,
                ],
            ].copy()

            testing_rows["seed"] = seed
            testing_rows["model"] = model_name
            testing_rows["fold"] = fold
            testing_rows["probability"] = probability
            testing_rows["prediction"] = prediction

            seed_predictions.append(testing_rows)

    return pd.concat(
        seed_predictions,
        ignore_index=True,
    )


def main():
    seeds = list(
        range(2000, 2000 + NUMBER_OF_SEEDS)
    )

    print(
        "Repeated seeds:",
        NUMBER_OF_SEEDS,
        flush=True,
    )

    print(
        "Parallel workers:",
        NUMBER_OF_WORKERS,
        flush=True,
    )

    prediction_tables = Parallel(
        n_jobs=NUMBER_OF_WORKERS,
        verbose=10,
    )(
        delayed(evaluate_seed)(seed)
        for seed in seeds
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    expected_rows = (
        NUMBER_OF_SEEDS
        * len(feature_sets)
        * len(data)
    )

    if len(predictions) != expected_rows:
        raise RuntimeError(
            "Unexpected prediction count: "
            f"{len(predictions)} versus {expected_rows}"
        )

    if predictions["probability"].isna().any():
        raise RuntimeError(
            "Missing probabilities detected."
        )

    metric_rows = []

    for (seed, model_name), group in (
        predictions.groupby(
            ["seed", "model"],
            sort=True,
        )
    ):
        y_true = group[
            target_column
        ].to_numpy(dtype=int)

        prediction = group[
            "prediction"
        ].to_numpy(dtype=int)

        probability = group[
            "probability"
        ].to_numpy(dtype=float)

        row_weights = np.ones(
            len(group),
            dtype=float,
        )

        source_sizes = group.groupby(
            source_column
        )["analysis_row_id"].transform("size")

        source_weights = (
            1.0
            / source_sizes.to_numpy(dtype=float)
        )

        for weighting, weights in [
            ("row_weighted", row_weights),
            ("source_balanced", source_weights),
        ]:
            metric_rows.append(
                {
                    "seed": int(seed),
                    "model": model_name,
                    "weighting": weighting,
                    **calculate_metrics(
                        y_true,
                        prediction,
                        probability,
                        weights,
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)

    delta_rows = []

    for weighting in [
        "row_weighted",
        "source_balanced",
    ]:
        selected = metrics[
            metrics["weighting"] == weighting
        ]

        baseline = (
            selected[
                selected["model"] == "baseline"
            ]
            .set_index("seed")
        )

        enhanced = (
            selected[
                selected["model"]
                == "baseline_plus_md"
            ]
            .set_index("seed")
        )

        for seed in seeds:
            for metric in METRICS:
                delta_rows.append(
                    {
                        "seed": int(seed),
                        "weighting": weighting,
                        "metric": metric,
                        "baseline": float(
                            baseline.loc[seed, metric]
                        ),
                        "baseline_plus_md": float(
                            enhanced.loc[seed, metric]
                        ),
                        "delta_md_minus_baseline": float(
                            enhanced.loc[seed, metric]
                            - baseline.loc[seed, metric]
                        ),
                    }
                )

    deltas = pd.DataFrame(delta_rows)

    summary_rows = []

    for (weighting, metric), group in (
        deltas.groupby(
            ["weighting", "metric"],
            sort=False,
        )
    ):
        values = group[
            "delta_md_minus_baseline"
        ].to_numpy(dtype=float)

        summary_rows.append(
            {
                "weighting": weighting,
                "metric": metric,
                "number_of_seeds": len(values),
                "mean_delta": float(
                    np.mean(values)
                ),
                "sd_delta": float(
                    np.std(values, ddof=1)
                ),
                "median_delta": float(
                    np.median(values)
                ),
                "q025_delta": float(
                    np.quantile(values, 0.025)
                ),
                "q975_delta": float(
                    np.quantile(values, 0.975)
                ),
                "minimum_delta": float(
                    np.min(values)
                ),
                "maximum_delta": float(
                    np.max(values)
                ),
                "fraction_positive": float(
                    np.mean(values > 0)
                ),
                "fraction_zero": float(
                    np.mean(
                        np.isclose(values, 0)
                    )
                ),
                "fraction_negative": float(
                    np.mean(values < 0)
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    prediction_stability = (
        predictions.groupby(
            [
                "model",
                "analysis_row_id",
                "lnp_id",
                source_column,
                target_column,
            ],
            as_index=False,
        )
        .agg(
            mean_probability=(
                "probability",
                "mean",
            ),
            sd_probability=(
                "probability",
                "std",
            ),
            minimum_probability=(
                "probability",
                "min",
            ),
            maximum_probability=(
                "probability",
                "max",
            ),
            active_prediction_fraction=(
                "prediction",
                "mean",
            ),
            minimum_prediction=(
                "prediction",
                "min",
            ),
            maximum_prediction=(
                "prediction",
                "max",
            ),
        )
    )

    prediction_stability[
        "classification_changes_across_seeds"
    ] = (
        prediction_stability[
            "minimum_prediction"
        ]
        != prediction_stability[
            "maximum_prediction"
        ]
    )

    METRIC_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTION_DIRECTORY
        / "md_seed_stability_predictions.csv",
        index=False,
    )

    metrics.to_csv(
        METRIC_DIRECTORY
        / "md_seed_stability_metrics.csv",
        index=False,
    )

    deltas.to_csv(
        METRIC_DIRECTORY
        / "md_seed_stability_paired_deltas.csv",
        index=False,
    )

    summary.to_csv(
        METRIC_DIRECTORY
        / "md_seed_stability_summary.csv",
        index=False,
    )

    prediction_stability.to_csv(
        TABLE_DIRECTORY
        / "md_seed_prediction_stability.csv",
        index=False,
    )

    primary = summary[
        (summary["weighting"] == "source_balanced")
        & (summary["metric"] == "mcc")
    ]

    print()
    print("===== SOURCE-BALANCED STABILITY =====")
    print(
        summary[
            summary["weighting"]
            == "source_balanced"
        ].to_string(index=False)
    )

    print()
    print("===== PRIMARY STABILITY RESULT =====")
    print(primary.to_string(index=False))

    changing_rows = prediction_stability[
        prediction_stability[
            "classification_changes_across_seeds"
        ]
    ]

    print()
    print(
        "Rows changing classification across seeds:",
        len(changing_rows),
    )

    if len(changing_rows):
        print(
            changing_rows[
                [
                    "model",
                    "lnp_id",
                    source_column,
                    target_column,
                    "mean_probability",
                    "sd_probability",
                    "active_prediction_fraction",
                ]
            ].to_string(index=False)
        )

    print()
    print(
        "PASS: every seed produced paired "
        "source-LOSO predictions."
    )

    print(
        "Written: results_v2/metrics/"
        "md_seed_stability_summary.csv"
    )

    print(
        "Written: results_v2/tables/"
        "md_seed_prediction_stability.csv"
    )


if __name__ == "__main__":
    main()
