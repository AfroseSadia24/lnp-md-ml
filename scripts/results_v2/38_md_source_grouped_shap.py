from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "results_v2"
    / "tables"
    / "exact_md_subset.csv"
)

BASELINE_NUMERIC_PATH = (
    ROOT
    / "results_v2"
    / "tables"
    / "baseline_numeric_features.txt"
)

BASELINE_CATEGORICAL_PATH = (
    ROOT
    / "results_v2"
    / "tables"
    / "baseline_categorical_features.txt"
)

MD_FEATURE_PATH = (
    ROOT
    / "results_v2"
    / "tables"
    / "md_features.txt"
)

REFERENCE_PREDICTION_PATH = (
    ROOT
    / "results_v2"
    / "predictions"
    / "md_source_loso_predictions.csv"
)

OUTPUT_DIRECTORY = ROOT / "results_v2" / "shap"
METRICS_DIRECTORY = ROOT / "results_v2" / "metrics"
PREDICTION_DIRECTORY = ROOT / "results_v2" / "predictions"
TABLE_DIRECTORY = ROOT / "results_v2" / "tables"
FIGURE_DIRECTORY = ROOT / "figures" / "results_v2"

TARGET = "transfection_active"
SOURCE = "md_source_lnp_id"

RANDOM_STATE_BASE = 2000
N_ESTIMATORS = 800
MIN_SAMPLES_LEAF = 2

N_JOBS = max(
    1,
    int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
)

FAMILY_BY_FEATURE = {
    "md_LNP_only_Rg_nm_mean_last100ns": "size_shape",
    "md_LNP_core_Rg_nm_mean_last100ns": "size_shape",
    "md_LNP_only_SASA_nm2_mean_last100ns": "size_shape",
    "md_RNA_Rg_nm_mean_last100ns": "size_shape",
    "md_LNP_RNA_Rg_nm_mean_last100ns": "size_shape",
    (
        "md_LNP_only_shape_anisotropy_kappa2_"
        "mean_last100ns"
    ): "size_shape",
    (
        "md_LNP_only_compactness_inv_Rg_1_per_nm_"
        "mean_last100ns"
    ): "size_shape",
    (
        "md_RNA_LNP_COM_distance_nm_mean_last100ns"
    ): "rna_association",
    (
        "md_RNA_LNP_mindist_nm_mean_last100ns"
    ): "rna_association",
    (
        "md_RNA_LNP_contacts_0p6nm_mean_last100ns"
    ): "rna_association",
    (
        "md_RNA_buried_fraction_contact_based_"
        "mean_last100ns"
    ): "rna_association",
    (
        "md_water_inside_LNP_core_RgSphere_count_"
        "fraction_of_total_water_mean_last100ns"
    ): "water_penetration",
}

FEATURE_DISPLAY = {
    "md_LNP_only_Rg_nm_mean_last100ns":
        "LNP radius of gyration",
    "md_LNP_core_Rg_nm_mean_last100ns":
        "Core radius of gyration",
    "md_LNP_only_SASA_nm2_mean_last100ns":
        "LNP solvent-accessible area",
    "md_RNA_Rg_nm_mean_last100ns":
        "RNA radius of gyration",
    "md_LNP_RNA_Rg_nm_mean_last100ns":
        "LNP-RNA radius of gyration",
    (
        "md_LNP_only_shape_anisotropy_kappa2_"
        "mean_last100ns"
    ): "Shape anisotropy",
    (
        "md_LNP_only_compactness_inv_Rg_1_per_nm_"
        "mean_last100ns"
    ): "LNP compactness",
    (
        "md_RNA_LNP_COM_distance_nm_mean_last100ns"
    ): "RNA-LNP center distance",
    (
        "md_RNA_LNP_mindist_nm_mean_last100ns"
    ): "RNA-LNP minimum distance",
    (
        "md_RNA_LNP_contacts_0p6nm_mean_last100ns"
    ): "RNA-LNP contacts",
    (
        "md_RNA_buried_fraction_contact_based_"
        "mean_last100ns"
    ): "RNA buried fraction",
    (
        "md_water_inside_LNP_core_RgSphere_count_"
        "fraction_of_total_water_mean_last100ns"
    ): "Core water fraction",
}

FAMILY_DISPLAY = {
    "size_shape": "Size and shape",
    "rna_association": "RNA association",
    "water_penetration": "Water penetration",
}

FAMILY_ORDER = [
    "size_shape",
    "rna_association",
    "water_penetration",
]

FAMILY_COLORS = {
    "size_shape": "#4472C4",
    "rna_association": "#ED7D31",
    "water_penetration": "#70AD47",
}


def read_feature_list(path: Path) -> list[str]:
    with open(path) as handle:
        return [
            line.strip()
            for line in handle
            if line.strip()
        ]


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_model(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
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
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )

    classifier = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=N_JOBS,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def source_balanced_training_weights(
    sources: pd.Series,
) -> np.ndarray:
    source_sizes = sources.groupby(
        sources
    ).transform("size")

    return (
        1.0
        / source_sizes.to_numpy(dtype=float)
    )


def active_class_shap_values(
    raw_values,
    number_rows: int,
    number_features: int,
) -> np.ndarray:
    if isinstance(raw_values, list):
        if len(raw_values) >= 2:
            values = np.asarray(raw_values[1])
        else:
            values = np.asarray(raw_values[0])
    elif hasattr(raw_values, "values"):
        values = np.asarray(raw_values.values)
    else:
        values = np.asarray(raw_values)

    if values.ndim == 2:
        result = values
    elif values.ndim == 3:
        if (
            values.shape[0] == number_rows
            and values.shape[1] == number_features
        ):
            class_index = 1 if values.shape[2] > 1 else 0
            result = values[:, :, class_index]
        elif (
            values.shape[1] == number_rows
            and values.shape[2] == number_features
        ):
            class_index = 1 if values.shape[0] > 1 else 0
            result = values[class_index, :, :]
        else:
            raise RuntimeError(
                "Unrecognized three-dimensional SHAP shape: "
                f"{values.shape}"
            )
    else:
        raise RuntimeError(
            "Unrecognized SHAP output shape: "
            f"{values.shape}"
        )

    expected_shape = (
        number_rows,
        number_features,
    )

    if result.shape != expected_shape:
        raise RuntimeError(
            f"Expected SHAP shape {expected_shape}, "
            f"received {result.shape}"
        )

    return result


def active_expected_value(explainer) -> float:
    expected = np.asarray(
        explainer.expected_value
    )

    if expected.ndim == 0:
        return float(expected)

    expected = expected.ravel()

    if len(expected) >= 2:
        return float(expected[1])

    return float(expected[0])


def locate_md_columns(
    transformed_names: np.ndarray,
    md_features: list[str],
) -> dict[str, int]:
    locations = {}

    for feature in md_features:
        exact_name = f"numeric__{feature}"

        matches = [
            position
            for position, name in enumerate(
                transformed_names
            )
            if (
                name == exact_name
                or name.endswith(f"__{feature}")
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one transformed column for "
                f"{feature}, found {matches}"
            )

        locations[feature] = matches[0]

    return locations


def zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=0)

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation == 0
    ):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
        )

    return (
        values - values.mean()
    ) / standard_deviation


def save_figure(
    figure,
    stem: str,
):
    png_path = FIGURE_DIRECTORY / f"{stem}.png"
    pdf_path = FIGURE_DIRECTORY / f"{stem}.pdf"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Written: {png_path}")
    print(f"Written: {pdf_path}")


def main():
    for directory in [
        OUTPUT_DIRECTORY,
        METRICS_DIRECTORY,
        PREDICTION_DIRECTORY,
        TABLE_DIRECTORY,
        FIGURE_DIRECTORY,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    baseline_numeric = read_feature_list(
        BASELINE_NUMERIC_PATH
    )

    baseline_categorical = read_feature_list(
        BASELINE_CATEGORICAL_PATH
    )

    md_features = read_feature_list(
        MD_FEATURE_PATH
    )

    if len(md_features) != 12:
        raise RuntimeError(
            f"Expected 12 MD features, found "
            f"{len(md_features)}"
        )

    features_without_family = [
        feature
        for feature in md_features
        if feature not in FAMILY_BY_FEATURE
    ]

    if features_without_family:
        raise RuntimeError(
            "MD features without family assignments: "
            f"{features_without_family}"
        )

    data = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    data[SOURCE] = data[SOURCE].astype(str)
    data["analysis_row_id"] = (
        data["analysis_row_id"].astype(str)
    )

    all_model_features = (
        baseline_numeric
        + baseline_categorical
        + md_features
    )

    missing_columns = [
        column
        for column in (
            all_model_features
            + [
                "analysis_row_id",
                "lnp_id",
                "paper_doi",
                SOURCE,
                TARGET,
            ]
        )
        if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing columns: {missing_columns}"
        )

    sources = sorted(
        data[SOURCE].unique()
    )

    if len(sources) != 6:
        raise RuntimeError(
            f"Expected six MD sources, found "
            f"{len(sources)}: {sources}"
        )

    shap_rows = []
    prediction_rows = []
    fold_audit_rows = []
    additivity_errors = []

    print("===== SOURCE-GROUPED SHAP =====")
    print("Rows:", len(data))
    print("Independent MD sources:", len(sources))
    print("MD features:", len(md_features))
    print("Parallel workers:", N_JOBS)
    print("SHAP version:", shap.__version__)
    print()

    for fold, held_source in enumerate(sources):
        training_mask = (
            data[SOURCE] != held_source
        )

        testing_mask = (
            data[SOURCE] == held_source
        )

        training = data.loc[
            training_mask
        ].copy()

        testing = data.loc[
            testing_mask
        ].copy()

        y_train = (
            training[TARGET]
            .astype(int)
            .to_numpy()
        )

        y_test = (
            testing[TARGET]
            .astype(int)
            .to_numpy()
        )

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                f"Training data for holdout "
                f"{held_source} lacks both classes."
            )

        training_weights = (
            source_balanced_training_weights(
                training[SOURCE]
            )
        )

        model = make_model(
            numeric_features=(
                baseline_numeric + md_features
            ),
            categorical_features=(
                baseline_categorical
            ),
            random_state=(
                RANDOM_STATE_BASE + fold
            ),
        )

        model.fit(
            training[all_model_features],
            y_train,
            classifier__sample_weight=(
                training_weights
            ),
        )

        probability = model.predict_proba(
            testing[all_model_features]
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        preprocessing = model.named_steps[
            "preprocessing"
        ]

        classifier = model.named_steps[
            "classifier"
        ]

        transformed_testing = (
            preprocessing.transform(
                testing[all_model_features]
            )
        )

        if hasattr(
            transformed_testing,
            "toarray",
        ):
            transformed_testing = (
                transformed_testing.toarray()
            )

        transformed_testing = np.asarray(
            transformed_testing,
            dtype=float,
        )

        transformed_names = np.asarray(
            preprocessing.get_feature_names_out(),
            dtype=str,
        )

        md_column_locations = locate_md_columns(
            transformed_names,
            md_features,
        )

        explainer = shap.TreeExplainer(
            classifier,
            feature_perturbation=(
                "tree_path_dependent"
            ),
        )

        raw_shap = explainer.shap_values(
            transformed_testing,
            check_additivity=False,
        )

        all_shap_values = (
            active_class_shap_values(
                raw_shap,
                number_rows=len(testing),
                number_features=(
                    transformed_testing.shape[1]
                ),
            )
        )

        expected_value = active_expected_value(
            explainer
        )

        reconstructed_probability = (
            expected_value
            + all_shap_values.sum(axis=1)
        )

        fold_additivity_error = float(
            np.max(
                np.abs(
                    reconstructed_probability
                    - probability
                )
            )
        )

        additivity_errors.append(
            fold_additivity_error
        )

        for test_position, (
            row_index,
            row,
        ) in enumerate(testing.iterrows()):
            prediction_rows.append(
                {
                    "analysis_row_id":
                        row["analysis_row_id"],
                    "lnp_id": row["lnp_id"],
                    "paper_doi": row["paper_doi"],
                    SOURCE: held_source,
                    TARGET: int(y_test[test_position]),
                    "fold": int(fold),
                    "model": "baseline_plus_md",
                    "prob_active": float(
                        probability[test_position]
                    ),
                    "prediction": int(
                        prediction[test_position]
                    ),
                }
            )

            for feature in md_features:
                feature_position = (
                    md_column_locations[feature]
                )

                shap_value = float(
                    all_shap_values[
                        test_position,
                        feature_position,
                    ]
                )

                shap_rows.append(
                    {
                        "analysis_row_id":
                            row["analysis_row_id"],
                        "lnp_id": row["lnp_id"],
                        "paper_doi":
                            row["paper_doi"],
                        SOURCE: held_source,
                        TARGET:
                            int(y_test[test_position]),
                        "fold": int(fold),
                        "feature": feature,
                        "display_name":
                            FEATURE_DISPLAY[feature],
                        "family":
                            FAMILY_BY_FEATURE[
                                feature
                            ],
                        "shap_value": shap_value,
                        "abs_shap_value":
                            abs(shap_value),
                        "md_value": float(
                            row[feature]
                        ),
                    }
                )

        fold_audit_rows.append(
            {
                "fold": int(fold),
                "held_source": held_source,
                "training_rows":
                    int(training_mask.sum()),
                "testing_rows":
                    int(testing_mask.sum()),
                "training_sources":
                    int(training[SOURCE].nunique()),
                "testing_sources":
                    int(testing[SOURCE].nunique()),
                "source_overlap": int(
                    len(
                        set(training[SOURCE])
                        & set(testing[SOURCE])
                    )
                ),
                "max_shap_additivity_error":
                    fold_additivity_error,
            }
        )

        print(
            f"Finished holdout {held_source}: "
            f"{len(testing)} rows, "
            f"additivity error="
            f"{fold_additivity_error:.3e}",
            flush=True,
        )

    shap_long = pd.DataFrame(shap_rows)

    predictions = pd.DataFrame(
        prediction_rows
    ).sort_values(
        [SOURCE, "analysis_row_id"]
    )

    fold_audit = pd.DataFrame(
        fold_audit_rows
    )

    expected_shap_rows = (
        len(data) * len(md_features)
    )

    if len(shap_long) != expected_shap_rows:
        raise RuntimeError(
            f"Expected {expected_shap_rows} SHAP rows, "
            f"found {len(shap_long)}"
        )

    if predictions["analysis_row_id"].nunique() != len(data):
        raise RuntimeError(
            "Not every experimental row received "
            "one out-of-fold prediction."
        )

    if fold_audit["source_overlap"].sum() != 0:
        raise RuntimeError(
            "Held-out MD source leakage detected."
        )

    source_feature = (
        shap_long.groupby(
            [
                SOURCE,
                "feature",
                "display_name",
                "family",
            ],
            as_index=False,
        )
        .agg(
            source_mean_shap=(
                "shap_value",
                "mean",
            ),
            source_mean_abs_shap=(
                "abs_shap_value",
                "mean",
            ),
            source_md_value=(
                "md_value",
                "mean",
            ),
            mapped_rows=(
                "analysis_row_id",
                "nunique",
            ),
        )
    )

    source_feature["descriptor_zscore"] = (
        source_feature.groupby(
            "feature",
            group_keys=False,
        )["source_md_value"]
        .transform(zscore)
    )

    feature_summary = (
        source_feature.groupby(
            [
                "feature",
                "display_name",
                "family",
            ],
            as_index=False,
        )
        .agg(
            mean_abs_shap=(
                "source_mean_abs_shap",
                "mean",
            ),
            sd_abs_shap=(
                "source_mean_abs_shap",
                "std",
            ),
            median_abs_shap=(
                "source_mean_abs_shap",
                "median",
            ),
            mean_signed_shap=(
                "source_mean_shap",
                "mean",
            ),
            minimum_signed_shap=(
                "source_mean_shap",
                "min",
            ),
            maximum_signed_shap=(
                "source_mean_shap",
                "max",
            ),
            number_sources=(
                SOURCE,
                "nunique",
            ),
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_summary["rank"] = (
        np.arange(
            1,
            len(feature_summary) + 1,
        )
    )

    row_family = (
        shap_long.groupby(
            [
                "analysis_row_id",
                "lnp_id",
                SOURCE,
                TARGET,
                "fold",
                "family",
            ],
            as_index=False,
        )
        .agg(
            family_signed_shap=(
                "shap_value",
                "sum",
            ),
            family_total_abs_shap=(
                "abs_shap_value",
                "sum",
            ),
            number_features=(
                "feature",
                "nunique",
            ),
        )
    )

    source_family = (
        row_family.groupby(
            [SOURCE, "family"],
            as_index=False,
        )
        .agg(
            source_mean_signed_shap=(
                "family_signed_shap",
                "mean",
            ),
            source_mean_total_abs_shap=(
                "family_total_abs_shap",
                "mean",
            ),
            number_features=(
                "number_features",
                "first",
            ),
            mapped_rows=(
                "analysis_row_id",
                "nunique",
            ),
        )
    )

    source_family[
        "source_mean_abs_shap_per_feature"
    ] = (
        source_family[
            "source_mean_total_abs_shap"
        ]
        / source_family["number_features"]
    )

    family_summary = (
        source_family.groupby(
            "family",
            as_index=False,
        )
        .agg(
            number_features=(
                "number_features",
                "first",
            ),
            mean_total_abs_shap=(
                "source_mean_total_abs_shap",
                "mean",
            ),
            sd_total_abs_shap=(
                "source_mean_total_abs_shap",
                "std",
            ),
            mean_abs_shap_per_feature=(
                "source_mean_abs_shap_per_feature",
                "mean",
            ),
            mean_signed_shap=(
                "source_mean_signed_shap",
                "mean",
            ),
            number_sources=(
                SOURCE,
                "nunique",
            ),
        )
    )

    family_summary["display_name"] = (
        family_summary["family"].map(
            FAMILY_DISPLAY
        )
    )

    family_summary = family_summary.sort_values(
        "mean_total_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    reproduction_audit = {
        "reference_file_available":
            REFERENCE_PREDICTION_PATH.exists(),
        "number_compared_rows": 0,
        "maximum_absolute_probability_difference":
            None,
        "status": "NOT_CHECKED",
    }

    if REFERENCE_PREDICTION_PATH.exists():
        reference = pd.read_csv(
            REFERENCE_PREDICTION_PATH
        )

        if "model" in reference.columns:
            reference = reference[
                reference["model"]
                == "baseline_plus_md"
            ].copy()

        reference["analysis_row_id"] = (
            reference["analysis_row_id"].astype(str)
        )

        comparison = predictions.merge(
            reference[
                [
                    "analysis_row_id",
                    "prob_active",
                ]
            ].rename(
                columns={
                    "prob_active":
                        "reference_prob_active"
                }
            ),
            on="analysis_row_id",
            how="inner",
            validate="one_to_one",
        )

        comparison[
            "absolute_probability_difference"
        ] = np.abs(
            comparison["prob_active"]
            - comparison["reference_prob_active"]
        )

        maximum_difference = float(
            comparison[
                "absolute_probability_difference"
            ].max()
        )

        reproduction_audit = {
            "reference_file_available": True,
            "number_compared_rows":
                int(len(comparison)),
            (
                "maximum_absolute_"
                "probability_difference"
            ): maximum_difference,
            "status": (
                "PASS"
                if maximum_difference <= 1e-8
                else "REVIEW"
            ),
        }

        comparison.to_csv(
            OUTPUT_DIRECTORY
            / "md_shap_prediction_reproduction_audit.csv",
            index=False,
        )

    shap_long.to_csv(
        OUTPUT_DIRECTORY
        / "md_source_loso_shap_long.csv",
        index=False,
    )

    source_feature.to_csv(
        OUTPUT_DIRECTORY
        / "md_source_feature_shap.csv",
        index=False,
    )

    feature_summary.to_csv(
        METRICS_DIRECTORY
        / "md_shap_feature_importance.csv",
        index=False,
    )

    row_family.to_csv(
        OUTPUT_DIRECTORY
        / "md_row_family_shap.csv",
        index=False,
    )

    source_family.to_csv(
        OUTPUT_DIRECTORY
        / "md_source_family_shap.csv",
        index=False,
    )

    family_summary.to_csv(
        METRICS_DIRECTORY
        / "md_shap_family_importance.csv",
        index=False,
    )

    predictions.to_csv(
        PREDICTION_DIRECTORY
        / "md_shap_loso_predictions.csv",
        index=False,
    )

    fold_audit.to_csv(
        TABLE_DIRECTORY
        / "md_shap_loso_fold_audit.csv",
        index=False,
    )

    summary = {
        "analysis": (
            "Source-grouped out-of-fold SHAP "
            "for the baseline-plus-MD ExtraTrees model"
        ),
        "number_rows": int(len(data)),
        "number_md_sources": int(len(sources)),
        "number_md_features":
            int(len(md_features)),
        "number_estimators": N_ESTIMATORS,
        "minimum_samples_leaf":
            MIN_SAMPLES_LEAF,
        "class_weight": "balanced",
        "source_balanced_training_weights": True,
        "maximum_shap_additivity_error":
            float(max(additivity_errors)),
        "prediction_reproduction_audit":
            reproduction_audit,
        "interpretation": (
            "Exploratory analysis based on six "
            "independent MD source vectors. "
            "Family-level interpretation is preferred "
            "because several descriptors are strongly "
            "correlated."
        ),
    }

    with open(
        OUTPUT_DIRECTORY
        / "md_source_grouped_shap_summary.json",
        "w",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    feature_plot = feature_summary.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    feature_colors = [
        FAMILY_COLORS[family]
        for family in feature_plot["family"]
    ]

    figure, axis = plt.subplots(
        figsize=(12, 9)
    )

    axis.barh(
        feature_plot["display_name"],
        feature_plot["mean_abs_shap"],
        color=feature_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    axis.set_xlabel(
        "Mean absolute out-of-fold SHAP value"
    )

    axis.set_ylabel("MD descriptor")

    axis.set_title(
        "Source-balanced MD descriptor importance"
    )

    legend_handles = [
        Patch(
            facecolor=FAMILY_COLORS[family],
            edgecolor="black",
            label=FAMILY_DISPLAY[family],
        )
        for family in FAMILY_ORDER
    ]

    axis.legend(
        handles=legend_handles,
        title="MD family",
        loc="lower right",
    )

    figure.tight_layout()

    save_figure(
        figure,
        "md_shap_feature_importance",
    )

    family_plot = (
        family_summary.set_index("family")
        .reindex(FAMILY_ORDER)
        .reset_index()
    )

    family_colors = [
        FAMILY_COLORS[family]
        for family in family_plot["family"]
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    axes[0].bar(
        family_plot["display_name"],
        family_plot["mean_total_abs_shap"],
        color=family_colors,
        edgecolor="black",
    )

    axes[0].set_ylabel(
        "Mean total absolute SHAP"
    )

    axes[0].set_title(
        "Total family contribution"
    )

    axes[1].bar(
        family_plot["display_name"],
        family_plot[
            "mean_abs_shap_per_feature"
        ],
        color=family_colors,
        edgecolor="black",
    )

    axes[1].set_ylabel(
        "Mean absolute SHAP per descriptor"
    )

    axes[1].set_title(
        "Feature-count-normalized contribution"
    )

    for axis in axes:
        axis.tick_params(
            axis="x",
            rotation=20,
        )

    figure.suptitle(
        "Source-balanced MD family SHAP importance",
        y=1.03,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "md_shap_family_importance",
    )

    signed_matrix = (
        source_family.pivot(
            index=SOURCE,
            columns="family",
            values="source_mean_signed_shap",
        )
        .reindex(
            index=sources,
            columns=FAMILY_ORDER,
        )
    )

    signed_matrix.columns = [
        FAMILY_DISPLAY[column]
        for column in signed_matrix.columns
    ]

    maximum_absolute = float(
        np.nanmax(
            np.abs(
                signed_matrix.to_numpy(
                    dtype=float
                )
            )
        )
    )

    if maximum_absolute == 0:
        maximum_absolute = 1.0

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        signed_matrix,
        cmap="vlag",
        center=0,
        vmin=-maximum_absolute,
        vmax=maximum_absolute,
        annot=True,
        fmt=".3f",
        linewidths=0.7,
        cbar_kws={
            "label": (
                "Mean signed SHAP contribution "
                "to active prediction"
            )
        },
        ax=axis,
    )

    axis.set_xlabel("MD descriptor family")
    axis.set_ylabel("Held-out MD source")
    axis.set_title(
        "Out-of-fold family SHAP contributions"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "md_shap_source_family_heatmap",
    )

    feature_order = (
        feature_summary.sort_values(
            "mean_abs_shap",
            ascending=False,
        )["feature"]
        .tolist()
    )

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    color_normalization = TwoSlopeNorm(
        vmin=-2.5,
        vcenter=0,
        vmax=2.5,
    )

    for feature_position, feature in enumerate(
        feature_order
    ):
        feature_data = (
            source_feature[
                source_feature["feature"] == feature
            ]
            .sort_values(SOURCE)
            .reset_index(drop=True)
        )

        offsets = np.linspace(
            -0.18,
            0.18,
            len(feature_data),
        )

        points = axis.scatter(
            feature_data["source_mean_shap"],
            (
                np.full(
                    len(feature_data),
                    feature_position,
                )
                + offsets
            ),
            c=feature_data[
                "descriptor_zscore"
            ],
            cmap="coolwarm",
            norm=color_normalization,
            s=85,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )

    axis.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axis.set_yticks(
        np.arange(len(feature_order))
    )

    axis.set_yticklabels(
        [
            FEATURE_DISPLAY[feature]
            for feature in feature_order
        ]
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Mean out-of-fold SHAP value"
    )

    axis.set_ylabel("MD descriptor")

    axis.set_title(
        "Source-balanced MD SHAP summary"
    )

    colorbar = figure.colorbar(
        points,
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label(
        "Descriptor z-score across six sources"
    )

    figure.tight_layout()

    save_figure(
        figure,
        "md_shap_source_balanced_summary",
    )

    print()
    print("===== FEATURE IMPORTANCE =====")
    print(
        feature_summary[
            [
                "rank",
                "display_name",
                "family",
                "mean_abs_shap",
                "mean_signed_shap",
            ]
        ].to_string(index=False)
    )

    print()
    print("===== FAMILY IMPORTANCE =====")
    print(
        family_summary[
            [
                "display_name",
                "number_features",
                "mean_total_abs_shap",
                "mean_abs_shap_per_feature",
                "mean_signed_shap",
            ]
        ].to_string(index=False)
    )

    print()
    print("===== ACCEPTANCE CHECKS =====")
    print("Six independent MD sources: PASS")
    print("No source overlap between train and test: PASS")
    print("Every row explained out of fold: PASS")
    print("Twelve MD descriptors extracted: PASS")
    print(
        "Maximum SHAP additivity error:",
        f"{max(additivity_errors):.3e}",
    )
    print(
        "Previous prediction reproduction:",
        reproduction_audit["status"],
    )

    print()
    print(
        "CAUTION: interpret family-level SHAP more "
        "strongly than individual descriptor rankings."
    )
    print(
        "This remains exploratory because only six "
        "independent MD vectors are available."
    )


if __name__ == "__main__":
    main()
