from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


INPUT = "data/raw/LNP_transfection_with_reused_MD_features.csv"

FEATURES = [
    "md_LNP_only_Rg_nm_mean_last100ns",
    "md_LNP_core_Rg_nm_mean_last100ns",
    "md_LNP_only_SASA_nm2_mean_last100ns",
    "md_RNA_Rg_nm_mean_last100ns",
    "md_LNP_RNA_Rg_nm_mean_last100ns",
    "md_RNA_LNP_COM_distance_nm_mean_last100ns",
    "md_RNA_LNP_mindist_nm_mean_last100ns",
    "md_RNA_LNP_contacts_0p6nm_mean_last100ns",
    "md_RNA_buried_fraction_contact_based_mean_last100ns",
    "md_LNP_only_shape_anisotropy_kappa2_mean_last100ns",
    "md_LNP_only_compactness_inv_Rg_1_per_nm_mean_last100ns",
    "md_water_inside_LNP_core_RgSphere_count_fraction_of_total_water_mean_last100ns",
]


def source_weights(groups):
    group_series = pd.Series(groups)

    sizes = group_series.groupby(
        group_series
    ).transform("size")

    return 1.0 / sizes.to_numpy(dtype=float)


def calculate_metrics(
    y_true,
    y_pred,
    probability,
    groups,
):
    weights = source_weights(groups)

    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                sample_weight=weights,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
                sample_weight=weights,
            )
        ),
        "inactive_pr_auc": float(
            average_precision_score(
                1 - y_true,
                1 - probability,
                sample_weight=weights,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
                sample_weight=weights,
            )
        ),
    }


def evaluate_assignment(
    data,
    assigned_features,
    sources,
):
    y = (
        data["transfection_active"]
        .astype(int)
        .to_numpy()
    )

    groups = data[
        "md_source_lnp_id"
    ].astype(str).to_numpy()

    all_predictions = np.empty(
        len(data),
        dtype=int,
    )

    all_probabilities = np.empty(
        len(data),
        dtype=float,
    )

    for fold, held_source in enumerate(sources):
        training_mask = groups != held_source
        testing_mask = groups == held_source

        y_train = y[training_mask]

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                f"Training fold lacks both classes: "
                f"{held_source}"
            )

        training_groups = groups[training_mask]

        training_table = pd.DataFrame(
            assigned_features[training_mask],
            columns=FEATURES,
        )

        training_table[
            "md_source_lnp_id"
        ] = training_groups

        representatives = (
            training_table.groupby(
                "md_source_lnp_id",
                sort=True,
            )[FEATURES]
            .first()
        )

        scaler = StandardScaler()

        representative_scaled = scaler.fit_transform(
            representatives
        )

        number_of_components = min(
            2,
            representative_scaled.shape[0] - 1,
            representative_scaled.shape[1],
        )

        pca = PCA(
            n_components=number_of_components,
            random_state=42,
        )

        pca.fit(representative_scaled)

        x_train = pca.transform(
            scaler.transform(
                assigned_features[training_mask]
            )
        )

        x_test = pca.transform(
            scaler.transform(
                assigned_features[testing_mask]
            )
        )

        training_weights = source_weights(
            training_groups
        )

        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42 + fold,
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=training_weights,
        )

        probability = model.predict_proba(
            x_test
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        all_probabilities[testing_mask] = probability
        all_predictions[testing_mask] = prediction

    return calculate_metrics(
        y,
        all_predictions,
        all_probabilities,
        groups,
    )


def main():
    data = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    missing_features = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing MD features: {missing_features}"
        )

    data = data[
        (
            data["md_features_available"]
            .fillna(0)
            .astype(int)
            == 1
        )
        & (data["lnp_id"].astype(str) != "803")
    ].copy()

    data = data.dropna(
        subset=FEATURES
        + [
            "md_source_lnp_id",
            "transfection_active",
        ]
    ).reset_index(drop=True)

    data["md_source_lnp_id"] = data[
        "md_source_lnp_id"
    ].astype(str)

    sources = sorted(
        data["md_source_lnp_id"].unique()
    )

    if len(sources) != 6:
        raise RuntimeError(
            f"Expected six MD sources, found "
            f"{len(sources)}: {sources}"
        )

    # Each source must have one unique MD vector.
    unique_counts = data.groupby(
        "md_source_lnp_id"
    )[FEATURES].nunique(dropna=False)

    if (unique_counts > 1).any().any():
        problem_sources = unique_counts[
            (unique_counts > 1).any(axis=1)
        ].index.tolist()

        raise RuntimeError(
            "Multiple MD vectors found within sources: "
            f"{problem_sources}"
        )

    source_vectors = (
        data.groupby(
            "md_source_lnp_id",
            sort=True,
        )[FEATURES]
        .first()
        .loc[sources]
    )

    groups = data[
        "md_source_lnp_id"
    ].to_numpy()

    source_to_position = {
        source: position
        for position, source in enumerate(sources)
    }

    identity_assignment = np.vstack(
        [
            source_vectors.loc[source].to_numpy(
                dtype=float
            )
            for source in groups
        ]
    )

    observed = evaluate_assignment(
        data,
        identity_assignment,
        sources,
    )

    print("===== OBSERVED METRICS =====")
    for name, value in observed.items():
        print(f"{name}: {value:.6f}")

    results = []

    all_permutations = list(
        itertools.permutations(sources)
    )

    for permutation_number, donors in enumerate(
        all_permutations,
        start=1,
    ):
        recipient_to_donor = dict(
            zip(sources, donors)
        )

        assigned_features = np.vstack(
            [
                source_vectors.loc[
                    recipient_to_donor[source]
                ].to_numpy(dtype=float)
                for source in groups
            ]
        )

        permutation_metrics = evaluate_assignment(
            data,
            assigned_features,
            sources,
        )

        row = {
            "permutation": permutation_number,
            "is_identity": bool(
                all(
                    recipient == donor
                    for recipient, donor in zip(
                        sources,
                        donors,
                    )
                )
            ),
            "assignment": ";".join(
                f"{recipient}<-{donor}"
                for recipient, donor in zip(
                    sources,
                    donors,
                )
            ),
            **permutation_metrics,
        }

        results.append(row)

        if permutation_number % 100 == 0:
            print(
                f"Completed {permutation_number}/"
                f"{len(all_permutations)} permutations",
                flush=True,
            )

    distribution = pd.DataFrame(results)

    metric_names = [
        "macro_f1",
        "mcc",
        "inactive_pr_auc",
        "roc_auc",
    ]

    p_values = {}

    for metric in metric_names:
        p_values[metric] = float(
            (
                distribution[metric]
                >= observed[metric] - 1e-12
            ).mean()
        )

    summary = {
        "test": (
            "Exact permutation of six MD vectors "
            "among six MD source groups"
        ),
        "number_of_permutations": int(
            len(distribution)
        ),
        "number_of_md_sources": int(
            len(sources)
        ),
        "number_of_rows": int(
            len(data)
        ),
        "features": FEATURES,
        "observed_metrics": observed,
        "exact_one_sided_p_values": p_values,
        "null_distribution": {
            metric: {
                "mean": float(
                    distribution[metric].mean()
                ),
                "median": float(
                    distribution[metric].median()
                ),
                "q025": float(
                    distribution[metric].quantile(
                        0.025
                    )
                ),
                "q975": float(
                    distribution[metric].quantile(
                        0.975
                    )
                ),
            }
            for metric in metric_names
        },
        "interpretation": (
            "Exploratory exact test with only six "
            "independent MD source simulations."
        ),
    }

    output_directory = Path(
        "results/md_exploratory"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution.to_csv(
        output_directory
        / "exact_permutation_distribution.csv",
        index=False,
    )

    with open(
        output_directory
        / "exact_permutation_summary.json",
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

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    for axis, metric in zip(
        axes.ravel(),
        metric_names,
    ):
        sns.histplot(
            data=distribution,
            x=metric,
            bins=25,
            color="#4C72B0",
            ax=axis,
        )

        axis.axvline(
            observed[metric],
            color="#C44E52",
            linestyle="--",
            linewidth=2,
            label=(
                f"Observed = "
                f"{observed[metric]:.3f}"
            ),
        )

        axis.set_title(
            f"{metric}\n"
            f"Exact p = {p_values[metric]:.4f}"
        )

        axis.legend()

    figure.tight_layout()

    figure.savefig(
        "figures/md_exact_permutation.png",
        dpi=300,
    )

    plt.close(figure)

    print()
    print("===== EXACT PERMUTATION RESULTS =====")
    print("Permutations:", len(distribution))

    for metric in metric_names:
        print(
            f"{metric}: "
            f"observed={observed[metric]:.4f}, "
            f"p={p_values[metric]:.4f}"
        )

    print()
    print(
        "Written: results/md_exploratory/"
        "exact_permutation_distribution.csv"
    )
    print(
        "Written: results/md_exploratory/"
        "exact_permutation_summary.json"
    )
    print(
        "Written: figures/"
        "md_exact_permutation.png"
    )


if __name__ == "__main__":
    main()
