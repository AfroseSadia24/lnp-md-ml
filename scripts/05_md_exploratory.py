from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
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


def calculate_metrics(y_true, y_pred, probability):
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "mcc": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "pr_auc_minority_inactive": float(
            average_precision_score(
                1 - y_true,
                1 - probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
    }


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing required MD features: {missing_features}"
        )

    data = df[
        (df["md_features_available"].fillna(0).astype(int) == 1)
        & (df["lnp_id"].astype(str) != "803")
    ].copy()

    data = data.dropna(
        subset=FEATURES
        + [
            "md_source_lnp_id",
            "transfection_active",
        ]
    ).reset_index(drop=True)

    if data.empty:
        raise RuntimeError("No complete MD-covered rows found")

    sources = sorted(
        data["md_source_lnp_id"].unique()
    )

    if len(sources) < 4:
        raise RuntimeError(
            "Fewer than four independent MD sources"
        )

    y = (
        data["transfection_active"]
        .astype(int)
        .to_numpy()
    )

    prediction_rows = []
    fold_results = []

    for fold, held_source in enumerate(sources):
        train_mask = (
            data["md_source_lnp_id"] != held_source
        ).to_numpy()

        test_mask = ~train_mask

        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()

        y_train = (
            train["transfection_active"]
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test["transfection_active"]
            .astype(int)
            .to_numpy()
        )

        if len(np.unique(y_train)) != 2:
            raise RuntimeError(
                f"Training fold lacks both classes: {held_source}"
            )

        # One representative vector per training MD source is
        # used to fit scaling and PCA. This prevents sources with
        # more reused rows from dominating the transformation.
        representatives = (
            train.groupby(
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
            scaler.transform(train[FEATURES])
        )

        x_test = pca.transform(
            scaler.transform(test[FEATURES])
        )

        # Give every MD source equal total training weight.
        source_sizes = train.groupby(
            "md_source_lnp_id"
        )["md_source_lnp_id"].transform("size")

        sample_weight = (
            1.0 / source_sizes.to_numpy(dtype=float)
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
            sample_weight=sample_weight,
        )

        probability = model.predict_proba(
            x_test
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        fold_result = {
            "fold": int(fold),
            "held_source": held_source,
            "testing_rows": int(test_mask.sum()),
            "active": int(y_test.sum()),
            "inactive": int((y_test == 0).sum()),
        }

        if len(np.unique(y_test)) == 2:
            fold_result.update(
                calculate_metrics(
                    y_test,
                    prediction,
                    probability,
                )
            )

        fold_results.append(fold_result)

        for position, (_, row) in enumerate(
            test.iterrows()
        ):
            prediction_rows.append(
                {
                    "lnp_id": row["lnp_id"],
                    "paper_doi": row["paper_doi"],
                    "md_source_lnp_id": held_source,
                    "y_true": int(y_test[position]),
                    "y_pred": int(prediction[position]),
                    "prob_active": float(
                        probability[position]
                    ),
                    "fold": int(fold),
                }
            )

        joblib.dump(
            {
                "features": FEATURES,
                "scaler": scaler,
                "pca": pca,
                "model": model,
                "held_source": held_source,
            },
            (
                "models/md_exploratory/"
                f"md_logistic_holdout_{held_source}.joblib"
            ),
        )

        print(
            f"Finished holdout {held_source}: "
            f"{len(test)} testing rows",
            flush=True,
        )

    predictions = pd.DataFrame(
        prediction_rows
    ).sort_values(
        ["md_source_lnp_id", "lnp_id"]
    )

    overall = calculate_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
        predictions["prob_active"].to_numpy(),
    )

    overall.update(
        {
            "model": "regularized_logistic_pca",
            "validation": "leave_one_md_source_out",
            "n_rows": int(len(predictions)),
            "n_md_sources": int(len(sources)),
            "n_original_md_features": int(len(FEATURES)),
            "n_pca_components": 2,
            "features": FEATURES,
            "fold_results": fold_results,
            "interpretation": (
                "Exploratory only due to six independent "
                "MD source simulations."
            ),
        }
    )

    predictions.to_csv(
        "results/md_exploratory/"
        "md_only_loso_predictions.csv",
        index=False,
    )

    with open(
        "results/md_exploratory/"
        "md_only_loso_metrics.json",
        "w",
    ) as handle:
        json.dump(
            overall,
            handle,
            indent=2,
        )

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(11, 7))

    plot_data = predictions.copy()
    plot_data["actual_label"] = plot_data[
        "y_true"
    ].map({0: "Inactive", 1: "Active"})

    sns.stripplot(
        data=plot_data,
        x="md_source_lnp_id",
        y="prob_active",
        hue="actual_label",
        dodge=True,
        size=10,
    )

    plt.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    plt.ylim(0, 1)
    plt.xlabel("Held-out MD source")
    plt.ylabel("Out-of-fold probability of active")
    plt.tight_layout()

    plt.savefig(
        "figures/md_exploratory_predictions.png",
        dpi=300,
    )

    plt.close()

    matrix = confusion_matrix(
        predictions["y_true"],
        predictions["y_pred"],
        labels=[0, 1],
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Inactive", "Active"],
        yticklabels=["Inactive", "Active"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()

    plt.savefig(
        "figures/md_exploratory_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    print()
    print("===== EXPLORATORY MD-ONLY MODEL =====")
    print("Rows:", overall["n_rows"])
    print("Independent MD sources:", overall["n_md_sources"])
    print("Macro-F1:", round(overall["macro_f1"], 4))
    print("MCC:", round(overall["mcc"], 4))
    print(
        "Inactive PR-AUC:",
        round(
            overall["pr_auc_minority_inactive"],
            4,
        ),
    )
    print("ROC-AUC:", round(overall["roc_auc"], 4))
    print()
    print(
        "WARNING: exploratory result based on only "
        "six independent MD sources."
    )
    print(
        "Written: results/md_exploratory/"
        "md_only_loso_predictions.csv"
    )
    print(
        "Written: results/md_exploratory/"
        "md_only_loso_metrics.json"
    )
    print(
        "Written: figures/"
        "md_exploratory_predictions.png"
    )
    print(
        "Written: figures/"
        "md_exploratory_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
