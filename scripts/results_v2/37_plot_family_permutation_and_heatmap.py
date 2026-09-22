from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

TABLE_DIRECTORY = ROOT / "results_v2/tables"
PERMUTATION_DIRECTORY = ROOT / "results_v2/permutation"
FIGURE_DIRECTORY = ROOT / "figures/results_v2"

DISTRIBUTION_PATH = (
    PERMUTATION_DIRECTORY
    / "md_family_exact_permutation_distribution.csv"
)

PVALUE_PATH = (
    PERMUTATION_DIRECTORY
    / "md_family_exact_permutation_pvalues.csv"
)

SOURCE_VECTOR_PATH = (
    TABLE_DIRECTORY
    / "md_unique_source_vectors.csv"
)

SUBSET_PATH = (
    TABLE_DIRECTORY
    / "exact_md_subset.csv"
)

MD_FEATURE_PATH = (
    TABLE_DIRECTORY
    / "md_features.txt"
)

FAMILY_MANIFEST_PATH = (
    TABLE_DIRECTORY
    / "md_family_manifest.csv"
)


FAMILY_ORDER = [
    "size_shape",
    "rna_association",
    "water_penetration",
]

FAMILY_DISPLAY = {
    "size_shape": "Size and shape",
    "rna_association": "RNA association",
    "water_penetration": "Water penetration",
}

SHORT_NAMES = {
    "md_LNP_only_Rg_nm_mean_last100ns":
        "LNP-only Rg",
    "md_LNP_core_Rg_nm_mean_last100ns":
        "Core Rg",
    "md_LNP_only_SASA_nm2_mean_last100ns":
        "LNP-only SASA",
    "md_RNA_Rg_nm_mean_last100ns":
        "RNA Rg",
    "md_LNP_RNA_Rg_nm_mean_last100ns":
        "LNP-RNA Rg",
    "md_RNA_LNP_COM_distance_nm_mean_last100ns":
        "RNA-LNP COM distance",
    "md_RNA_LNP_mindist_nm_mean_last100ns":
        "RNA-LNP minimum distance",
    "md_RNA_LNP_contacts_0p6nm_mean_last100ns":
        "RNA-LNP contacts",
    "md_RNA_buried_fraction_contact_based_mean_last100ns":
        "RNA buried fraction",
    "md_LNP_only_shape_anisotropy_kappa2_mean_last100ns":
        "Shape anisotropy",
    "md_LNP_only_compactness_inv_Rg_1_per_nm_mean_last100ns":
        "Compactness (1/Rg)",
    "md_water_inside_LNP_core_RgSphere_count_fraction_of_total_water_mean_last100ns":
        "Core water fraction",
}


def read_list(path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def identify_column(
    table,
    candidates,
    description,
):
    for column in candidates:
        if column in table.columns:
            return column

    raise RuntimeError(
        f"Could not identify {description}. "
        f"Columns found: {table.columns.tolist()}"
    )


def plot_permutation_distributions():
    distribution = pd.read_csv(
        DISTRIBUTION_PATH
    )

    pvalues = pd.read_csv(
        PVALUE_PATH
    )

    required_distribution = {
        "family",
        "is_identity",
        "delta_mcc",
    }

    missing = (
        required_distribution
        - set(distribution.columns)
    )

    if missing:
        raise RuntimeError(
            "Permutation distribution is missing: "
            f"{sorted(missing)}"
        )

    mcc_results = pvalues[
        pvalues["metric"] == "mcc"
    ].copy()

    if len(mcc_results) != 3:
        raise RuntimeError(
            "Expected three family-level MCC tests."
        )

    sns.set_theme(
        style="whitegrid",
        context="talk",
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6.5),
        sharey=False,
    )

    family_colors = {
        "size_shape": "#4C78A8",
        "rna_association": "#F58518",
        "water_penetration": "#54A24B",
    }

    for axis, family in zip(
        axes,
        FAMILY_ORDER,
    ):
        family_distribution = distribution[
            distribution["family"] == family
        ].copy()

        family_result = mcc_results[
            mcc_results["family"] == family
        ]

        if len(family_distribution) != 720:
            raise RuntimeError(
                f"{family} has "
                f"{len(family_distribution)} "
                "permutations instead of 720."
            )

        if len(family_result) != 1:
            raise RuntimeError(
                f"Missing MCC result for {family}."
            )

        family_result = family_result.iloc[0]

        observed = float(
            family_result["observed_delta"]
        )

        raw_p = float(
            family_result["exact_one_sided_p"]
        )

        holm_p = float(
            family_result[
                "holm_adjusted_p_mcc"
            ]
        )

        sns.histplot(
            data=family_distribution,
            x="delta_mcc",
            bins=28,
            color=family_colors[family],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
            ax=axis,
        )

        axis.axvline(
            0,
            color="#555555",
            linestyle=":",
            linewidth=1.8,
            label="No improvement",
        )

        axis.axvline(
            observed,
            color="#B22222",
            linestyle="--",
            linewidth=2.8,
            label="Observed",
        )

        axis.set_title(
            FAMILY_DISPLAY[family],
            fontweight="bold",
            pad=12,
        )

        axis.set_xlabel(
            "MCC difference versus baseline"
        )

        axis.set_ylabel(
            "Number of assignments"
        )

        result_text = (
            f"Observed ΔMCC = {observed:.3f}\n"
            f"Exact p = {raw_p:.4f}\n"
            f"Holm p = {holm_p:.4f}"
        )

        axis.text(
            0.04,
            0.96,
            result_text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#BBBBBB",
                "alpha": 0.94,
            },
        )

        axis.legend(
            frameon=False,
            loc="upper right",
            fontsize=10,
        )

    figure.suptitle(
        "Exact source-level permutation tests for MD descriptor families",
        fontsize=19,
        fontweight="bold",
        y=1.02,
    )

    figure.text(
        0.5,
        -0.01,
        (
            "Null distributions contain all 720 assignments "
            "of six MD vectors to six source groups."
        ),
        ha="center",
        fontsize=11,
    )

    figure.tight_layout(
        rect=[0, 0.04, 1, 0.96]
    )

    png_path = (
        FIGURE_DIRECTORY
        / "md_family_exact_permutation_mcc.png"
    )

    pdf_path = (
        FIGURE_DIRECTORY
        / "md_family_exact_permutation_mcc.pdf"
    )

    figure.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return png_path, pdf_path


def plot_source_heatmap():
    md_features = read_list(
        MD_FEATURE_PATH
    )

    vectors = pd.read_csv(
        SOURCE_VECTOR_PATH
    )

    subset = pd.read_csv(
        SUBSET_PATH,
        low_memory=False,
    )

    manifest = pd.read_csv(
        FAMILY_MANIFEST_PATH
    )

    source_column = identify_column(
        vectors,
        [
            "md_source_lnp_id",
            "source",
            "md_source",
        ],
        "source column",
    )

    family_column = identify_column(
        manifest,
        [
            "family",
            "md_family",
            "feature_family",
        ],
        "family column",
    )

    feature_column = identify_column(
        manifest,
        [
            "feature",
            "md_feature",
            "column",
            "descriptor",
        ],
        "feature column",
    )

    manifest["normalized_family"] = (
        manifest[family_column].map(
            normalize_name
        )
    )

    ordered_features = []

    family_sizes = {}

    for family in FAMILY_ORDER:
        selected = manifest[
            manifest["normalized_family"]
            == family
        ][feature_column].astype(str).tolist()

        family_sizes[family] = len(selected)
        ordered_features.extend(selected)

    if set(ordered_features) != set(md_features):
        raise RuntimeError(
            "Family manifest does not match "
            "the approved MD feature list."
        )

    missing_features = sorted(
        set(ordered_features)
        - set(vectors.columns)
    )

    if missing_features:
        raise RuntimeError(
            f"Source-vector table is missing: "
            f"{missing_features}"
        )

    vectors[source_column] = (
        vectors[source_column].astype(str)
    )

    subset["md_source_lnp_id"] = (
        subset["md_source_lnp_id"].astype(str)
    )

    label_summary = (
        subset.groupby(
            "md_source_lnp_id",
            as_index=False,
        )
        .agg(
            mapped_rows=(
                "transfection_active",
                "size",
            ),
            active=(
                "transfection_active",
                "sum",
            ),
        )
    )

    label_summary["inactive"] = (
        label_summary["mapped_rows"]
        - label_summary["active"]
    )

    label_summary["active_fraction"] = (
        label_summary["active"]
        / label_summary["mapped_rows"]
    )

    vectors = vectors.merge(
        label_summary,
        left_on=source_column,
        right_on="md_source_lnp_id",
        how="left",
        validate="one_to_one",
    )

    vectors = vectors.sort_values(
        source_column
    ).reset_index(drop=True)

    numeric_values = vectors[
        ordered_features
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise RuntimeError(
            "Nonfinite MD values found."
        )

    feature_means = numeric_values.mean(
        axis=0
    )

    feature_standard_deviations = (
        numeric_values.std(
            axis=0,
            ddof=0,
        )
    )

    if np.any(
        feature_standard_deviations == 0
    ):
        raise RuntimeError(
            "Constant MD descriptor found."
        )

    standardized = (
        numeric_values - feature_means
    ) / feature_standard_deviations

    row_labels = []

    for _, row in vectors.iterrows():
        row_labels.append(
            (
                f"{row[source_column]}  "
                f"({int(row.get('active', row.get('active_y', row.get('active_x'))))}/"
                f"{int(row.get('mapped_rows', row.get('mapped_rows_y', row.get('mapped_rows_x'))))} active)"
            )
        )

    short_feature_names = [
        SHORT_NAMES.get(
            feature,
            feature.replace(
                "md_",
                "",
            ),
        )
        for feature in ordered_features
    ]

    zscore_table = pd.DataFrame(
        standardized,
        columns=ordered_features,
    )

    zscore_table.insert(
        0,
        "md_source_lnp_id",
        vectors[source_column].to_numpy(),
    )

    zscore_table.insert(
        1,
        "mapped_rows",
        vectors.get("mapped_rows", vectors.get("mapped_rows_y", vectors.get("mapped_rows_x"))).to_numpy(),
    )

    zscore_table.insert(
        2,
        "active",
        vectors.get("active", vectors.get("active_y", vectors.get("active_x"))).to_numpy(),
    )

    zscore_table.insert(
        3,
        "inactive",
        vectors.get("inactive", vectors.get("inactive_y", vectors.get("inactive_x"))).to_numpy(),
    )

    zscore_table.insert(
        4,
        "active_fraction",
        vectors.get("active_fraction", vectors.get("active_fraction_y", vectors.get("active_fraction_x"))).to_numpy(),
    )

    zscore_output = (
        TABLE_DIRECTORY
        / "md_source_descriptor_zscores.csv"
    )

    zscore_table.to_csv(
        zscore_output,
        index=False,
    )

    absolute_limit = max(
        2.0,
        float(np.abs(standardized).max()),
    )

    sns.set_theme(
        style="white",
        context="talk",
    )

    figure, axis = plt.subplots(
        figsize=(19, 8),
    )

    sns.heatmap(
        standardized,
        cmap="vlag",
        center=0,
        vmin=-absolute_limit,
        vmax=absolute_limit,
        annot=True,
        fmt=".1f",
        annot_kws={
            "fontsize": 9,
        },
        linewidths=0.8,
        linecolor="white",
        xticklabels=short_feature_names,
        yticklabels=row_labels,
        cbar_kws={
            "label": (
                "Descriptor z-score across "
                "six MD sources"
            ),
            "shrink": 0.82,
        },
        ax=axis,
    )

    boundary = 0

    family_centers = {}

    for family in FAMILY_ORDER:
        size = family_sizes[family]

        family_centers[family] = (
            boundary + size / 2
        )

        boundary += size

        if boundary < len(ordered_features):
            axis.axvline(
                boundary,
                color="#222222",
                linewidth=2.2,
            )

    for family in FAMILY_ORDER:
        axis.text(
            family_centers[family],
            1.055,
            FAMILY_DISPLAY[family],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    axis.set_title(
        (
            "Standardized MD descriptor profiles "
            "for six independent LNP sources"
        ),
        fontsize=18,
        fontweight="bold",
        pad=52,
    )

    axis.set_xlabel(
        "MD descriptor",
        labelpad=15,
    )

    axis.set_ylabel(
        "MD source and observed active rows",
        labelpad=12,
    )

    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=42,
        ha="right",
        rotation_mode="anchor",
    )

    axis.set_yticklabels(
        axis.get_yticklabels(),
        rotation=0,
    )

    figure.text(
        0.5,
        0.01,
        (
            "Each descriptor was standardized across "
            "the six unique source vectors. "
            "Positive values are above the source mean."
        ),
        ha="center",
        fontsize=11,
    )

    figure.subplots_adjust(
        left=0.19,
        right=0.96,
        top=0.79,
        bottom=0.31,
    )

    png_path = (
        FIGURE_DIRECTORY
        / "md_source_descriptor_heatmap.png"
    )

    pdf_path = (
        FIGURE_DIRECTORY
        / "md_source_descriptor_heatmap.pdf"
    )

    figure.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return (
        png_path,
        pdf_path,
        zscore_output,
    )


def main():
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    permutation_png, permutation_pdf = (
        plot_permutation_distributions()
    )

    (
        heatmap_png,
        heatmap_pdf,
        zscore_output,
    ) = plot_source_heatmap()

    print(
        "===== RESULTS_V2 FIGURES CREATED ====="
    )

    print(f"Written: {permutation_png}")
    print(f"Written: {permutation_pdf}")
    print(f"Written: {heatmap_png}")
    print(f"Written: {heatmap_pdf}")
    print(f"Written: {zscore_output}")


if __name__ == "__main__":
    main()
