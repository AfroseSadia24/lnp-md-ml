from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import (
    leaves_list,
    linkage,
)
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")
TABLES = ROOT / "results_v2/tables"
FIGURES = ROOT / "figures/results_v2"

DATA_PATH = TABLES / "exact_md_subset.csv"
MD_LIST_PATH = TABLES / "md_features.txt"
FAMILY_PATH = TABLES / "md_family_manifest.csv"

TARGET = "transfection_active"
SOURCE = "md_source_lnp_id"


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

md_features = [
    value.strip()
    for value in MD_LIST_PATH.read_text().splitlines()
    if value.strip()
]

if len(data) != 13:
    raise RuntimeError(
        f"Expected 13 rows, found {len(data)}."
    )

if len(md_features) != 12:
    raise RuntimeError(
        f"Expected 12 MD descriptors, "
        f"found {len(md_features)}."
    )

required = [
    TARGET,
    SOURCE,
    "lnp_id",
    "paper_doi",
] + md_features

missing_columns = [
    column
    for column in required
    if column not in data.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing columns: {missing_columns}"
    )

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="raise",
).astype(int)

data[SOURCE] = data[SOURCE].astype(str)

for column in md_features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


within_source_counts = data.groupby(
    SOURCE
)[md_features].nunique(dropna=False)

if (within_source_counts > 1).any().any():
    problem_sources = within_source_counts.loc[
        (within_source_counts > 1).any(axis=1)
    ].index.tolist()

    raise RuntimeError(
        "More than one MD vector occurs within "
        f"these sources: {problem_sources}"
    )


source_labels = (
    data.groupby(
        SOURCE,
        as_index=False,
    )
    .agg(
        mapped_rows=("lnp_id", "size"),
        active=(TARGET, "sum"),
        doi_groups=(
            "paper_doi",
            "nunique",
        ),
        mapped_lnp_ids=(
            "lnp_id",
            lambda values: ";".join(
                sorted(
                    values.astype(str).unique()
                )
            ),
        ),
    )
)

source_labels["inactive"] = (
    source_labels["mapped_rows"]
    - source_labels["active"]
)

source_labels["active_fraction"] = (
    source_labels["active"]
    / source_labels["mapped_rows"]
)


representatives = (
    data.groupby(
        SOURCE,
        sort=True,
    )[md_features]
    .first()
    .reset_index()
)

unique_table = source_labels.merge(
    representatives,
    on=SOURCE,
    how="inner",
    validate="one_to_one",
)

if len(unique_table) != 6:
    raise RuntimeError(
        f"Expected six independent MD sources, "
        f"found {len(unique_table)}."
    )


values = unique_table[
    md_features
].to_numpy(dtype=float)

missing_total = int(
    np.isnan(values).sum()
)

infinite_total = int(
    np.isinf(values).sum()
)

if missing_total != 0:
    raise RuntimeError(
        f"Found {missing_total} missing MD values."
    )

if infinite_total != 0:
    raise RuntimeError(
        f"Found {infinite_total} infinite MD values."
    )


family_lookup = {}

if FAMILY_PATH.exists():
    family_table = pd.read_csv(FAMILY_PATH)

    family_lookup = dict(
        zip(
            family_table["feature"],
            family_table["family"],
        )
    )


audit_rows = []

for feature in md_features:
    feature_values = unique_table[
        feature
    ].to_numpy(dtype=float)

    standard_deviation = float(
        np.std(
            feature_values,
            ddof=0,
        )
    )

    feature_range = float(
        np.max(feature_values)
        - np.min(feature_values)
    )

    audit_rows.append(
        {
            "feature": feature,
            "family": family_lookup.get(
                feature,
                "unassigned",
            ),
            "independent_sources": len(
                feature_values
            ),
            "unique_values": int(
                pd.Series(
                    feature_values
                ).nunique()
            ),
            "missing": int(
                np.isnan(feature_values).sum()
            ),
            "infinite": int(
                np.isinf(feature_values).sum()
            ),
            "minimum": float(
                np.min(feature_values)
            ),
            "maximum": float(
                np.max(feature_values)
            ),
            "mean": float(
                np.mean(feature_values)
            ),
            "standard_deviation": (
                standard_deviation
            ),
            "range": feature_range,
            "constant": bool(
                standard_deviation == 0
            ),
        }
    )


quality_audit = pd.DataFrame(
    audit_rows
)

quality_audit.to_csv(
    TABLES
    / "md_descriptor_quality_audit.csv",
    index=False,
)

unique_table.to_csv(
    TABLES
    / "md_unique_source_vectors.csv",
    index=False,
)

source_labels.to_csv(
    TABLES
    / "md_source_label_summary.csv",
    index=False,
)


rounded_vectors = unique_table[
    md_features
].round(12)

vector_signatures = rounded_vectors.astype(
    str
).agg("|".join, axis=1)

duplicate_vector_count = int(
    vector_signatures.duplicated(
        keep=False
    ).sum()
)


nonconstant_features = quality_audit.loc[
    ~quality_audit["constant"],
    "feature",
].tolist()

constant_features = quality_audit.loc[
    quality_audit["constant"],
    "feature",
].tolist()

if len(nonconstant_features) < 2:
    raise RuntimeError(
        "Fewer than two nonconstant MD "
        "descriptors are available."
    )


correlation = unique_table[
    nonconstant_features
].corr(
    method="spearman"
)

correlation.to_csv(
    TABLES
    / "md_spearman_correlation.csv"
)


pair_rows = []

for first_index, first_feature in enumerate(
    nonconstant_features
):
    for second_index in range(
        first_index + 1,
        len(nonconstant_features),
    ):
        second_feature = (
            nonconstant_features[
                second_index
            ]
        )

        rho, p_value = spearmanr(
            unique_table[first_feature],
            unique_table[second_feature],
        )

        pair_rows.append(
            {
                "feature_1": first_feature,
                "family_1": family_lookup.get(
                    first_feature,
                    "unassigned",
                ),
                "feature_2": second_feature,
                "family_2": family_lookup.get(
                    second_feature,
                    "unassigned",
                ),
                "spearman_rho": float(rho),
                "absolute_rho": float(abs(rho)),
                "p_value": float(p_value),
            }
        )


pairs = pd.DataFrame(pair_rows)

# Benjamini-Hochberg diagnostic adjustment.
if not pairs.empty:
    p_values = pairs[
        "p_value"
    ].to_numpy(dtype=float)

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted_ranked = (
        ranked
        * len(ranked)
        / np.arange(
            1,
            len(ranked) + 1,
        )
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.minimum(
        adjusted_ranked,
        1.0,
    )

    adjusted = np.empty_like(
        adjusted_ranked
    )

    adjusted[order] = adjusted_ranked

    pairs["bh_adjusted_p"] = adjusted


pairs = pairs.sort_values(
    [
        "absolute_rho",
        "p_value",
    ],
    ascending=[
        False,
        True,
    ],
)

pairs.to_csv(
    TABLES
    / "md_all_correlation_pairs.csv",
    index=False,
)

high_pairs = pairs.loc[
    pairs["absolute_rho"] >= 0.80
].copy()

high_pairs.to_csv(
    TABLES
    / "md_high_correlation_pairs.csv",
    index=False,
)


# Cluster descriptors by absolute correlation.
distance = (
    1.0
    - correlation.abs().to_numpy()
)

distance = np.nan_to_num(
    distance,
    nan=1.0,
    posinf=1.0,
    neginf=1.0,
)

distance = (
    distance + distance.T
) / 2.0

np.fill_diagonal(
    distance,
    0.0,
)

condensed_distance = squareform(
    distance,
    checks=False,
)

linkage_matrix = linkage(
    condensed_distance,
    method="average",
)

feature_order = leaves_list(
    linkage_matrix
)

ordered_features = [
    nonconstant_features[index]
    for index in feature_order
]

ordered_correlation = correlation.loc[
    ordered_features,
    ordered_features,
]

ordered_correlation.to_csv(
    TABLES
    / "md_spearman_correlation_clustered.csv"
)


display_names = {
    "md_LNP_only_Rg_nm_mean_last100ns":
        "LNP Rg",
    "md_LNP_core_Rg_nm_mean_last100ns":
        "Core Rg",
    "md_LNP_only_SASA_nm2_mean_last100ns":
        "LNP SASA",
    "md_RNA_Rg_nm_mean_last100ns":
        "RNA Rg",
    "md_LNP_RNA_Rg_nm_mean_last100ns":
        "LNP-RNA Rg",
    (
        "md_RNA_LNP_COM_distance_nm_"
        "mean_last100ns"
    ):
        "RNA-LNP COM distance",
    (
        "md_RNA_LNP_mindist_nm_"
        "mean_last100ns"
    ):
        "RNA-LNP minimum distance",
    (
        "md_RNA_LNP_contacts_0p6nm_"
        "mean_last100ns"
    ):
        "RNA-LNP contacts",
    (
        "md_RNA_buried_fraction_contact_"
        "based_mean_last100ns"
    ):
        "RNA buried fraction",
    (
        "md_LNP_only_shape_anisotropy_"
        "kappa2_mean_last100ns"
    ):
        "Shape anisotropy",
    (
        "md_LNP_only_compactness_inv_"
        "Rg_1_per_nm_mean_last100ns"
    ):
        "Compactness",
    (
        "md_water_inside_LNP_core_RgSphere_"
        "count_fraction_of_total_water_"
        "mean_last100ns"
    ):
        "Internal water fraction",
}

ordered_labels = [
    display_names.get(
        feature,
        feature,
    )
    for feature in ordered_features
]


figure_size = max(
    9,
    0.72 * len(ordered_features),
)

figure, axis = plt.subplots(
    figsize=(
        figure_size,
        figure_size - 0.5,
    )
)

image = axis.imshow(
    ordered_correlation.to_numpy(),
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="equal",
)

axis.set_xticks(
    np.arange(
        len(ordered_labels)
    )
)

axis.set_yticks(
    np.arange(
        len(ordered_labels)
    )
)

axis.set_xticklabels(
    ordered_labels,
    rotation=55,
    ha="right",
    fontsize=9,
)

axis.set_yticklabels(
    ordered_labels,
    fontsize=9,
)

for row in range(
    len(ordered_features)
):
    for column in range(
        len(ordered_features)
    ):
        value = ordered_correlation.iloc[
            row,
            column,
        ]

        text_color = (
            "white"
            if abs(value) >= 0.65
            else "black"
        )

        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color=text_color,
        )

axis.set_title(
    "Spearman correlation among independent MD vectors",
    fontsize=14,
    pad=14,
)

colorbar = figure.colorbar(
    image,
    ax=axis,
    fraction=0.046,
    pad=0.04,
)

colorbar.set_label(
    "Spearman correlation",
    fontsize=11,
)

figure.text(
    0.5,
    0.01,
    (
        "Correlations use one vector per MD source "
        "(n = 6). Values are exploratory."
    ),
    ha="center",
    fontsize=9,
    color="#444444",
)

figure.tight_layout(
    rect=[0, 0.04, 1, 1]
)

png_path = (
    FIGURES
    / "md_unique_vector_spearman_correlation.png"
)

pdf_path = (
    FIGURES
    / "md_unique_vector_spearman_correlation.pdf"
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


print("===== UNIQUE MD VECTOR AUDIT =====")
print("Experimental rows:", len(data))
print("Independent MD sources:", len(unique_table))
print("MD descriptors:", len(md_features))
print("Missing MD values:", missing_total)
print("Infinite MD values:", infinite_total)
print("Constant descriptors:", len(constant_features))
print(
    "Duplicate source vectors:",
    duplicate_vector_count,
)

if constant_features:
    print(
        "Constant feature names:",
        constant_features,
    )

print()
print("===== SOURCE LABEL SUMMARY =====")

print(
    source_labels[
        [
            SOURCE,
            "mapped_rows",
            "active",
            "inactive",
            "active_fraction",
            "doi_groups",
        ]
    ].to_string(index=False)
)

print()
print(
    "===== HIGH CORRELATION PAIRS "
    "(ABS RHO >= 0.80) ====="
)

if high_pairs.empty:
    print("None")
else:
    print(
        high_pairs[
            [
                "feature_1",
                "feature_2",
                "spearman_rho",
                "p_value",
                "bh_adjusted_p",
            ]
        ].to_string(index=False)
    )

print()
print("===== ACCEPTANCE CHECKS =====")
print("One MD vector per source: PASS")
print("All MD values finite: PASS")
print("Correlation uses six source vectors: PASS")

print()
print(
    "Written: results_v2/tables/"
    "md_unique_source_vectors.csv"
)

print(
    "Written: results_v2/tables/"
    "md_descriptor_quality_audit.csv"
)

print(
    "Written: results_v2/tables/"
    "md_high_correlation_pairs.csv"
)

print(
    "Written: figures/results_v2/"
    "md_unique_vector_spearman_correlation.png"
)

print(
    "Written: figures/results_v2/"
    "md_unique_vector_spearman_correlation.pdf"
)
