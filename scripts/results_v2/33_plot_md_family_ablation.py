from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

INPUT = (
    ROOT
    / "results_v2/metrics/"
    "md_family_ablation_family_effects.csv"
)

OUTPUT_TABLE = (
    ROOT
    / "results_v2/tables/"
    "md_family_ablation_mcc_plot_data.csv"
)

FIGURE_DIRECTORY = (
    ROOT / "figures/results_v2"
)


data = pd.read_csv(INPUT)

plot_data = data.loc[
    data["weighting"].eq("source_balanced")
    & data["metric"].eq("mcc")
].copy()

expected_families = {
    "size_shape",
    "rna_association",
    "water_penetration",
}

if set(plot_data["family"]) != expected_families:
    raise RuntimeError(
        "Unexpected or missing MD families."
    )

family_order = [
    "size_shape",
    "rna_association",
    "water_penetration",
]

display_names = {
    "size_shape": "Size and shape",
    "rna_association": "RNA association",
    "water_penetration": "Water penetration",
}

plot_data["family"] = pd.Categorical(
    plot_data["family"],
    categories=family_order,
    ordered=True,
)

plot_data = plot_data.sort_values(
    "family"
).reset_index(drop=True)

plot_data["display_name"] = (
    plot_data["family"]
    .astype(str)
    .map(display_names)
)

plot_data.to_csv(
    OUTPUT_TABLE,
    index=False,
)


positions = np.arange(
    len(plot_data)
)

width = 0.35

add_values = plot_data[
    "add_family_delta_vs_baseline"
].to_numpy(dtype=float)

removal_values = plot_data[
    "leave_family_out_effect"
].to_numpy(dtype=float)


figure, axis = plt.subplots(
    figsize=(9, 6.5)
)

bars_add = axis.bar(
    positions - width / 2,
    add_values,
    width,
    color="#4C78A8",
    label="Added alone to baseline",
)

bars_remove = axis.bar(
    positions + width / 2,
    removal_values,
    width,
    color="#F58518",
    label="Full MD minus model without family",
)

axis.axhline(
    0,
    color="black",
    linewidth=1.2,
)

axis.set_xticks(
    positions
)

axis.set_xticklabels(
    plot_data["display_name"],
    fontsize=11,
)

axis.set_ylabel(
    "Change in source-balanced MCC",
    fontsize=12,
)

axis.set_xlabel(
    "MD descriptor family",
    fontsize=12,
)

axis.set_title(
    "Exploratory MD descriptor-family ablation",
    fontsize=14,
    pad=14,
)

axis.grid(
    axis="y",
    linestyle=":",
    alpha=0.3,
)

axis.legend(
    frameon=False,
    fontsize=10,
    loc="upper right",
)


def label_bars(bars):
    for bar in bars:
        value = bar.get_height()

        offset = (
            0.018
            if value >= 0
            else -0.035
        )

        vertical_alignment = (
            "bottom"
            if value >= 0
            else "top"
        )

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=10,
        )


label_bars(bars_add)
label_bars(bars_remove)

minimum = min(
    add_values.min(),
    removal_values.min(),
)

maximum = max(
    add_values.max(),
    removal_values.max(),
)

axis.set_ylim(
    minimum - 0.12,
    maximum + 0.16,
)

figure.text(
    0.5,
    0.01,
    (
        "Exploratory analysis: 13 observations, "
        "6 independent MD sources; overall exact "
        "MD permutation p = 0.0597."
    ),
    ha="center",
    fontsize=9,
    color="#444444",
)

figure.tight_layout(
    rect=[0, 0.045, 1, 1]
)

png_path = (
    FIGURE_DIRECTORY
    / "md_family_ablation_mcc_effects.png"
)

pdf_path = (
    FIGURE_DIRECTORY
    / "md_family_ablation_mcc_effects.pdf"
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

print("===== MD FAMILY ABLATION FIGURE =====")
print(
    plot_data[
        [
            "display_name",
            "number_features",
            "add_family_delta_vs_baseline",
            "leave_family_out_effect",
        ]
    ].to_string(index=False)
)

print()
print("Written:", png_path)
print("Written:", pdf_path)
