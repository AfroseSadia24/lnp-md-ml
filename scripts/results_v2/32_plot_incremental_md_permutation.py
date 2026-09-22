from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

DISTRIBUTION_PATH = (
    ROOT
    / "results_v2/permutation/"
    "incremental_md_exact_permutation_distribution.csv"
)

PVALUES_PATH = (
    ROOT
    / "results_v2/permutation/"
    "incremental_md_exact_permutation_pvalues.csv"
)

FIGURE_DIRECTORY = (
    ROOT / "figures/results_v2"
)

COLUMN = "source_balanced_delta_mcc"


distribution = pd.read_csv(
    DISTRIBUTION_PATH
)

tests = pd.read_csv(
    PVALUES_PATH
)

if len(distribution) != 720:
    raise RuntimeError(
        f"Expected 720 assignments, found "
        f"{len(distribution)}."
    )

if COLUMN not in distribution.columns:
    raise RuntimeError(
        f"Missing column: {COLUMN}"
    )

primary = tests.loc[
    tests["weighting"].eq("source_balanced")
    & tests["metric"].eq("mcc")
]

if len(primary) != 1:
    raise RuntimeError(
        "Primary MCC test was not found."
    )

primary = primary.iloc[0]

values = distribution[
    COLUMN
].to_numpy(dtype=float)

observed = float(
    primary["observed_delta"]
)

p_value = float(
    primary["exact_one_sided_p"]
)

null_mean = float(
    primary["null_mean_delta"]
)

q025 = float(
    primary["null_q025"]
)

q975 = float(
    primary["null_q975"]
)

number_extreme = int(
    np.sum(values >= observed - 1e-12)
)

calculated_p = (
    number_extreme
    / len(values)
)

if not np.isclose(
    calculated_p,
    p_value,
):
    raise RuntimeError(
        "Stored and calculated p-values differ."
    )


figure, axis = plt.subplots(
    figsize=(9, 6.5)
)

axis.hist(
    values,
    bins=25,
    color="#4C78A8",
    edgecolor="white",
    linewidth=0.8,
    alpha=0.90,
)

axis.axvspan(
    q025,
    q975,
    color="#72B7B2",
    alpha=0.16,
    label=(
        "Central 95% of permutation distribution"
    ),
)

axis.axvline(
    null_mean,
    color="#333333",
    linestyle=":",
    linewidth=2.0,
    label=f"Null mean = {null_mean:.3f}",
)

axis.axvline(
    observed,
    color="#C44E52",
    linestyle="--",
    linewidth=2.8,
    label=f"Observed gain = {observed:.3f}",
)

axis.set_xlabel(
    "MD minus baseline source-balanced MCC",
    fontsize=12,
)

axis.set_ylabel(
    "Number of MD-vector assignments",
    fontsize=12,
)

axis.set_title(
    "Exact permutation test of incremental MD contribution",
    fontsize=14,
    pad=14,
)

axis.text(
    0.02,
    0.96,
    (
        f"Exact one-sided p = {p_value:.4f}\n"
        f"{number_extreme} of {len(values)} "
        "assignments matched or exceeded observed"
    ),
    transform=axis.transAxes,
    ha="left",
    va="top",
    fontsize=11,
    bbox={
        "boxstyle": "round,pad=0.4",
        "facecolor": "white",
        "edgecolor": "#999999",
        "alpha": 0.95,
    },
)

axis.grid(
    axis="y",
    linestyle=":",
    alpha=0.30,
)

axis.legend(
    frameon=False,
    fontsize=9,
    loc="upper left",
    bbox_to_anchor=(0, 0.78),
)

figure.tight_layout()

png_path = (
    FIGURE_DIRECTORY
    / "incremental_md_exact_permutation_mcc.png"
)

pdf_path = (
    FIGURE_DIRECTORY
    / "incremental_md_exact_permutation_mcc.pdf"
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

print("===== PERMUTATION FIGURE =====")
print("Assignments:", len(values))
print("Observed MCC gain:", round(observed, 6))
print("Assignments as or more extreme:", number_extreme)
print("Exact p-value:", round(p_value, 6))
print("Null 95% interval:", round(q025, 6), round(q975, 6))
print()
print("Written:", png_path)
print("Written:", pdf_path)
