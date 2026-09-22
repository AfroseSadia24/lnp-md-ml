from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

INPUT = (
    ROOT
    / "results_v2/predictions/"
    "baseline_vs_md_oof.csv"
)

FIGURE_DIR = ROOT / "figures/results_v2"
TABLE_DIR = ROOT / "results_v2/tables"

TARGET = "transfection_active"

MODEL_ORDER = [
    "baseline",
    "baseline_plus_md",
]

MODEL_LABELS = {
    "baseline": "Formulation + chemistry",
    "baseline_plus_md": (
        "Formulation + chemistry + MD"
    ),
}

MODEL_COLORS = {
    "baseline": "#2166AC",
    "baseline_plus_md": "#B2182B",
}


predictions = pd.read_csv(
    INPUT,
    low_memory=False,
)

required_columns = {
    "analysis_row_id",
    TARGET,
    "model",
    "prob_active",
}

missing = required_columns - set(
    predictions.columns
)

if missing:
    raise RuntimeError(
        f"Missing prediction columns: {sorted(missing)}"
    )

if set(predictions["model"]) != set(MODEL_ORDER):
    raise RuntimeError(
        "Unexpected model names in OOF predictions."
    )

model_counts = predictions.groupby(
    "model"
)["analysis_row_id"].nunique()

if not (model_counts == 13).all():
    raise RuntimeError(
        f"Expected 13 OOF rows per model:\n{model_counts}"
    )

truth_check = predictions.groupby(
    "analysis_row_id"
)[TARGET].nunique()

if (truth_check != 1).any():
    raise RuntimeError(
        "Target labels differ between model predictions."
    )


def configure_axis(axis, xlabel, ylabel):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.tick_params(labelsize=10)
    axis.grid(
        alpha=0.25,
        linestyle=":",
    )


roc_rows = []
pr_rows = []

for model_name in MODEL_ORDER:
    model_data = predictions.loc[
        predictions["model"].eq(model_name)
    ].copy()

    y_true = model_data[TARGET].to_numpy(
        dtype=int
    )

    probability = model_data[
        "prob_active"
    ].to_numpy(dtype=float)

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        probability,
    )

    precision, recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    for index in range(len(fpr)):
        roc_rows.append(
            {
                "model": model_name,
                "false_positive_rate": fpr[index],
                "true_positive_rate": tpr[index],
                "threshold": roc_thresholds[index],
            }
        )

    for index in range(len(recall)):
        pr_rows.append(
            {
                "model": model_name,
                "recall": recall[index],
                "precision": precision[index],
                "threshold": (
                    pr_thresholds[index]
                    if index < len(pr_thresholds)
                    else None
                ),
            }
        )


roc_points = pd.DataFrame(roc_rows)
pr_points = pd.DataFrame(pr_rows)

roc_points.to_csv(
    TABLE_DIR / "oof_roc_curve_points.csv",
    index=False,
)

pr_points.to_csv(
    TABLE_DIR / "oof_pr_curve_points.csv",
    index=False,
)


# Separate ROC figure
figure, axis = plt.subplots(
    figsize=(7, 6)
)

for model_name in MODEL_ORDER:
    model_data = predictions.loc[
        predictions["model"].eq(model_name)
    ]

    y_true = model_data[TARGET].to_numpy(
        dtype=int
    )

    probability = model_data[
        "prob_active"
    ].to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(
        y_true,
        probability,
    )

    auc = roc_auc_score(
        y_true,
        probability,
    )

    axis.plot(
        fpr,
        tpr,
        linewidth=2.5,
        color=MODEL_COLORS[model_name],
        label=(
            f"{MODEL_LABELS[model_name]} "
            f"(AUC = {auc:.3f})"
        ),
    )

axis.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Random ranking",
)

configure_axis(
    axis,
    "False positive rate",
    "True positive rate",
)

axis.set_title(
    "Out-of-fold ROC curves",
    fontsize=14,
)

axis.legend(
    frameon=False,
    fontsize=9,
    loc="lower right",
)

figure.tight_layout()

figure.savefig(
    FIGURE_DIR
    / "oof_roc_baseline_vs_md.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(figure)


# Separate precision-recall figure
figure, axis = plt.subplots(
    figsize=(7, 6)
)

prevalence = predictions.loc[
    predictions["model"].eq("baseline"),
    TARGET,
].mean()

for model_name in MODEL_ORDER:
    model_data = predictions.loc[
        predictions["model"].eq(model_name)
    ]

    y_true = model_data[TARGET].to_numpy(
        dtype=int
    )

    probability = model_data[
        "prob_active"
    ].to_numpy(dtype=float)

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    average_precision = (
        average_precision_score(
            y_true,
            probability,
        )
    )

    axis.plot(
        recall,
        precision,
        linewidth=2.5,
        color=MODEL_COLORS[model_name],
        label=(
            f"{MODEL_LABELS[model_name]} "
            f"(AP = {average_precision:.3f})"
        ),
    )

axis.axhline(
    prevalence,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"Active prevalence = {prevalence:.3f}",
)

configure_axis(
    axis,
    "Recall",
    "Precision",
)

axis.set_title(
    "Out-of-fold precision-recall curves",
    fontsize=14,
)

axis.legend(
    frameon=False,
    fontsize=9,
    loc="lower left",
)

figure.tight_layout()

figure.savefig(
    FIGURE_DIR
    / "oof_pr_baseline_vs_md.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(figure)


# Combined publication figure
figure, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)

roc_axis, pr_axis = axes

for model_name in MODEL_ORDER:
    model_data = predictions.loc[
        predictions["model"].eq(model_name)
    ]

    y_true = model_data[TARGET].to_numpy(
        dtype=int
    )

    probability = model_data[
        "prob_active"
    ].to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(
        y_true,
        probability,
    )

    roc_auc = roc_auc_score(
        y_true,
        probability,
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    average_precision = (
        average_precision_score(
            y_true,
            probability,
        )
    )

    roc_axis.plot(
        fpr,
        tpr,
        linewidth=2.5,
        color=MODEL_COLORS[model_name],
        label=(
            f"{MODEL_LABELS[model_name]} "
            f"(AUC = {roc_auc:.3f})"
        ),
    )

    pr_axis.plot(
        recall,
        precision,
        linewidth=2.5,
        color=MODEL_COLORS[model_name],
        label=(
            f"{MODEL_LABELS[model_name]} "
            f"(AP = {average_precision:.3f})"
        ),
    )

roc_axis.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    linewidth=1.5,
)

pr_axis.axhline(
    prevalence,
    color="gray",
    linestyle="--",
    linewidth=1.5,
)

configure_axis(
    roc_axis,
    "False positive rate",
    "True positive rate",
)

configure_axis(
    pr_axis,
    "Recall",
    "Precision",
)

roc_axis.set_title(
    "A. ROC curve",
    fontsize=14,
)

pr_axis.set_title(
    "B. Precision-recall curve",
    fontsize=14,
)

roc_axis.legend(
    frameon=False,
    fontsize=8,
    loc="lower right",
)

pr_axis.legend(
    frameon=False,
    fontsize=8,
    loc="lower left",
)

figure.tight_layout()

figure.savefig(
    FIGURE_DIR
    / "oof_curves_baseline_vs_md_combined.png",
    dpi=400,
    bbox_inches="tight",
)

plt.close(figure)


print("===== OOF CURVE RESULTS =====")

for model_name in MODEL_ORDER:
    model_data = predictions.loc[
        predictions["model"].eq(model_name)
    ]

    y_true = model_data[TARGET]
    probability = model_data["prob_active"]

    print(
        MODEL_LABELS[model_name],
        f"ROC-AUC={roc_auc_score(y_true, probability):.4f}",
        (
            "Active PR-AUC="
            f"{average_precision_score(y_true, probability):.4f}"
        ),
    )

print()
print(
    "PASS: all curves use out-of-fold "
    "predictions only."
)

print(
    "Written: figures/results_v2/"
    "oof_roc_baseline_vs_md.png"
)

print(
    "Written: figures/results_v2/"
    "oof_pr_baseline_vs_md.png"
)

print(
    "Written: figures/results_v2/"
    "oof_curves_baseline_vs_md_combined.png"
)
