from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")
TABLES = ROOT / "results_v2/tables"

DATA_PATH = TABLES / "exact_md_subset.csv"
FOLD_PATH = TABLES / "doi_grouped_folds.csv"
GROUP_AUDIT_PATH = TABLES / "doi_group_audit.csv"
OVERLAP_PATH = TABLES / "doi_fold_md_source_overlap.csv"
CONFIG_PATH = TABLES / "doi_grouped_fold_config.json"

TARGET = "transfection_active"
GROUP = "paper_doi"
MD_SOURCE = "md_source_lnp_id"


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

required = [
    "analysis_row_id",
    "lnp_id",
    TARGET,
    GROUP,
    MD_SOURCE,
]

missing = [
    column
    for column in required
    if column not in data.columns
]

if missing:
    raise RuntimeError(
        f"Required columns are missing: {missing}"
    )

if len(data) != 13:
    raise RuntimeError(
        f"Expected 13 rows, found {len(data)}."
    )

data[TARGET] = data[TARGET].astype(int)
data[GROUP] = data[GROUP].astype(str)
data[MD_SOURCE] = data[MD_SOURCE].astype(str)

if data[TARGET].nunique() != 2:
    raise RuntimeError(
        "Both target classes are required."
    )


group_audit = (
    data.groupby(GROUP, as_index=False)
    .agg(
        rows=("lnp_id", "size"),
        active=(TARGET, "sum"),
        md_sources=(
            MD_SOURCE,
            lambda values: ";".join(
                sorted(set(values))
            ),
        ),
        lnp_ids=(
            "lnp_id",
            lambda values: ";".join(
                sorted(
                    set(values.astype(str))
                )
            ),
        ),
    )
)

group_audit["inactive"] = (
    group_audit["rows"]
    - group_audit["active"]
)

group_audit.to_csv(
    GROUP_AUDIT_PATH,
    index=False,
)


def evaluate_assignment(fold_values):
    statistics = []

    for fold in sorted(np.unique(fold_values)):
        test_mask = fold_values == fold
        train_mask = ~test_mask

        y_test = data.loc[
            test_mask,
            TARGET,
        ]

        y_train = data.loc[
            train_mask,
            TARGET,
        ]

        statistics.append(
            {
                "fold": int(fold),
                "test_rows": int(test_mask.sum()),
                "test_active": int(y_test.sum()),
                "test_inactive": int(
                    (y_test == 0).sum()
                ),
                "test_doi_groups": int(
                    data.loc[
                        test_mask,
                        GROUP,
                    ].nunique()
                ),
                "train_rows": int(train_mask.sum()),
                "train_active": int(y_train.sum()),
                "train_inactive": int(
                    (y_train == 0).sum()
                ),
            }
        )

    statistics = pd.DataFrame(statistics)

    valid = bool(
        (statistics["test_active"] > 0).all()
        and (statistics["test_inactive"] > 0).all()
        and (statistics["train_active"] > 0).all()
        and (statistics["train_inactive"] > 0).all()
    )

    global_rate = data[TARGET].mean()

    balance_score = float(
        statistics["test_rows"].std(ddof=0)
        + (
            (
                statistics["test_active"]
                / statistics["test_rows"]
            )
            - global_rate
        ).abs().mean()
    )

    return valid, balance_score, statistics


selected = None

# Search only using group size and class balance.
# No model performance is used to choose the folds.
for number_of_folds in [4, 3, 2]:
    valid_candidates = []

    for seed in range(42, 1042):
        splitter = StratifiedGroupKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=seed,
        )

        fold_values = np.full(
            len(data),
            -1,
            dtype=int,
        )

        try:
            splits = list(
                splitter.split(
                    data,
                    data[TARGET],
                    groups=data[GROUP],
                )
            )
        except ValueError:
            continue

        for fold, (_, test_indices) in enumerate(
            splits
        ):
            fold_values[test_indices] = fold

        valid, score, statistics = (
            evaluate_assignment(fold_values)
        )

        if valid:
            valid_candidates.append(
                {
                    "seed": seed,
                    "score": score,
                    "fold_values": fold_values.copy(),
                    "statistics": statistics,
                }
            )

    if valid_candidates:
        selected = min(
            valid_candidates,
            key=lambda item: (
                item["score"],
                item["seed"],
            ),
        )

        selected["number_of_folds"] = (
            number_of_folds
        )
        break


if selected is None:
    raise RuntimeError(
        "No valid DOI-grouped split was found in "
        "which every test fold contains both classes."
    )


fold_values = selected["fold_values"]
fold_statistics = selected["statistics"]

fold_table = data[
    [
        "analysis_row_id",
        "lnp_id",
        GROUP,
        TARGET,
        MD_SOURCE,
    ]
].copy()

fold_table["fold"] = fold_values

if (fold_table["fold"] < 0).any():
    raise RuntimeError(
        "Some rows were not assigned to a fold."
    )

doi_fold_counts = fold_table.groupby(
    GROUP
)["fold"].nunique()

if (doi_fold_counts != 1).any():
    raise RuntimeError(
        "DOI leakage detected across folds."
    )

fold_table.to_csv(
    FOLD_PATH,
    index=False,
)


overlap_rows = []

for fold in sorted(fold_table["fold"].unique()):
    test = fold_table[
        fold_table["fold"] == fold
    ]

    train = fold_table[
        fold_table["fold"] != fold
    ]

    test_sources = set(test[MD_SOURCE])
    train_sources = set(train[MD_SOURCE])
    overlap = sorted(test_sources & train_sources)

    overlap_rows.append(
        {
            "fold": int(fold),
            "test_md_sources": ";".join(
                sorted(test_sources)
            ),
            "train_md_sources": ";".join(
                sorted(train_sources)
            ),
            "overlapping_md_sources": ";".join(
                overlap
            ),
            "number_overlapping": len(overlap),
        }
    )

overlap_table = pd.DataFrame(overlap_rows)

overlap_table.to_csv(
    OVERLAP_PATH,
    index=False,
)


config = {
    "validation": "StratifiedGroupKFold",
    "group_column": GROUP,
    "target_column": TARGET,
    "number_of_folds": int(
        selected["number_of_folds"]
    ),
    "random_state": int(selected["seed"]),
    "selection_rule": (
        "Largest valid fold count among 4, 3, and 2. "
        "Every train and test fold must contain both "
        "classes. Ties use class and row balance only."
    ),
    "number_of_rows": int(len(data)),
    "number_of_doi_groups": int(
        data[GROUP].nunique()
    ),
}

with open(CONFIG_PATH, "w") as handle:
    json.dump(
        config,
        handle,
        indent=2,
    )


print("===== DOI GROUP AUDIT =====")
print(group_audit.to_string(index=False))

print()
print("===== SELECTED FOLD CONFIGURATION =====")
print(
    "Number of folds:",
    selected["number_of_folds"],
)
print("Random state:", selected["seed"])
print(
    "DOI groups:",
    data[GROUP].nunique(),
)

print()
print("===== FOLD CLASS COUNTS =====")
print(fold_statistics.to_string(index=False))

print()
print("===== MD SOURCE OVERLAP AUDIT =====")
print(overlap_table.to_string(index=False))

print()
print("===== ACCEPTANCE CHECKS =====")
print("Every row assigned once: PASS")
print("No DOI appears in multiple folds: PASS")
print("Every training fold has both classes: PASS")
print("Every testing fold has both classes: PASS")

if (overlap_table["number_overlapping"] > 0).any():
    print(
        "Repeated MD sources cross some DOI folds: "
        "EXPECTED, but source-grouped sensitivity "
        "analysis will be mandatory."
    )

print()
print(f"Written: {FOLD_PATH}")
print(f"Written: {GROUP_AUDIT_PATH}")
print(f"Written: {OVERLAP_PATH}")
print(f"Written: {CONFIG_PATH}")
