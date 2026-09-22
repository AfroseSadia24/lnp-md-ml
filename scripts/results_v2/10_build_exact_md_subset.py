from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

INPUT = (
    ROOT
    / "data/raw/"
    "LNP_transfection_with_reused_MD_features.csv"
)

METRICS_JSON = (
    ROOT
    / "results/md_exploratory/"
    "md_only_loso_metrics.json"
)

OUTPUT = (
    ROOT
    / "results_v2/tables/"
    "exact_md_subset.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "results_v2/tables/"
    "exact_md_subset_audit.csv"
)

SOURCE_OUTPUT = (
    ROOT
    / "results_v2/tables/"
    "exact_md_subset_by_source.csv"
)

MD_LIST_OUTPUT = (
    ROOT
    / "results_v2/tables/"
    "md_features.txt"
)

TARGET = "transfection_active"
GROUP = "paper_doi"
LNP_ID = "lnp_id"
MD_SOURCE = "md_source_lnp_id"
MD_FLAG = "md_features_available"


DEFAULT_MD_FEATURES = [
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
    (
        "md_water_inside_LNP_core_RgSphere_"
        "count_fraction_of_total_water_mean_last100ns"
    ),
]


if not INPUT.is_file():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT}"
    )

data = pd.read_csv(INPUT, low_memory=False)

required_columns = [
    LNP_ID,
    TARGET,
    GROUP,
    MD_SOURCE,
    MD_FLAG,
]

missing_required = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_required:
    raise RuntimeError(
        "Missing required columns: "
        f"{missing_required}"
    )

# Recover the same MD feature list used in the
# completed exploratory model.
if METRICS_JSON.is_file():
    with open(METRICS_JSON) as handle:
        previous_result = json.load(handle)

    md_features = previous_result.get(
        "features",
        DEFAULT_MD_FEATURES,
    )
else:
    md_features = DEFAULT_MD_FEATURES

missing_md = [
    column
    for column in md_features
    if column not in data.columns
]

if missing_md:
    raise RuntimeError(
        "The following approved MD features are missing:\n"
        + "\n".join(missing_md)
    )

if len(md_features) != 12:
    raise RuntimeError(
        "Expected 12 approved MD features, but found "
        f"{len(md_features)}."
    )

# Ensure all MD descriptors are numeric.
for column in md_features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce",
)

data[MD_FLAG] = pd.to_numeric(
    data[MD_FLAG],
    errors="coerce",
)

data[LNP_ID] = data[LNP_ID].astype(str).str.strip()
data[GROUP] = data[GROUP].astype(str).str.strip()
data[MD_SOURCE] = (
    data[MD_SOURCE]
    .astype(str)
    .str.strip()
)

valid_target = data[TARGET].isin([0, 1])

valid_group = (
    data[GROUP].notna()
    & ~data[GROUP].isin(
        ["", "nan", "None", "NA"]
    )
)

valid_md_flag = data[MD_FLAG].eq(1)

# LNP803 is excluded because its MD system was not accepted.
valid_md_source = (
    data[MD_SOURCE].notna()
    & ~data[MD_SOURCE].isin(
        ["", "nan", "None", "NA", "LNP803", "803"]
    )
)

valid_lnp_id = ~data[LNP_ID].isin(
    ["803", "LNP803"]
)

valid_md_values = (
    data[md_features].notna().all(axis=1)
    & np.isfinite(
        data[md_features].to_numpy(dtype=float)
    ).all(axis=1)
)

eligible = (
    valid_target
    & valid_group
    & valid_md_flag
    & valid_md_source
    & valid_lnp_id
    & valid_md_values
)

exact = data.loc[eligible].copy()

# Preserve the original CSV row position.
exact.insert(
    0,
    "analysis_row_id",
    data.index[eligible].to_numpy(),
)

exact[TARGET] = exact[TARGET].astype(int)

if exact.empty:
    raise RuntimeError(
        "The exact MD subset contains zero rows."
    )

if exact[TARGET].nunique() != 2:
    raise RuntimeError(
        "The exact MD subset does not contain both classes."
    )

if exact[md_features].isna().any().any():
    raise RuntimeError(
        "Missing MD values remain in the exact subset."
    )

if not np.isfinite(
    exact[md_features].to_numpy(dtype=float)
).all():
    raise RuntimeError(
        "Infinite MD values remain in the exact subset."
    )

if exact[LNP_ID].isin(["803", "LNP803"]).any():
    raise RuntimeError(
        "LNP803 was not excluded by formulation ID."
    )

if exact["analysis_row_id"].duplicated().any():
    raise RuntimeError(
        "Duplicated analysis row IDs were found."
    )

if exact[MD_SOURCE].isin(
    ["LNP803", "803"]
).any():
    raise RuntimeError(
        "LNP803 was not excluded correctly."
    )

audit = pd.DataFrame(
    {
        "item": [
            "input_rows",
            "exact_md_rows",
            "active_rows",
            "inactive_rows",
            "doi_groups",
            "independent_md_sources",
            "approved_md_features",
            "excluded_rows",
        ],
        "value": [
            len(data),
            len(exact),
            int(exact[TARGET].eq(1).sum()),
            int(exact[TARGET].eq(0).sum()),
            int(exact[GROUP].nunique()),
            int(exact[MD_SOURCE].nunique()),
            len(md_features),
            int(len(data) - len(exact)),
        ],
    }
)

source_summary = (
    exact.groupby(MD_SOURCE, as_index=False)
    .agg(
        rows=(LNP_ID, "size"),
        unique_lnp_ids=(LNP_ID, "nunique"),
        active=(TARGET, "sum"),
        doi_groups=(GROUP, "nunique"),
    )
)

source_summary["inactive"] = (
    source_summary["rows"]
    - source_summary["active"]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

exact.to_csv(
    OUTPUT,
    index=False,
)

audit.to_csv(
    AUDIT_OUTPUT,
    index=False,
)

source_summary.to_csv(
    SOURCE_OUTPUT,
    index=False,
)

MD_LIST_OUTPUT.write_text(
    "\n".join(md_features) + "\n"
)

print("===== EXACT MD SUBSET AUDIT =====")
print(audit.to_string(index=False))

print()
print("===== ROWS BY MD SOURCE =====")
print(source_summary.to_string(index=False))

print()
print("===== ACCEPTANCE CHECKS =====")
print("Both target classes present: PASS")
print("All MD values complete and finite: PASS")
print("LNP803 excluded: PASS")
print("Analysis row IDs unique: PASS")

print()
print(f"Written: {OUTPUT}")
print(f"Written: {AUDIT_OUTPUT}")
print(f"Written: {SOURCE_OUTPUT}")
print(f"Written: {MD_LIST_OUTPUT}")
