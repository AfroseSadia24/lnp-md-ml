from pathlib import Path
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")

DATA_PATH = (
    ROOT
    / "results_v2/tables/"
    "exact_md_subset.csv"
)

MD_LIST_PATH = (
    ROOT
    / "results_v2/tables/"
    "md_features.txt"
)

AUDIT_PATH = (
    ROOT
    / "results_v2/tables/"
    "feature_column_audit.csv"
)

NUMERIC_PATH = (
    ROOT
    / "results_v2/tables/"
    "baseline_numeric_candidates.txt"
)

CATEGORICAL_PATH = (
    ROOT
    / "results_v2/tables/"
    "baseline_categorical_candidates.txt"
)

REVIEW_PATH = (
    ROOT
    / "results_v2/tables/"
    "columns_requiring_review.txt"
)


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

md_features = [
    line.strip()
    for line in MD_LIST_PATH.read_text().splitlines()
    if line.strip()
]

if len(data) != 13:
    raise RuntimeError(
        f"Expected 13 exact MD rows, found {len(data)}."
    )

if len(md_features) != 12:
    raise RuntimeError(
        f"Expected 12 approved MD features, "
        f"found {len(md_features)}."
    )


TARGET_COLUMNS = {
    "target",
    "transfection_active",
    "transfection_label",
    "binary_target",
    "class_label",
    "y",
}

METADATA_COLUMNS = {
    "analysis_row_id",
    "lnp_id",
    "formulation_id",
    "paper_doi",
    "doi",
    "paper_title",
    "reference",
    "citation",
    "source",
    "source_file",
    "row_id",
    "fold",
    "split",
    "md_source_lnp_id",
    "md_features_available",
    "md_match_type",
}

STRUCTURE_TEXT_COLUMNS = {
    "smiles",
    "canonical_smiles",
    "isomeric_smiles",
    "ionizable_lipid_smiles",
    "structure",
    "inchi",
    "inchikey",
}

SUSPICIOUS_PATTERN = re.compile(
    r"("
    r"target|transfection|outcome|response|"
    r"efficacy|expression|toxicity|"
    r"label|prediction|probability|"
    r"observed|measured|fold|split"
    r")",
    flags=re.IGNORECASE,
)


records = []

for column in data.columns:
    series = data[column]
    unique_count = int(series.nunique(dropna=True))
    missing_count = int(series.isna().sum())
    numeric = bool(is_numeric_dtype(series))

    lower = column.lower()

    if column in md_features:
        role = "approved_md_feature"

    elif lower.startswith("md_"):
        role = "excluded_other_md"

    elif lower in TARGET_COLUMNS:
        role = "target"

    elif (
        lower in METADATA_COLUMNS
        or lower.endswith("_id")
    ):
        role = "metadata"

    elif lower in STRUCTURE_TEXT_COLUMNS:
        role = "structure_text"

    elif SUSPICIOUS_PATTERN.search(lower):
        role = "requires_leakage_review"

    elif unique_count <= 1:
        role = "constant_in_exact_subset"

    elif numeric:
        role = "numeric_baseline_candidate"

    else:
        role = "categorical_baseline_candidate"

    records.append(
        {
            "column": column,
            "dtype": str(series.dtype),
            "nonmissing": int(series.notna().sum()),
            "missing": missing_count,
            "unique_values": unique_count,
            "role": role,
        }
    )


audit = pd.DataFrame(records)

audit.to_csv(
    AUDIT_PATH,
    index=False,
)

numeric_candidates = audit.loc[
    audit["role"].eq("numeric_baseline_candidate"),
    "column",
].tolist()

categorical_candidates = audit.loc[
    audit["role"].eq(
        "categorical_baseline_candidate"
    ),
    "column",
].tolist()

review_columns = audit.loc[
    audit["role"].eq("requires_leakage_review"),
    "column",
].tolist()

NUMERIC_PATH.write_text(
    "\n".join(numeric_candidates) + "\n"
)

CATEGORICAL_PATH.write_text(
    "\n".join(categorical_candidates) + "\n"
)

REVIEW_PATH.write_text(
    "\n".join(review_columns) + "\n"
)


print("===== FEATURE ROLE COUNTS =====")
print(
    audit["role"]
    .value_counts()
    .rename_axis("role")
    .reset_index(name="columns")
    .to_string(index=False)
)

print()
print("===== CATEGORICAL BASELINE CANDIDATES =====")

categorical_audit = audit.loc[
    audit["role"].eq(
        "categorical_baseline_candidate"
    ),
    [
        "column",
        "unique_values",
        "missing",
    ],
]

if categorical_audit.empty:
    print("None")
else:
    print(
        categorical_audit.to_string(index=False)
    )

print()
print("===== COLUMNS REQUIRING LEAKAGE REVIEW =====")

review_audit = audit.loc[
    audit["role"].eq("requires_leakage_review"),
    [
        "column",
        "dtype",
        "unique_values",
        "missing",
    ],
]

if review_audit.empty:
    print("None")
else:
    print(
        review_audit.to_string(index=False)
    )

print()
print("===== SUMMARY =====")
print("Exact rows:", len(data))
print("Total columns:", data.shape[1])
print("Approved MD features:", len(md_features))
print(
    "Numeric baseline candidates:",
    len(numeric_candidates),
)
print(
    "Categorical baseline candidates:",
    len(categorical_candidates),
)
print(
    "Columns requiring review:",
    len(review_columns),
)

print()
print(f"Written: {AUDIT_PATH}")
print(f"Written: {NUMERIC_PATH}")
print(f"Written: {CATEGORICAL_PATH}")
print(f"Written: {REVIEW_PATH}")
