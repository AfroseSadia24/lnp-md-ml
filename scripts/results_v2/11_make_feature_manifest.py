from pathlib import Path

import pandas as pd


ROOT = Path("/home/sa594/LNP_ML/hpc_lnp_ml")
TABLES = ROOT / "results_v2/tables"

DATA_PATH = TABLES / "exact_md_subset.csv"
NUMERIC_CANDIDATES_PATH = (
    TABLES / "baseline_numeric_candidates.txt"
)
MD_PATH = TABLES / "md_features.txt"


data = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

numeric_candidates = [
    value.strip()
    for value in NUMERIC_CANDIDATES_PATH
    .read_text()
    .splitlines()
    if value.strip()
]

md_features = [
    value.strip()
    for value in MD_PATH.read_text().splitlines()
    if value.strip()
]


numeric_exclusions = {
    "paper_year": "study metadata",
    "particle_size_nm_std": (
        "post-formulation measured property"
    ),
    "pdi_std": (
        "post-formulation measured property"
    ),
    "zeta_potential_mv_std": (
        "post-formulation measured property"
    ),
    "encapsulation_efficiency_percent_std": (
        "post-formulation measured property"
    ),
}

categorical_features = [
    "ionizable_lipid",
    "peg_lipid",
    "helper_lipid",
    "mixing_method",
    "buffer",
    "bio_administration_route",
]

categorical_exclusions = {
    "loading_capacity_std": (
        "post-formulation measured property"
    ),
    "ionizable_lipid_canonical_smiles": (
        "raw structure text; represented by "
        "RDKit and Morgan descriptors"
    ),
    "transfection_label_source": (
        "target-label metadata"
    ),
    "target_type": (
        "possible target information"
    ),
}


baseline_numeric = [
    column
    for column in numeric_candidates
    if column not in numeric_exclusions
]


formulation_numeric = [
    "ionizable_lipid_mol_percent_final",
    "peg_lipid_mol_percent_final",
    "sterol_lipid_mol_percent_final",
    "helper_lipid_mol_percent_final",
    "buffer_ph",
]

chemistry_numeric = [
    column
    for column in baseline_numeric
    if (
        column.startswith("rdkit_")
        or column.startswith("atom_count_")
        or column.startswith("tail_")
        or column.startswith("morgan_")
        or column in {
            "ester_count",
            "hydroxyl_count",
            "ether_count",
        }
    )
]

cargo_numeric = [
    column
    for column in baseline_numeric
    if column.startswith("cargo_")
]

classified_numeric = set(
    formulation_numeric
    + chemistry_numeric
    + cargo_numeric
)

unclassified_numeric = sorted(
    set(baseline_numeric) - classified_numeric
)

if unclassified_numeric:
    raise RuntimeError(
        "Unclassified numeric baseline features: "
        f"{unclassified_numeric}"
    )


baseline_features = (
    formulation_numeric
    + categorical_features
    + chemistry_numeric
    + cargo_numeric
)

enhanced_features = (
    baseline_features
    + md_features
)


required_features = list(
    dict.fromkeys(
        baseline_features
        + md_features
        + list(numeric_exclusions)
        + list(categorical_exclusions)
    )
)

missing_features = [
    column
    for column in required_features
    if column not in data.columns
]

if missing_features:
    raise RuntimeError(
        "Features missing from the exact subset: "
        f"{missing_features}"
    )

if len(baseline_numeric) != 74:
    raise RuntimeError(
        "Expected 74 numeric baseline features, "
        f"found {len(baseline_numeric)}."
    )

if len(categorical_features) != 6:
    raise RuntimeError(
        "Expected 6 categorical baseline features."
    )

if len(md_features) != 12:
    raise RuntimeError(
        "Expected 12 approved MD features."
    )

if len(baseline_features) != 80:
    raise RuntimeError(
        "Expected 80 total baseline features, "
        f"found {len(baseline_features)}."
    )

if len(enhanced_features) != 92:
    raise RuntimeError(
        "Expected 92 enhanced features, "
        f"found {len(enhanced_features)}."
    )

if len(baseline_features) != len(
    set(baseline_features)
):
    raise RuntimeError(
        "Duplicate baseline features detected."
    )

if set(baseline_features) & set(md_features):
    raise RuntimeError(
        "Baseline and MD feature blocks overlap."
    )


def write_list(filename, values):
    path = TABLES / filename
    path.write_text(
        "\n".join(values) + "\n"
    )
    return path


written_paths = [
    write_list(
        "formulation_numeric_features.txt",
        formulation_numeric,
    ),
    write_list(
        "chemistry_numeric_features.txt",
        chemistry_numeric,
    ),
    write_list(
        "cargo_numeric_features.txt",
        cargo_numeric,
    ),
    write_list(
        "baseline_numeric_features.txt",
        baseline_numeric,
    ),
    write_list(
        "baseline_categorical_features.txt",
        categorical_features,
    ),
    write_list(
        "baseline_features.txt",
        baseline_features,
    ),
    write_list(
        "enhanced_features.txt",
        enhanced_features,
    ),
]


manifest_rows = []

for column in formulation_numeric:
    manifest_rows.append(
        {
            "feature": column,
            "block": "formulation",
            "data_type": "numeric",
            "use": "baseline_and_enhanced",
            "reason": "",
        }
    )

for column in categorical_features:
    block = (
        "experimental_context"
        if column == "bio_administration_route"
        else "formulation"
    )

    manifest_rows.append(
        {
            "feature": column,
            "block": block,
            "data_type": "categorical",
            "use": "baseline_and_enhanced",
            "reason": "",
        }
    )

for column in chemistry_numeric:
    manifest_rows.append(
        {
            "feature": column,
            "block": "chemistry",
            "data_type": "numeric",
            "use": "baseline_and_enhanced",
            "reason": "",
        }
    )

for column in cargo_numeric:
    manifest_rows.append(
        {
            "feature": column,
            "block": "cargo",
            "data_type": "numeric",
            "use": "baseline_and_enhanced",
            "reason": "",
        }
    )

for column in md_features:
    manifest_rows.append(
        {
            "feature": column,
            "block": "md",
            "data_type": "numeric",
            "use": "enhanced_only",
            "reason": "",
        }
    )

for column, reason in numeric_exclusions.items():
    manifest_rows.append(
        {
            "feature": column,
            "block": "excluded",
            "data_type": "numeric",
            "use": "excluded",
            "reason": reason,
        }
    )

for column, reason in categorical_exclusions.items():
    manifest_rows.append(
        {
            "feature": column,
            "block": "excluded",
            "data_type": "categorical",
            "use": "excluded",
            "reason": reason,
        }
    )


manifest = pd.DataFrame(manifest_rows)

manifest.to_csv(
    TABLES / "feature_manifest.csv",
    index=False,
)


summary = pd.DataFrame(
    [
        {
            "feature_set": "baseline_numeric",
            "number_of_features": len(
                baseline_numeric
            ),
        },
        {
            "feature_set": "baseline_categorical",
            "number_of_features": len(
                categorical_features
            ),
        },
        {
            "feature_set": "baseline_total",
            "number_of_features": len(
                baseline_features
            ),
        },
        {
            "feature_set": "approved_md",
            "number_of_features": len(
                md_features
            ),
        },
        {
            "feature_set": "enhanced_total",
            "number_of_features": len(
                enhanced_features
            ),
        },
        {
            "feature_set": "explicitly_excluded",
            "number_of_features": (
                len(numeric_exclusions)
                + len(categorical_exclusions)
            ),
        },
    ]
)

summary.to_csv(
    TABLES / "feature_manifest_summary.csv",
    index=False,
)


print("===== FROZEN FEATURE MANIFEST =====")
print(summary.to_string(index=False))

print()
print("===== EXCLUDED FEATURES =====")

print(
    manifest.loc[
        manifest["use"].eq("excluded"),
        ["feature", "reason"],
    ].to_string(index=False)
)

print()
print("===== ACCEPTANCE CHECKS =====")
print("No target columns included: PASS")
print("No metadata columns included: PASS")
print("No measured characterization included: PASS")
print("Baseline and MD blocks do not overlap: PASS")
print("All manifest features exist in dataset: PASS")

print()
for path in written_paths:
    print(f"Written: {path}")

print(
    "Written:",
    TABLES / "feature_manifest.csv",
)
print(
    "Written:",
    TABLES / "feature_manifest_summary.csv",
)
