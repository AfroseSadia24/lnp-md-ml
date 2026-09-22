from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import normalize_doi, sha256, stable_lipid_key, write_json


RENAME = {
    "lnp_id": "id_lnp_id",
    "transfection_active": "target_transfection_active",
    "transfection_label_source": "confounder_label_source",
    "paper_year": "confounder_paper_year",
    "ionizable_lipid_canonical_smiles": "struct_ionizable_smiles",
    "bio_administration_route": "confounder_administration_route",
    "target_type": "feat_cargo_type",
}


def classify_column(c: str) -> str | None:
    if c.startswith("morgan_r2_2048_bit_"):
        return "feat_fp_" + c.removeprefix("morgan_r2_2048_bit_")
    if c.startswith("rdkit_3d_"):
        return "feat_chem3d_" + c.removeprefix("rdkit_3d_")
    if c.startswith("rdkit_") or c.startswith("atom_count_") or c.endswith("_count") or c.startswith("tail_") or c.startswith("has_"):
        return "feat_chem2d_" + c
    if c.startswith("cargo_"):
        return "feat_cargo_" + c.removeprefix("cargo_")
    if c in {"ionizable_lipid_mol_percent_final", "peg_lipid_mol_percent_final", "sterol_lipid_mol_percent_final", "helper_lipid_mol_percent_final", "mixing_method", "solvent", "buffer", "buffer_ph", "ionizable_lipid", "peg_lipid", "sterol_lipid", "helper_lipid"}:
        return "feat_form_" + c
    if c in {"particle_size_nm_std", "pdi_std", "zeta_potential_mv_std", "encapsulation_efficiency_percent_std", "loading_capacity_std"}:
        return "feat_phys_" + c
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--exclude-id-803", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, low_memory=False)
    if args.exclude_id_803:
        raw = raw[raw["lnp_id"].astype(str) != "803"].copy()

    data = pd.DataFrame(index=raw.index)
    data["id_lnp_id"] = raw["lnp_id"].astype(str)
    data["group_paper_doi"] = raw["paper_doi"].map(normalize_doi)
    data["group_lipid_key"] = [stable_lipid_key(s, i) for s, i in zip(raw["ionizable_lipid_canonical_smiles"], raw["lnp_id"])]
    data["target_transfection_active"] = pd.to_numeric(raw["transfection_active"], errors="coerce")
    data["confounder_label_source"] = raw["transfection_label_source"].fillna("unreported").astype(str)
    data["confounder_paper_year"] = raw["paper_year"]
    data["confounder_administration_route"] = raw["bio_administration_route"].fillna("unreported").astype(str)
    data["struct_ionizable_smiles"] = raw["ionizable_lipid_canonical_smiles"]
    data["structure_available"] = raw["ionizable_lipid_canonical_smiles"].notna().astype(int)

    mapped = {}
    categorical_sources = {
        "ionizable_lipid", "peg_lipid", "sterol_lipid", "helper_lipid",
        "mixing_method", "solvent", "buffer", "target_type", "cargo_sequence_type",
    }
    for c in raw.columns:
        new = classify_column(c)
        if new:
            mapped[new] = raw[c] if c in categorical_sources else pd.to_numeric(raw[c], errors="coerce")
    if mapped:
        data = pd.concat([data, pd.DataFrame(mapped, index=raw.index)], axis=1)

    data = data[data["target_transfection_active"].notna()].reset_index(drop=True)
    feature_cols = [c for c in data if c.startswith("feat_")]
    low_fill = [c for c in feature_cols if data[c].notna().mean() < 0.10]
    data = data.drop(columns=low_fill)

    y = data["target_transfection_active"].astype(int).to_numpy()
    groups = data["group_paper_doi"].to_numpy()
    grouped = np.full(len(data), -1, dtype=int)
    for fold, (_, te) in enumerate(GroupKFold(5).split(data, y, groups)):
        grouped[te] = fold
    random_folds = np.full(len(data), -1, dtype=int)
    for fold, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(data, y)):
        random_folds[te] = fold

    suffix = "_exclude803" if args.exclude_id_803 else ""
    modelready = outdir / f"transfection_modelready_v1{suffix}.csv"
    data.to_csv(modelready, index=False)
    base = data[["id_lnp_id", "group_paper_doi", "group_lipid_key", "target_transfection_active"]].copy()
    base.assign(fold=grouped).to_csv(outdir / f"folds_grouped_v1{suffix}.csv", index=False)
    base.assign(fold=random_folds).to_csv(outdir / f"folds_random_v1{suffix}.csv", index=False)

    dictionary = pd.DataFrame({
        "processed_column": data.columns,
        "block": [c.split("_", 2)[0] + "_" if "_" in c else "audit" for c in data.columns],
        "dtype": data.dtypes.astype(str).values,
        "fill_fraction": data.notna().mean().values,
        "model_input": [c.startswith("feat_") for c in data.columns],
    })
    Path("reports").mkdir(exist_ok=True)
    dictionary.to_csv("reports/data_dictionary.csv", index=False)
    write_json({
        "input_sha256": sha256(args.input), "output_sha256": sha256(modelready),
        "rows": len(data), "columns": len(data.columns), "feature_columns": sum(c.startswith("feat_") for c in data),
        "doi_groups": int(data.group_paper_doi.nunique()), "lipid_keys": int(data.group_lipid_key.nunique()),
        "target_counts": data.target_transfection_active.value_counts().to_dict(), "dropped_low_fill": low_fill,
        "excluded_id_803": args.exclude_id_803,
    }, f"reports/preprocess_manifest_v1{suffix}.json")
    print(f"PASS: {modelready}")


if __name__ == "__main__":
    main()
