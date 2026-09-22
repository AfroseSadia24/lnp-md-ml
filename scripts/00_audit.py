from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import normalize_doi, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/qc_report.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    required = {
        "lnp_id", "transfection_active", "transfection_label_source", "paper_doi",
        "ionizable_lipid_canonical_smiles",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    if df.columns.duplicated().any():
        raise SystemExit("Duplicate column names detected")

    doi = df["paper_doi"].map(normalize_doi)
    row803 = df[df["lnp_id"].astype(str) == "803"]
    report = {
        "input": str(Path(args.input).resolve()),
        "sha256": sha256(args.input),
        "rows": len(df),
        "columns": len(df.columns),
        "target_counts": df["transfection_active"].value_counts(dropna=False).to_dict(),
        "raw_doi_groups": int(df["paper_doi"].nunique(dropna=False)),
        "normalized_doi_groups": int(doi.nunique(dropna=False)),
        "unique_lnp_ids": int(df["lnp_id"].nunique(dropna=False)),
        "unique_structure_strings": int(df["ionizable_lipid_canonical_smiles"].nunique(dropna=True)),
        "missing_structures": int(df["ionizable_lipid_canonical_smiles"].isna().sum()),
        "duplicate_lnp_ids": int(df["lnp_id"].duplicated().sum()),
        "id_803_rows": row803.to_dict(orient="records"),
        "id_803_warning": "ID 803 has a curator-assigned inactive label without a published value; retain and run exclusion sensitivity.",
        "columns_below_10pct_fill": [c for c in df if df[c].notna().mean() < 0.10],
        "zero_variance_columns": [c for c in df if df[c].nunique(dropna=False) <= 1],
    }
    write_json(report, args.output)
    print(f"PASS: wrote {args.output}")
    print(report)


if __name__ == "__main__":
    main()

