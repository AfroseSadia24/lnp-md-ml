# Input schemas

Place the current file at:

`data/raw/LNP_transfection_binary_target_with_803.csv`

Future MD data should be one row per canonical ionizable lipid and contain:

- `lipid_key`: canonical SMILES produced by the same standardization function used for the main data.
- `feat_md_*`: complete MD descriptor columns.
- Mechanistic block prefixes such as `feat_md_ionization_*`, `feat_md_lipid_rna_*`, `feat_md_structural_*`, `feat_md_water_ion_*`, and `feat_md_dynamics_*`.

Future organ data must contain a binary organ label and DOI. It should be processed as a separate endpoint through the same fold-safe harness. Do not merge organ labels into the transfection target.

