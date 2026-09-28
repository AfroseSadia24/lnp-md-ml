# LNP Nucleic Acid Delivery Project
## Reproducibility and Reproduction Guide

**Project type:** LNP formulation + molecular dynamics + machine learning  
**Primary scientific goal:** Test whether molecular-dynamics-derived descriptors add transferable information to machine-learning models of LNP transfection activity.  
**Current project status:** Exploratory MD integration completed for a small independent-source panel.  
**Documentation date:** 28 September 2026

---

# 1. Project overview

This project connects four layers:

1. **Published LNP formulation data**
2. **Molecular and formulation descriptors**
3. **Martini 3 coarse-grained molecular dynamics**
4. **Leakage-controlled machine learning**

The central idea is:

> Experimental descriptors describe what was formulated. MD descriptors describe properties of the assembled LNP-RNA system.

The project was designed to test whether this additional physical representation can improve prediction of LNP transfection activity when evaluation is performed on genuinely independent studies and MD sources.

The project is not yet a definitive biological validation of MD-enhanced prediction. The current MD analysis contains only six independent MD source vectors. The MD results are therefore exploratory.

---

# 2. Scientific objectives

## 2.1 Main objective

Build a reproducible pipeline that combines:

- LNP formulation variables
- lipid identities and molar ratios
- physicochemical properties
- cargo information
- RDKit molecular descriptors
- Morgan fingerprints
- Martini 3 MD descriptors
- binary transfection labels

and evaluate their ability to predict LNP transfection activity.

## 2.2 Specific objectives

1. Curate LNP formulation and transfection records.
2. Resolve inconsistent or ambiguous labels.
3. Generate chemical descriptors from lipid structures.
4. Build standardized LNP-RNA Martini 3 systems.
5. Run consistent GROMACS simulations.
6. Extract structural, RNA-association, dynamic, lipid-organization, and hydration descriptors.
7. Construct leakage-controlled ML datasets.
8. Compare random splits with DOI-grouped validation.
9. Evaluate MD features with leave-one-MD-source-out validation.
10. Perform MD-family ablation.
11. Perform exact source-level permutation testing.
12. Test seed stability.
13. Interpret the MD model using SHAP.
14. Preserve every input, intermediate file, model prediction, and simulation artifact required for future reproduction.

---

# 3. Current evidence and important conclusions

## 3.1 Full experimental benchmark

The frozen raw transfection table contained:

- **452 rows**
- **508 columns**
- **335 active**
- **117 inactive**
- **49 DOI groups**

After resolving the ID 803 record, the strong-label benchmark contained:

- **451 formulations**
- **335 active**
- **116 inactive**
- **48 DOI groups**

The inactive class represented approximately 25.7% of the strong-label dataset.

## 3.2 Feature inventory

The main structure/tabular feature inventory included:

- formulation variables
- physicochemical variables
- cargo descriptors
- RDKit 2D descriptors
- RDKit 3D descriptors
- Morgan fingerprints

The documented inventory contained:

| Feature block | Approximate number |
|---|---:|
| RDKit 2D | 42 |
| RDKit 3D | 13 |
| Morgan fingerprint | 414 |
| Total candidate structure/tabular variables | 501 |
| Default model variables after removing zero variance | 500 |

A zero-variance cargo feature was removed.

Structure coverage was incomplete:

- 378 rows had usable structures
- 74 rows did not

The field `pka_estimate` was excluded from the leakage-clean model because its provenance and availability were not consistent across rows.

---

# 4. Data sources

The project used information derived from:

- LNP Atlas
- LaNCE
- supporting literature
- curated formulation records
- lipid structures and SMILES
- MD simulations generated specifically for this project

Important project source papers included:

- **A Comprehensive Dataset of Lipid Nanoparticle Compositions and Properties for Nucleic Acid Delivery**
- **Review of machine learning for lipid nanoparticle formulation and process development**
- **High-throughput platforms for machine learning-guided lipid nanoparticle design**
- **Machine-Learning Framework to Predict the Performance of Lipid Nanoparticles for Nucleic Acid Delivery**
- **High-throughput, AI-assisted design and optimization of lipid nanoparticles for drug delivery**
- **Designing lipid nanoparticles using a transformer-based neural network**
- **Artificial intelligence-guided design of lipid nanoparticles for mRNA delivery**
- **LUMI-lab: A foundation model-driven autonomous platform enabling discovery of ionizable lipid designs for mRNA delivery**
- **Advancing RNA delivery with Ionizable lipid nanoparticles: the roles of microfluidics and machine learning**
- **Artificial Intelligence In The Rational Design Of Lipid Nanoparticles For mRNA Therapeutics**
- **Challenges and opportunities in computational studies for lipid nanoparticle development**
- **High-throughput barcoding of nanoparticles identifies cationic, degradable lipid-like materials for mRNA delivery**
- **Immunogenicity of lipid nanoparticles and its impact on the efficacy of mRNA vaccines and therapeutics**
- **Development of Lipid Nanoparticle Formulation for the Repeated Administration of mRNA Therapeutics**

These papers are background and methodological references. The actual numerical project results should always be reproduced from the frozen project files, not reconstructed from literature values.

---

# 5. Important file structure

The intended project organization is:

```text
hpc_lnp_ml/
│
├── data/
│   ├── raw/
│   │   ├── LNP_transfection_strong_labels_451.csv
│   │   └── original_source_tables/
│   │
│   └── processed/
│       └── transfection_modelready_v1.csv
│
├── md/
│   ├── source_tables/
│   ├── topologies/
│   ├── systems/
│   ├── trajectories/
│   └── descriptors/
│
├── scripts/
│   ├── preprocessing/
│   ├── feature_generation/
│   ├── modeling/
│   ├── md/
│   └── analysis/
│
├── results/
│   ├── main_results.csv
│   ├── predictions/
│   └── results_v2/
│
├── figures/
│   ├── model_benchmark.png
│   ├── split_scheme_gap.png
│   └── results_v2/
│
├── reports/
│   └── qc_report.json
│
├── md_unique_source_vectors.csv
├── md_descriptor_quality_audit.csv
├── md_source_loso_by_source.csv
├── md_source_loso_paired_deltas.csv
├── md_family_ablation_summary.csv
├── md_seed_stability_summary.csv
├── md_shap_feature_importance.csv
└── md_shap_family_importance.csv
```

The actual repository may contain additional files. Do not delete them simply because they are not listed here.

---

# 6. HPC environment

The working project location was:

```bash
/home/sa594/LNP_ML/hpc_lnp_ml
```

The environment activation script was:

```bash
/home/sa594/LNP_ML/transfection_binary/environment/activate_lnp_ml.sh
```

Activate the environment before running the ML pipeline:

```bash
source /home/sa594/LNP_ML/transfection_binary/environment/activate_lnp_ml.sh
```

Then move to the project:

```bash
cd /home/sa594/LNP_ML/hpc_lnp_ml
```

Before reproduction, record versions:

```bash
python --version
python -c "import sklearn; print('scikit-learn', sklearn.__version__)"
python -c "import rdkit; print('RDKit', rdkit.__version__)"
python -c "import xgboost; print('XGBoost', xgboost.__version__)"
python -c "import lightgbm; print('LightGBM', lightgbm.__version__)"
python -c "import shap; print('SHAP', shap.__version__)"
gmx --version
```

Also record:

- Martini 3 version
- martinize2 version
- Packmol version
- Python version
- operating system
- compiler/toolchain if relevant
- HPC module versions

---

# 7. Data integrity and frozen inputs

Before any preprocessing:

1. Copy the raw input into `data/raw/`.
2. Never edit the frozen raw file.
3. Calculate a SHA256 hash.
4. Save the hash in a manifest.
5. Record row count and column count.
6. Record label counts.
7. Record DOI count.
8. Record the exact source filename.

Example:

```bash
sha256sum data/raw/LNP_transfection_strong_labels_451.csv
```

For the earlier raw binary-target file, the preprocessing history included a hash change after preprocessing. The exact current hash should be recalculated from the file being used for reproduction rather than copied from an old notebook.

---

# 8. Dataset cleaning

## 8.1 Primary label

The main target was:

```text
target_transfection_active_binary_manual
```

Encoding:

```text
active   = 1
inactive = 0
```

## 8.2 ID 803

ID 803 was identified as problematic and was not accepted into the strong-label benchmark.

The strong benchmark therefore uses:

```text
451 rows
```

instead of the original:

```text
452 rows
```

## 8.3 Excluded predictor types

The leakage-clean model should not use:

- paper identifiers
- paper titles
- author names
- journal names
- raw nucleic-acid sequences
- target-derived fields
- label provenance fields
- outcome-derived fields
- inconsistent `pka_estimate`

Do not allow any feature that directly or indirectly reveals the label.

---

# 9. Chemistry feature generation

## 9.1 RDKit

Generate molecular descriptors from lipid structures.

The project used:

- RDKit 2D descriptors
- RDKit 3D descriptors
- Morgan fingerprints

The structure workflow should preserve:

```text
LNP ID
ionizable lipid ID
canonical SMILES
structure availability
descriptor generation status
```

## 9.2 2D descriptors

The documented 2D block contained 42 descriptors.

Examples include:

- molecular weight
- logP
- polar surface area
- hydrogen-bond counts
- topology-related descriptors
- charge-related descriptors

## 9.3 3D descriptors

The documented 3D block contained 13 conformer-derived descriptors.

These capture:

- molecular geometry
- spatial distribution
- shape
- size
- conformational properties

## 9.4 Morgan fingerprints

The project used a 414-bit Morgan fingerprint block.

These are binary structural features.

---

# 10. Cargo descriptors

Cargo information was incorporated where available.

The project considered:

- nucleic-acid composition
- GC/A/U/G/C information
- approximate molecular weight
- cargo identity
- luciferase/FVII/NanoLuc-related assay information where available

A zero-variance cargo feature:

```text
cargo_cargo_is_pdna
```

was removed from the default modeling feature set.

Do not assume that all cargo descriptors are available for every row. Missingness must be handled inside the training fold.

---

# 11. Machine-learning preprocessing

All preprocessing must occur **inside the training portion of each fold**.

This is critical.

## Numeric variables

Use:

```text
median imputation
```

## Categorical variables

Use:

```text
constant missing category
+
one-hot encoding
```

## Logistic regression

Use scaled numeric features.

## Tree models

Use the fold-local encoded matrix without global scaling.

## Class imbalance

Use class weighting.

Do not use synthetic oversampling across group boundaries.

---

# 12. ML models

The full benchmark included:

1. Majority-class baseline
2. Regularized logistic regression
3. Random forest
4. XGBoost
5. LightGBM

The strict MD-linked comparison used:

```text
ExtraTrees
```

with the confirmed configuration:

```text
n_estimators = 800
min_samples_leaf = 2
max_features = "sqrt"
class_weight = "balanced"
n_jobs = all available CPU workers
```

The prediction threshold was:

```text
probability >= 0.5 -> active
probability < 0.5  -> inactive
```

The main data-preparation seed was:

```text
42
```

---

# 13. Evaluation strategy

The project deliberately compared several splitting strategies.

## 13.1 Stratified random split

Purpose:

- conventional benchmark
- diagnostic only

Problem:

Related papers and related lipid chemistries can appear in both training and test sets.

Therefore random-split performance should not be treated as the primary estimate of generalization.

## 13.2 DOI-grouped 5-fold cross-validation

Complete papers are kept together.

Purpose:

- test generalization to unseen studies
- reduce study-specific leakage

This is the primary benchmark for study-level generalization.

## 13.3 Joint DOI and lipid-component split

This was used as a stricter leakage audit.

The goal was to prevent both:

- DOI overlap
- lipid-component overlap

The connected-component split produced highly unequal fold sizes.

A balanced component assignment was also tested as a sensitivity analysis.

## 13.4 Leave-one-MD-source-out

For MD integration:

```text
hold out one independent MD source
train on all other sources
predict the held-out source
repeat for all six sources
```

This is the primary MD-linked evaluation.

---

# 14. Primary metrics

Because the strong dataset is imbalanced, headline metrics were:

- Macro F1
- Matthews correlation coefficient (MCC)
- Inactive-class precision-recall AUC
- ROC AUC

Accuracy was not the main metric.

Cluster-aware bootstrap intervals were calculated for selected full-dataset metrics.

---

# 15. Baseline results that must be preserved

The random-split diagnostic showed substantially better apparent performance than grouped validation.

Documented examples:

```text
Random-split logistic regression:
Macro F1 = 0.695
MCC      = 0.403
```

A documented LightGBM random-split result included:

```text
ROC AUC = 0.780
Inactive PR AUC = 0.583
```

DOI-grouped validation reduced the best documented performance substantially:

```text
Best documented grouped Macro F1 ≈ 0.522
Best documented grouped MCC     ≈ 0.053
```

The exact current values should always be read from:

```text
results/main_results.csv
```

Do not manually recreate them from rounded values in reports.

---

# 16. Molecular dynamics methodology

## 16.1 Force field

The project used:

```text
Martini 3
```

with GROMACS.

The MD simulations were coarse-grained LNP-RNA simulations.

## 16.2 Main simulation software

- GROMACS
- Martini 3
- martinize2
- Packmol
- VMD
- OVITO
- Python analysis scripts

Packmol was used to construct starting coordinates.

Packmol did not calculate forces or perform dynamics.

GROMACS performed:

- preprocessing
- energy minimization
- equilibration
- production dynamics
- trajectory processing
- standard analyses

---

# 17. LNP components

Depending on the formulation, the simulations included:

### Ionizable lipids

Examples:

- MC3-related topologies
- S1D for SM-102 systems
- OF-02 where required

### Helper lipids

Examples:

- DSPC
- DPPC
- DOPE

### Cholesterol

```text
CHOL
```

### PEG lipid

```text
DMG-PEG2000
```

when present.

### RNA

RNA was converted into a Martini-compatible coarse-grained representation using `martinize2` and the corresponding RNA topology files.

---

# 18. Standard LNP-RNA MD protocol

## 18.1 General sequence

```text
Build system
    ↓
Check molecule counts
    ↓
Check charge
    ↓
Solvate
    ↓
Add counterions
    ↓
Energy minimization
    ↓
10 ns NVT
    ↓
20 ns NPT
    ↓
200 ns production
    ↓
Analyze 100–200 ns
    ↓
Generate 377-column descriptor table
```

## 18.2 Temperature

```text
310 K
```

## 18.3 Equilibration

```text
NVT = 10 ns
NPT = 20 ns
```

## 18.4 Production

```text
200 ns
```

## 18.5 Analysis window

Use:

```text
100–200 ns
```

for the standard cross-formulation descriptor calculation.

## 18.6 Time step

Standard:

```text
5 fs
```

Fallback:

```text
2 fs
```

A 20 fs FAST configuration was used only when a system passed the relevant stability checks.

---

# 19. RNA model

The standardized mRNA model used:

```text
189 A
168 U
```

Total:

```text
357 RNA beads
```

The RNA was represented as a Martini-compatible coarse-grained molecule.

For the standardized neutralized system, the documented system used:

```text
42 Na+
0 added Cl-
```

when this configuration was required to neutralize the RNA/system charge.

Important:

Do not add salt simply because another MD protocol used salt. Reproduce the exact frozen system configuration for each formulation.

---

# 20. System construction checklist

For every MD system:

```text
[ ] Confirm LNP ID
[ ] Confirm source formulation
[ ] Confirm lipid identities
[ ] Confirm molar ratios
[ ] Confirm RNA type
[ ] Confirm RNA sequence/model
[ ] Confirm lipid topology
[ ] Confirm RNA topology
[ ] Confirm PEG topology
[ ] Confirm cholesterol topology
[ ] Confirm helper lipid topology
[ ] Check total molecular counts
[ ] Check total charge
[ ] Add water
[ ] Add neutralizing ions
[ ] Check initial geometry
[ ] Run energy minimization
[ ] Inspect energy minimization
[ ] Run NVT
[ ] Inspect temperature
[ ] Run NPT
[ ] Inspect density and box
[ ] Run production
[ ] Inspect trajectory
[ ] Check periodic boundary behavior
[ ] Check LINCS/constraint warnings
[ ] Check energy
[ ] Check density
[ ] Check box dimensions
[ ] Analyze 100–200 ns
[ ] Generate descriptor CSV
[ ] Validate descriptor CSV
```

---

# 21. Completed MD systems

The project history includes the following systems:

| LNP | Status | Note |
|---|---|---|
| 193 | Completed | Independent MD source |
| 194 | Completed | Independent MD source |
| 195 | Completed | Inactive independent source |
| 509 | Completed after rebuild | Periodic-boundary issue required rebuild |
| 781 | Completed | Lung-targeting formulation |
| 782 | Completed | Independent MD source |
| 854 | Completed after stability work | LINCS instability required corrective rerun |
| 858 | Completed | MC3-based reference/FAST production system |
| 911 | Started / linked through matching source profile | Used in later mapping context |
| 803 | Not accepted | Energy minimization showed Lennard-Jones overflow |
| 825 | Excluded | ASO cargo did not match mRNA-focused scope |

The strict MD-linked statistical analysis used six independent MD source vectors:

```text
193
194
195
782
854
858
```

---

# 22. Composition-matched MD reuse

A critical reproducibility rule is:

> If multiple experimental rows have the same LNP composition represented by one MD source, the same MD descriptor vector can be assigned to those rows.

However:

- preserve the original experimental row ID
- preserve the original DOI
- preserve the MD source ID
- do not pretend the repeated rows are independent MD simulations

The MD source identifier must remain available after mapping.

This prevents accidental pseudoreplication.

---

# 23. MD source mapping

The documented strict source mapping was:

| MD source | Mapped rows | Active | Inactive | LNP IDs |
|---|---:|---:|---:|---|
| LNP193 | 1 | 1 | 0 | 193 |
| LNP194 | 1 | 1 | 0 | 194 |
| LNP195 | 1 | 0 | 1 | 195 |
| LNP782 | 2 | 2 | 0 | 782, 875 |
| LNP854 | 2 | 1 | 1 | 783, 911 |
| LNP858 | 6 | 1 | 5 | 636, 687, 781, 922, 923, 931 |

Total:

```text
13 mapped experimental rows
6 independent MD sources
```

This distinction is extremely important when interpreting statistical power.

---

# 24. MD descriptor schema

Each validated simulation produced a fixed:

```text
377-column
```

paper-ready descriptor file.

The schema contained:

```text
13 metadata fields
364 quantitative descriptor fields
```

The descriptor families included:

- global structure
- RNA structure and position
- RNA-lipid association
- lipid organization
- dynamics
- hydration
- stability summaries

---

# 25. MD descriptor categories

## 25.1 Global structure

Examples:

- LNP radius of gyration
- core radius of gyration
- SASA
- shape anisotropy
- hydrodynamic/size-related quantities

## 25.2 RNA structure and position

Examples:

- RNA radius of gyration
- combined LNP-RNA radius of gyration
- RNA center distance
- RNA minimum distance

## 25.3 RNA-lipid association

Examples:

- RNA-LNP contacts within 0.6 nm
- RNA buried fraction
- RNA-LNP center distance
- RNA-LNP minimum distance

## 25.4 Lipid organization

Examples:

- radial distributions
- density profiles
- shell thickness
- component distributions
- core/surface partitioning

## 25.5 Dynamics

Examples:

- mean squared displacement
- drift
- molecular fluctuations
- mobility summaries

## 25.6 Hydration

Examples:

- RNA-water contacts
- LNP-water contacts
- core water fraction
- ion-water contacts

## 25.7 Stability

Examples:

- first-to-last change
- standard deviation
- late-window mean
- trajectory drift

---

# 26. Strict 12-feature MD model

Because only six independent MD sources were available, the full 364-dimensional MD block was not used for the strict confirmatory-style analysis.

A reduced set of 12 descriptors was used.

### Size and shape

1. LNP radius of gyration
2. Core radius of gyration
3. LNP solvent-accessible area
4. RNA radius of gyration
5. Combined LNP-RNA radius of gyration
6. Shape anisotropy
7. Compactness / inverse radius

### RNA association

8. RNA-LNP center distance
9. RNA-LNP minimum distance
10. RNA-LNP contacts within 0.6 nm
11. RNA buried fraction

### Water penetration

12. Core water fraction

These features had:

- six independent-source values
- six unique values
- zero missing values
- zero infinite values
- no constant feature

Units include:

- nm for distances/radii
- nm² for SASA
- counts for contacts
- fractions for buried fraction and water fraction

---

# 27. MD descriptor quality control

Every descriptor file must be checked for:

```text
exact column count = 377
duplicate column names = 0
empty cells = 0
NA-like values = 0
infinite values = 0
```

Also verify:

- metadata columns
- source ID
- analysis window
- units
- residue selection
- trajectory length
- topology consistency

A descriptor audit should record:

```text
minimum
maximum
mean
standard deviation
missingness
infinity count
constant status
```

---

# 28. MD-linked machine learning

The strict MD model used:

```text
ExtraTrees
```

and the same baseline experimental feature representation was used for:

```text
baseline
baseline + MD
```

The only intended difference between the two models is the addition of the selected MD descriptors.

This makes the paired comparison interpretable.

---

# 29. Row-weighted vs source-balanced evaluation

Two weighting schemes were used.

## Row weighted

Each mapped experimental row receives equal weight.

Problem:

A single MD source can represent multiple experimental rows.

## Source balanced

Each row receives:

```text
1 / number_of_rows_from_that_MD_source
```

This makes each independent MD source contribute approximately equally.

For the small MD dataset, source-balanced analysis is especially important.

---

# 30. MD results

The strongest documented MD signal came from the:

```text
size and shape
```

family.

Adding this family produced a documented source-balanced MCC increase of:

```text
+0.442
```

The exact source-level permutation p-value was:

```text
0.0667
```

Holm-adjusted:

```text
0.175
```

Therefore this result should be described as exploratory.

No MD family passed the 0.05 significance threshold in the documented analysis.

---

# 31. SHAP results

SHAP analysis used `TreeExplainer` on each held-out ExtraTrees model.

Additivity was checked for each holdout.

The five most influential MD descriptors in the documented analysis were:

1. Core radius of gyration
2. Combined LNP-RNA radius of gyration
3. LNP radius of gyration
4. Compactness
5. Core water fraction

The family-level mean absolute SHAP values were approximately:

```text
Size / shape       0.08308
RNA association    0.01831
Water penetration  0.01113
```

The individual ranks are exploratory because:

- descriptors are correlated
- only six independent sources were available
- the source-to-source variation was large

Do not present the SHAP ranking as a causal mechanism.

---

# 32. Exact permutation testing

The strict MD-source permutation analysis used:

```text
720 source-label assignments
```

because six sources produce:

```text
6! = 720
```

possible source-level label assignments.

The purpose is to test whether the observed source-level improvement could arise from arbitrary source-label assignment.

This is preferable to treating 13 mapped rows as 13 independent MD observations.

---

# 33. Seed stability

The paired MD analysis was repeated across:

```text
200 random seeds
```

The output should be preserved in:

```text
md_seed_stability_summary.csv
```

Always retain:

- seed
- fold assignments
- model result
- baseline metric
- baseline + MD metric
- delta metric

---

# 34. Recommended analysis sequence

The complete ML analysis should be run in this order:

```text
1. Freeze raw dataset
2. Verify hash
3. Clean labels
4. Remove ID 803
5. Build strong-label table
6. Audit structures
7. Generate RDKit features
8. Generate Morgan fingerprints
9. Generate cargo/formulation features
10. Build leakage-clean model matrix
11. Run random-split benchmark
12. Run DOI-grouped benchmark
13. Run joint DOI/lipid leakage audit
14. Freeze MD mapping
15. Validate MD descriptor files
16. Create unique MD source vectors
17. Run baseline MD-source LOSO
18. Run baseline + MD LOSO
19. Calculate paired deltas
20. Run MD-family ablation
21. Run exact source permutation
22. Run 200-seed stability
23. Run SHAP
24. Generate figures
25. Archive all predictions and metadata
```

---

# 35. Important script history

The project used advanced analysis scripts including:

```text
04_train_advanced
05_md_exploratory
06_md_exact_permutation
```

Earlier stages included preprocessing, baseline training, feature construction, and MD descriptor preparation.

When reproducing:

```bash
find . -maxdepth 3 -type f | sort
```

Use this to confirm the actual script names in the repository before execution.

Do not rename scripts during reproduction unless the run manifest is updated.

---

# 36. Recommended command sequence

Start from the repository:

```bash
cd /home/sa594/LNP_ML/hpc_lnp_ml
```

Activate the environment:

```bash
source /home/sa594/LNP_ML/transfection_binary/environment/activate_lnp_ml.sh
```

Inspect repository:

```bash
find . -maxdepth 3 -type f | sort
```

Check Git status:

```bash
git status
```

Record commit:

```bash
git rev-parse HEAD
```

Record Python:

```bash
python --version
```

Record installed packages:

```bash
pip freeze > reproduction_environment_pip_freeze.txt
```

Then execute the pipeline in dependency order.

---

# 37. What must never be changed silently

For a true reproduction, do not silently change:

- input dataset
- ID 803 handling
- label definition
- DOI grouping
- lipid grouping
- train/test folds
- random seeds
- imputation rules
- scaling rules
- class weights
- MD source mapping
- MD source IDs
- descriptor names
- analysis window
- MD simulation temperature
- MD trajectory duration
- force field
- RNA model
- topology
- box construction
- water model
- ionization assumptions

If any item changes, create a new experiment ID.

---

# 38. Reproduction experiment IDs

Use a naming system such as:

```text
EXP_001_raw_freeze
EXP_002_feature_generation
EXP_003_random_baseline
EXP_004_doi_grouped_baseline
EXP_005_joint_leakage_audit
EXP_006_md_source_loso
EXP_007_md_ablation
EXP_008_md_permutation
EXP_009_md_seed_stability
EXP_010_md_shap
```

For each experiment save:

```text
experiment_id
date
git_commit
input_hash
environment
script
arguments
random_seed
output_files
notes
```

---

# 39. MD archive requirements

For every simulation, archive:

```text
topology files
coordinate files
GROMACS .mdp files
.tpr
.xtc
.edr
.log
.gro
checkpoint files
index files
Packmol input
Packmol output
RNA topology
lipid topology
force-field files
descriptor extraction scripts
descriptor CSV
QC report
```

Do not keep only the final CSV.

The trajectory and topology are required to reproduce or audit the descriptor.

---

# 40. GROMACS QC

For each simulation inspect:

## Energy

Check for:

- exploding energy
- abnormal potential energy
- unexpected drift

## Temperature

Check:

```text
310 K
```

within the expected thermostat behavior.

## Pressure

Inspect the NPT pressure behavior.

## Density

Check stabilization after NPT.

## Box dimensions

Check:

- expected dimensions
- sudden jumps
- abnormal shrinkage
- periodic-boundary artifacts

## Constraints

Look for:

```text
LINCS WARNING
```

or repeated constraint failures.

## Trajectory

Inspect with:

- VMD
- OVITO

Check for:

- broken molecules
- extreme overlaps
- unphysical deformation
- RNA leaving the intended system
- periodic-boundary artifacts

---

# 41. Known MD troubleshooting history

## LNP509

A periodic-boundary shift required rebuilding the system in a larger:

```text
40 nm
```

box.

## LNP854

A LINCS instability occurred around:

```text
24.6335 ns
```

and required corrective simulation settings.

## LNP803

Energy minimization showed:

```text
Lennard-Jones overflow
```

and the system was not accepted as a valid MD source.

These failures must remain documented. Do not delete failed runs from the archive.

---

# 42. Figures to reproduce

Important project figures include:

```text
figures/model_benchmark.png
figures/split_scheme_gap.png
```

and the newer MD analysis figures:

```text
figures/results_v2/
```

Important figure types:

1. Model benchmark
2. Random vs grouped performance
3. Split-scheme gap
4. MD source mapping
5. Baseline vs MD paired performance
6. MD descriptor family ablation
7. Exact permutation distribution
8. Seed stability
9. SHAP feature importance
10. SHAP family importance
11. MD descriptor correlation matrix
12. Formulation-level MD profiles

All predictive figures should use out-of-fold predictions.

Never create a performance figure from training predictions.

---

# 43. Output files that should be preserved

Core files:

```text
data/raw/LNP_transfection_strong_labels_451.csv
data/processed/transfection_modelready_v1.csv
reports/qc_report.json
results/main_results.csv
results/predictions/*_grouped.csv
```

MD files:

```text
md_unique_source_vectors.csv
md_descriptor_quality_audit.csv
md_source_loso_by_source.csv
md_source_loso_paired_deltas.csv
md_family_ablation_summary.csv
md_seed_stability_summary.csv
md_shap_feature_importance.csv
md_shap_family_importance.csv
```

These files are the minimum numerical record needed to reconstruct the current conclusions.

---

# 44. Data lineage

The complete lineage should be:

```text
Literature
   |
   v
LNP Atlas / LaNCE / supporting records
   |
   v
Raw formulation table
   |
   v
Label cleaning + ID 803 resolution
   |
   v
451-row strong-label benchmark
   |
   +-----------------------+
   |                       |
   v                       v
Chemical features          MD source selection
   |                       |
   v                       v
RDKit + Morgan             Martini 3 + GROMACS
   |                       |
   |                       v
   |                  377-column MD descriptors
   |                       |
   +-----------+-----------+
               |
               v
       Leakage-clean ML table
               |
               v
       Grouped cross-validation
               |
               v
       Baseline vs baseline+MD
               |
      +--------+---------+
      |        |         |
      v        v         v
   Ablation  Permutation SHAP
      |
      v
Mechanistic hypotheses
```

---

# 45. Why random splitting is not sufficient

LNP literature data contain repeated:

- lipid chemistry
- formulation families
- papers
- experimental procedures

A random split can place related formulations in both training and test sets.

This can produce apparently strong performance without demonstrating generalization to a new study.

Therefore:

```text
random split = diagnostic
DOI-grouped split = primary study-level benchmark
MD-source LOSO = primary MD-linked benchmark
```

---

# 46. Current scientific interpretation

The current project supports the following cautious interpretation:

1. The data pipeline is reproducible and substantially audited.
2. Random-split performance is much more optimistic than grouped validation.
3. The literature contains strong study and chemistry dependence.
4. MD descriptors provide an additional physical representation of the LNP-RNA system.
5. Size and shape descriptors showed the clearest exploratory signal in the six-source MD analysis.
6. RNA association and water penetration also provide potentially useful information.
7. The present number of independent MD systems is too small for a definitive claim that MD improves biological prediction across the broader formulation space.
8. More independent MD systems are needed before expanding to very large MD feature sets or complex neural fusion.

---

# 47. What should be done next

## Priority 1: Increase independent MD sources

Target:

```text
20–30 independent systems
```

then:

```text
40–60 systems
```

The new systems should increase diversity in:

- ionizable lipid scaffold
- helper lipid
- PEG level
- formulation ratio
- cargo
- transfection label
- DOI
- chemical feature space

## Priority 2: Add inactive formulations

The current strict source panel contains only:

- one purely inactive source
- two mixed-label sources

More inactive and mixed-response systems are needed.

## Priority 3: Replicate selected MD systems

For a smaller anchor panel:

```text
at least 3 independent trajectory seeds
```

should be run.

This estimates descriptor uncertainty.

## Priority 4: Freeze the MD protocol

Before scaling up, freeze:

- protonation assumptions
- box size
- output frequency
- force field
- RNA representation
- analysis window
- water model
- ion treatment
- trajectory length

## Priority 5: Improve ML validation

Use:

```text
nested grouped cross-validation
```

for model selection.

All preprocessing must remain inside training folds.

## Priority 6: Reduce correlated MD features

Before confirmatory modeling:

- calculate feature correlations
- remove redundant features
- retain representative features
- compare family-level summaries

## Priority 7: Add assay context

Future models should explicitly represent:

- cell line
- species
- route
- dose
- tissue
- assay type
- time point

A single binary label should not be expected to absorb all experimental differences.

---

# 48. Future targets

The project was initially designed to expand beyond binary transfection activity.

Potential future targets:

- organ targeting
- expression level
- toxicity
- biodistribution
- tissue-specific transfection
- intracellular trafficking
- stability

These should be added only after sufficient target-specific data are available.

---

# 49. Future multimodal model

A later model can combine:

```text
Lipid structure
       +
Formulation
       +
Process
       +
Cargo
       +
MD
       +
Assay context
       |
       v
Multimodal ML
```

Possible future architectures:

- gradient boosting on engineered features
- graph neural networks
- transformer-based formulation models
- multimodal fusion networks
- uncertainty-aware models

Complex neural fusion should be delayed until the number of independent MD sources is substantially larger.

---

# 50. Reproducibility checklist

## Dataset

- [ ] Raw dataset frozen
- [ ] SHA256 recorded
- [ ] 452-row original table archived
- [ ] ID 803 resolution documented
- [ ] 451-row strong dataset generated
- [ ] 335 active / 116 inactive verified
- [ ] 48 DOI groups verified
- [ ] Structure coverage audited
- [ ] Outcome-derived fields removed

## Chemistry

- [ ] RDKit version recorded
- [ ] SMILES preserved
- [ ] 2D descriptors generated
- [ ] 3D descriptors generated
- [ ] Morgan fingerprints generated
- [ ] Invalid structures logged

## ML

- [ ] Environment frozen
- [ ] Random seed 42 recorded
- [ ] Fold assignments saved
- [ ] Preprocessing inside folds
- [ ] Random split run
- [ ] DOI-grouped CV run
- [ ] Joint DOI/lipid leakage audit run
- [ ] Model hyperparameters saved
- [ ] OOF predictions saved
- [ ] Metrics saved

## MD

- [ ] Martini 3 version recorded
- [ ] GROMACS version recorded
- [ ] Topologies archived
- [ ] RNA model archived
- [ ] Packmol input archived
- [ ] Initial coordinates archived
- [ ] MDP files archived
- [ ] TPR archived
- [ ] XTC archived
- [ ] EDR archived
- [ ] LOG archived
- [ ] 310 K verified
- [ ] 10 ns NVT verified
- [ ] 20 ns NPT verified
- [ ] 200 ns production verified
- [ ] 100–200 ns analysis verified
- [ ] Descriptor count = 377
- [ ] Descriptor QC passed

## MD-ML

- [ ] MD source mapping frozen
- [ ] Unique source vectors generated
- [ ] Six-source LOSO run
- [ ] Baseline and baseline+MD paired
- [ ] Source-balanced weighting run
- [ ] Family ablation run
- [ ] Exact 720-assignment permutation run
- [ ] 200-seed stability run
- [ ] SHAP run
- [ ] SHAP additivity verified

## Archive

- [ ] Git commit recorded
- [ ] Environment exported
- [ ] Raw hashes recorded
- [ ] Model predictions archived
- [ ] Figures archived
- [ ] Simulation files archived
- [ ] README/Markdown updated
- [ ] Final experiment manifest created

---

# 51. Minimal reproduction

If the complete simulations already exist, the fastest reproduction is:

```bash
cd /home/sa594/LNP_ML/hpc_lnp_ml

source /home/sa594/LNP_ML/transfection_binary/environment/activate_lnp_ml.sh

git rev-parse HEAD

sha256sum data/raw/LNP_transfection_strong_labels_451.csv

python --version

find . -maxdepth 3 -type f | sort
```

Then:

```text
1. Run preprocessing
2. Run feature generation
3. Run baseline ML
4. Run DOI-grouped ML
5. Validate MD descriptor files
6. Build MD source mapping
7. Run MD-source LOSO
8. Run ablation
9. Run exact permutation
10. Run seed stability
11. Run SHAP
12. Recreate figures
13. Compare all outputs with archived results
```

If the original MD trajectories are unavailable, the ML/descriptor analysis can be reproduced only if the validated 377-column descriptor files are available. The MD simulation itself cannot be considered fully reproduced without the topology, coordinates, force-field files, MDP files, and trajectory.

---

# 52. What counts as a successful reproduction

A reproduction is successful only when:

1. The same frozen dataset is used.
2. The same label definition is used.
3. The same ID 803 treatment is used.
4. The same feature exclusions are used.
5. The same grouped folds are used.
6. The same MD source mapping is used.
7. The same 12 MD descriptors are used for the strict analysis.
8. The same model configuration is used.
9. The same seeds are used.
10. The same OOF predictions are generated within numerical tolerance.
11. The same aggregate metrics are recovered within numerical tolerance.
12. The same qualitative MD-source patterns are recovered.

---

# 53. Project status at documentation freeze

### Completed

- [x] Literature/data curation
- [x] Binary transfection dataset
- [x] Strong-label 451-row benchmark
- [x] Chemistry features
- [x] RDKit descriptors
- [x] Morgan fingerprints
- [x] Leakage audits
- [x] Random ML benchmark
- [x] DOI-grouped benchmark
- [x] Martini 3 LNP-RNA simulations for selected systems
- [x] 377-column MD descriptor schema
- [x] Six-source MD-linked analysis
- [x] MD-family ablation
- [x] Exact source permutation
- [x] 200-seed stability analysis
- [x] SHAP analysis
- [x] Reproducibility inventory

### Not yet complete

- [ ] 20–30 independent MD source panel
- [ ] 40–60 source panel
- [ ] More inactive MD systems
- [ ] Three-seed MD replication panel
- [ ] Prospective experimental validation
- [ ] Final organ-targeting model
- [ ] Final toxicity model
- [ ] Large multimodal neural model

---

# 54. Final reproducibility principle

The most important rule for continuing this project is:

> Never improve the apparent result by weakening the validation scheme.

The main scientific value of the project is not a high random-split score. The important workflow is the combination of:

```text
careful data curation
+
leakage control
+
standardized MD
+
independent-source validation
+
mechanistic descriptor analysis
```

Future improvements should increase the number and diversity of independent observations rather than relying on increasingly complex models applied to the same small source set.

---

# 55. Key project references

## Dataset and LNP background

Song S, Baek J, Seo S. *A Comprehensive Dataset of Lipid Nanoparticle Compositions and Properties for Nucleic Acid Delivery*. Scientific Data, 2026.

## ML framework

Kumar G, Ardekani AM. *Machine-Learning Framework to Predict the Performance of Lipid Nanoparticles for Nucleic Acid Delivery*. ACS Applied Bio Materials, 2025.

## ML review

Dorsey PJ, et al. *Review of machine learning for lipid nanoparticle formulation and process development*. Journal of Pharmaceutical Sciences, 2024.

## High-throughput LNP design

Hanna AR, Issadore DA, Mitchell MJ. *High-throughput platforms for machine learning-guided lipid nanoparticle design*. Nature Reviews Materials, 2025.

## Transformer-based LNP modeling

Chan A, et al. *Designing lipid nanoparticles using a transformer-based neural network*. Nature Nanotechnology, 2025.

## AI-assisted LNP design

Su K, et al. *Artificial intelligence-guided design of lipid nanoparticles for mRNA delivery*. Acta Pharmaceutica Sinica B, 2025.

## Computational LNP methods

Oh Y, et al. *Challenges and opportunities in computational studies for lipid nanoparticle development*. npj Drug Discovery, 2025.

## High-throughput AI review

Zeng J, et al. *High-throughput, AI-assisted design and optimization of lipid nanoparticles for drug delivery*.

## High-throughput barcoding

Xue L, et al. *High-throughput barcoding of nanoparticles identifies cationic, degradable lipid-like materials for mRNA delivery to the lungs in female preclinical models*. Nature Communications, 2024.

## LNP immunogenicity

Lee Y, et al. *Immunogenicity of lipid nanoparticles and its impact on the efficacy of mRNA vaccines and therapeutics*. Experimental & Molecular Medicine, 2023.

---

# 56. Source documentation used to build this guide

This reproduction guide was assembled from the project's stored documentation and uploaded literature, especially:

- `LNP_Nucleic_Acid_Project_Complete_Progress_Report.pdf`
- `LNP_Nucleic_Acid_Project_Complete_Progress_Report.docx`
- `LNP_Nucleic_Acid_Project_Comprehensive_Report.pdf`
- `LNP_MD_ML_Result_Execution_Guide(1).pdf`
- LNP dataset and ML review papers
- LNP AI and high-throughput design papers
- the project's stored HPC/ML workflow history

The numerical values in this guide should be checked against the frozen CSV result files when preparing a publication.

---

# 57. One-page reproduction map

```text
RAW DATA
  |
  |-- freeze + hash
  |
  v
LABEL CLEANING
  |
  |-- remove/resolve ID 803
  |
  v
451 STRONG-LABEL DATASET
  |
  +-------------------------------+
  |                               |
  v                               v
CHEMISTRY FEATURES                MD
  |                               |
  |-- RDKit 2D                    |-- Martini 3
  |-- RDKit 3D                    |-- GROMACS
  |-- Morgan                      |-- 310 K
  |-- cargo                       |-- 10 ns NVT
  |-- formulation                 |-- 20 ns NPT
  |                               |-- 200 ns production
  |                               |-- 100–200 ns analysis
  |                               |
  |                               v
  |                         377-COLUMN MD FILES
  |                               |
  +---------------+---------------+
                  |
                  v
        LEAKAGE-CLEAN ML TABLE
                  |
        +---------+---------+
        |         |         |
        v         v         v
     RANDOM     DOI-CV    JOINT AUDIT
        |
        v
  BASELINE MODELS
        |
        +-------------------------+
        |                         |
        v                         v
  MD SOURCE MAPPING        SIX-SOURCE LOSO
                                  |
                           +------+------+
                           |      |      |
                           v      v      v
                       ABLATION PERM.  SHAP
                           |
                           v
                 MECHANISTIC HYPOTHESES
                           |
                           v
                 MORE INDEPENDENT MD
                           |
                           v
                PROSPECTIVE VALIDATION
```

---

# 58. Reproduction log template

Copy this section into a new file for every future reproduction:

```text
Experiment ID:
Date:
Operator:
Git commit:
Branch:
Raw data filename:
Raw data SHA256:
Processed data filename:
Python version:
RDKit version:
scikit-learn version:
XGBoost version:
LightGBM version:
SHAP version:
GROMACS version:
Martini version:
martinize2 version:
Packmol version:

Dataset rows:
Dataset columns:
Active:
Inactive:
DOI groups:

MD sources:
MD mapping file:
MD descriptor schema:
MD analysis window:

Random seed:
Model:
Hyperparameters:

Output directory:

Results:
Random split:
DOI grouped:
MD baseline:
MD + baseline:
Ablation:
Permutation:
Seed stability:
SHAP:

Notes:
```

---

# 59. Final file inventory

At minimum, keep these files together when moving the project to a new computer or HPC system:

```text
README_REPRODUCTION.md

data/raw/LNP_transfection_strong_labels_451.csv
data/processed/transfection_modelready_v1.csv

reports/qc_report.json

results/main_results.csv
results/predictions/

md_unique_source_vectors.csv
md_descriptor_quality_audit.csv
md_source_loso_by_source.csv
md_source_loso_paired_deltas.csv
md_family_ablation_summary.csv
md_seed_stability_summary.csv
md_shap_feature_importance.csv
md_shap_family_importance.csv

figures/
scripts/

MD topologies
MD coordinate files
MD MDP files
MD TPR files
MD XTC files
MD EDR files
MD logs
RNA topology files
Martini force-field files

environment lock/export
Git repository
```

This is the minimum archive needed to make the project reproducible rather than merely reproducible-looking.
