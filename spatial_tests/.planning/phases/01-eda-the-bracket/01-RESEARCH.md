# Phase 1: EDA (the bracket) — Research

**Researched:** 2026-06-22
**Domain:** Diagnostic statistics, structure-based classification, spatial transcriptomics
**Confidence:** HIGH (all major claims verified against on-disk files, installed library APIs, or official
sklearn/squidpy documentation)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 Scope:** Run the full bracket for liver, human only this pass. The other organs (kidney,
  brain, heart) and rodent labels/structure-floor are deferred. Exception: cross-species liver basal
  diagnostics stay in (human-vs-rodent-liver basal correlation using the GSE272564 control arm +
  ortholog map; OOD distance of healthy liver from the cancer-line manifold). These need no rodent
  labels.

- **D-02 Halt Gate 2:** Gap metric = AUROC. Gate fires when the 95% paired bootstrap CI of
  (measured-ceiling AUROC - structure-floor AUROC) includes 0 (default 10,000 resamples). Fired gate
  means stop-and-REFRAME (negative is publishable), write `HALT_REASON.md`.

- **D-03 Floor:** Logistic regression + random forest on ECFP4/Morgan fingerprints (2048-bit, radius
  2). SMILES joined by REUSING sibling v0.5 tables first (dili_canonical.csv, dilist_smiles_resolved.csv,
  drugbank_smiles_index.csv); fall back to TDC DILI / PubChem only for residual gaps. Report label
  entropy, class balance, AUPRC base rate.

- **D-04 Liver label:** DILIrank primary/gating (vDILIConcern → binary, Wang/Li convention); DILIst
  secondary expanded-coverage cross-check.

- **D-05 Ceiling:** REUSE v0.5 Wang/Li LINCS measured DE (wangli_measured_de.npy + wangli_profiles.csv)
  as the human liver ceiling. Ceiling diagnostics: effective rank (PCA participation ratio), per-gene
  mutual information vs label, measured-signature-only baseline AUROC. Record provenance (SHA/source) in
  this project's MANIFEST.md. Organs/species with no measured data: report "no measured ceiling — floor
  only".

### Claude's Discretion

- OOD-distance method: Mahalanobis vs kNN (researcher picks; report which).
- Exact bootstrap resample count if 10,000 is too slow (floor 2,000).
- Morgan fingerprint bit length / radius if 2048/r2 underperforms.
- Which annotation field in the liver basal .h5ad provides the published region labels
  (published-annotations-default is locked; researcher resolves the field).

### Deferred Ideas (OUT OF SCOPE)

- Kidney / brain / heart brackets — same EDA machinery, run at each organ's training phase.
- Rodent structure-floor + rodent labels — deferred to the rodent pass.
- Rodent toxicogenomics ceiling (Open TG-GATEs / DrugMatrix) — acquire + bracket when the rodent arm
  starts.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EDA-01 | Label entropy / class balance / AUPRC base rates per organ+species; structure-space diagnostics + structure-only floor (logistic + RF) | D-03, D-04: DILIrank column layout verified, SMILES join coverage quantified, RDKit/sklearn confirmed in env |
| EDA-02 | Where measured data exists: effective rank, per-gene mutual information, measured-signature-only baseline (the ceiling) | D-05: wangli_measured_de.npy shape (5517, 978) verified on disk, wangli_profiles.csv columns confirmed, participation ratio formula provided |
| EDA-03 | Region diagnostics: basal-profile similarity matrix, SVG retention (Moran's I via squidpy), gene coverage. Cross-species diagnostics: ortholog overlap, human-vs-rodent basal correlation, OOD distance from cancer-line manifold | squidpy 1.8.2 API verified, yu2022 region annotation field confirmed (`category` in l5_category.csv), kuppe heart annotation field confirmed (`cell_type_original`), PDG cancer-line manifold location confirmed on disk |

</phase_requirements>

---

## Summary

Phase 1 runs entirely on files already on disk from Phase 0. The three logical work streams are:
(1) label + structure-only floor (EDA-01); (2) measured-biology ceiling (EDA-02); (3) region and
cross-species diagnostics (EDA-03). All required libraries are installed in `dili_v04_env`. All data
files and their SHAs have been verified.

**SMILES coverage gap (acquisition risk):** Joining DILIrank to the v0.5 sibling tables covers ~49.5%
of DILIrank drugs (661 / 1336 after dili_canonical + drugbank join). The Wang/Li ceiling covers 628
drugs with labels; intersection with DILIrank is 254. The floor classifier runs on the SMILES-joined
subset (non-zero coverage), which is sufficient for the bracket. For residual gaps, TDC-DILI (via
`pytdc`, installed) and PubChem name lookup are the fallback per D-03, but these should be pursued only
after the join; they are not blocking on day one.

**Cancer-line manifold reference:** The PDG diseased baseline is on disk at
`/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv`
(10 cell lines × 10,716 genes). This is the manifold reference for OOD distance. It must be logged in
MANIFEST.md with its SHA (computed: `e1e38f118c91be64065bdb8458479bfe3b0fc0a9c118a7e5352523f5a2306fb1`)
before use.

**GSE252772 mouse kidney:** Blocked on R→anndata conversion (`rpy2` not installed in `dili_v04_env`).
This dataset is not in scope for Phase 1 (liver-only per D-01), so no blocker exists here.

**Primary recommendation:** Implement three independent scripts (or a single `scripts/run_p1_eda.py`
with sub-commands): `floor`, `ceiling`, `diagnostics`. Each writes to `results/tables/P1_eda.md`.
Gate the halt-gate evaluation last so it can read both floor and ceiling numbers.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SMILES join + fingerprint featurization | Script / data layer | No model code | Pure data prep; RDKit is a data library |
| Label loading + binary encoding | Script / data layer | — | Reads raw xlsx; no model |
| Structure-only floor (LR + RF) | Script / classifier | sklearn | Stateless fit-predict on vectors |
| Measured-DE ceiling loading | Script / data layer | — | Reads pre-built .npy; no recomputation |
| PCA participation ratio | Script / stats | numpy | Formula applied to matrix |
| Per-gene mutual information | Script / stats | sklearn | mutual_info_classif on array |
| Measured-ceiling AUROC | Script / stats | sklearn | roc_auc_score |
| Paired bootstrap CI of gap | Script / stats | numpy | Resampling loop over shared drug set |
| Pseudobulk aggregation | src/spatial/pseudobulk.py | — | Already implemented; caller supplies obs_col |
| Moran's I (SVG retention) | squidpy.gr.spatial_autocorr | AnnData graph | squidpy 1.8.2 installed |
| Region similarity matrix | Script / stats | scipy.spatial | Pearson/cosine on pseudobulk profiles |
| Human-vs-rodent basal correlation | Script / stats | src/spatial/orthology.py | OrthologTable.pairs filters genes |
| OOD distance (healthy vs manifold) | Script / stats | scipy / sklearn | Mahalanobis recommended (see below) |
| MANIFEST.md provenance entries | Manual / script output | — | SHA computed, entries written before use |

---

## Standard Stack

### Core (all verified in `dili_v04_env`)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| squidpy | 1.8.2 | Spatial autocorrelation (Moran's I), neighbor graphs | Installed, verified; `sq.gr.spatial_autocorr` is the canonical API |
| sklearn | 1.8.0 | LR, RF, `mutual_info_classif`, `roc_auc_score`, `average_precision_score` | Already in env; covers all EDA-01/02 stats |
| RDKit | 2024.09.4 | ECFP4/Morgan fingerprints (2048-bit r2) | Installed; the canonical cheminformatics library |
| numpy | (env) | Matrix ops, bootstrap resampling | Core dependency |
| scipy | (env) | `mahalanobis`, `pearsonr`, PCA via `scipy.sparse.linalg.svds` | Installed |
| anndata | (env) | Read .h5ad files; pseudobulk input | Installed (note: use `anndata.read_h5ad`, not `scanpy.read_h5ad` — numba incompatibility in base env) |
| pandas | (env) | xlsx reading, csv I/O, label tables | Core dependency |
| h5py | (env) | Reading .h5 count matrices inside zips | Core dependency |
| pytdc | 0.4.17 | TDC DILI fallback for residual SMILES gaps | Installed per MANIFEST |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.spatial.distance.mahalanobis (Mahalanobis OOD) | sklearn kNN distance | Mahalanobis is the correct parametric choice for a small reference manifold (10 cell lines); requires covariance inversion. kNN is non-parametric and robust when n is larger. With only 10 points, kNN distance is poorly conditioned. Use Mahalanobis with ridge regularization (`np.cov(X.T) + alpha * I`). |
| sq.gr.spatial_neighbors_knn (squidpy 1.8.2) | sq.gr.spatial_neighbors (deprecated) | `spatial_neighbors` is deprecated as of 1.7.0 and will be removed in 1.9.0. Use `spatial_neighbors_knn` (n_neighs=6 for Visium hexagonal grid default). |
| sklearn.feature_selection.mutual_info_classif | mutual_info_regression | `mutual_info_classif` is correct for the binary DILI label; regression variant is for continuous targets. |

---

## Data File Inventory (verified on disk)

### Reused from v0.5 dili_downstream (must be logged in MANIFEST.md before use — not yet logged)

| File | Path | Shape / Size | SHA256 | Use |
|------|------|-------------|--------|-----|
| wangli_measured_de.npy | `../dili_downstream/data/processed/wangli_measured_de.npy` | (5517, 978) float32 | `aaeb07ce13222af49c873be8652b01597a4c638119efd306d4f4d7c04b9397f5` | Measured DE matrix for ceiling |
| wangli_profiles.csv | `../dili_downstream/data/processed/wangli_profiles.csv` | 5517 rows × 12 cols | `963ea3e836b061a6eb2fa066ef5634b4cf9e9cfda8537ea45e98ef1ec3cf872e` | Profile metadata; columns: profile_id, compound_name, brd_id, cell_id, time_h, dose_str, dose_um, smiles, dili_binary, dili_severity, tuple_key, usage |
| dili_canonical.csv | `../dili_downstream/data/processed/dili_canonical.csv` | 1118 rows × 9 cols | `8e30f71278fad1391eacab0fc287d2898361249cd119ade810f86212243a3525` | SMILES join (cols: pert_id, drug_name, smiles, canonical_smiles, scaffold, dili_binary, dili_severity, in_lincs, in_pdg); join key: drug_name (normalize to lowercase) |
| dilist_smiles_resolved.csv | `../dili_downstream/data/processed/dilist_smiles_resolved.csv` | 1118 rows × 6 cols | `b57be7557b52fd6c3f094e718e560a522452895ee7c317a67084b1c83ebbdf26` | SMILES join; cols: DILIST_ID, drug_name, name_lower, smiles, canonical_smiles, source |
| drugbank_smiles_index.csv | `../dili_downstream/data/processed/drugbank_smiles_index.csv` | 38917 rows × 3 cols | `2ad9eb4d642f954dfdc8aa30deb0a4089b99292af0917af4df0e034c05134a5e` | SMILES fallback; cols: name_lower, name, smiles; join key: name_lower |

### Cancer-line manifold reference (must be logged in MANIFEST.md before use — not yet logged)

| File | Path | Shape | SHA256 | Use |
|------|------|-------|--------|-----|
| pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv | `/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` | (10, 10716) — 10 cell lines × 10,716 genes | `e1e38f118c91be64065bdb8458479bfe3b0fc0a9c118a7e5352523f5a2306fb1` | OOD distance manifold reference |

### Label files (on disk, this project)

| File | Path | Key Column | Binary Encoding |
|------|------|-----------|-----------------|
| dilirank.xlsx | `data/raw/labels/dilirank/dilirank.xlsx` | `vDILI-Concern` (read with header=1) | vMost-DILI-concern + vLess-DILI-concern + vMOST-DILI-concern → 1; vNo-DILI-concern + vNo-DILI-Concern → 0; Ambiguous-DILI-concern → exclude. 1336 rows. Positive: ~568, Negative: ~414, Ambiguous: 354 |
| dilist.xlsx | `data/raw/labels/dilist/dilist.xlsx` | `DILIst Classification ` (note trailing space) | 1 → positive, 0 → negative; 1279 rows (768 positive, 511 negative) |

### Spatial basal files (on disk, this project)

| Dataset | Path | Status | Region Annotation |
|---------|------|--------|------------------|
| yu2022_liver (L5) | `data/raw/spatial/yu2022_liver/L5_upload.zip` | Zip; contains `L5_upload/filtered_feature_bc_matrix.h5` + `L5_upload/l5_category.csv` | `l5_category.csv`: barcode→category; values: Zone1, Zone2_1, Zone2_2, Zone2_3, Zone3, portal_area, central_area (7 regions) |
| yu2022_liver (L18) | `data/raw/spatial/yu2022_liver/L18_upload.zip` | Zip; same structure | Same `category` column (load from `L18_upload/l18_category.csv`) |
| gse272564_mouse_liver_ctrl | `data/raw/spatial/gse272564_mouse_liver_ctrl/GSE272564_RAW.tar` | TAR; per-sample 10x mtx.gz format | APAP0h sample (GSM8404653) = 0h control arm; samples: APAP0h, APAP3h, APAP6h, APAP24h |

---

## Architecture Patterns

### System Architecture Diagram

```
[yu2022_liver zips]  [gse272564 ctrl TAR]       [dilirank.xlsx / dilist.xlsx]
         |                    |                            |
   [unzip+load h5]    [untar+load mtx.gz]        [label_loader.py]
         |                    |                            |
   [pseudobulk.py] ←  obs_col='category'          [binary encoding]
         |                    |                            |
   [region_profiles]   [mouse_profiles]           [compound_list]
         |                    |                            |
         v                    v                            v
   [squidpy neighbor graph]  [ortholog map]         [SMILES join cascade]
   [sq.gr.spatial_autocorr]  [orthology.py]     dili_canonical→drugbank→TDC
         |                    |                            |
   [SVG retention %]   [cross-species correlation]  [ECFP4 fingerprints (RDKit)]
                                                          |
                              [wangli_measured_de.npy] (5517×978)
                              [wangli_profiles.csv]        |
                                       |           [LR + RF classifiers]
                              [align to DILIrank]          |
                              [drug-level aggregate] [floor AUROC]
                                       |                   |
                              [PCA participation ratio]    |
                              [per-gene MI vs label]       |
                              [ceiling AUROC]              |
                                       |                   |
                                       +---------+---------+
                                                 |
                              [paired bootstrap CI(ceiling - floor)]
                                                 |
                                    [Halt Gate 2 evaluation]
                                                 |
                                    [results/tables/P1_eda.md]
```

### Recommended Project Structure

New files for Phase 1 only. Do not modify existing `src/spatial/` library files.

```
scripts/
└── run_p1_eda.py          # Entry point (sub-commands: floor, ceiling, diagnostics, all)
src/spatial/
└── eda/                   # New sub-package (pure library, no I/O)
    ├── __init__.py
    ├── labels.py           # DILIrank/DILIst loading + binary encoding
    ├── smiles_join.py      # SMILES cascade: dili_canonical → drugbank → TDC fallback
    ├── fingerprints.py     # ECFP4 via RDKit (2048-bit, radius 2)
    ├── floor.py            # LR + RF floor + bootstrap seeds
    ├── ceiling.py          # Measured-DE ceiling: participation ratio, MI, AUROC
    ├── bootstrap.py        # Paired bootstrap CI of the AUROC gap
    └── region_diagnostics.py  # SVG retention, basal similarity, OOD, cross-species
results/tables/
└── P1_eda.md              # Deliverable (written by run_p1_eda.py)
tests/
└── spatial/
    ├── test_eda_labels.py
    ├── test_eda_fingerprints.py
    ├── test_eda_floor.py
    ├── test_eda_ceiling.py
    └── test_eda_bootstrap.py
```

---

## Pattern 1: DILIrank loading and binary encoding (Wang/Li convention)

**What:** Read dilirank.xlsx (header row at row index 1), normalize vDILI-Concern to binary,
exclude Ambiguous.

**When to use:** EDA-01 floor, EDA-02 ceiling label alignment.

```python
# Source: verified against dilirank.xlsx on disk (2026-06-22)
import pandas as pd

def load_dilirank(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    # Normalize case variations (vMOST vs vMost both present)
    concern = df["vDILI-Concern"].str.lower().str.strip()
    pos_mask = concern.isin({"vmost-dili-concern", "vless-dili-concern"})
    neg_mask = concern.isin({"vno-dili-concern", "vno-dili-concern"})  # catches both case variants
    df = df[pos_mask | neg_mask].copy()
    df["dili_binary"] = pos_mask[pos_mask | neg_mask].astype(int)
    df["name_lower"] = df["CompoundName"].str.lower().str.strip()
    return df[["LTKBID", "CompoundName", "name_lower", "dili_binary", "SeverityClass"]]

# Result: 1336 total, ~568 positive, ~414 negative, 354 excluded (Ambiguous)
```

## Pattern 2: SMILES join cascade (D-03)

**What:** Join DILIrank/DILIst drugs to canonical SMILES via three sibling tables in priority order.
**When to use:** Before fingerprint featurization in EDA-01.

```python
# Source: verified column names from file inspection (2026-06-22)
# Join key: name_lower (lowercase CompoundName)

def join_smiles_cascade(labels_df: pd.DataFrame,
                        dili_canonical_path: str,
                        drugbank_path: str) -> pd.DataFrame:
    # Layer 1: dili_canonical (has canonical_smiles column)
    c1 = pd.read_csv(dili_canonical_path)
    c1["name_lower"] = c1["drug_name"].str.lower().str.strip()
    merged = labels_df.merge(c1[["name_lower", "canonical_smiles"]].rename(
        columns={"canonical_smiles": "smiles"}), on="name_lower", how="left")

    # Layer 2: drugbank_smiles_index for residual (name_lower already the key)
    missing_mask = merged["smiles"].isna()
    c2 = pd.read_csv(drugbank_path)  # cols: name_lower, name, smiles
    residual = merged[missing_mask][["name_lower"]].merge(c2[["name_lower", "smiles"]],
                                                           on="name_lower", how="left")
    merged.loc[missing_mask, "smiles"] = residual["smiles"].values

    # Layer 3: TDC fallback for still-missing (pytdc available)
    # ... (implement if gap > 10% after layers 1+2)
    return merged

# Coverage after layers 1+2 on DILIrank: ~661/1336 (~49.5%)
# The floor runs on covered drugs; uncovered are excluded and reported
```

## Pattern 3: ECFP4/Morgan fingerprints via RDKit

**What:** 2048-bit, radius 2 Morgan fingerprints for the structure-only floor.

```python
# Source: RDKit 2024.09.4 verified in dili_v04_env (2026-06-22)
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_ecfp4(smiles_list: list[str],
                    n_bits: int = 2048,
                    radius: int = 2) -> np.ndarray:
    fps = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            valid_mask.append(False)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.frombuffer(fp.ToBitString().encode(), dtype='u1') - ord('0'))
            valid_mask.append(True)
    return np.array(fps, dtype=np.uint8), np.array(valid_mask)
```

## Pattern 4: PCA participation ratio (effective rank)

**What:** Participation ratio = (sum_i lambda_i)^2 / sum_i lambda_i^2 where lambda_i are
eigenvalues of the covariance matrix. Numerically stable via SVD on the centered data matrix.

```python
# Source: formula from Roy & Cover (information theory); implementation verified numerically
import numpy as np

def participation_ratio(X: np.ndarray) -> float:
    """Effective rank via PCA participation ratio.
    X shape: (n_samples, n_genes). Returns float."""
    X_centered = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    lambdas = s ** 2  # eigenvalues of covariance (up to 1/(n-1) factor; cancels)
    lambdas = lambdas[lambdas > 0]  # drop numerical zero
    pr = (lambdas.sum()) ** 2 / (lambdas ** 2).sum()
    return float(pr)

# For wangli_measured_de.npy (5517, 978): expect PR in range [50, 300] — signals
# that the 978-dimensional space has meaningful low-rank structure
```

## Pattern 5: Per-gene mutual information vs binary DILI label

**What:** `sklearn.feature_selection.mutual_info_classif` on the measured DE matrix vs binary label.
The function handles discretization internally (continuous features, binary target).

```python
# Source: sklearn 1.8.0 verified in env (2026-06-22)
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def per_gene_mi(X: np.ndarray, y: np.ndarray,
                random_state: int = 42) -> np.ndarray:
    """X: (n_drugs, n_genes), y: binary labels. Returns MI per gene (n_genes,)."""
    mi = mutual_info_classif(X, y, discrete_features=False,
                              n_neighbors=3, random_state=random_state)
    return mi  # shape (n_genes,); report top-k, mean, median

# Report: fraction of genes with MI > 0 (non-zero signal); top-20 gene names
```

## Pattern 6: Squidpy Moran's I for SVG retention

**What:** Compute Moran's I on the raw Visium spots, then on the pseudobulk profiles, to check what
fraction of spatially variable genes (SVGs) from the raw data survive into the pseudobulk used as
model input.

**Key prerequisite:** `adata.obsm["spatial"]` must contain (x, y) pixel coordinates.
For yu2022 (loaded from zip), coordinates come from `tissue_positions_list.csv` and must be loaded
into `adata.obsm["spatial"]` after reading the h5.

**API:** Use `sq.gr.spatial_neighbors_knn` (not deprecated `spatial_neighbors`), then
`sq.gr.spatial_autocorr(adata, mode='moran')`. Result is stored in `adata.uns["moranI"]`.

```python
# Source: squidpy 1.8.2 API verified in dili_v04_env (2026-06-22)
import squidpy as sq
import anndata as ad

def compute_moran_svgs(adata: ad.AnnData,
                       spatial_key: str = "spatial",
                       n_neighs: int = 6,
                       n_hvg: int = 2000) -> dict:
    """
    1. Build k-NN graph on spatial coordinates (n_neighs=6 for Visium hex grid).
    2. Compute Moran's I for top HVGs (or all genes if n < 2000).
    3. Return dict with svg_count, svg_fraction, top_svgs.

    Prerequisite: adata.obsm[spatial_key] populated with (x, y) coords.
    """
    sq.gr.spatial_neighbors_knn(adata, spatial_key=spatial_key,
                                 n_neighs=n_neighs, key_added="spatial")
    # Subset to HVGs for speed (or pass genes= argument directly)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=None,
                            connectivity_key="spatial_connectivities",
                            transformation=True, copy=False)
    moran_df = adata.uns["moranI"]  # DataFrame: index=gene, columns=['I', 'pval_norm', ...]
    svgs = moran_df[moran_df["pval_norm"] < 0.05]
    return {
        "svg_count": len(svgs),
        "svg_fraction": len(svgs) / len(moran_df),
        "top_svgs": svgs.nlargest(20, "I").index.tolist(),
    }

# SVG retention metric: fraction of raw-Visium SVGs whose gene symbol appears in
# the 10,716-gene MultiDCP model space (from P0_coverage.md: yu2022_liver = 99.8%).
# "Retention" = (# raw SVGs in model space) / (# raw SVGs total).
```

## Pattern 7: Paired bootstrap CI of the AUROC gap (D-02)

**What:** Correct paired resampling over the SHARED drug set where both floor and ceiling have
predictions. 10,000 resamples of (drug_i, label_i, floor_prob_i, ceiling_prob_i); compute AUROC
gap per resample; 2.5th / 97.5th percentile CI.

**Critical:** floor and ceiling must be intersected to the shared drug set before the paired test.
The ceiling is profile-level (multiple profiles per drug); aggregate to drug-level before the paired
test (e.g., mean probability across profiles for each drug).

```python
# Source: standard paired bootstrap methodology; implementation verified conceptually (2026-06-22)
import numpy as np
from sklearn.metrics import roc_auc_score

def paired_bootstrap_auroc_gap(y: np.ndarray,
                                floor_probs: np.ndarray,
                                ceil_probs: np.ndarray,
                                n_resamples: int = 10_000,
                                seed: int = 42) -> dict:
    """
    y, floor_probs, ceil_probs: aligned arrays over the SHARED drug set.
    Returns dict with gap_observed, ci_lower, ci_upper, gate_fires.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = np.empty(n_resamples)
    obs_gap = roc_auc_score(y, ceil_probs) - roc_auc_score(y, floor_probs)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            gaps[i] = np.nan
            continue
        gaps[i] = roc_auc_score(y_b, ceil_probs[idx]) - roc_auc_score(y_b, floor_probs[idx])
    gaps = gaps[~np.isnan(gaps)]
    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "gap_observed": obs_gap,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "gate_fires": ci_lo <= 0,  # True = Halt Gate 2 fires
        "n_resamples_valid": len(gaps),
    }
```

## Pattern 8: OOD distance — Mahalanobis vs kNN decision

**Recommendation:** Use **Mahalanobis** with ridge regularization (`alpha=1e-3` or `alpha=1e-2`).

**Rationale:** The cancer-line manifold has exactly 10 points in 10,716-gene space (PDG diseased
avg). With n=10, kNN distances are sensitive to the choice of k (k=1..5 spans the whole dataset)
and give little information. Mahalanobis, with regularization, estimates whether the healthy Visium
pseudobulk profiles lie inside the covariance structure of the cancer-line training distribution.
Report the regularization alpha and the mean Mahalanobis distance per liver zone.

```python
# Source: scipy.spatial.distance.mahalanobis verified available (2026-06-22)
import numpy as np
from scipy.spatial.distance import mahalanobis

def ood_mahalanobis(manifold: np.ndarray,  # (10, 10716) cancer-line matrix
                    query: np.ndarray,      # (n_regions, 10716) Visium pseudobulk
                    alpha: float = 1e-2) -> np.ndarray:
    """
    Compute Mahalanobis distance of each query row from the manifold distribution.
    manifold: (n_ref, d); query: (n_query, d).
    Returns (n_query,) distances.
    """
    cov = np.cov(manifold.T)  # (d, d) — singular for d >> n_ref
    cov_reg = cov + alpha * np.eye(cov.shape[0])
    vi = np.linalg.inv(cov_reg)
    mean_ref = manifold.mean(axis=0)
    return np.array([mahalanobis(q, mean_ref, vi) for q in query])

# NOTE: d=10716 >> n=10 means full covariance is rank-deficient. Two safe alternatives:
# (a) Project to PCA top-k (k=9 max for 10 points) before Mahalanobis — loses spatial info.
# (b) Use the LINCS L1000 978-gene subspace instead (ortholog-aligned): both manifold and
#     query projected to 978 genes → well-conditioned covariance (n=10, d=978 still tricky
#     but manageable with alpha).
# RECOMMENDED IMPLEMENTATION: project both manifold and query to the 978 landmark genes
# present in both spaces, apply ridge-regularized Mahalanobis in that subspace.
```

## Pattern 9: Human-vs-rodent liver basal correlation

**What:** Pearson correlation between human (yu2022) and mouse (GSE272564 APAP0h) pseudobulk liver
profiles, restricted to the one-to-one ortholog map from `src/spatial/orthology.py`.

**Region alignment:** yu2022 has 7 annotated zones (Zone1..Zone3, portal_area, central_area).
GSE272564 APAP0h has no published zone annotations — use whole-sample pseudobulk as the mouse
reference. Report correlation at the whole-tissue level; note that zone-level mouse correlation
requires annotation work deferred to later phases.

```python
# Source: orthology.py interface verified (2026-06-22)
from src.spatial.orthology import build_one2one_orthologs, OrthologTable
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def human_mouse_liver_correlation(human_profile: np.ndarray,  # (n_genes_human,)
                                   human_genes: list[str],
                                   mouse_profile: np.ndarray,  # (n_genes_mouse,)
                                   mouse_genes: list[str],
                                   ortholog_table: OrthologTable) -> dict:
    ot = ortholog_table.pairs  # cols: human_symbol, mouse_symbol, ...
    h_to_m = dict(zip(ot["human_symbol"], ot["mouse_symbol"]))
    h_idx = {g: i for i, g in enumerate(human_genes)}
    m_idx = {g: i for i, g in enumerate(mouse_genes)}
    h_vals, m_vals = [], []
    for h_gene, m_gene in h_to_m.items():
        if h_gene in h_idx and m_gene in m_idx:
            h_vals.append(human_profile[h_idx[h_gene]])
            m_vals.append(mouse_profile[m_idx[m_gene]])
    r, p = pearsonr(np.array(h_vals), np.array(m_vals))
    return {"pearson_r": r, "p_value": p, "n_genes_compared": len(h_vals)}
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fingerprint similarity / hashing | Custom bit-vector hash | RDKit `GetMorganFingerprintAsBitVect` | RDKit handles stereochemistry, valence, aromatic perception correctly |
| Multiple-testing correction for SVG p-values | Custom BH procedure | squidpy `corr_method='fdr_bh'` (calls statsmodels under the hood) | Built into `spatial_autocorr` via `corr_method` argument |
| Bootstrap CI | Manual percentile loop | The pattern above; do NOT use statsmodels bootstrap — it re-fits models | The paired constraint requires custom sampling logic anyway |
| AUROC with ties | Manual sort/trapezoid | `sklearn.metrics.roc_auc_score` | Handles ties correctly |
| AUPRC base rate | Custom precision-recall | `sklearn.metrics.average_precision_score` | Handles class imbalance; equal to area under PR curve |

---

## Common Pitfalls

### Pitfall 1: DILIrank header is on row 1 (not row 0)
**What goes wrong:** `pd.read_excel(..., header=0)` returns the FDA title row as column names.
**Why it happens:** The xlsx has a one-line title row before the real header.
**How to avoid:** Always `pd.read_excel(path, header=1)`. Verified: produces columns `LTKBID`,
`CompoundName`, `SeverityClass`, `LabelSection`, `vDILI-Concern`, `Comment`.
**Warning signs:** Column names look like "Drug Induced Liver Injury Rank..." — wrong row.

### Pitfall 2: Case variation in vDILI-Concern values
**What goes wrong:** Treating "vMOST-DILI-concern" (2 rows) and "vMost-DILI-concern" (215 rows)
as distinct categories loses positive labels.
**Why it happens:** The Excel file has two spellings.
**How to avoid:** Normalize with `.str.lower().str.strip()` before matching.

### Pitfall 3: DILIst column name has a trailing space
**What goes wrong:** `df["DILIst Classification"]` raises KeyError.
**Why it happens:** On-disk column is `"DILIst Classification "` (trailing space).
**How to avoid:** Use `df.columns.str.strip()` after loading, or hardcode the space in lookups.
Verified on disk.

### Pitfall 4: squidpy `spatial_neighbors` is deprecated (1.7.0+)
**What goes wrong:** Using `sq.gr.spatial_neighbors(...)` raises a deprecation warning and will
break in squidpy 1.9.0. The key_added behavior also changed.
**How to avoid:** Use `sq.gr.spatial_neighbors_knn(adata, n_neighs=6, key_added="spatial")`.
The resulting connectivity key is `"spatial_connectivities"` (pass to `spatial_autocorr` as
`connectivity_key`).

### Pitfall 5: yu2022 liver files are in ZIP, not h5ad
**What goes wrong:** Code tries `anndata.read_h5ad(zip_path)` — fails.
**Why it happens:** The figshare download is a zip containing a 10x Visium directory structure.
**How to avoid:** Use `zipfile.ZipFile` to extract `filtered_feature_bc_matrix.h5` and
`tissue_positions_list.csv` into a temp directory, then use `scanpy.read_10x_h5` (or equivalent)
to load. The region labels are in `l5_category.csv` (not in the h5 obs), joined on barcode.

**Exact file structure inside zip:**
```
L5_upload/
├── filtered_feature_bc_matrix.h5     # expression counts
├── l5_category.csv                   # barcode → category (region annotation)
├── spatial/tissue_positions_list.csv # spatial coordinates (x, y)
└── ...
```

### Pitfall 6: GSE272564 APAP0h is the control sample — not a separate file
**What goes wrong:** Code looks for a "control" directory that doesn't exist.
**Why it happens:** All four timepoints (APAP0h, APAP3h, APAP6h, APAP24h) ship in a single
`GSE272564_RAW.tar`. The 0h timepoint (GSM8404653) is the basal/control condition.
**How to avoid:** Extract the tar, then filter to `*APAP0h*` files (barcodes, features, matrix).

### Pitfall 7: Mahalanobis on full 10,716-gene space is rank-deficient
**What goes wrong:** `np.linalg.inv(np.cov(manifold.T))` fails or is numerically unstable when
n=10 << d=10,716.
**How to avoid:** Project both manifold and query to the 978 landmark gene subspace first
(well-conditioned with alpha regularization), or use PCA to reduce to n-1=9 components. Document
which projection was used in P1_eda.md.

### Pitfall 8: Ceiling drug-level aggregation before paired bootstrap
**What goes wrong:** wangli_profiles.csv has 5517 rows but only 628 unique drugs. If floor
probabilities are at drug-level and ceiling probabilities at profile-level, the paired bootstrap
misaligns.
**How to avoid:** Aggregate ceiling predictions to drug level (mean predicted probability across
all profiles for that drug) before the paired bootstrap. The SHARED drug set is the intersection
of (floor drugs with SMILES) and (ceiling drugs in wangli_profiles). Report the intersection size.

### Pitfall 9: `anndata.read_h5ad` vs `scanpy.read_h5ad` — numba conflict
**What goes wrong:** `import scanpy` fails with `numba.ImportError: NumPy 2.0 or less required`
when using the base conda env (NumPy 2.2 installed there).
**How to avoid:** Use `conda run -n dili_v04_env` for all EDA scripts. Within scripts, use
`import anndata` directly (not `import scanpy`), which does not import numba. Or use
`scanpy` inside the dili_v04_env where numba and numpy are compatible.

---

## Runtime State Inventory

> This phase is not a rename/refactor/migration phase. No runtime state inventory needed.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| squidpy | EDA-03 Moran's I | Yes | 1.8.2 | — |
| sklearn | EDA-01/02 floor/ceiling/MI | Yes | 1.8.0 | — |
| RDKit | EDA-01 fingerprints | Yes | 2024.09.4 | — |
| anndata | EDA-03 h5ad loading | Yes | (in dili_v04_env) | — |
| scipy | OOD distance, pearsonr | Yes | (in dili_v04_env) | — |
| pytdc | TDC DILI fallback SMILES | Yes | 0.4.17 | PubChem API |
| rpy2 | GSE252772 .rds conversion | No | — | Out of scope (liver-only P1) |
| wangli_measured_de.npy | EDA-02 ceiling | Yes | — | No fallback (hard dependency) |
| PDG cancer-line manifold CSV | EDA-03 OOD distance | Yes (sibling path) | — | Use PCA-9 of the available 10 cell lines |

**Missing dependencies with no fallback for this phase:** None (rpy2 is only needed for kidney, deferred).

**Note on numba/numpy conflict:** Running scripts in the base conda env will fail on `import scanpy`.
All scripts must explicitly target `dili_v04_env`. Set the shebang or invocation to
`conda run -n dili_v04_env python scripts/run_p1_eda.py`.

---

## Validation Architecture

Nyquist validation is enabled (`workflow.nyquist_validation` key absent from config → treated as
enabled).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (in dili_v04_env) |
| Config file | none (implicit discovery) |
| Quick run command | `conda run -n dili_v04_env pytest tests/spatial/ -x -q` |
| Full suite command | `conda run -n dili_v04_env pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EDA-01 | DILIrank binary encoding excludes Ambiguous | unit | `pytest tests/spatial/test_eda_labels.py::test_dilirank_binary_encoding -x` | No — Wave 0 |
| EDA-01 | SMILES join cascade: dili_canonical first, drugbank fallback | unit | `pytest tests/spatial/test_eda_labels.py::test_smiles_join_cascade -x` | No — Wave 0 |
| EDA-01 | ECFP4 fingerprint: 2048-bit radius-2 on known SMILES | unit | `pytest tests/spatial/test_eda_fingerprints.py::test_ecfp4_shape -x` | No — Wave 0 |
| EDA-01 | Floor AUROC > 0.5 (better than random) on DILIrank | smoke | `pytest tests/spatial/test_eda_floor.py::test_floor_auroc_above_chance -x` | No — Wave 0 |
| EDA-02 | Participation ratio formula on synthetic matrix | unit | `pytest tests/spatial/test_eda_ceiling.py::test_participation_ratio -x` | No — Wave 0 |
| EDA-02 | Per-gene MI returns correct shape | unit | `pytest tests/spatial/test_eda_ceiling.py::test_per_gene_mi_shape -x` | No — Wave 0 |
| EDA-02 | Ceiling AUROC ≥ floor AUROC on shared drug set | smoke | `pytest tests/spatial/test_eda_ceiling.py::test_ceiling_ge_floor -x` | No — Wave 0 |
| EDA-02 | Paired bootstrap CI width > 0 | unit | `pytest tests/spatial/test_eda_bootstrap.py::test_bootstrap_ci_nonzero_width -x` | No — Wave 0 |
| EDA-03 | Moran's I returns DataFrame with gene index | unit | `pytest tests/spatial/test_eda_region.py::test_moran_returns_dataframe -x` | No — Wave 0 |
| EDA-03 | Human-mouse correlation uses only one-to-one orthologs | unit | `pytest tests/spatial/test_eda_region.py::test_cross_species_ortholog_filter -x` | No — Wave 0 |

All tests are missing — Wave 0 must create the test files and `tests/spatial/__init__.py`.

### Sampling Rate

- Per task commit: `conda run -n dili_v04_env pytest tests/spatial/ -x -q`
- Per wave merge: `conda run -n dili_v04_env pytest tests/ -q`
- Phase gate: Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/spatial/__init__.py` — package init
- [ ] `tests/spatial/test_eda_labels.py` — covers EDA-01 (label loading, SMILES join)
- [ ] `tests/spatial/test_eda_fingerprints.py` — covers EDA-01 (ECFP4)
- [ ] `tests/spatial/test_eda_floor.py` — covers EDA-01 (floor AUROC smoke)
- [ ] `tests/spatial/test_eda_ceiling.py` — covers EDA-02 (participation ratio, MI, ceiling AUROC)
- [ ] `tests/spatial/test_eda_bootstrap.py` — covers EDA-02 (paired bootstrap CI)
- [ ] `tests/spatial/test_eda_region.py` — covers EDA-03 (Moran's I, cross-species correlation)

---

## Security Domain

No authentication, credentials, or user data involved. Web API calls (TDC, PubChem fallback) are
read-only public endpoints and do not handle sensitive data. ASVS categories V2/V3/V4/V6 do not
apply. V5 (input validation) applies minimally: SMILES strings from the join tables must be
validated by RDKit (`MolFromSmiles` returning `None` on parse failure) before fingerprint generation —
this is already captured in Pattern 3 above.

---

## Open Questions

1. **Region annotations for andrews_liver (GSE185477)**
   - What we know: `usable_as_input=True`, `region_annotation_source="reference"` (not manual).
   - What's unclear: Which obs column provides region labels in the GSE185477 RAW.tar h5 files?
   - Recommendation: Defer to the plan — andrews_liver is the backup to yu2022_liver for EDA-03.
     Use yu2022_liver (which has verified `l5_category.csv` annotation) as the primary. Probe
     andrews_liver obs columns only if a second sample is needed for robustness.

2. **TDC DILI SMILES gap quantification**
   - What we know: After dili_canonical + drugbank join, ~675 DILIrank drugs have no SMILES (~50.5%).
   - What's unclear: How many of those have entries in TDC's DILI dataset vs truly SMILES-less entries
     (e.g., biologics, salts not in DrugBank).
   - Recommendation: Run TDC lookup as the first task in the floor wave; report how many additional
     drugs are recovered. The floor quality depends on coverage — if coverage stays below 40% of
     non-Ambiguous DILIrank, flag in P1_eda.md.

3. **GSE272564 zone annotations for mouse liver**
   - What we know: The APAP0h sample (GSM8404653) has barcodes, features, and matrix on disk.
     No published zone annotation is available — only timepoint metadata.
   - What's unclear: Whether any spot-level liver zone annotation (periportal/pericentral) is in the
     GEO metadata for the 0h sample.
   - Recommendation: Use whole-sample pseudobulk for the mouse reference in EDA-03. Note "no zone
     annotation available" in P1_eda.md. Zone-level rodent correlation is deferred.

4. **Mahalanobis dimensionality: full 10,716 vs 978 landmark subspace**
   - What we know: PDG manifold is (10, 10,716). Full covariance is rank-deficient. Landmark genes
     (978) are a well-characterized subset present in both the manifold and the Visium data.
   - What's unclear: Which projection better captures the biological question (does healthy liver
     look like cancer cell lines in the model's feature space?).
   - Recommendation: Use the 978-gene landmark projection for the primary OOD distance report.
     Report dimension and alpha in P1_eda.md. Full 10,716 with heavy ridge regularization is an
     optional sensitivity check if time allows.

---

## Acquisition Risks and Blockers

| Risk | Impact | Mitigation |
|------|--------|-----------|
| TDC DILI / PubChem SMILES gap | Floor covers ~50% of DILIrank drugs after join; classifier may be underpowered | Run fallback early; report coverage; if < 40% non-Ambiguous, flag in P1_eda.md |
| yu2022 L5_upload.zip extraction: coordinate loading | Moran's I requires `adata.obsm["spatial"]` which must be populated from `tissue_positions_list.csv` inside the zip | Load positions separately and assign to adata.obsm after h5 load (see Pattern 5 notes) |
| wangli_measured_de.npy not yet in spatial MANIFEST.md | D-05 provenance discipline requires MANIFEST entry before use | Wave 0 task: add all 5 reused v0.5 files + PDG manifold CSV to MANIFEST.md with verified SHAs (SHAs computed above) |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | DILIrank Ambiguous-DILI-concern drugs are excluded (not labeled as positive or negative) | Pattern 1 | If convention is "Ambiguous = positive," floor/ceiling AUROC changes; verify against Wang/Li 2020 methods |
| A2 | wangli_profiles.csv `dili_binary` column uses the same Wang/Li convention as will be applied to DILIrank | Ceiling loading | Misalignment would make the ceiling AUROC meaningless; confirm column provenance |
| A3 | `adata.obsm["spatial"]` key is the correct key for yu2022 coordinates after loading | Moran's I pattern | If key differs, `sq.gr.spatial_neighbors_knn` fails; probe `adata.obsm.keys()` at runtime |

All other factual claims (file shapes, library versions, column names, SHA256s) are VERIFIED.

---

## Sources

### Primary (HIGH confidence — verified against on-disk files or installed library APIs)

- File inspection: `wangli_measured_de.npy` (5517, 978), `wangli_profiles.csv` columns, `dilirank.xlsx`
  header structure, `dilist.xlsx` classification column, `dili_canonical.csv` columns, `drugbank_smiles_index.csv`
  columns, `l5_category.csv` region annotation field, kuppe_heart `cell_type_original`, GSE272564 sample names
  — all verified by running Python against on-disk data (2026-06-22)
- `squidpy 1.8.2` API: `sq.gr.spatial_autocorr`, `sq.gr.spatial_neighbors_knn` — verified via
  `help()` in dili_v04_env (2026-06-22)
- `sklearn 1.8.0`: `mutual_info_classif`, `roc_auc_score` — verified import in dili_v04_env
- `RDKit 2024.09.4`: `GetMorganFingerprintAsBitVect` — verified import in dili_v04_env
- PDG cancer-line manifold: `pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` shape (10, 10716)
  and path — verified by reading file (2026-06-22)
- SMILES coverage calculation: 49.5% of DILIrank covered after dili_canonical + drugbank join —
  verified by Python join on live files (2026-06-22)
- SHA256s for all 6 files requiring MANIFEST entries — computed from on-disk files (2026-06-22)

### Secondary (MEDIUM confidence — standard statistical methodology)

- Participation ratio formula `(sum lambda)^2 / sum lambda^2`: Roy & Cover formulation; widely used
  in computational biology for effective rank estimation
- Paired bootstrap AUROC CI: standard methodology; percentile CI with paired resampling is the
  accepted approach for comparing two classifiers on the same test set
- Mahalanobis with ridge regularization for low-n/high-d OOD detection: standard ML practice;
  alternative to kNN for small reference sets

---

## Metadata

**Confidence breakdown:**
- Data file verification: HIGH — all key files read directly on-disk
- Label encoding: HIGH — verified column structure; A1 assumption flagged
- Library APIs: HIGH — verified in installed dili_v04_env
- Statistical formulas: MEDIUM — standard methodology, not re-derived
- SMILES gap estimate: HIGH — computed from actual join

**Research date:** 2026-06-22
**Valid until:** 2026-07-22 (stable libraries; squidpy deprecation of `spatial_neighbors` locks in
at 1.9.0 — check before upgrading env)
