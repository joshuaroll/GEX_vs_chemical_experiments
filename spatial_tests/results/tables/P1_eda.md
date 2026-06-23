# Phase 1 EDA Report: The Bracket (liver/human)

**Generated:** 2026-06-23T05:08:47Z
**Sub-command:** all

## EDA-01: Structure-Only Floor (DILIrank, liver/human)

**Label set:** DILIrank 2.0 (FDA LTKB); Wang/Li convention (D-04).
**Non-Ambiguous drugs:** 982 (568 positive, 414 negative).
**SMILES coverage:** 50.8% (499 / 982 non-Ambiguous).

**Floor dataset:** 499 drugs with valid ECFP4 fingerprints (2048-bit, radius 2; RDKit excluded: 0).

### Floor metrics

| Metric | Value |
|--------|-------|
| Label entropy (bits) | 0.9528 |
| Class balance (positive fraction) | 0.6273 |
| AUPRC base rate | 0.6273 |
| LR floor AUROC (mean, seeds 0-2) | 0.7207 |
| RF floor AUROC (mean, seeds 0-2) | 0.7625 |
| n drugs (floor dataset) | 499 |

### DILIst secondary cross-check (D-04)

| Label set | n total | n with SMILES | Positive fraction | LR AUROC (seed 0) | Notes |
|-----------|---------|---------------|-------------------|-------------------|-------|
| DILIst (secondary, D-04) | 1279 | 1118 | 0.613 | 0.6385 | -- |

## EDA-02: Measured-Biology Ceiling (liver/human)

**Source:** Wang/Li LINCS measured DE (wangli_measured_de.npy, shape (5517, 978), float32; already differential expression, DE rule satisfied).
**Profiles in DILIrank:** 2648 profiles from 227 unique drugs.

### Ceiling metrics

| Metric | Value |
|--------|-------|
| PCA participation ratio (effective rank) | 18.5 |
| MI fraction nonzero | 0.9366 |
| Ceiling AUROC (drug-level, LR OOF) | 0.4336 |
| n drugs (drug-level, ceiling) | 227 |

## Gap + Halt Gate 2 (EDA-01/02, D-02)

**Shared drug set:** 227 drugs (floor ECFP4 ∩ ceiling LINCS).

| Metric | Value |
|--------|-------|
| Floor AUROC (LR, shared set) | 0.6106 |
| Ceiling AUROC (drug-grouped OOF, shared set) | 0.4336 |
| Gap (ceiling - floor) | -0.1770 |
| 95% CI [lo, hi] | [-0.3161, -0.0326] |
| Bootstrap resamples (valid) | 10,000 / 10,000 |
| Shared-set class balance | 189 pos / 38 neg (83.3% positive) |
| Halt Gate 2 | **FIRES** (stop-and-REFRAME per D-02) |

### Interpretation caveat (read before acting on the gate)

The ceiling AUROC above is a **leakage-free, drug-grouped** estimate (StratifiedGroupKFold over compound; a drug's profiles never straddle train/test). An earlier profile-level CV inflated the ceiling via per-drug memorization (one drug carries up to 784 profiles) and is not used. Two limits bound how much this gate can say:

1. **Underpowered drug-level set.** Only 38 negative drugs in the 227-drug shared set drive a wide CI; the test has little power to resolve a small gap.
2. **Unit of analysis.** At the *profile* level (Wang/Li's published setup) the measured DE reproduces their benchmark (AUROC ~0.79-0.93), so a near-chance *drug-level* ceiling reflects the harder, drug-disjoint, small-n comparison here -- **not** an absence of measured-biology DILI signal. Treat a firing as 'inconclusive at the drug level on this set', not as a clean biological null, when deciding the D-02 reframe.

## EDA-03: Region Distinguishability (liver/human, yu2022 L5)

**Moran's I SVGs (pval_norm < 0.05):** 3909 (36.6% of 10693 model-space genes tested; Moran's I run on MultiDCP 10,716-gene subset for speed).
**SVG retention in model space:** 20/20 top SVGs in MultiDCP space (100.0%; all SVGs are in model space by construction since Moran's I was run on the model-space gene subset).
**Top-5 SVGs:** CYP3A4, CYP2E1, ADH1B, APOC1, APOC3.

### Basal similarity matrix (zone pairwise Pearson r)

| Zone | Zone1 | Zone2_1 | Zone2_2 | Zone2_3 | Zone3 | central_area | portal_area |
|------|------|------|------|------|------|------|------|
| Zone1 | 1.000 | 0.994 | 0.998 | 0.994 | 0.988 | 0.996 | 0.994 |
| Zone2_1 | 0.994 | 1.000 | 0.992 | 0.995 | 0.993 | 0.998 | 0.992 |
| Zone2_2 | 0.998 | 0.992 | 1.000 | 0.997 | 0.991 | 0.996 | 0.995 |
| Zone2_3 | 0.994 | 0.995 | 0.997 | 1.000 | 0.998 | 0.998 | 0.995 |
| Zone3 | 0.988 | 0.993 | 0.991 | 0.998 | 1.000 | 0.995 | 0.991 |
| central_area | 0.996 | 0.998 | 0.996 | 0.998 | 0.995 | 1.000 | 0.997 |
| portal_area | 0.994 | 0.992 | 0.995 | 0.995 | 0.991 | 0.997 | 1.000 |

**Spot counts per zone:** Zone1=829, Zone2_1=491, Zone2_2=345, Zone2_3=680, Zone3=555, central_area=179, portal_area=302 | unannotated=796

**Gene coverage (yu2022 Visium vs 10,716-gene MultiDCP space):** see P0_coverage.md (>99% confirmed in Phase 0).

## EDA-03: Human-Rodent Basal Concordance (liver, yu2022 human vs GSE272564 APAP0h mouse)

**Mouse liver note:** GSE272564 APAP0h (GSM8404653) -- no zone annotation available; whole-sample pseudobulk used as mouse reference (Open Q3; zone-level rodent correlation deferred).

### Cross-species correlation (ortholog-filtered)

| Metric | Value |
|--------|-------|
| Pearson r | 0.9265 |
| p-value | 0.00e+00 |
| n genes compared | 14,888 |
| Ortholog map | one-to-one (Ensembl BioMart release 116) |
| Ortholog pairs total | 15,956 |

### OOD Distance from Cancer-Line Manifold (PDG diseased baseline)

**Method:** Mahalanobis, 978-gene landmark subspace, alpha=1e-2.
**Landmark subspace:** 978-gene (PDG manifold ∩ Visium; capped at 978).

| Zone | Mahalanobis distance |
|------|---------------------|
| Zone1 | 3177.2269 |
| Zone2_1 | 6650.0649 |
| Zone2_2 | 3174.3081 |
| Zone2_3 | 4841.4636 |
| Zone3 | 5730.1684 |
| central_area | 4936.9828 |
| portal_area | 2935.2208 |


---
*Run elapsed: 181.8s*
