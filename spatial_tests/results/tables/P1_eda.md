# Phase 1 EDA Report: The Bracket (liver/human)

**Generated:** 2026-06-24T03:31:12Z
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
| Ceiling AUROC (drug-level, leakage-free) | 0.4336 |
| n drugs (drug-level, ceiling) | 227 |

### Leakage decomposition (Phase 1 headline)

Profile-level measured-DE AUROC under a leaky split (a drug's profiles in both train and test, the Wang/Li-style setup) vs a drug-disjoint split (held-out drugs):

| Profile-level evaluation | AUROC |
|--------|-------|
| Leaky (drug in train+test; benchmark-style) | 0.9120 |
| Drug-disjoint (honest, held-out drugs) | 0.6052 |
| **Drug-leakage inflation** | **+0.3068** |

The honest profile-level measured ceiling (0.605) is comparable to the structure floor, and the leaky number (0.912) reproduces/exceeds the published Wang/Li benchmark (~0.798). The gap between them is drug-identity memorization, indicating the benchmark headline is substantially drug-leakage-inflated.

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

### Power (Hanley-McNeil SE)

| Quantity | Value |
|--------|-------|
| Floor AUROC SE | 0.0474 |
| Ceiling AUROC SE | 0.0524 |
| Min detectable gap @80% power (conservative, indep SE) | 0.1977 |
| Observed |gap| | 0.1770 |

With only 38 negative drugs the conservative minimum detectable gap (0.198) exceeds the observed |gap| (0.177); the paired-bootstrap significance comes from cancelling shared per-drug noise (same drugs in floor and ceiling) and the margin is thin.

### Interpretation caveat (read before acting on the gate)

The ceiling AUROC above is a **leakage-free, drug-grouped** estimate (StratifiedGroupKFold over compound). Three points bound the conclusion:

1. **Headline = drug leakage.** At the profile level the honest drug-disjoint measured AUROC is 0.605 vs a leaky 0.912 (+0.307 inflation) -- the Wang/Li-style benchmark is substantially drug-leakage-inflated (see EDA-02 leakage decomposition).
2. **Measured ~= structure at the fair level.** The honest profile-level ceiling (0.605) is comparable to the structure floor (0.611); the more negative drug-aggregated gap is noise from collapsing many profiles onto few drugs.
3. **Underpowered.** Only 38 negative drugs; the gate firing is marginal (see Power above). Treat this as 'measured biology adds no lift over structure, benchmark is leakage-inflated', not as a clean 'structure beats biology' result.

### Profile-level drug-disjoint head-to-head (supporting sensitivity view, D-06)

This **profile-level drug-disjoint** comparison is the **supporting, better-powered sensitivity view** (D-06). The PRIMARY gate operates on the drug-level drug-disjoint comparison above; Halt Gate 2's firing is unchanged by this row. Both numbers sit on the SAME aligned profile rows (2648 profiles from 227 drugs) and the SAME StratifiedGroupKFold drug folds (seed=42) as the ceiling's leakage decomposition, so it is a true apples-to-apples head-to-head.

| Profile-level drug-disjoint AUROC | Value |
|--------|-------|
| Structure floor (ECFP4, profile-disjoint) | 0.5461 |
| Measured ceiling (LINCS DE, profile-disjoint) | 0.6052 |
| Ceiling minus floor (profile-disjoint) | +0.0590 |

(For reference, the leaky profile-level ceiling is 0.9120; see the EDA-02 leakage decomposition.)

### Completed 2x2: floor x ceiling, leaky x disjoint (D-09)

|          | Leaky (paper's random-profile split) | Drug-disjoint (honest) |
|----------|--------------------------------------|------------------------|
| Ceiling  | 0.9120    | 0.6052 |
| Floor    | 0.9989      | 0.5461   |

All four cells sit on the SAME kept profile set (2648 profiles, 227 drugs) for consistency. **Diagnostic reading:** if floor-leaky (0.999) is approximately ceiling-leaky (0.912, ~0.912), then the Wang/Li headline (~0.798) is largely drug-identity memorization that chemical structure reproduces on its own -- measured biology adds little even in its own favorable (leaky) setup. This is the cell that "explains the paper" (D-09): the leaky column is the diagnostic, the drug-disjoint column is the fair verdict.

### Profile-level drug-disjoint gap -- 95% paired-bootstrap CI (D-09)

| Profile-level drug-disjoint gap (ceiling - floor) | Value |
|--------|-------|
| Gap (point) | +0.0590 |
| 95% CI [lo, hi] | [0.0259, 0.0921] |
| Bootstrap resamples (valid) | 10,000 / 10,000 |

Both OOF vectors come from the SAME kept profile rows (2648 profiles, 227 drugs) and the SAME StratifiedGroupKFold drug folds (seed=42), so this is a paired, leakage-free estimate of the +0.059 profile-disjoint gap. This is documentation for honesty, not a new hard gate; the PRIMARY drug-level Halt Gate 2 above is unchanged.

### Leakage-discipline guidance (D-07)

Drug-disjoint (group-aware) cross-validation is the **recommended** evaluation for every ceiling/floor/AUROC comparison across this milestone, including Phase 2+ predicted-signature evals. Enforcement is per-phase planner guidance, not a project hard rule. The +0.31 AUROC benchmark drug-leakage finding (profile-level 0.912 leaky vs 0.605 drug-disjoint) stays documented in this report (see the EDA-02 Leakage decomposition section).

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
*Run elapsed: 331.5s*
