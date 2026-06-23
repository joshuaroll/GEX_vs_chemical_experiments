# Phase 1: EDA (the bracket) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-22
**Phase:** 1-EDA (the bracket)
**Areas discussed:** P1 scope, Halt Gate 2, structure-only floor, liver label set, measured-biology ceiling, gap threshold, SMILES join, ceiling source

---

## P1 organ/species scope

| Option | Description | Selected |
|--------|-------------|----------|
| Liver-human first only | Full bracket for liver-human now; other organs + rodent deferred to their training phases. Matches per-organ halt gate. | ✓ |
| All 4 organs, human first | Brackets for liver/kidney/brain/heart (human), rodent deferred. | |
| All 4 organs, both species | Full bracket for every organ × {human, rodent} in one pass. | |

**User's choice:** Liver-human first only
**Notes:** Cross-species *liver* basal diagnostics stay in scope (need no rodent labels); only non-liver organs and rodent labels/floor/ceiling are deferred.

---

## Halt Gate 2 operationalization + negative-result stance

| Option | Description | Selected |
|--------|-------------|----------|
| Numeric gap + stop-and-reframe | Fixed AUROC threshold; fired gate → reframe (negative publishable). | |
| Numeric gap + stop-and-consult | Fixed threshold; fired gate halts for professor sign-off. | |
| Report gap, no hard gate | Advisory gap only, never auto-halt. | |

**User's choice:** Numeric gap + stop-and-reframe
**Notes:** Confirms and locks ROADMAP Proposed Decision #3 (negative result is publishable). Refined to a bootstrap-CI gate (see "Gap threshold" below).

---

## Structure-only floor featurization

| Option | Description | Selected |
|--------|-------------|----------|
| ECFP4/Morgan fingerprints | Morgan (ECFP4, 2048-bit) → logistic + RF. Transparent, reproduces Wang/Li convention. | ✓ |
| Reuse a frozen encoder embedding | ChemBERTa/MolFormer embedding as floor features. | |
| Both, report side by side | Run floor with both featurizations. | |

**User's choice:** ECFP4/Morgan fingerprints
**Notes:** SMILES join reuses the sibling v0.5 resolved tables before any fresh resolution.

---

## Liver label set + SMILES join

| Option | Description | Selected |
|--------|-------------|----------|
| DILIrank + TDC/PubChem SMILES | DILIrank primary; SMILES via TDC/PubChem; DILIst secondary. | ✓ |
| DILIst + TDC/PubChem SMILES | DILIst primary; DILIrank as stricter cross-check. | |
| Both, report each | Run bracket for both label sets. | |

**User's choice:** DILIrank + TDC/PubChem SMILES
**Notes:** DILIrank gates; DILIst reported as expanded-coverage cross-check. SMILES reuse `dili_canonical.csv` / `dilist_smiles_resolved.csv` / `drugbank_smiles_index.csv` from the sibling project first.

---

## Gap threshold + metric

| Option | Description | Selected |
|--------|-------------|----------|
| AUROC gap < 0.02 | Fixed cut mirroring the parent TRAIN gate granularity. | |
| AUROC gap < 0.05 | More conservative fixed cut. | |
| AUROC gap with bootstrap CI | Gate fires if 95% bootstrap CI of the gap includes 0. | ✓ |

**User's choice:** AUROC gap with bootstrap CI
**Notes:** 95% paired bootstrap, default 10,000 resamples (P4 convention). Statistically clean go/no-go.

---

## Measured-biology ceiling source

| Option | Description | Selected |
|--------|-------------|----------|
| Acquire LINCS+TG, liver-anchored | Pull measured signatures; ceiling where they exist, mark the rest "floor only". | (direction) |
| Reuse v0.5 Wang/Li LINCS | Reuse the 6,000 curated measured DE profiles from dili_downstream as the human liver ceiling. | ✓ |
| Fresh LINCS L1000 pull | Acquire LINCS independently. | |

**User's choice:** Reuse v0.5 Wang/Li LINCS (`wangli_measured_de.npy` + `wangli_profiles.csv`)
**Notes:** Record as external input in this project's MANIFEST with SHA/provenance. Rodent toxicogenomics ceiling deferred; organs with no measured data reported as "no measured ceiling — floor only".

---

## Claude's Discretion

- OOD-distance method (Mahalanobis vs kNN).
- Exact bootstrap resample count if 10,000 is too slow (floor 2,000).
- Morgan fingerprint bit length / radius if 2048/r2 underperforms.
- Which annotation field in the liver basal `.h5ad` supplies the published region labels.

## Deferred Ideas

- Kidney / brain / heart EDA brackets — at each organ's training phase (heart last).
- Rodent structure-floor + rodent labels — rodent pass.
- Rodent toxicogenomics ceiling (Open TG-GATEs / DrugMatrix) — rodent pass.
