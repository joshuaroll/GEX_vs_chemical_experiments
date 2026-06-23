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

---

# Reframe session (2026-06-23, post-Halt-Gate-2)

**Date:** 2026-06-23
**Trigger:** Halt Gate 2 fired on first execution; checkpoint investigation found the measured ceiling (0.56) was a drug-leakage artifact (profile CV splitting one drug's up-to-784 profiles across train/test). Honest decomposition: profile-level measured AUROC 0.912 leaky vs 0.605 drug-disjoint (+0.31); measured ≈ structure floor (~0.61) at the fair level.
**Areas discussed:** Unit of analysis; Leakage as a result + downstream discipline
**Areas offered but not selected:** Power the gate (expand negatives); Redefine Halt Gate 2 / milestone go-no-go

## Unit of analysis (→ D-06)

| Option | Description | Selected |
|--------|-------------|----------|
| Drug-level primary, profile-level support | Gate on drug-level drug-disjoint AUROC (the unit predicted); report profile-level drug-disjoint as a powered sensitivity view. | ✓ |
| Profile-level primary | Gate on drug-disjoint profile-level AUROC (more samples but over-weights high-profile-count drugs; not the per-drug unit). | |
| Report both, gate on neither alone | Fire only if both units agree. | |

**User's choice:** Drug-level primary, profile-level support
**Notes:** Floor and ceiling must sit on the same drug-disjoint footing; planner must add the floor's profile-level-disjoint number for a true head-to-head.

## Leakage as a result + downstream discipline (→ D-07)

| Option | Description | Selected |
|--------|-------------|----------|
| Lock project-wide rule + primary writeup | Hard rule: drug-grouped CV for every comparison Phases 1-5; +0.31 finding as a primary writeup. | |
| P1 + Phase 2 only, note in P1_eda.md | Apply to P1/P2; finding documented in P1_eda.md. | |
| Guidance, not a hard rule | Document drug-disjoint as recommended guidance; enforcement left to each phase's planner. | ✓ |

**User's choice:** Guidance, not a hard rule
**Notes:** Leakage finding stays documented in P1_eda.md (leakage decomposition table); no separate primary writeup forced; not a locked hard rule.

## Deferred / flagged (reframe)
- Power the gate (expand negatives: DILIrank ∪ DILIst, relax Ambiguous, scaffold-level) — deferred to next gate review (38 negatives → underpowered).
- Redefine Halt Gate 2 / milestone go-no-go — not reframed; ⚠ open flag: honest measured ceiling ≈ structure floor, so the predicted-signature milestone bet is a steep hill (Phase 2 stays alive as the still-untested question).
