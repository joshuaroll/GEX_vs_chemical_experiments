# Phase 2 Split Summary

Generated from `data/processed/dili_canonical.csv` (1118 rows) with seed=42.
_generated_at: 2026-05-06T17:29:20.500094+00:00_

## Scaffold split

Murcko scaffold split (80/10/10), stratified on `dili_binary`. Sizes: train=894, val=112, test=112.

**Class balance per slice:**

- train: 62.1% (555/894 positive)
- val: 56.2% (63/112 positive)
- test: 59.8% (67/112 positive)

**Stratified:** yes.

**Tanimoto train-test max-similarity histogram (scaffold split):**

| Tanimoto bucket | Count | Bar |
| --- | ---: | --- |
| 0.0–0.1 | 0 |  |
| 0.1–0.2 | 2 | ██ |
| 0.2–0.3 | 26 | ██████████████████████████ |
| 0.3–0.4 | 28 | ████████████████████████████ |
| 0.4–0.5 | 24 | ████████████████████████ |
| 0.5–0.6 | 13 | █████████████ |
| 0.6–0.7 | 13 | █████████████ |
| 0.7–0.8 | 6 | ██████ |
| 0.8–0.9 | 0 |  |
| 0.9–1.0 | 0 |  |

## Cluster split

Tanimoto 0.4 single-linkage cluster split (80/10/10) on Morgan FPs (radius=2, nBits=2048). Sizes: train=894, val=112, test=112.

**Class balance per slice (stratified):**

- train: 61.2% (547/894 positive)
- val: 59.8% (67/112 positive)
- test: 63.4% (71/112 positive)

**Tanimoto train-test max-similarity histogram (cluster split):**

| Tanimoto bucket | Count | Bar |
| --- | ---: | --- |
| 0.0–0.1 | 0 |  |
| 0.1–0.2 | 1 | █ |
| 0.2–0.3 | 58 | ██████████████████████████████████████████████████ |
| 0.3–0.4 | 53 | ██████████████████████████████████████████████████ |
| 0.4–0.5 | 0 |  |
| 0.5–0.6 | 0 |  |
| 0.6–0.7 | 0 |  |
| 0.7–0.8 | 0 |  |
| 0.8–0.9 | 0 |  |
| 0.9–1.0 | 0 |  |

## TDC-DILI scaffold split

`tdc.single_pred.Tox(name='DILI_Hong')` with scaffold split (TDC default 70/10/20, NOT 80/10/10 — explicit divergence).

- pytdc version: `0.4.17`
- dataset size: 475
- counts: train=332, val=47, test=96

## Three transfer slices

Slices nested inside `D_DILI_test` for per-slice transfer reporting in Phase 5 (per `02-CONTEXT.md` §"Three transfer slices structure").

- |test_in_pdg| = 88
- |test_drug_novel| = 24
- |test_drug_and_scaffold_novel| = 24

_Halt-gate input is `|test_drug_novel|`; threshold is 30 — see Halt gate evaluation section below._

## Upstream-train filter diagnostics

Option (a) leakage discipline (`upstream_filter.filter_upstream_train`): exclude any PDG drug whose Murcko scaffold is in S_test OR whose lowercased drug_name appears in DILIst-test.

- |D_PDG_total| = 6125
- |excluded_by_scaffold| = 258
- |excluded_by_pert_id| = 88 _(actual matching is by lowercased drug_name; the legacy `pert_id` label is preserved per 02-CONTEXT.md prose)_
- |intersection| = 81
- |D_PDG_train_after_exclusion| = 5860

## Halt gate evaluation

HALT-GATE FAIL: |D_DILI_test \ D_PDG| = 24 (threshold 30) -- STOP and re-discuss before Phase 3

The drug-novel slice is below the locked threshold of 30. STOP and re-discuss before Phase 3 — running upstream training without a meaningful transfer-test slice would invalidate the headline claim. See `HALT_REASON.md` in the phase directory for context.
