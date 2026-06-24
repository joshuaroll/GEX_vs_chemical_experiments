---
phase: 01-eda-the-bracket
plan: 07
subsystem: eda
tags: [dili, ecfp4, group-cv, leakage, profile-disjoint, head-to-head, halt-gate, reframe, spatial]

requires:
  - phase: 01-eda-the-bracket (plans 01-06)
    provides: floor.py, ceiling.py, run_p1_eda.py driver, P1_eda.md, HALT_REASON.md, leakage decomposition
provides:
  - "src/spatial/eda/floor.py: floor_profile_disjoint_auroc (profile-level drug-disjoint structure AUROC, StratifiedGroupKFold grouped by compound, per-fold StandardScaler+LR, seed=42)"
  - "scripts/run_p1_eda.py: profile->drug ECFP4 expansion on the ceiling's aligned profile rows + D-07 guidance writer"
  - "results/tables/P1_eda.md: profile-level drug-disjoint head-to-head row (floor beside ceiling) + D-07 note"
  - "tests/spatial/test_eda_floor.py: no-drug-leakage regression test for the profile-disjoint floor"
affects: [phase-1-reframe-gap-closure, milestone-leakage-discipline]

tech-stack:
  added: []
  patterns: ["floor and ceiling on identical drug-disjoint folds at profile granularity (true head-to-head)", "drug-disjoint CV recommended milestone-wide (D-07, per-phase guidance not a hard rule)"]

key-files:
  created:
    - .planning/phases/01-eda-the-bracket/01-07-SUMMARY.md
  modified:
    - src/spatial/eda/floor.py
    - scripts/run_p1_eda.py
    - tests/spatial/test_eda_floor.py
    - results/tables/P1_eda.md

key-decisions:
  - "floor_profile_disjoint_auroc mirrors compute_ceiling exactly (StratifiedGroupKFold + per-fold StandardScaler + class-weighted LR, seed=42) so floor and ceiling sit on identical drug-disjoint folds"
  - "Profile-level drug-disjoint comparison is the SUPPORTING, better-powered sensitivity view (D-06); the PRIMARY gate stays drug-level drug-disjoint and Halt Gate 2 firing is unchanged"
  - "D-07 leakage-discipline guidance written into P1_eda.md: drug-disjoint CV recommended for every ceiling/floor/AUROC comparison milestone-wide; per-phase planner guidance, not a project hard rule"

requirements-completed: [EDA-01, EDA-03]
metrics:
  duration: ~7m
  completed: 2026-06-23
---

# Phase 1 Plan 07: Profile-level drug-disjoint floor head-to-head (D-06/D-07) Summary

Closed the single reframe gap (D-06): the structure floor now has a profile-level, drug-disjoint AUROC computed on the SAME aligned profile rows and SAME StratifiedGroupKFold drug folds the measured ceiling uses, so the profile-level sensitivity view is a true floor-vs-ceiling head-to-head; added the D-07 leakage-discipline guidance note. The halt stands.

## What was built

- **`floor_profile_disjoint_auroc(fps, y, groups, n_splits=5, seed=42)`** in `src/spatial/eda/floor.py`. Each profile row carries its drug's ECFP4 fingerprint; folds use `StratifiedGroupKFold` grouped by the lowercased compound key with a per-fold `StandardScaler` + class-weighted `LogisticRegression(max_iter=2000)` pipeline, exactly mirroring `compute_ceiling`. Reuses `_validate_inputs` and adds a `len(groups) != len(y)` guard; degenerate (<2 groups) falls back to `StratifiedKFold` like the ceiling. Returns AUROC at profile granularity (no drug aggregation). Added to `__all__`.
- **Regression test** `test_floor_profile_disjoint_no_drug_leakage` in `tests/spatial/test_eda_floor.py`, mirroring `test_ceiling_no_drug_leakage` (10 drugs, 8 profiles/drug, per-drug identity dim) and asserting AUROC < 0.75. RED (ImportError) → GREEN.
- **Driver wiring** in `scripts/run_p1_eda.py`: `run_floor` now returns `floor_fp_by_drug` (name_lower → ECFP4 row, reusing fingerprints already in scope — no SMILES re-resolution); `run_ceiling` exposes `de_aligned`, `y_profiles`, `drug_key`; `run_gap_and_gate` expands the floor to profile granularity on the ceiling's exact aligned rows and drug folds (seed=42), writes the head-to-head section and the D-07 guidance note.
- **Regenerated `results/tables/P1_eda.md`** from `run_p1_eda.py all` (seed=42).

## Head-to-head numbers produced (profile-level drug-disjoint)

| Profile-level drug-disjoint AUROC | Value |
|--------|-------|
| Structure floor (ECFP4, profile-disjoint) | 0.5461 |
| Measured ceiling (LINCS DE, profile-disjoint) | 0.6052 |
| Ceiling minus floor (profile-disjoint) | +0.0590 |

Both numbers sit on 2648 aligned profiles from 227 drugs, same StratifiedGroupKFold drug folds (seed=42) as the ceiling's leakage decomposition. (Leaky profile-level ceiling = 0.9120, for reference.) This is the supporting, better-powered sensitivity view (D-06); it does not move the PRIMARY drug-level gate.

## Halt / verdict status (unchanged)

- Halt Gate 2 still **FIRES** (stop-and-REFRAME, D-02). The driver writes `HALT_REASON.md` and exits 1 — the correct, expected outcome for this gap-closure plan.
- `HALT_REASON.md` is byte-identical (no diff); `gate_fires: True` preserved.
- The EDA-02 "Leakage decomposition (Phase 1 headline)" section and its 0.912 leaky / 0.605 disjoint / +0.31 inflation numbers are preserved (D-07 reference target).
- The PRIMARY "Gap + Halt Gate 2" section and the drug-level gate are untouched.
- Phase 2 remains BLOCKED. Executing 01-07 does NOT un-halt the phase.

## Scope boundary held

No model/training code touched, no negative-set expansion, no SMILES re-resolution, no other organs/species, and the PRIMARY gate / firing / `HALT_REASON.md` left intact — all per the plan's HARD scope boundary and 01-CONTEXT.md Deferred Ideas.

## Deviations from Plan

None - plan executed exactly as written. The new floor function reuses the existing `_validate_inputs` helper and the exact ceiling pipeline; the driver reuses the in-scope fingerprints and the ceiling's profile→drug alignment with no new data reads.

## Verification

- `pytest tests/spatial/test_eda_floor.py -k "profile_disjoint or above_chance"` — 2 passed.
- `pytest tests/spatial/test_eda_ceiling.py` — 4 passed (no regression).
- `run_p1_eda.py all` regenerates P1_eda.md, fires Halt Gate 2, writes HALT_REASON.md, exits 1 (expected).
- All plan acceptance-criteria grep/AST checks pass: `floor_profile_disjoint_auroc` (2 hits) + `floor_fp_by_drug` (4) + `"drug_key"` (2) in the driver; head-to-head + D-07 headings present in driver and report; +0.31 decomposition + FIRES preserved.

## Self-Check: PASSED

- FOUND: src/spatial/eda/floor.py (floor_profile_disjoint_auroc)
- FOUND: scripts/run_p1_eda.py (head-to-head + D-07 wiring)
- FOUND: tests/spatial/test_eda_floor.py (profile_disjoint regression test)
- FOUND: results/tables/P1_eda.md (regenerated, both new sections, FIRES preserved)
- FOUND commit a9b866b (Task 1), a609cfd (Task 2), 8fd07ee (Task 3)
