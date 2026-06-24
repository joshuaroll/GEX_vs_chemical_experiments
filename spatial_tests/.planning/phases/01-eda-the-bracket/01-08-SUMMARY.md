---
phase: 01-eda-the-bracket
plan: 08
subsystem: eda-bracket
tags: [floor, ceiling, leakage, 2x2, paired-bootstrap, un-halt, D-08, D-09, D-10]
requires: [01-06, 01-07]
provides:
  - "floor_profile_leaky_auroc (random StratifiedKFold, NO groups, seed=42) -- the missing floor-leaky 2x2 cell"
  - "floor_profile_disjoint_oof (per-profile OOF vector matching the disjoint scalar) for paired bootstrap"
  - "completed floor x ceiling, leaky x disjoint 2x2 + 'explains the paper' diagnostic in P1_eda.md"
  - "95% paired-bootstrap CI on the profile-level drug-disjoint gap (+0.059, CI [0.0259, 0.0921])"
  - "dated RESOLUTION note in HALT_REASON.md unblocking Phase 2 (D-10)"
affects:
  - src/spatial/eda/floor.py
  - scripts/run_p1_eda.py
  - tests/spatial/test_eda_floor.py
  - results/tables/P1_eda.md
  - .planning/phases/01-eda-the-bracket/HALT_REASON.md
tech-stack:
  added: []
  patterns:
    - "TDD RED->GREEN for new pure-library functions (floor.py)"
    - "single source of truth for CV folds: floor_profile_disjoint_auroc delegates to floor_profile_disjoint_oof"
    - "kept_idx capture to slice de_aligned -> de_kept so floor/ceiling OOF align on identical rows + drug folds"
key-files:
  created:
    - .planning/phases/01-eda-the-bracket/01-08-SUMMARY.md
  modified:
    - src/spatial/eda/floor.py
    - scripts/run_p1_eda.py
    - tests/spatial/test_eda_floor.py
    - results/tables/P1_eda.md
    - .planning/phases/01-eda-the-bracket/HALT_REASON.md
decisions:
  - "D-08/D-10 honored: the halt lifts by completing the honest 2x2 documentation, regardless of gap sign; PRIMARY drug-level gate still fires, driver still exits 1."
  - "D-09 honored: floor-leaky cell added (no negative-set expansion, no gate re-powering); paired-bootstrap CI is documentation, not a new gate."
  - "floor_profile_disjoint_auroc refactored to delegate to floor_profile_disjoint_oof so scalar and OOF vector cannot drift."
metrics:
  duration: "~10 min"
  completed: 2026-06-23
  tasks: 3
  files: 5
---

# Phase 01 Plan 08: 2x2-Completion Gap-Closure (Un-Halt) Summary

Added the missing floor-leaky cell and a paired-bootstrap CI on the profile-disjoint gap to complete the floor x ceiling, leaky x disjoint 2x2, then recorded the D-10 un-halt; the PRIMARY drug-level gate still fires and the driver still exits 1 by design.

## What was built

**Task 1 (TDD RED->GREEN, `src/spatial/eda/floor.py`):**
- `floor_profile_leaky_auroc(fps, y, n_splits=5, seed=42)` -- LR on ECFP4 under a random `StratifiedKFold` profile split with NO `groups`, mirroring `_leakage_decomposition`'s leaky path exactly. The missing floor-leaky 2x2 cell (D-09).
- `floor_profile_disjoint_oof(fps, y, groups, n_splits=5, seed=42)` -- returns the per-profile OOF positive-class probability vector (float64) for the disjoint floor. `floor_profile_disjoint_auroc` now delegates to it and wraps in `roc_auc_score`, so the scalar and the OOF vector share one fold definition and cannot drift.
- Both added to `__all__`. Two regression tests added: a mirror-opposite test locking leaky-as-non-grouped (leaky AUROC > disjoint AUROC > on the per-drug-identity fixture, leaky > 0.75), and an OOF-matches-scalar test (within 1e-9). RED confirmed (ImportError), then GREEN (4 passed).

**Task 2 (`scripts/run_p1_eda.py`, additive, inside the existing 01-07 head-to-head block):**
- Imported the two new functions.
- Computed floor-leaky on the existing kept profile set (`fps_prof`, `y_kept`).
- Wrote the completed 2x2 markdown table plus a one-line diagnostic reading containing "explains the paper".
- Captured `kept_idx` in the head-to-head loop to slice `de_aligned -> de_kept`, recomputed the ceiling-disjoint OOF on exactly the kept rows + same `StratifiedGroupKFold` drug folds (seed=42), and ran `paired_bootstrap_auroc_gap` (10,000 resamples, seed=42) on the aligned floor/ceiling OOF vectors. Wrote the CI section with the "documentation for honesty, not a new hard gate" caveat.

**Task 3 (regenerate + un-halt):**
- Regenerated `results/tables/P1_eda.md` via `run_p1_eda.py all` (seed=42). Driver exited 1 by design (PRIMARY drug-level Halt Gate 2 still fires).
- Appended a dated `## RESOLUTION (2026-06-23)` note to `HALT_REASON.md` recording that the 2x2 is complete and Phase 2 (Halt Gate 3) is unblocked per D-10, with the original halt content and `gate_fires:** True` intact.

## Results (regenerated, seed=42)

Completed 2x2 on the kept profile set (2648 profiles, 227 drugs):

|          | Leaky (paper's random-profile split) | Drug-disjoint (honest) |
|----------|--------------------------------------|------------------------|
| Ceiling  | 0.9120 | 0.6052 |
| Floor    | 0.9989 | 0.5461 |

- Floor-leaky (0.999) meets/exceeds ceiling-leaky (0.912): the leaky setup is largely drug-identity memorization that chemical structure reproduces on its own. This is the cell that "explains the paper" -- the Wang/Li headline (~0.798) is inflated by drug leakage, and measured biology adds little even in its own favorable setup.
- Profile-level drug-disjoint gap (ceiling - floor) = +0.0590, 95% paired-bootstrap CI [0.0259, 0.0921], 10,000/10,000 valid resamples (documentation, not a gate).
- PRIMARY drug-aggregated gate unchanged: gap -0.1770, CI [-0.3161, -0.0326], `gate_fires=True`, driver exit 1.

## Deviations from Plan

None - plan executed exactly as written. Scope boundary held: no negative-set expansion, no gate re-powering, no SMILES re-resolution, no other organs, no model/training code; the PRIMARY gate, drug-level bootstrap, Power section, Interpretation caveat, D-07 note, EDA-02 +0.31 decomposition, `_write_halt_reason`, and `gate_fires`/`sys.exit(1)` logic are untouched.

## TDD Gate Compliance

- RED: `b7f2dc5 test(01-08): add failing tests ...` (functions absent, ImportError).
- GREEN: `02dd37e feat(01-08): add floor_profile_leaky_auroc + floor_profile_disjoint_oof` (4 tests pass).
- No separate REFACTOR commit needed; the delegate refactor of `floor_profile_disjoint_auroc` landed in the GREEN commit with the existing disjoint test still passing.

## Note on HALT_REASON.md regeneration order

`run_p1_eda.py all` rewrites `HALT_REASON.md` via `_write_halt_reason` each run (PRIMARY gate fires). The RESOLUTION note was therefore appended AFTER the Task-3 driver run, on top of freshly regenerated original halt content -- so the original analysis and `gate_fires:** True` are present verbatim and the RESOLUTION sits below them. Any future `run_p1_eda.py all` will overwrite the file and drop the appended RESOLUTION; the un-halt decision is the canonical record in `.planning/STATE.md` / `01-CONTEXT.md` (D-10), and the RESOLUTION note can be re-appended if the report is regenerated.

## Self-Check: PASSED

- src/spatial/eda/floor.py FOUND; floor_profile_leaky_auroc + floor_profile_disjoint_oof present + in __all__.
- scripts/run_p1_eda.py FOUND; 2x2, "explains the paper", CI section, kept_idx/de_kept present; parses.
- tests/spatial/test_eda_floor.py FOUND; 4 GREEN (leaky/disjoint/above_chance/oof).
- results/tables/P1_eda.md FOUND; Completed 2x2, explains the paper, 95% paired-bootstrap CI (D-09), +0.31 decomposition, 01-07 head-to-head, D-07 note, FIRES all present.
- HALT_REASON.md FOUND; RESOLUTION + 2026-06-23 + gate_fires:** True all present.
- Commits FOUND: b7f2dc5, 02dd37e, 3611f6c, 6e26a4a.
