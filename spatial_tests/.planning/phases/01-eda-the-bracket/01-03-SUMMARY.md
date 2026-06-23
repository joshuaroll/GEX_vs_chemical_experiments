---
phase: 01-eda-the-bracket
plan: 03
subsystem: eda
tags: [sklearn, logistic-regression, random-forest, ecfp4, auroc, named-tuple, floor-classifier]

# Dependency graph
requires:
  - phase: 01-eda-the-bracket
    plan: 01
    provides: "RED test contract: test_floor_auroc_above_chance in tests/spatial/test_eda_floor.py"
  - phase: 01-eda-the-bracket
    plan: 02
    provides: "fingerprints.py (smiles_to_ecfp4), labels.py (load_dilirank), smiles_join.py (join_smiles_cascade)"
provides:
  - "FloorResult NamedTuple: lr_auroc, rf_auroc, label_entropy, class_balance, auprc_base_rate, n_drugs, n_drugs_with_smiles"
  - "compute_floor(fps, y, seeds): mean OOF AUROC across >=3 seeds via StratifiedKFold(5) LR + RF on ECFP4"
  - "floor_probabilities(fps, y, seed): OOF LR positive-class probability vector for paired bootstrap (plan 06)"
affects: [01-04-ceiling, 01-06-bootstrap, scripts/run_p1_eda.py]

# Tech tracking
tech-stack:
  added: [sklearn.linear_model.LogisticRegression, sklearn.ensemble.RandomForestClassifier, sklearn.model_selection.StratifiedKFold cross_val_predict, sklearn.metrics.roc_auc_score]
  patterns: [NamedTuple output schema (orthology.py convention), multi-seed mean-std AUROC (XC-06), log.info completion with quantitative counts, ValueError guard with !r formatting (T-01-05)]

key-files:
  created:
    - src/spatial/eda/floor.py
  modified: []

key-decisions:
  - "LR class_weight='balanced' + RF class_weight='balanced': required for imbalanced DILIrank (568 pos / 414 neg from ~982 total post-Ambiguous-exclusion)"
  - "n_jobs=-1 on RF to use all cores during cross_val_predict; deterministic due to fixed random_state per seed"
  - "auprc_base_rate == float(y.mean()): the PR no-skill baseline is positive prevalence by definition"
  - "floor_probabilities takes seed (int, not Sequence) to return a single aligned OOF vector for plan 06 bootstrap"
  - "fps cast to float32 inside compute_floor: sklearn LR/RF accept float32 and it halves memory vs float64 at n_drugs scale"

patterns-established:
  - "NamedTuple result schema: copy orthology.py structure (7-field FloorResult with docstring per field)"
  - "Multi-seed AUROC: log both mean and std via log.info; return mean in result"
  - "Input validation first: _validate_inputs helper called at top of every public function"

requirements-completed: [EDA-01]

# Metrics
duration: 2 min
completed: 2026-06-23
---

# Phase 01 Plan 03: Floor Classifier Summary

**LR + RF structure-only floor on ECFP4 fingerprints via StratifiedKFold cross_val_predict, with per-drug OOF probs for the paired bootstrap gap test**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-23T03:15:56Z
- **Completed:** 2026-06-23T03:18:42Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Implemented `FloorResult` NamedTuple with all 7 fields from the plan interfaces contract
- `compute_floor` runs LR + RF with StratifiedKFold(5) per seed, averages AUROC across seeds (XC-06), logs std
- `floor_probabilities` exposes aligned OOF LR probs for plan 06 paired bootstrap gap test
- All 5 acceptance criteria passed; `test_floor_auroc_above_chance` GREEN

## Task Commits

1. **Task 1: floor.py LR + RF floor on ECFP4** - `863cc4b` (feat)

## Files Created/Modified

- `src/spatial/eda/floor.py` - FloorResult NamedTuple + compute_floor + floor_probabilities; pure library, no I/O

## Decisions Made

- Used `LogisticRegression(max_iter=1000, class_weight='balanced')` to handle class imbalance and convergence on 2048-bit sparse fingerprints.
- Used `RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)` for speed; deterministic via fixed seed per fold.
- `fps` cast to float32 inside the function: sklearn handles it efficiently and halves memory vs float64 at this scale.
- `auprc_base_rate = float(y.mean())`: the PR no-skill baseline is positive prevalence by definition (no-skill classifier predicts mean everywhere).
- `floor_probabilities` accepts a single `seed: int` (not a `Sequence`) so plan 06 can pass one seed and get a single aligned OOF vector aligned to the shared drug set.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Test passed on first run (16.55s wall time for 20-drug separable synthetic matrix with 3 seeds x 5-fold CV).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `floor.py` is ready for plan 04 (ceiling), plan 05 (bootstrap input), and plan 06 (paired bootstrap gap test via `floor_probabilities`).
- Plans 04-06 (ceiling, bootstrap, region_diagnostics) still have pre-existing RED import errors from plan 01 stubs; those are the next wave of implementations.
- No blockers.

## Self-Check: PASSED

- `src/spatial/eda/floor.py`: FOUND
- Commit `863cc4b`: FOUND
- `01-03-SUMMARY.md`: FOUND

---
*Phase: 01-eda-the-bracket*
*Completed: 2026-06-23*
