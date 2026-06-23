---
phase: 01-eda-the-bracket
plan: 04
subsystem: eda
tags: [ceiling, bootstrap, auroc, participation-ratio, mutual-information, halt-gate]

requires:
  - phase: 01-eda-the-bracket
    plan: 01-01
    provides: "Nyquist RED test contracts for ceiling and bootstrap"
  - phase: 01-eda-the-bracket
    plan: 01-02
    provides: "labels.py and shared data-prep patterns"
  - phase: 01-eda-the-bracket
    plan: 01-03
    provides: "floor.py with floor_probabilities for paired bootstrap alignment"

provides:
  - "src/spatial/eda/ceiling.py: load_ceiling, participation_ratio, per_gene_mi, compute_ceiling (drug-level AUROC)"
  - "src/spatial/eda/bootstrap.py: BootstrapResult NamedTuple + paired_bootstrap_auroc_gap (10,000 resamples)"
  - "EDA-02 upper bracket: measured-biology ceiling with effective rank, MI diagnostics, and gate decision"

affects:
  - "01-05 (region diagnostics): ceiling.py patterns"
  - "01-06 (run script): calls load_ceiling, compute_ceiling, floor_probabilities, paired_bootstrap_auroc_gap"

tech-stack:
  added: []
  patterns:
    - "participation_ratio via SVD (np.linalg.svd, full_matrices=False); lambdas = s**2; drop zeros"
    - "per_gene_mi via mutual_info_classif(discrete_features=False, n_neighbors=3)"
    - "Drug-level aggregation (Pitfall 8): profile OOF probs grouped by compound_name, mean taken before AUROC"
    - "BootstrapResult NamedTuple mirrors OrthologTable pattern from orthology.py"
    - "Degenerate resamples skipped with NaN sentinel, dropped before percentile CI"

key-files:
  created:
    - "src/spatial/eda/ceiling.py"
    - "src/spatial/eda/bootstrap.py"
  modified: []

key-decisions:
  - "Drug-level aggregation in compute_ceiling: profile-level OOF probs computed first, then mean per drug before roc_auc_score (Pitfall 8 compliance)"
  - "gate_fires = ci_lower <= 0 (not ci_lower < 0): includes 0 fires the gate per D-02 spec"
  - "NaN-sentinel approach for degenerate resamples: gaps[i] = nan, then valid_gaps = gaps[~np.isnan(gaps)] -- avoids conditional append overhead in inner loop"
  - "load_ceiling raises FileNotFoundError (not caught) so calling script can report 'no measured ceiling -- floor only' per D-05 honesty rule"

patterns-established:
  - "Pitfall-8 pattern: always aggregate profile -> drug before AUROC; compute_ceiling handles this internally"
  - "Halt Gate 2 operationalization: BootstrapResult.gate_fires is the machine-readable gate signal for plan 06"

requirements-completed: [EDA-02]

duration: 3min
completed: 2026-06-23
---

# Phase 1 Plan 4: Measured-Biology Ceiling and Paired Bootstrap Gap Test Summary

**Participation-ratio ceiling (SVD effective rank + per-gene MI + drug-level AUROC) and 10,000-resample paired bootstrap CI operationalizing Halt Gate 2, pure libraries, 5/5 tests green.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-23T03:21:29Z
- **Completed:** 2026-06-23T03:24:52Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `ceiling.py` delivers `participation_ratio` (SVD-based effective rank), `per_gene_mi` (sklearn `mutual_info_classif`), and `compute_ceiling` (drug-level LR OOF -> AUROC), satisfying EDA-02.
- `bootstrap.py` delivers `BootstrapResult` NamedTuple and `paired_bootstrap_auroc_gap` (10,000 resamples, percentile CI, NaN-skip for degenerate draws), operationalizing Halt Gate 2 via `gate_fires`.
- Both T-01-06 threat mitigations active: `load_ceiling` raises `ValueError` on row-count mismatch; `paired_bootstrap_auroc_gap` raises `ValueError` on mismatched array lengths.

## Task Commits

Each task was committed atomically:

1. **Task 1: ceiling.py** - `b2a0042` (feat)
2. **Task 2: bootstrap.py** - `4ce55f2` (feat)

**Plan metadata:** (committed below)

## Files Created/Modified

- `src/spatial/eda/ceiling.py` - `load_ceiling`, `participation_ratio`, `per_gene_mi`, `compute_ceiling`
- `src/spatial/eda/bootstrap.py` - `BootstrapResult`, `paired_bootstrap_auroc_gap`

## Decisions Made

- Drug-level aggregation happens inside `compute_ceiling`: profile OOF probs are computed first at profile level by LogisticRegression + StratifiedKFold, then grouped by `compound_name` (lowercase) and averaged. AUROC is computed on the aggregated drug-level probs. This keeps the Pitfall-8 compliance transparent to callers.
- `gate_fires = ci_lower <= 0` uses `<=` not `<` to fire on exact-zero lower bound, matching D-02 ("CI includes 0").
- NaN-sentinel pattern for degenerate bootstrap resamples: assign `nan` in-loop, filter with `~np.isnan` after the loop. Avoids conditional list appends inside the tight loop.
- `load_ceiling` does not catch `FileNotFoundError` -- it propagates to the caller so `run_p1_eda.py` can print "no measured ceiling -- floor only" per the D-05 honesty rule.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `ceiling.py` and `bootstrap.py` are ready for plan 01-05 (region diagnostics) and plan 01-06 (run script).
- Plan 01-06 calls: `load_ceiling`, `compute_ceiling`, `floor_probabilities` (from `floor.py`), `paired_bootstrap_auroc_gap` -- all available.
- The `ceiling_probs` dict returned by `compute_ceiling` is keyed by drug name (lowercase), making plan-06 shared-drug-set alignment straightforward.

---
*Phase: 01-eda-the-bracket*
*Completed: 2026-06-23*

## Self-Check: PASSED

Files verified on disk:
- `src/spatial/eda/ceiling.py`: FOUND
- `src/spatial/eda/bootstrap.py`: FOUND

Commits verified in git log:
- b2a0042: FOUND
- 4ce55f2: FOUND

Acceptance criteria re-verified:
- `grep -c "mutual_info_classif" ceiling.py` = 5 (>= 1): PASS
- `grep -Ec "groupby|drug.level|compound_name" ceiling.py` = 13 (>= 1): PASS
- `grep -c "class BootstrapResult" bootstrap.py` = 1 (>= 1): PASS
- `grep -c "gate_fires" bootstrap.py` = 10 (>= 1): PASS
- `grep -Ec "10_000|10000" bootstrap.py` = 1 (>= 1): PASS
- `grep -c "percentile" bootstrap.py` = 8 (>= 1): PASS
- All 5 tests green: PASS
