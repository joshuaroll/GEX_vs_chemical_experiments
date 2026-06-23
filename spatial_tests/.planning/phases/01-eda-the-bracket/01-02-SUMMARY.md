---
phase: 01-eda-the-bracket
plan: "02"
subsystem: eda
tags: [dili, rdkit, pandas, ecfp4, smiles, label-encoding, wang-li]

requires:
  - phase: 01-eda-the-bracket
    provides: Wave-0 Nyquist RED test scaffold (test_eda_labels.py, test_eda_fingerprints.py)

provides:
  - src/spatial/eda/__init__.py (blank package init)
  - src/spatial/eda/labels.py (load_dilirank, load_dilist with Wang/Li binary encoding)
  - src/spatial/eda/smiles_join.py (join_smiles_cascade: 3-layer cascade with coverage flag)
  - src/spatial/eda/fingerprints.py (smiles_to_ecfp4: 2048-bit radius-2, RDKit-validated)

affects:
  - 01-03 (floor.py consumes labels.py + smiles_join.py + fingerprints.py)
  - 01-04 (ceiling label alignment consumes labels.py)
  - 01-05 (bootstrap consumes floor/ceiling outputs)

tech-stack:
  added: [rdkit 2024.09.4 (fingerprints), pandas (xlsx/csv I/O)]
  patterns:
    - orthology.py header pattern (from __future__ import annotations, logging.getLogger, __all__, NamedTuple)
    - gene_alignment.py coverage-warning pattern (low-coverage log.warning at < 40%)
    - V5 input validation: MolFromSmiles None -> zero-vector + valid_mask False (no raise)

key-files:
  created:
    - src/spatial/eda/__init__.py
    - src/spatial/eda/labels.py
    - src/spatial/eda/smiles_join.py
    - src/spatial/eda/fingerprints.py
  modified: []

key-decisions:
  - "LabelTable NamedTuple defined but load_dilirank/load_dilist return plain DataFrames (matches test contracts)"
  - "pos_mask[keep_mask].astype(int).values used for dili_binary assignment to avoid index alignment on filtered df"
  - "smiles_to_ecfp4 returns (fps, valid_mask) tuple; zero-vector + valid_mask=False for unparseable SMILES (T-01-03)"
  - "TDC fallback in join_smiles_cascade is a callable hook (not hard pytdc dependency); triggered only when gap > 10%"

patterns-established:
  - "EDA subpackage: blank __init__.py; callers import from submodules directly"
  - "All loaders call pd.read_excel/pd.read_csv(path, ...) so tests can monkeypatch or supply real tmp files"

requirements-completed: [EDA-01]

duration: 4min
completed: 2026-06-23
---

# Phase 01 Plan 02: EDA Data-Prep Library Summary

**EDA-01 shared data-prep library: DILIrank/DILIst Wang/Li binary encoding, 3-layer SMILES cascade join with 40% coverage flag, and RDKit ECFP4 (2048-bit radius-2) fingerprinting with V5 input validation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-23T03:08:59Z
- **Completed:** 2026-06-23T03:12:57Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `labels.py`: load_dilirank (header=1, vDILI-Concern lowercase-normalized, Ambiguous excluded, vMost/vMOST -> 1, vNo -> 0, name_lower key added) and load_dilist (trailing-space column fixed via df.columns.str.strip())
- `smiles_join.py`: 3-layer cascade (dili_canonical -> drugbank -> optional TDC callable); logs warning when SMILES coverage < 40% of input rows
- `fingerprints.py`: smiles_to_ecfp4 returning (fps (n,2048) uint8, valid_mask (n,) bool); MolFromSmiles None-check prevents crash on bad SMILES (T-01-03 mitigation)
- All 5 plan-01 Nyquist tests now GREEN (were ModuleNotFoundError in plan 01)

## Task Commits

1. **Task 1: labels.py + __init__.py** - `5d0aae5` (feat)
2. **Task 2: smiles_join.py + fingerprints.py** - `1b043d9` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `src/spatial/eda/__init__.py` - blank package init; callers import from submodules directly
- `src/spatial/eda/labels.py` - load_dilirank/load_dilist with Wang/Li binary encoding and LabelTable NamedTuple
- `src/spatial/eda/smiles_join.py` - join_smiles_cascade (3-layer, coverage-flagged) with TDC callable hook
- `src/spatial/eda/fingerprints.py` - smiles_to_ecfp4 (2048-bit radius-2, MolFromSmiles None-check)

## Decisions Made

- `load_dilirank` and `load_dilist` return plain `pd.DataFrame` (not `LabelTable`) because the test contracts call `.set_index()` and `pd.isna()` directly on the return value. `LabelTable` is defined for callers that want richer metadata.
- `pos_mask[keep_mask].astype(int).values` (not `.values` without indexing) to prevent pandas index misalignment when assigning `dili_binary` into the filtered copy.
- TDC fallback in `join_smiles_cascade` takes an optional callable rather than importing `pytdc` directly; this keeps the module pure and avoids a network call at import time.

## Deviations from Plan

None - plan executed exactly as written. Both modules were created in a single pass; the smiles_join was also created alongside labels.py (Task 1) because the test file imports it at module level and would have caused an ImportError otherwise, but this matches the plan's file scope for Task 2.

## Issues Encountered

The label test file imports `from src.spatial.eda.smiles_join import join_smiles_cascade` at module level, which meant both `labels.py` and `smiles_join.py` had to exist before the Task 1 label test could be collected. `smiles_join.py` and `fingerprints.py` were implemented before running the Task 1 test suite; all acceptance criteria were then verified per-task before committing.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- EDA-01 data-prep library complete; plans 03 (floor) and 04 (ceiling label alignment) can import from `src.spatial.eda.labels`, `src.spatial.eda.smiles_join`, and `src.spatial.eda.fingerprints`
- 5 / 10 plan-01 Nyquist tests are now GREEN; remaining 5 (floor, ceiling, bootstrap, region) will be addressed in plans 03-05
- No blockers

## Self-Check: PASSED

- `src/spatial/eda/__init__.py` exists on disk
- `src/spatial/eda/labels.py` exists on disk
- `src/spatial/eda/smiles_join.py` exists on disk
- `src/spatial/eda/fingerprints.py` exists on disk
- Commits `5d0aae5` and `1b043d9` present in git log
- 5 targeted tests pass under conda run -n dili_v04_env pytest

---
*Phase: 01-eda-the-bracket*
*Completed: 2026-06-23*
