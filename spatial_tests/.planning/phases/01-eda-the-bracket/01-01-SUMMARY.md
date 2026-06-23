---
phase: 01-eda-the-bracket
plan: "01"
subsystem: testing
tags: [pytest, nyquist, eda, tdd, red-phase, manifest, provenance]

requires:
  - phase: 00-foundation
    provides: MANIFEST.md table structure, tests/spatial/ package, conda env dili_v04_env

provides:
  - 6 Nyquist RED test files under tests/spatial/ (Wave 0 scaffold for EDA bracket)
  - MANIFEST.md Phase 1 section with SHA256-verified provenance for all reused inputs

affects:
  - plans 01-02 through 01-06 (must turn these RED tests GREEN)
  - scripts/run_p1_eda.py (imports tested by these files)

tech-stack:
  added: []
  patterns:
    - "Nyquist test-first: test files created before library implementation (Wave 0)"
    - "monkeypatch pattern: pandas.read_excel monkeypatched to inject synthetic fixture"
    - "SimpleNamespace duck-typed AnnData and OrthologTable for test isolation"
    - "importorskip guard for optional squidpy dependency in test_eda_region.py"

key-files:
  created:
    - tests/spatial/test_eda_labels.py
    - tests/spatial/test_eda_fingerprints.py
    - tests/spatial/test_eda_floor.py
    - tests/spatial/test_eda_ceiling.py
    - tests/spatial/test_eda_bootstrap.py
    - tests/spatial/test_eda_region.py
  modified:
    - MANIFEST.md

key-decisions:
  - "tests/spatial/__init__.py already existed (empty); no action needed"
  - "All 6 SHA256s verified on-disk before writing to MANIFEST.md -- no drift detected"
  - "RED state confirmed: 6 ModuleNotFoundError on src.spatial.eda imports; 124 existing tests unaffected"
  - "importorskip('squidpy') guard used in test_eda_region.py::test_moran_returns_dataframe per plan spec"

requirements-completed: [EDA-01, EDA-02, EDA-03]

duration: 4min
completed: 2026-06-23
---

# Phase 1 Plan 1: EDA Wave-0 Nyquist Tests + MANIFEST Provenance Summary

**6 Nyquist RED test files with synthetic-only fixtures and SHA256-verified provenance for 6 reused external inputs in MANIFEST.md, gating Wave 1+ EDA library implementation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-23T03:01:31Z
- **Completed:** 2026-06-23T03:05:46Z
- **Tasks:** 2
- **Files modified:** 7 (6 created + 1 modified)

## Accomplishments

- 6 test files created with exact function names from 01-VALIDATION.md (all 10 required test functions present)
- All fixtures are in-memory synthetic (np.ndarray, pd.DataFrame, SimpleNamespace) -- no real data file reads in any test
- MANIFEST.md Phase 1 section appended with all 6 reused-input rows; SHA256s verified on-disk before writing
- Confirmed: `tests/spatial/__init__.py` already exists (empty, correct)
- Confirmed RED state: 124 existing tests collect cleanly; 6 new files fail import on `src.spatial.eda` (expected until plans 02-05)

## Task Commits

1. **Task 1: Create 7 Nyquist test files** - `bfe6799` (test)
2. **Task 2: Record reused-input provenance in MANIFEST.md** - `f0d0635` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `tests/spatial/test_eda_labels.py` - EDA-01: DILIrank binary encoding + SMILES join cascade tests
- `tests/spatial/test_eda_fingerprints.py` - EDA-01: ECFP4 shape, valid_mask, nonzero bits tests
- `tests/spatial/test_eda_floor.py` - EDA-01: floor AUROC > 0.5 smoke test on separable fingerprints
- `tests/spatial/test_eda_ceiling.py` - EDA-02: participation ratio, per-gene MI shape, ceiling >= floor AUROC
- `tests/spatial/test_eda_bootstrap.py` - EDA-02: paired bootstrap CI nonzero width + gate_fires on identical probs
- `tests/spatial/test_eda_region.py` - EDA-03: Moran SVG dict keys + cross-species ortholog filter
- `MANIFEST.md` - Phase 1 section with 6 SHA256-verified reused-input rows

## Decisions Made

- `tests/spatial/__init__.py` was already an empty file from Phase 0; plan called for creating it but it already existed correctly -- no action taken (not a deviation, the file matched the spec).
- All 6 SHA256 values computed with `sha256sum` before writing to MANIFEST.md; all matched RESEARCH.md values exactly. No drift.
- `importorskip("squidpy")` guard added to `test_moran_returns_dataframe` per plan spec; squidpy 1.8.2 is in `dili_v04_env` so this test will not be skipped in CI.

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None. SHA256 verification was clean; RED test state confirmed by `--collect-only`.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness

- Wave 0 complete: all 6 test files exist with correct function names
- MANIFEST.md provenance gate satisfied for all reused inputs
- Ready for plans 01-02 through 01-06 (EDA library implementation, Wave 1+)
- 124 pre-existing tests remain green (unaffected by this plan)

---
*Phase: 01-eda-the-bracket*
*Completed: 2026-06-23*

## Self-Check: PASSED

- `tests/spatial/test_eda_labels.py` exists: FOUND
- `tests/spatial/test_eda_fingerprints.py` exists: FOUND
- `tests/spatial/test_eda_floor.py` exists: FOUND
- `tests/spatial/test_eda_ceiling.py` exists: FOUND
- `tests/spatial/test_eda_bootstrap.py` exists: FOUND
- `tests/spatial/test_eda_region.py` exists: FOUND
- `tests/spatial/__init__.py` exists: FOUND
- Commit bfe6799 exists: FOUND (test(01-01): add 6 Nyquist RED test files)
- Commit f0d0635 exists: FOUND (feat(01-01): record Phase 1 reused-input provenance)
- MANIFEST.md contains wangli_measured_de.npy: FOUND
- MANIFEST.md contains pdg_diseased manifold: FOUND
- All 6 SHA256s in MANIFEST.md: FOUND (grep count=3+3)
- def test_dilirank_binary_encoding in test_eda_labels.py: FOUND
- def test_bootstrap_ci_nonzero_width in test_eda_bootstrap.py: FOUND
- test_eda_ceiling.py has >= 3 def test_ functions: FOUND (count=3)
- RED state confirmed: 6 import errors on src.spatial.eda (expected)
