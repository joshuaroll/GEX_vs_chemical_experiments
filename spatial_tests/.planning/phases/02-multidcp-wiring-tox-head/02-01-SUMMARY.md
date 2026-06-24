---
phase: 02-multidcp-wiring-tox-head
plan: 01
subsystem: spatial-test-scaffold
tags: [tdd, nyquist, wave-0, red-scaffold, pytest, gpu-marker]
requires: []
provides:
  - tests/spatial/conftest.py (shared Phase-2 fixtures)
  - pytest.ini (gpu marker registration)
  - RED contract tests for WIRE-01/02/03 (rule-B DE, 3-vector cache, ToxHead, apap_validation)
affects:
  - 02-02 (WIRE-01 region_signature seam + cache) — flips rule-B/3-vector/N_PDG tests + gpu smoke GREEN
  - 02-03 (WIRE-02 tox_head) — flips test_tox_head.py GREEN
  - 02-04 (WIRE-03 apap_validation) — flips test_apap_validation.py GREEN
tech-stack:
  added: []
  patterns:
    - NamedTuple-output nn.Module test (mirror test_region_combiner.py)
    - ortholog-mapped intersection-flagged Pearson test (mirror test_eda_region.py)
    - duck-typed AnnData fixture (mirror test_pseudobulk.py)
    - gpu-marked real-inference smoke, imports deferred so file deselects cleanly under -m "not gpu"
key-files:
  created:
    - tests/spatial/conftest.py
    - tests/spatial/test_tox_head.py
    - tests/spatial/test_model_load.py
    - tests/spatial/test_apap_validation.py
    - pytest.ini
  modified:
    - tests/spatial/test_region_signature.py
decisions:
  - "Force RED on de_convention string, not compute_de math: current rule-A math (pt - rb) is numerically identical to rule-B (treated - control), so the executable RED signal is the manifest convention string + the 3-vector cache schema + the N_PDG import."
  - "Defer N_PDG/RegionSignatureCacher imports into test_model_load.py bodies so the gpu-marked file collects/deselects cleanly under -m 'not gpu' instead of erroring at module import (would otherwise break a combined pure run)."
metrics:
  duration: ~10 min
  completed: 2026-06-24
  tasks: 3
  files: 6
---

# Phase 2 Plan 01: Wave-0 Nyquist Test Scaffold Summary

Wrote the RED test scaffolds for every Phase-2 behavior in 02-VALIDATION.md (WIRE-01/02/03), added the shared `tests/spatial/conftest.py`, and registered the `gpu` pytest marker — locking the rule-B DE / D-03 3-vector cache / ToxHead-masking / zone-Pearson / Halt-Gate-3 contracts as executable tests before the implementation plans (02-02/03/04) run.

## What was built

Three TDD tasks, each committed atomically. All new contract tests are RED (the source they target lands in Waves 1-2), which is the correct, intended Wave-0 state. The 141 pre-existing pure tests stay GREEN.

### Task 1 — conftest + gpu marker + region_signature RED (WIRE-01) — commit 73b9304
- `tests/spatial/conftest.py`: moved the shared region-signature fixtures (`three_genes`, `two_regions`, `two_pert_ids`, `small_manifest`, `region_basal_map`) out of `test_region_signature.py` and added the new Phase-2 fixtures `synthetic_treated_control`, `gene_order_slice` (hardcoded 16-symbol synthetic — no real-data read), `zone_marker_adata` (duck-typed AnnData with planted pericentral/periportal zones), and `ortholog_table_small`.
- `pytest.ini`: registers the `gpu` marker (and `network`) so `-m "not gpu"` selects the pure suite with no unknown-marker warning.
- `test_region_signature.py`: added `TestRuleBAndThreeVectorCache` — RED on the rule-B `de_convention` string (D-02), the 3-vector cache schema `treated_array`/`control_array`/`de_array` (D-03), and `N_PDG == 10716` (D-01). Pre-existing 55 tests still GREEN.

### Task 2 — tox_head RED + gpu real-inference smoke (WIRE-02 / WIRE-01) — commit 3fa9b72
- `test_tox_head.py` (analog `test_region_combiner.py`): zero-GEX-channel → finite logit (condition A masking), combiner feed `[B,n_regions,10716]→[B,10716]→logit`, `d_dr=0` optional, `ToxHeadOutput` NamedTuple. RED on missing `src.spatial.tox_head` (lands 02-03).
- `test_model_load.py` (`pytestmark = pytest.mark.gpu`, NOT mocked, Hard Rule 1): strict 0/0 load of row-17 `best_model.pt` + one forward → finite `[10716]` in ~[0,1] (Pitfall 1). Imports of `N_PDG`/`RegionSignatureCacher` deferred into the test bodies so the file deselects cleanly under `-m "not gpu"` (2 deselected) and collects under `-m gpu` (2 tests). RED on `load_model` NotImplementedError + missing `N_PDG` (land 02-02).

### Task 3 — apap_validation RED (WIRE-03 / Halt Gate 3) — commit 5560e91
- `test_apap_validation.py` (analog `test_eda_region.py` + `test_pseudobulk.py`): zone assignment from canonical markers (D-07), measured DE = APAP_zone − ctrl_zone with mouse→human ortholog alignment reporting `n_genes_compared` and `present_mask` flag-not-zero (D-08), per-zone Pearson dict + `<2` ValueError guard, pericentral-keyed Halt-Gate-3 fire/no-fire (`<0.3`, D-08/D-09). RED on missing `src.spatial.apap_validation` (lands 02-04). Pure: `grep -c '/raid' == 0`.

## Verification

| Check | Result |
|-------|--------|
| Full pure spatial suite (excl. 3 Wave-0 scaffolds) | 141 passed, 3 expected-RED (rule-B/3-vector/N_PDG) |
| `test_model_load.py` under `-m "not gpu"` | 2 deselected (clean) |
| `test_model_load.py` under `-m gpu` | 2 collected, no marker warning |
| `test_tox_head.py` / `test_apap_validation.py` under `-m "not gpu"` | RED via collection ImportError (intended) |
| Purity gate (`grep -c '/raid'`) in pure test files | 0 |
| No production source under `src/spatial/` modified | confirmed |

## Deviations from Plan

None — plan executed as written. One implementation refinement worth noting (not a behavior change vs. the plan's intent): in `test_model_load.py` the `N_PDG`/`RegionSignatureCacher` imports were moved from module scope into the test/fixture bodies. The plan showed module-top imports, but a module-top import of the not-yet-existing `N_PDG` makes pytest error during collection even under `-m "not gpu"` (collection precedes marker deselection), which would break a combined pure-suite run. Deferring the imports keeps the gpu file cleanly deselected by the pure suite while still RED when actually run under `-m gpu`. This satisfies the must-have that `-m "not gpu"` selects the pure suite cleanly.

## Known Stubs

None. These are intentional RED test scaffolds, not stubbed production code. The targeted source symbols (rule-B `de_convention`, 3-vector cache, `N_PDG`, `ToxHead`, `apap_validation`) are owned by Waves 1-2 (plans 02-02/03/04) and are explicitly out of scope for this plan (WRITE TESTS ONLY).

## Self-Check: PASSED

Created files verified present; commits verified in git log.
