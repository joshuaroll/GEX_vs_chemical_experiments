---
phase: 00-dataset-acquisition-manifest
plan: "01"
subsystem: registry
tags: [datasets, registry, accession-correction, tdd, data-path-tests, biomart-fixture]
dependency_graph:
  requires: []
  provides:
    - "src/spatial/datasets.py — corrected SpatialDataset registry with 6 new fields including slug (single source of truth)"
    - "tests/test_data_paths.py — Wave-0 test suite for DATA-01/02/03 (2 pass offline, 6 skip cleanly)"
    - "tests/fixtures/biomart_chr21_sample.tsv — offline BioMart fixture with mixed orthology types"
  affects:
    - "scripts/download_spatial.py (Plan 02) — must read entry.slug, not slugify name"
    - "tests/test_data_paths.py — test_ortholog_one2one uses this fixture once orthology.py exists"
tech_stack:
  added: []
  patterns:
    - "NamedTuple extension (keyword args preserve positional safety)"
    - "Single-slug source-of-truth pattern (entry.slug shared between download driver and test)"
    - "Offline BioMart fixture for deterministic ortholog tests"
    - "Network-gated tests via needs_net / pytest.mark.skipif(TDC_NETWORK_TESTS)"
key_files:
  created:
    - "spatial_tests/src/spatial/datasets.py (full rewrite: NamedTuple extended; all entries corrected and converted to keyword args)"
    - "spatial_tests/tests/test_data_paths.py (new: 8 test functions covering DATA-01/02/03)"
    - "spatial_tests/tests/fixtures/biomart_chr21_sample.tsv (new: 10-row offline fixture with mixed orthology_type)"
  modified: []
decisions:
  - "slug field is the single source of truth for on-disk directory names — no independent slugification in download driver or test suite (prevents name-drift desync bug)"
  - "test_raw_datasets_present skips on .gitkeep-only RAW dir (checks for actual subdirectories, not just any file)"
  - "Comments in datasets.py explain corrections (GSE189994, GSE144239) without putting stale strings in accession field values — test passes on accession field not full file text"
  - "Siletti entry relabeled platform=snRNA-seq, whole_transcriptome=False, usable_as_input=False per DOC-09 correction"
  - "APAP series (GSE280652, GSE272564) added as validation-only entries (usable_as_input=False)"
  - "Rodent basal-context Visium candidates added (mouse liver/kidney/brain) with region_annotation_source note for Pitfall 5"
metrics:
  duration_minutes: 6
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 0
  completed_date: "2026-06-20"
---

# Phase 00 Plan 01: Registry Correction and Wave-0 Test Scaffold Summary

Corrected SpatialDataset registry (6 new fields, keyword-arg conversion, all four DOC-09 accession corrections applied) and created the Wave-0 test suite with an offline BioMart fixture. Tests run green offline; artifact-dependent tests skip cleanly.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Correct and extend the SpatialDataset registry | 32a394a | src/spatial/datasets.py, tests/test_data_paths.py (initial) |
| 2 | Create offline BioMart fixture | 5f61dff | tests/fixtures/biomart_chr21_sample.tsv |

## Verification Results

- `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py -q`: **2 passed, 6 skipped** (clean offline run)
- `conda run -n dili_v04_env python -m pytest tests/ -q`: **126 passed, 6 skipped** (124 pre-existing + 2 new pass)
- `grep -c "GSE189994\|GSE144239" src/spatial/datasets.py`: **2** (comments only, not accession field values — stale accessions absent from SPATIAL_DATASETS entries)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_raw_datasets_present failed when RAW dir had only .gitkeep**
- **Found during:** Task 2 verification (full test run)
- **Issue:** The skip guard `not any(RAW.iterdir())` evaluated to True because `.gitkeep` exists in `data/raw/spatial/`. The test then ran and failed on all 17 dataset slugs being absent.
- **Fix:** Changed skip condition to check for actual subdirectories: `dataset_dirs = [p for p in RAW.iterdir() if p.is_dir()]`; skip if `not dataset_dirs`.
- **Files modified:** `tests/test_data_paths.py`
- **Commit:** 32a394a (fix applied in same commit before final push)

### TDD Gate Note

The TDD flow was: (1) write test_data_paths.py with failing tests against the old registry — confirmed RED failure against `GSE189994` assertion; (2) write corrected datasets.py — confirmed GREEN. Both files were new (untracked) so they landed in a single commit (32a394a) rather than separate RED/GREEN commits. The chronological sequence was correct: test written before implementation.

## Known Stubs

None. The registry carries accurate accessions verified against live sources (Figshare API, NCBI GEO, HTTP HEAD — 2026-06-20).

The `tests/fixtures/biomart_chr21_sample.tsv` contains 10 real chr21 gene symbols with mixed orthology types. It is a reference annotation fixture, not synthetic experimental data.

The following tests skip (not stubs — they correctly await later plans):
- `test_raw_datasets_present` — awaits Plan 02 downloads
- `test_whole_transcriptome_gate` — awaits Plan 02 + gene symbol list
- `test_manifest_complete` — awaits Plan 03 MANIFEST.md
- `test_squidpy_available` — awaits `pip install squidpy` (DATA-02 task in Plan 03)
- `test_ortholog_one2one` — awaits Plan 02 `src/spatial/orthology.py`
- `test_coverage_report_exists` — awaits Plan 04 coverage computation

## Threat Flags

None. The threat mitigations from the plan's `<threat_model>` are both implemented:
- T-00-01 (Tampering): `test_accessions_corrected` is in place as a regression guard — stale accessions GSE189994/GSE144239 cannot re-enter the registry without failing CI.
- T-00-02 (Information disclosure): fixture uses only public chr21 reference gene symbols; no PII or secrets.

## Self-Check: PASSED

- `src/spatial/datasets.py`: FOUND (18-entry corrected registry with 12 NamedTuple fields)
- `tests/test_data_paths.py`: FOUND (8 test functions, all collect)
- `tests/fixtures/biomart_chr21_sample.tsv`: FOUND (11 lines = 1 header + 10 data rows)
- `32a394a`: FOUND in git log
- `5f61dff`: FOUND in git log
