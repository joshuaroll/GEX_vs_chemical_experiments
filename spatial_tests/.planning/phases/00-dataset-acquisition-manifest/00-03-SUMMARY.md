---
phase: 00-dataset-acquisition-manifest
plan: "03"
subsystem: orthology
tags: [orthologs, biomart, one2one, pure-library, xc-08, xc-10, tdd]
dependency_graph:
  requires:
    - "tests/fixtures/biomart_chr21_sample.tsv (Plan 01 offline fixture)"
    - "src/spatial/gene_alignment.py (pure-library idiom to mirror)"
  provides:
    - "src/spatial/orthology.py — pure OrthologTable filter (build_one2one_orthologs, ortholog_report)"
    - "scripts/build_orthologs.py — BioMart fetch + cache + write one2one TSV"
    - "data/raw/spatial/biomart/orthologs_raw_116_20260620.tsv — raw BioMart TSV (Ensembl 116)"
    - "data/raw/spatial/biomart/orthologs_raw_116_20260620.meta — provenance sidecar (XC-10)"
    - "data/processed/spatial/orthologs_h_m_r_one2one.tsv — 15956 one2one human-mouse-rat pairs"
  affects:
    - "Plan 04 (P0_orthologs.md) — calls ortholog_report(table) from orthology.py"
    - "Phase 2 (gene_alignment.py extend) — one2one table restricts cross-species gene alignment"
tech_stack:
  added: []
  patterns:
    - "Pure-library / side-effect split: orthology.py takes DataFrame/Path; scripts/ does the GET"
    - "NamedTuple result container (OrthologTable) with numpydoc Attributes block"
    - "Warn-on-high-dropped-fraction idiom (mirrors gene_alignment.py warn-on-low)"
    - "XC-10 time-leakage discipline: Ensembl release + query date baked into raw TSV filename"
    - "HALT_REASON.md exit on BioMart failure (XC-01 no-fabrication)"
key_files:
  created:
    - "spatial_tests/src/spatial/orthology.py (pure library, 314 lines)"
    - "spatial_tests/scripts/build_orthologs.py (side-effecting BioMart fetcher)"
  modified:
    - "spatial_tests/tests/test_data_paths.py (Rule 1 fix: skip guard now slug-aware)"
  data_artifacts_on_disk:
    - "data/raw/spatial/biomart/orthologs_raw_116_20260620.tsv (219938 rows, 19.1 MB, gitignored)"
    - "data/raw/spatial/biomart/orthologs_raw_116_20260620.meta (provenance sidecar)"
    - "data/processed/spatial/orthologs_h_m_r_one2one.tsv (15956 one2one pairs, gitignored)"
decisions:
  - "build_one2one_orthologs() accepts pd.DataFrame, pathlib.Path, or str — offline-testable against fixture without network"
  - "High dropped fraction (92.75%) is expected and correct: BioMart returns all human genes (including no-homolog rows); strict mutual one2one is ~15k of ~220k input rows"
  - "Ensembl release 116 confirmed from registry (2026-06-20); baked into raw TSV filename for XC-10 reproducibility"
  - "Raw BioMart TSV is gitignored (data/raw/ rule in umbrella .gitignore); MANIFEST.md in Plan 04 will record SHA256 + release"
metrics:
  duration_minutes: 12
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 1
  completed_date: "2026-06-20"
---

# Phase 00 Plan 03: Ortholog Map (BioMart one2one filter) Summary

Pure `src/spatial/orthology.py` filters a cached Ensembl BioMart TSV to mutual human-mouse-rat one-to-one orthologs with a reported dropped fraction (XC-08); side-effecting `scripts/build_orthologs.py` fetches Ensembl release 116, caches the raw TSV with date-stamped filename (XC-10), and writes 15,956 one2one pairs to `data/processed/spatial/`.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement pure src/spatial/orthology.py (TDD) | 3d7e6d6 | src/spatial/orthology.py |
| 2 | scripts/build_orthologs.py — BioMart fetch, cache, write one2one TSV | c292242 | scripts/build_orthologs.py, tests/test_data_paths.py |

## Verification Results

- `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py::test_ortholog_one2one -x -q`: **1 passed** (offline, from fixture)
- `grep -L "import requests" src/spatial/orthology.py`: **FOUND** (purity check: orthology.py has no requests import)
- `data/processed/spatial/orthologs_h_m_r_one2one.tsv`: **EXISTS, 15957 lines** (header + 15956 one2one pairs)
- `conda run -n dili_v04_env python -m pytest tests/ -q`: **127 passed, 5 skipped** (no regressions)
- Ortholog report:

| Metric | Value |
|--------|-------|
| n_input | 219938 |
| n_one2one | 15956 |
| dropped_fraction | 92.75% |

The 92.75% dropped fraction is **expected**: BioMart returns all human genes including those with no mouse/rat homolog, many-to-many pairs, and empty Ensembl IDs. The 15,956 strict mutual one2one pairs is the correct order of magnitude for human-mouse-rat ortholog discipline.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Ensembl release probe regex failed to match BioMart registry format**
- **Found during:** Task 2 — first run of build_orthologs.py returned release "unknown"
- **Issue:** The `_probe_ensembl_release()` regex pattern `_gene_ensembl_(\d+)` expected the older BioMart registry format. Ensembl 116 returns `database="ensembl_mart_116"` (not `_gene_ensembl_`).
- **Fix:** Extended regex to `_(?:mart|gene_ensembl)_(\d+)` to match both formats. Confirmed release 116 extracted correctly on second run.
- **Files modified:** `scripts/build_orthologs.py`
- **Resolution:** Re-ran script with `--release 116` to produce properly-named `orthologs_raw_116_20260620.tsv`; removed stale `orthologs_raw_unknown_20260620.*` files.

**2. [Rule 1 - Bug] test_raw_datasets_present skip guard triggered by biomart/ subdirectory**
- **Found during:** Task 2 verification (full suite run after creating data/raw/spatial/biomart/)
- **Issue:** The test's skip guard `dataset_dirs = [p for p in RAW.iterdir() if p.is_dir()]` found `data/raw/spatial/biomart/` and concluded downloads were present — causing the test to run and fail on missing dataset slugs.
- **Fix:** Changed guard to only count directories whose names match a registry slug: `[p for p in RAW.iterdir() if p.is_dir() and p.name in all_slugs]`. This preserves the original intent (skip when no dataset downloads present) while ignoring auxiliary dirs like `biomart/`.
- **Files modified:** `tests/test_data_paths.py`
- **Commit:** c292242

### TDD Gate Note

Task 1 was executed with TDD (`tdd="true"`). The RED state was a skip (not a failure) because `test_ortholog_one2one` uses `pytest.skip` on `ImportError` when `orthology.py` doesn't exist — the test is written defensively to support incremental builds. After writing `orthology.py` (GREEN), the test passed. The chronological sequence was correct: test file existed (Plan 01) before implementation.

## Known Stubs

None. `build_one2one_orthologs` receives and filters real BioMart data; `ortholog_report` computes from real filter output. No placeholder values flow to any output.

The data files are gitignored per umbrella repo `.gitignore` rules (`*/data/raw/`, `*/data/processed/`). MANIFEST.md (Plan 04) will record SHA256 + license + Ensembl release for these artifacts.

## Threat Flags

None beyond the plan's declared threat model. The mitigations are implemented:
- **T-00-07 (Tampering/silent mis-pairing):** One2one filter applied to BOTH species columns; empty Ensembl IDs dropped; `n_one2one <= unique human genes` invariant asserted in test_ortholog_one2one.
- **T-00-08 (Repudiation/reproducibility):** Ensembl release 116 + query date 2026-06-20 baked into raw TSV filename; `.meta` sidecar records URL, columns, XML query prefix.
- **T-00-09 (Elevation of privilege):** BioMart response parsed as text/TSV via pandas; never eval'd.

## Self-Check: PASSED

- `src/spatial/orthology.py`: FOUND (314 lines, no `import requests`)
- `scripts/build_orthologs.py`: FOUND (contains martservice, build_one2one_orthologs, HALT_REASON)
- `data/processed/spatial/orthologs_h_m_r_one2one.tsv`: FOUND (15957 lines on disk, gitignored)
- `3d7e6d6`: FOUND in git log
- `c292242`: FOUND in git log
