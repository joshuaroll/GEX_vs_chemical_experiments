---
phase: 00-dataset-acquisition-manifest
verified: 2026-06-21T00:23:07Z
status: human_needed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification: false
human_verification:
  - test: "Confirm maynard_dlpfc (spatialLIBD) whole-gene-space coverage is achievable at Phase 1"
    expected: "The full DLPFC Visium objects can be obtained via spatialLIBD/Bioconductor at Phase 1; metadata-only CSV on disk is an accepted P0 placeholder because expected_files=() and the plan documented this as a 'manifest only; full objects Plan 04' deliverable"
    why_human: "Only a metadata CSV (2.6 KB) exists on disk for the maynard_dlpfc directory; the full multi-GB Visium objects were not downloaded; the entry is usable_as_input=True and whole_transcriptome=True; a human must confirm that the P1 EDA can still gate on this or that it is accepted as a P0 partial"
  - test: "Confirm chen_brain_mtg is accepted as a DE-results-only download (not a Visium feature matrix)"
    expected: "The only file in chen_brain_mtg/ is a DESeq2 normalized expression .txt.gz — NOT a raw Visium feature_bc_matrix; however the entry is whole_transcriptome=True and usable_as_input=True; a human must confirm this dataset can still serve as a usable input or must be demoted before Phase 1"
    why_human: "The GSE200474 supplementary only contains DE results, not the raw Visium h5 matrix; the registry marks it as usable_as_input=True/whole_transcriptome=True; whether this download is sufficient for Phase 1 use is a data quality judgment, not a code check"
  - test: "Confirm DIRIL kidney TODO is accepted as a recorded gap and not a DATA-01 blocker"
    expected: "data/raw/labels/diril/DIRIL_TODO.txt records that the journal-gated supplement URL is unresolved; the decision to not fabricate and not halt was recorded in 00-02-SUMMARY.md; a human must confirm this is acceptable to proceed to Phase 1"
    why_human: "Whether DIRIL kidney supplement absence blocks DATA-01 (it is a required per-organ toxicity label per REQUIREMENTS.md) or is acceptable as a recorded TODO requires a research judgment call"
  - test: "Confirm the 3 latent code-quality bugs from 00-REVIEW.md are accepted as latent for Phase 0"
    expected: "CR-01 (Figshare article ID hardcoded, not read from entry.accession), CR-02 (KPMP non-fatal path is dead code since access_mechanism='geo_supp'), CR-03 (MD5 mismatch is non-fatal warning, not halt) are all LATENT for this run — current data is correct — but will bite if a second figshare entry is added or KPMP GEO 404s in a re-run"
    why_human: "The review classifies these as critical findings. Whether they must be fixed before Phase 1 proceeds or can be deferred is a risk-acceptance decision"
---

# Phase 0: Dataset Acquisition & MANIFEST Verification Report

**Phase Goal:** Every planned input dataset on disk, versioned, with gene-space coverage + ortholog map; no model code.
**Verified:** 2026-06-21T00:23:07Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Every planned input dataset (whole-transcriptome Visium basal, rodent spatial, APAP validation, rodent toxicogenomics, per-organ tox labels) is on disk and versioned, with corrected accessions applied | VERIFIED (with noted caveats below) | 13 slug dirs under data/raw/spatial/ totaling 16 GB; yu2022_liver (1.8 GB, md5-OK), andrews_liver (1.2 GB), lake_kpmp_kidney (4.6 GB), gse280652_apap_liver (527 MB), gse252772_mouse_kidney (4.9 GB), gse233983_mouse_brain (1.2 GB), etc.; labels: dilist.xlsx, dilirank.xlsx, sider/meddra_all_se.tsv.gz all on disk; accession corrections confirmed in registry (GSE189994/GSE144239 only in comments, not in accession fields; 22321447 present; GSE183456+GSE183279 present) |
| 2 | MANIFEST.md records paths/versions/SHAs/licenses for all datasets and frozen MultiDCP/CheMoE checkpoints, and records the squidpy env additions | VERIFIED | MANIFEST.md (9.6 KB) contains SHA256+license+Whole-transcriptome table rows for all downloaded datasets; checkpoint rows with SHA256 for best_model.pt present; squidpy 1.8.2 recorded in MANIFEST and in MANIFEST_env_snapshot.yml |
| 3 | src/spatial/orthology.py produces a human-mouse-rat one-to-one ortholog map and reports the dropped many-to-many fraction | VERIFIED | src/spatial/orthology.py (11.8 KB, 314 lines); exports build_one2one_orthologs, ortholog_report, OrthologTable; no requests import (pure library confirmed); one2one filter verified against fixture (5/10 kept, 50% dropped); data/processed/spatial/orthologs_h_m_r_one2one.tsv: 15,956 one2one pairs (15,957 lines with header); P0_orthologs.md reports n_input=219938, n_one2one=15956, dropped=92.75%; Ensembl release 116 pinned (XC-10) |
| 4 | tests/test_data_paths.py is green and per-Visium-dataset coverage against the 10,716-gene space is reported | VERIFIED | Full suite: 132 passed (0 skipped); test_data_paths.py 8/8 pass including test_whole_transcriptome_gate, test_manifest_complete, test_squidpy_available; P0_coverage.md: human datasets (yu2022_liver 99.8%, andrews_liver 99.5%, lake_kpmp_kidney 99.5%) all PASS; rodent datasets pass genome-scale gate (n_genes=32245) |

**Score:** 4/4 truths verified (with human-needed caveats on partial downloads and DIRIL TODO)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/spatial/datasets.py` | Corrected SpatialDataset registry with whole_transcriptome + slug fields | VERIFIED | 18.4 KB; 12 NamedTuple fields confirmed; 19 SpatialDataset entries; stale accessions in comments only; all 6 new fields populated |
| `tests/test_data_paths.py` | Wave-0 test suite; contains test_accessions_corrected | VERIFIED | 13.4 KB; 8 test functions, all passing including test_accessions_corrected |
| `tests/fixtures/biomart_chr21_sample.tsv` | Offline BioMart fixture >=5 rows | VERIFIED | 1.3 KB; 11 lines (header + 10 data rows with mixed orthology types) |
| `scripts/download_spatial.py` | Registry-driven download driver; entry.slug; HALT_REASON | VERIFIED | 20.9 KB; imports SPATIAL_DATASETS; 21 .slug usages; 6 HALT_REASON references; expected_files assertion present |
| `scripts/compute_sha256.py` | Streaming SHA-256 helper | VERIFIED | 2.4 KB; hashlib.sha256 present |
| `scripts/download_labels.py` | Per-organ label fetcher; HALT_REASON gate | VERIFIED | 18.1 KB; HALT_REASON gate present; SIDER-only brain labels; DIRIL TODO note written |
| `src/spatial/orthology.py` | Pure one2one ortholog builder; exports build_one2one_orthologs, ortholog_report, OrthologTable | VERIFIED | 11.8 KB; all 3 exports in __all__; no requests import |
| `scripts/build_orthologs.py` | BioMart fetch + cache; martservice; build_one2one_orthologs; HALT_REASON | VERIFIED | 11.7 KB; all key patterns confirmed |
| `data/processed/spatial/orthologs_h_m_r_one2one.tsv` | Cached one2one TSV; >=100 lines | VERIFIED | 15,957 lines (header + 15,956 pairs); 1.2 MB |
| `MANIFEST.md` | SHA256+license+Whole-transcriptome rows + checkpoint SHAs | VERIFIED | 9.6 KB; Whole-transcriptome column confirmed; best_model.pt SHA confirmed; squidpy version recorded |
| `results/tables/P0_coverage.md` | Per-Visium coverage vs 10,716 space + gate | VERIFIED | 2.3 KB; all coverage rows present; PASS/FAIL gate column; Halt Gate 1 NOT FIRED |
| `results/tables/P0_orthologs.md` | Ortholog dropped-fraction report | VERIFIED | 1.5 KB; n_input/n_one2one/dropped_fraction reported |
| `scripts/report_coverage.py` | Coverage reader for compressed archives | VERIFIED | 16.2 KB; archive-format dispatch; cross-species gate split |
| `data/raw/spatial/<slug>/` dirs (13) | All planned dataset dirs with expected files on disk | VERIFIED (partial caveats below) | 13 slug dirs confirmed; 16 GB total; yu2022_liver L5+L18 (md5 OK); lake_kpmp_kidney 4.6 GB; maynard_dlpfc has metadata only (expected_files=()); chen_brain_mtg has DE results only (not Visium matrix); see Human Verification #1 and #2 |
| `data/raw/labels/` (dilist, dilirank, sider) | Per-organ tox labels on disk | VERIFIED (diril TODO noted) | dilist.xlsx 49 KB; dilirank.xlsx 108 KB; sider/meddra_all_se.tsv.gz 2.3 MB; diril has DIRIL_TODO.txt only (journal-gated) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| tests/test_data_paths.py | src/spatial/datasets.py | from src.spatial.datasets import SPATIAL_DATASETS | VERIFIED | Import confirmed; test_accessions_corrected asserts corrected accessions; test_registry_schema asserts schema |
| tests/test_data_paths.py | src/spatial/gene_alignment.coverage_fraction | coverage_fraction() called in test_whole_transcriptome_gate | VERIFIED | coverage_fraction import present; test passes |
| scripts/download_spatial.py | src/spatial/datasets.py | imports SPATIAL_DATASETS; dispatches on access_mechanism; dirs from entry.slug | VERIFIED | 21 .slug references; SPATIAL_DATASETS import confirmed |
| scripts/build_orthologs.py | src/spatial/orthology.build_one2one_orthologs | calls pure filter after BioMart fetch | VERIFIED | build_one2one_orthologs and martservice both present |
| src/spatial/orthology.py | ortholog_one2one filter | keep rows where mouse_type == rat_type == 'ortholog_one2one' | VERIFIED | Filter confirmed correct against fixture (5/10 kept); data/processed/spatial/ TSV produced |
| results/tables/P0_coverage.md | src/spatial/gene_alignment.coverage_fraction | coverage_fraction(visium_var_names, multidcp_10716_symbols) | VERIFIED | report_coverage.py wires coverage_fraction; report confirms 99.5-99.8% for human datasets |
| MANIFEST.md | /raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt | frozen-checkpoint SHA row | VERIFIED | SHA256 row present in MANIFEST |

---

### Data-Flow Trace (Level 4)

Not applicable for Phase 0 — no components render dynamic model outputs. All artifacts are data files, registry definitions, and pure filter functions. The region_signature.py seam remains at NotImplementedError (confirmed: lines 550-554, 604-607 raise NotImplementedError for load_model and _call_model). No model inference data flows in P0.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| squidpy importable in dili_v04_env | conda run -n dili_v04_env python -c "import squidpy; print(squidpy.__version__)" | 1.8.2 | PASS |
| Full test suite green (132 tests) | conda run -n dili_v04_env python -m pytest tests/ -q | 132 passed in 7.54s | PASS |
| all 8 data-path tests pass (no skips) | conda run -n dili_v04_env python -m pytest tests/test_data_paths.py -v | 8/8 PASSED | PASS |
| ortholog filter correct on fixture | build_one2one_orthologs on 10-row fixture | n_input=10, n_one2one=5, dropped=50% | PASS |
| Stale accessions absent from accession fields | grep "GSE189994\|GSE144239" src/spatial/datasets.py | 2 matches (comments only, not accession values) | PASS |
| NotImplementedError seam untouched | grep "raise NotImplementedError" src/spatial/region_signature.py | lines 550, 604 — both raise, no model code | PASS |
| Ortholog TSV 15,956 one2one pairs | wc -l data/processed/spatial/orthologs_h_m_r_one2one.tsv | 15957 lines | PASS |
| 16 GB of spatial data on disk | du -sh data/raw/spatial/ | 16G | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| DATA-01 | Plans 01, 02 | Acquire all input datasets, versioned, on disk with corrected accessions | VERIFIED (with human caveats on maynard_dlpfc, chen_brain_mtg, DIRIL) | 13 dataset dirs on disk; accession corrections confirmed; expected_files assertions passed for all entries with expected_files defined; 3 items need human judgment (see Human Verification) |
| DATA-02 | Plans 01, 04 | MANIFEST.md + squidpy env record | VERIFIED | MANIFEST.md with SHA256+license+Whole-transcriptome; squidpy 1.8.2 in MANIFEST and env snapshot; conda env export saved |
| DATA-03 | Plans 01, 03, 04 | Gene-space coverage + ortholog map | VERIFIED | src/spatial/orthology.py pure library; data/processed/spatial/orthologs_h_m_r_one2one.tsv (15,956 pairs); P0_coverage.md with human 99.5-99.8% PASS; P0_orthologs.md with 92.75% dropped fraction |

No orphaned requirements: DATA-01/02/03 are all mapped to Phase 0 and all satisfied at the code+data level. REQUIREMENTS.md traceability table shows all three as "Pending" (table not yet updated to "Complete" — acceptable; status is in ROADMAP.md and SUMMARY.md).

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/download_spatial.py | 65, 163 | FIGSHARE_ARTICLE_ID = "22321447" hardcoded constant; never reads entry.accession | Warning (code quality) | Breaks single-source-of-truth invariant for Figshare; if a second figshare entry is added, both will silently download article 22321447. Current run is correct because only one figshare entry exists. Documented in 00-REVIEW.md CR-01. |
| src/spatial/datasets.py + scripts/download_spatial.py | datasets.py:161 | lake_kpmp_kidney access_mechanism="geo_supp" but the non-fatal KPMP handler routes on "kpmp" — dead code path | Warning (latent) | If GSE183456_RAW.tar returns 404 in a re-run, the "log and continue" path is unreachable and Halt Gate 1 fires for the whole run. Current data is on disk. Documented in 00-REVIEW.md CR-02. |
| scripts/download_spatial.py | 217-224 | MD5 mismatch on Figshare download is log.warning only; execution continues; file stays on disk | Warning (integrity) | Corrupt/truncated download is not halted. Current files passed md5 OK. Documented in 00-REVIEW.md CR-03. |
| scripts/report_coverage.py | multiple | Bare except Exception: pass in archive readers; unreadable archive indistinguishable from "format not recognized" | Warning (gate integrity) | A corrupt download silently records N/A in coverage report rather than firing Halt Gate 1. Current archives are intact. Documented in 00-REVIEW.md WR-03. |

All four are LATENT for this run (current data is correct and on disk). None of them caused a goal failure in Phase 0 because: (a) only one figshare entry exists; (b) KPMP GEO succeeded; (c) Figshare md5 was OK; (d) no corrupt archives are present. They are code-quality risks for Phase 1 re-runs or dataset additions.

---

### Human Verification Required

#### 1. maynard_dlpfc partial download acceptance

**Test:** Open data/raw/spatial/maynard_dlpfc/ and confirm that only metadata_spatialLIBD.csv (2.6 KB) is present; then confirm the full Visium objects for the DLPFC dataset will be obtainable at Phase 1 (spatialLIBD/Bioconductor) and that the metadata-only state is an accepted P0 partial.

**Expected:** Either (a) this is accepted as P0-complete because the plan documented it as "manifest only; full objects Plan 04" and expected_files=() for this entry, so no expected-file assertion applies — or (b) the full Visium objects must be downloaded before Phase 1 proceeds.

**Why human:** The maynard_dlpfc entry is usable_as_input=True and whole_transcriptome=True in the registry, but only a 2.6 KB metadata CSV is on disk. The coverage report correctly shows N/A for this entry (no readable feature matrix). The plan justified this as a Plan 04 deliverable, but Plan 04 is now complete and the full objects were not downloaded. Whether this is acceptable to Phase 1 (which needs basal gene expression for region diagnostics) is a research scope judgment.

#### 2. chen_brain_mtg DE-results-only download acceptance

**Test:** Open data/raw/spatial/chen_brain_mtg/ and confirm only GSE200474_Deseq2_normalized_gene_expression_with_annotations.txt.gz (15 MB) is present. Determine whether this DE results file can serve as a usable input for Phase 1 EDA or whether the raw Visium feature matrix must be obtained.

**Expected:** Either (a) the DE results file is sufficient for Phase 1 analysis (usable_as_input demoted to False for this entry pending Visium matrix acquisition), or (b) the raw Visium h5 matrix must be downloaded before Phase 1.

**Why human:** GSE200474 supplementary on GEO contains only the DESeq2 normalized expression file, not a raw Visium feature_bc_matrix.h5. The entry is marked usable_as_input=True but the only available download is a derived analysis output. Whether this is acceptable or requires finding an alternative source is a research judgment.

#### 3. DIRIL kidney TODO acceptance as Phase 0 gap

**Test:** Read data/raw/labels/diril/DIRIL_TODO.txt and confirm the journal-gated supplement URL for Connor 2024 kidney toxicity labels is acceptable as a recorded TODO rather than a DATA-01 blocker.

**Expected:** Either (a) this is explicitly accepted as a Phase 1 prerequisite — the kidney organ track cannot be fully processed until DIRIL is resolved — or (b) an alternative kidney label set must be identified and acquired before Phase 1 proceeds.

**Why human:** REQUIREMENTS.md DATA-01 lists "DIRIL kidney" as one of the per-organ toxicity labels to acquire. The download failed (Elsevier CDN 404 on journal supplement). The plan's recorded decision was to write a TODO note and not fabricate labels per XC-01. Whether this is an acceptable P0 terminal state or a gap that blocks Phase 1's kidney organ track is a scope and risk decision.

#### 4. CR-01/CR-02/CR-03 latent code bugs — fix before Phase 1 or accept as deferred

**Test:** Review 00-REVIEW.md findings CR-01 (Figshare hardcoded ID), CR-02 (KPMP dead-code non-fatal path), CR-03 (MD5 mismatch non-fatal) and decide whether these must be fixed before Phase 1 executes or whether they are accepted as latent risks.

**Expected:** Either (a) fixes are applied to scripts/download_spatial.py before Phase 1 proceeds (recommended to prevent re-run failures), or (b) explicit acceptance that these bugs are low-risk for Phase 1 (which does not re-download spatial data).

**Why human:** The bugs are latent — they did not affect this run's data. But CR-02 means any Phase 0 re-run where KPMP GEO returns 404 will halt the entire run unexpectedly. CR-01 means adding a second Figshare dataset silently downloads the wrong data. CR-03 means a corrupt Figshare download is not caught. Whether the code debt must be resolved before proceeding is a risk-tolerance decision.

---

## Summary

Phase 0's primary goal — datasets on disk, MANIFEST with SHA provenance, ortholog map, coverage report, no model code — is **substantively achieved**. Every measurable, code-verifiable success criterion passes:

- 13 dataset directories with 16 GB of spatial transcriptomics data on disk
- 15,956 one-to-one human-mouse-rat orthologs produced by a pure, offline-testable library
- MANIFEST.md with complete SHA256/license/Whole-transcriptome provenance for all datasets and both frozen checkpoints
- squidpy 1.8.2 installed and importable
- gene-space coverage confirmed for 3 human Visium datasets (99.5-99.8%)
- region_signature.py NotImplementedError seam untouched
- Full test suite: 132 passed

The `human_needed` status is driven by four items that require research-scope judgment rather than code verification:

1. maynard_dlpfc — only metadata CSV on disk, not Visium objects
2. chen_brain_mtg — only DE results file, not raw Visium matrix
3. DIRIL kidney — journal-gated TODO, not fabricated but not acquired
4. Three latent code-quality bugs from the code review (CR-01/02/03)

The first three items do not prevent proceeding if the plan intent is confirmed (items #1 and #2 are N/A in the coverage gate; item #3 is a recorded decision). The latent bugs (item #4) did not corrupt any data in this run. A human must confirm these are acceptable terminal states before Phase 1 begins.

---

_Verified: 2026-06-21T00:23:07Z_
_Verifier: Claude (gsd-verifier)_
