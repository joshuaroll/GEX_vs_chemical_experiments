---
phase: 00-dataset-acquisition-manifest
verified: 2026-06-22T00:00:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: 4/4
  gaps_closed:
    - "DIRIL kidney labels TODO — acquired from FDA; diril_dataset_508.xlsx on disk (21 MB)"
    - "chen_brain_mtg wrong-accession — corrected to GSE220442; real Visium counts+images.tar.gz (451 MB) on disk; filtered_feature_bc_matrix.h5 confirmed inside archive"
    - "maynard_dlpfc partial-download question — demoted to usable_as_input=False; chen_brain_mtg is the verified human brain input; no open UAT item"
    - "CR-01 Figshare hardcoded ID — fixed; article ID now parsed from entry.accession; test_cr01_figshare_id_parsed_from_accession PASSES"
    - "CR-02 KPMP dead-code non-fatal path — fixed; lake_kpmp_kidney now access_mechanism='kpmp'; test_cr02_lake_kpmp_uses_nonfatal_kpmp_mechanism PASSES"
    - "CR-03 MD5 mismatch non-fatal — fixed; mismatch now removes file and fires Halt Gate 1; test_cr03_md5_mismatch_is_fatal PASSES"
    - "Heart scope activation (2026-06-21) — kuppe_heart usable_as_input=True (4 control .h5ad, Zenodo 6578047, CC BY 4.0, 15730 genes, coverage 0.834 > 0.80); DICTrank 1318 drugs on disk"
    - "Content/shape guard implemented — src/spatial/data_validation.py enforces counts-presence; test_usable_inputs_have_counts PASSES over all 8 usable datasets"
  gaps_remaining: []
  regressions: []
---

# Phase 0: Dataset Acquisition & MANIFEST Verification Report

**Phase Goal:** Every planned input dataset is on disk, versioned, with verified gene-space coverage and a usable cross-species ortholog map — and nothing else (no model code).
**Verified:** 2026-06-22T00:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure and scope expansion (heart activated 2026-06-21)

---

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Every planned input dataset (whole-transcriptome Visium basal, rodent spatial, APAP validation, rodent toxicogenomics, per-organ tox labels) is on disk and versioned, with corrected accessions applied | VERIFIED | 8 usable_as_input=True datasets confirmed with real counts: yu2022_liver, andrews_liver, lake_kpmp_kidney, chen_brain_mtg, kuppe_heart, gse272564_mouse_liver_ctrl, gse252772_mouse_kidney (.rds), gse233983_mouse_brain. All 4 organs have tox labels: DILIst+DILIrank (liver), DIRIL 317 drugs (kidney), SIDER (brain), DICTrank 1318 drugs (heart). 17 GB spatial data on disk. test_usable_inputs_have_counts PASSES; test_raw_datasets_present PASSES |
| 2 | MANIFEST.md records paths/versions/SHAs/licenses for all datasets and frozen MultiDCP/CheMoE checkpoints, and records the squidpy env additions | VERIFIED | MANIFEST.md contains SHA256+license+Whole-transcriptome rows for all datasets (incl. kuppe_heart 4×.h5ad, DIRIL, DICTrank, SIDER, DILIst, DILIrank); both frozen checkpoint SHAs present; squidpy 1.8.2 recorded; test_manifest_complete PASSES |
| 3 | src/spatial/orthology.py produces a human-mouse-rat one-to-one ortholog map and reports the dropped many-to-many fraction | VERIFIED | src/spatial/orthology.py (314 lines); data/processed/spatial/orthologs_h_m_r_one2one.tsv 15,957 lines (15,956 one2one pairs); P0_orthologs.md: n_input=219938, n_one2one=15956, dropped=92.75%; Ensembl release 116; test_ortholog_one2one PASSES |
| 4 | tests/test_data_paths.py is green and per-Visium-dataset coverage against the 10,716-gene space is reported | VERIFIED | Full suite: 151 passed (0 failures); test_data_paths.py 8/8 PASS; test_whole_transcriptome_gate checks all usable_as_input=True Visium datasets dynamically incl. kuppe_heart (0.834 > 0.80 confirmed); P0_coverage.md present with Halt Gate 1 NOT FIRED; kuppe coverage verified by direct h5ad read (8935/10716 = 0.834) |

**Score:** 4/4 truths verified

---

### Scope Decisions Honored (Not Gaps)

The following items were raised as open questions in the previous verification. All have been resolved and are documented here as accepted scope decisions, not defects:

| Item | Decision | Evidence |
|------|----------|---------|
| Heart activated (deferral rescinded) | Heart is the 4th in-scope organ per user direction 2026-06-21 | PROJECT.md, STATE.md, REQUIREMENTS.md all updated; datasets.py kuppe_heart usable_as_input=True; test_heart_in_scope_kuppe_usable PASSES |
| DICTrank has no SMILES | Recorded known limitation; name→structure join needed at Phase 1/SPLIT phase | MANIFEST.md note: "NO SMILES column (keyed by drug/active-ingredient name -> needs structure join)"; not a DATA-01 defect |
| gse252772_mouse_kidney in .rds format | Real data, needs R→anndata conversion at Phase 1/2 | 4.9 GB on disk; 24 GSM*_obj.rds.gz members inside GSE252772_RAW.tar; MANIFEST note present |
| P0_coverage.md missing kuppe_heart row | Coverage report was generated before heart activation; coverage verified two ways: (a) test_whole_transcriptome_gate dynamically checks kuppe_heart h5ad files and PASSES, (b) direct read confirmed 0.834; stale report is documentation gap, not a gate failure | Direct h5ad read: 8935/10716 = 0.834; test passes |
| MANIFEST kuppe_heart note stale ("Heart deferred -> usable_as_input=False") | Note text was written during initial download (pre-activation); code-of-record (datasets.py) correctly has usable_as_input=True; test_heart_in_scope_kuppe_usable enforces this | datasets.py line 337: usable_as_input=True; test PASSES |
| maynard_dlpfc / abedini_kidney / canela_kidney / kanemaru_heart demoted | Intentional; documented in AUDIT notes in datasets.py and MANIFEST; chen_brain_mtg (GSE220442) is the verified human brain input | datasets.py usable_as_input=False for all four; MANIFEST notes present |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/spatial/datasets.py` | Registry with 8 usable_as_input=True entries, corrected accessions, kuppe_heart activated | VERIFIED | 18.4 KB; 8 usable entries confirmed by direct Python query; kuppe_heart usable_as_input=True; lake_kpmp_kidney access_mechanism='kpmp' (CR-02 fix); no FIGSHARE_ARTICLE_ID constant (CR-01 fix) |
| `src/spatial/data_validation.py` | Content/shape guard; dataset_has_counts + validate_usable_inputs | VERIFIED | 181 lines; functions present; enforced in download script and test suite |
| `src/spatial/orthology.py` | Pure one2one ortholog builder; exports build_one2one_orthologs, ortholog_report, OrthologTable | VERIFIED | 314 lines; all 3 exports in __all__; no I/O (pure library) |
| `tests/test_data_paths.py` | 8 data-path tests; test_whole_transcriptome_gate incl. kuppe_heart | VERIFIED | 8 passed; dynamically iterates usable_as_input=True+Visium datasets; kuppe_heart .h5ad files found and coverage checked |
| `tests/test_data_validation.py` | 19 tests incl. CR-01/02/03 regression tests + test_usable_inputs_have_counts + test_heart_in_scope_kuppe_usable + test_dictrank_labels_present_if_downloaded | VERIFIED | 19 passed; all CR regression tests PASS; test_usable_inputs_have_counts PASSES over all 8 usable datasets on disk |
| `data/raw/spatial/yu2022_liver/` | L5_upload.zip + L18_upload.zip | VERIFIED | Both on disk; SHA256 verified in MANIFEST |
| `data/raw/spatial/andrews_liver/` | GSE185477_RAW.tar | VERIFIED | On disk; 33514 genes; MANIFEST SHA present |
| `data/raw/spatial/lake_kpmp_kidney/` | GSE183456_RAW.tar | VERIFIED | 4.6 GB on disk; MANIFEST SHA present |
| `data/raw/spatial/chen_brain_mtg/` | GSE220442_counts_and_images.tar.gz (corrected from GSE200474) | VERIFIED | 451 MB on disk; 6 sections with filtered_feature_bc_matrix.h5 + matrix.mtx.gz inside; MANIFEST SHA present |
| `data/raw/spatial/kuppe_heart/` | Visium_control_P1/P7/P8/P17.h5ad (4 files) | VERIFIED | All 4 .h5ad files on disk; P1 confirmed: 4269 spots × 15730 genes, X=csr_matrix (real counts); MANIFEST SHA for all 4 |
| `data/raw/spatial/gse272564_mouse_liver_ctrl/` | GSE272564_RAW.tar | VERIFIED | 32245 genes; MANIFEST SHA present |
| `data/raw/spatial/gse252772_mouse_kidney/` | GSE252772_RAW.tar (Seurat .rds format) | VERIFIED | 4.9 GB; 24 GSM*_obj.rds.gz inside; MANIFEST notes .rds format |
| `data/raw/spatial/gse233983_mouse_brain/` | GSE233983_RAW.tar | VERIFIED | 32245 genes; MANIFEST SHA present |
| `data/raw/labels/dilist/dilist.xlsx` | DILIst liver labels | VERIFIED | 49 KB on disk; MANIFEST SHA present |
| `data/raw/labels/dilirank/dilirank.xlsx` | DILIrank liver labels | VERIFIED | 108 KB on disk; MANIFEST SHA present |
| `data/raw/labels/diril/diril_dataset_508.xlsx` | DIRIL kidney labels (317 drugs) | VERIFIED | 21 MB on disk; DIRIL_TODO.txt removed; MANIFEST SHA present; test_data_validation confirms 317 rows |
| `data/raw/labels/dictrank/dictrank_dataset_508.xlsx` | DICTrank heart labels (1318 drugs) | VERIFIED | 93 KB on disk; MANIFEST SHA present; test_dictrank_labels_present_if_downloaded PASSES (1318 rows, DICT-concern column confirmed) |
| `data/raw/labels/sider/meddra_all_se.tsv.gz` | SIDER brain SOC labels | VERIFIED | 2.3 MB on disk; MANIFEST SHA present |
| `data/processed/spatial/orthologs_h_m_r_one2one.tsv` | 15,956 one2one human-mouse-rat pairs | VERIFIED | 15,957 lines (header + 15,956 pairs); 1.2 MB; MANIFEST SHA present |
| `MANIFEST.md` | SHA256+license+Whole-transcriptome for all datasets incl. heart additions; checkpoint SHAs | VERIFIED | All rows present; kuppe_heart 4 rows; DICTrank row; DIRIL row; checkpoint SHAs; squidpy version |
| `results/tables/P0_coverage.md` | Per-dataset coverage gate evidence | VERIFIED | Present; human datasets 99.5-99.8% PASS; Halt Gate 1 NOT FIRED (kuppe_heart coverage verified via live test, not static table row) |
| `results/tables/P0_orthologs.md` | n_input/n_one2one/dropped_fraction | VERIFIED | Present; n_input=219938, n_one2one=15956, dropped=92.75% |
| `src/spatial/region_signature.py` seam | raise NotImplementedError — no model code | VERIFIED | Lines 550 and 604 raise NotImplementedError; no load_model or _call_model real implementation |
| `scripts/download_spatial.py` | CR-01/02/03 fixed; no FIGSHARE_ARTICLE_ID constant | VERIFIED | CR regression tests all PASS; test_cr01_no_hardcoded_figshare_constant PASSES |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| tests/test_data_paths.py | src/spatial/datasets.py | SPATIAL_DATASETS import | VERIFIED | Import confirmed; test_accessions_corrected and test_registry_schema pass |
| tests/test_data_paths.py | src/spatial/gene_alignment.coverage_fraction | coverage_fraction() in test_whole_transcriptome_gate | VERIFIED | Iterates all usable_as_input=True+Visium entries incl. kuppe_heart; all pass > 0.80 |
| tests/test_data_validation.py | src/spatial/datasets.py | SPATIAL_DATASETS import | VERIFIED | 19 tests pass; CR regression tests verify download_spatial.py behavior via registry |
| tests/test_data_validation.py | src/spatial/data_validation.py | dataset_has_counts, validate_usable_inputs | VERIFIED | test_usable_inputs_have_counts validates all 8 on-disk usable datasets have real count artifacts |
| scripts/download_spatial.py | src/spatial/datasets.py | entry.accession parsed for figshare ID | VERIFIED | CR-01 fixed; test_cr01_figshare_id_parsed_from_accession PASSES |
| scripts/download_spatial.py | lake_kpmp_kidney via kpmp dispatcher | access_mechanism='kpmp' | VERIFIED | CR-02 fixed; test_cr02_lake_kpmp_uses_nonfatal_kpmp_mechanism PASSES |
| MANIFEST.md | frozen checkpoints | SHA256 rows for both best_model.pt paths | VERIFIED | MultiDCP-CheMoE + MultiDCP-PDG SHA rows present |
| src/spatial/orthology.py | data/processed/spatial/orthologs_h_m_r_one2one.tsv | build_one2one_orthologs filter | VERIFIED | 15,956 pairs; Ensembl release 116 |

---

### Data-Flow Trace (Level 4)

Not applicable for Phase 0. No components render dynamic model outputs. All artifacts are data files, registry definitions, and pure filter functions. The `region_signature.py` seam remains at `NotImplementedError` (lines 550 and 604). No model inference data flows in P0 per XC-04.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite 151 passed | `conda run -n dili_v04_env python -m pytest -q` | 151 passed, 49 warnings in 9.37s | PASS |
| test_usable_inputs_have_counts | `pytest tests/test_data_validation.py::test_usable_inputs_have_counts -v` | 1 passed | PASS |
| All 19 test_data_validation.py tests | `pytest tests/test_data_validation.py -q` | 19 passed | PASS |
| All 8 test_data_paths.py tests | `pytest tests/test_data_paths.py -v` | 8 passed | PASS |
| CR-01 regression: no hardcoded Figshare ID | `pytest tests/test_data_validation.py::test_cr01_no_hardcoded_figshare_constant` | PASSED | PASS |
| CR-02 regression: KPMP uses non-fatal path | `pytest tests/test_data_validation.py::test_cr02_lake_kpmp_uses_nonfatal_kpmp_mechanism` | PASSED | PASS |
| CR-03 regression: MD5 mismatch is fatal | `pytest tests/test_data_validation.py::test_cr03_md5_mismatch_is_fatal` | PASSED | PASS |
| Heart in scope, kuppe usable | `pytest tests/test_data_validation.py::test_heart_in_scope_kuppe_usable` | PASSED | PASS |
| DICTrank 1318 drugs | `pytest tests/test_data_validation.py::test_dictrank_labels_present_if_downloaded` | PASSED | PASS |
| kuppe_heart real counts | direct anndata read Visium_control_P1.h5ad | shape (4269, 15730), X=csr_matrix | PASS |
| kuppe_heart coverage vs MultiDCP 10716 | coverage_fraction(kuppe_genes, mdcp) | 0.834 (8935/10716) > 0.80 | PASS |
| squidpy importable | `conda run -n dili_v04_env python -c "import squidpy; print(squidpy.__version__)"` | 1.8.2 | PASS |
| NotImplementedError seam untouched | `grep "raise NotImplementedError" src/spatial/region_signature.py` | lines 550, 604 | PASS |
| Ortholog TSV 15,956 pairs | `wc -l data/processed/spatial/orthologs_h_m_r_one2one.tsv` | 15957 lines | PASS |
| gse252772_mouse_kidney real .rds | tarfile member scan | 24 GSM*_obj.rds.gz | PASS |
| chen_brain_mtg real Visium counts | tarfile member scan GSE220442 | filtered_feature_bc_matrix.h5 present | PASS |
| DIRIL_TODO.txt removed | `ls data/raw/labels/diril/` | diril_dataset_508.xlsx only | PASS |
| 17 GB spatial data on disk | `du -sh data/raw/spatial/` | 17G | PASS |

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| DATA-01 | 00-01, 00-02 | Acquire all input datasets, versioned, on disk with corrected accessions; all per-organ tox labels incl. DIRIL kidney and DICTrank heart | SATISFIED | 8 usable datasets on disk with real counts (test_usable_inputs_have_counts PASSES); all 4 per-organ label sets on disk; DIRIL acquired from FDA; DICTrank acquired from FDA; REQUIREMENTS.md DATA-01 text explicitly includes "DICTrank heart [activated 2026-06-21]" |
| DATA-02 | 00-01, 00-04 | MANIFEST.md + squidpy env record | SATISFIED | MANIFEST.md with SHA256+license+Whole-transcriptome; all heart rows added; squidpy 1.8.2 in MANIFEST and MANIFEST_env_snapshot.yml; test_manifest_complete PASSES |
| DATA-03 | 00-01, 00-03, 00-04 | Gene-space coverage + ortholog map | SATISFIED | src/spatial/orthology.py; 15,956 one2one pairs; P0_coverage.md Halt Gate 1 NOT FIRED; kuppe_heart coverage verified at 0.834 > 0.80 via live test; test_ortholog_one2one PASSES; test_coverage_report_exists PASSES |

No orphaned requirements. DATA-01/02/03 are the only P0 requirements; all satisfied.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| MANIFEST.md | 42 | Kuppe heart note text says "Heart deferred -> usable_as_input=False" — stale relative to 2026-06-21 activation | Info | Documentation inconsistency only; code-of-record (datasets.py) correctly has usable_as_input=True; test enforces this; no functional impact |
| results/tables/P0_coverage.md | — | Missing kuppe_heart row (report generated before heart activation) | Info | Not a gate failure; coverage confirmed two other ways (live test_whole_transcriptome_gate PASSES; direct h5ad read = 0.834); Halt Gate 1 still NOT FIRED |

All previously-blocking anti-patterns (CR-01/02/03 and the content-guard absence) are now fixed and regression-tested. No blockers or warnings remain.

---

### Human Verification Required

None. All four previous human-needed items are resolved:

1. **maynard_dlpfc** — RESOLVED: demoted to usable_as_input=False; chen_brain_mtg (GSE220442) is the verified human brain input.
2. **chen_brain_mtg** — RESOLVED: wrong-accession fixed to GSE220442; real Visium filtered_feature_bc_matrix.h5 confirmed inside archive.
3. **DIRIL kidney** — RESOLVED: diril_dataset_508.xlsx acquired from FDA (21 MB, 317 drugs, SMILES present); DIRIL_TODO.txt removed.
4. **CR-01/02/03 latent bugs** — RESOLVED: all three fixed in scripts/download_spatial.py + src/spatial/datasets.py; regression tests added and all PASS.

The heart activation (item added in this re-verification scope) is fully code-verified: kuppe_heart is usable_as_input=True, 4 .h5ad files on disk, coverage 0.834 > 0.80 confirmed by both the live test gate and direct file read, DICTrank on disk with 1318 drugs confirmed by test. No human judgment required.

---

## Summary

Phase 0 goal is **fully achieved** across both the original scope and the 2026-06-21 scope expansion (heart activation + CR-01/02/03 fixes + content guard). All four ROADMAP success criteria pass:

1. **8 usable datasets on disk with real counts** — liver (yu2022_liver, andrews_liver), kidney (lake_kpmp_kidney), brain (chen_brain_mtg), heart (kuppe_heart, 4 .h5ad), mouse liver/kidney/brain. All 4 per-organ tox label sets present (DILIst+DILIrank, DIRIL, SIDER, DICTrank). `test_usable_inputs_have_counts` is the enforcement test.
2. **MANIFEST.md complete** — all datasets + kuppe_heart 4-file block + DICTrank row + checkpoint SHAs + squidpy version. `test_manifest_complete` passes.
3. **Ortholog map** — 15,956 human-mouse-rat one2one pairs, Ensembl release 116, 92.75% many-to-many dropped. Pure library, offline-testable.
4. **Coverage gate** — all human basal Visium > 99.5%; kuppe_heart 0.834 > 0.80; Halt Gate 1 NOT FIRED. `test_whole_transcriptome_gate` dynamically checks every usable_as_input+Visium entry including kuppe_heart.

**XC-04 intact**: `region_signature.py` raises `NotImplementedError` at lines 550 and 604 — no model code in P0.

**Test count:** 151 passed (was 132 at previous verification; +19 from test_data_validation.py CR regression and content-guard tests).

---

_Verified: 2026-06-22T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — previous status human_needed; all UAT items resolved + scope expanded_
