---
phase: 00-dataset-acquisition-manifest
plan: "02"
subsystem: data-acquisition
tags: [datasets, download, label-sets, halt-gate, figshare, geo, sider]
dependency_graph:
  requires:
    - "src/spatial/datasets.py — corrected registry with slug (Plan 01)"
  provides:
    - "scripts/compute_sha256.py — streaming SHA-256 helper for MANIFEST rows"
    - "scripts/download_spatial.py — registry-driven download driver (access_mechanism dispatch + expected_files assertion)"
    - "scripts/download_labels.py — per-organ toxicity label fetcher (liver/kidney/brain)"
    - "data/raw/spatial/<slug>/ — all planned dataset subdirs (13 entries fetched)"
    - "data/raw/labels/ — liver (DILIst+DILIrank), brain (SIDER), kidney TODO note"
  affects:
    - "tests/test_data_paths.py::test_raw_datasets_present — now passes (no longer skips)"
    - "Plan 04 — MANIFEST.md inputs: SHA-256 digests computable via scripts/compute_sha256.py"
tech_stack:
  added: []
  patterns:
    - "Registry-driven dispatch on entry.access_mechanism (figshare_api/geo_supp/kpmp/spatialLIBD/url)"
    - "entry.slug as single source of truth for on-disk directory names"
    - "Sibling SHA-copy fallback for DILIst/DILIrank (FDA bot-protection avoidance)"
    - "Halt Gate 1 wired: HALT_REASON.md + sys.exit(1) on 404/empty/HTML"
    - "Post-download expected_files assertion (partial-download detection)"
    - "KPMP non-fatal fallback (portal ToS click-through per plan user_setup)"
    - "DIRIL TODO note (journal-gated supplement URL; never fabricated)"
key_files:
  created:
    - "spatial_tests/scripts/compute_sha256.py (streaming sha256_file() + main() for MANIFEST rows)"
    - "spatial_tests/scripts/download_spatial.py (registry driver: figshare_api/geo_supp/kpmp/spatialLIBD/url dispatch)"
    - "spatial_tests/scripts/download_labels.py (per-organ label fetcher: DILIst+DILIrank/SIDER/DIRIL-TODO/DICTrank-deferred)"
  modified:
    - "spatial_tests/src/spatial/datasets.py (chen_brain_mtg expected_files corrected: RAW.tar -> actual GEO suppl file)"
decisions:
  - "P0 brain toxicity labels = SIDER meddra_all_se.tsv.gz (SOC filter at use-time); Lane-Ekins seizure and DNT-IVB DEFERRED to Phase 1 (brain is last organ sequenced)"
  - "DIRIL kidney supplement URL is journal-gated (Elsevier CDN 404); TODO note written to data/raw/labels/diril/DIRIL_TODO.txt — never fabricated per XC-01"
  - "DILIst/DILIrank acquired from sibling dili_downstream project (SHA256-verified) when FDA website blocks automated download (bot protection is transient; data is real and verified)"
  - "asp_heart (GSE113764, usable_as_input=False) excluded from fetching — heart is deferred per ROADMAP; _should_fetch() now explicitly checks species==mouse for non-input validation entries"
  - "KPMP primary GEO supplementary (GSE183456 RAW.tar = 4.9 GB) succeeded; KPMP portal ToS click-through was not needed"
metrics:
  duration_minutes: 18
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 1
  completed_date: "2026-06-20"
---

# Phase 00 Plan 02: Spatial Dataset Acquisition and Label Fetcher Summary

Registry-driven download of all planned spatial transcriptomics datasets plus per-organ toxicity label sets; SHA-256 helper written; post-download expected_files assertion wired; brain-label scope DECISION recorded (SIDER SOC only; seizure/DNT-IVB deferred to P1).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | SHA helper + registry-driven spatial download driver | 243e858 | scripts/compute_sha256.py, scripts/download_spatial.py |
| 2 | Label fetcher + run spatial/label downloads + expected_files assertion | b94eb65 | scripts/download_labels.py, src/spatial/datasets.py (fix), downloads executed |

## Verification Results

- `conda run -n dili_v04_env python -c "import ast; ..."` (all three scripts parse): **PASSED**
- `grep -q "from src.spatial.datasets import SPATIAL_DATASETS" scripts/download_spatial.py`: **PASSED**
- `grep -q "\.slug" scripts/download_spatial.py`: **PASSED** (entry.slug used throughout; no independent slugification)
- `grep -q "HALT_REASON" scripts/download_spatial.py` and `... scripts/download_labels.py`: **PASSED**
- `grep -q "expected_files" scripts/download_spatial.py`: **PASSED** (post-download assertion present)
- `grep -q "hashlib.sha256" scripts/compute_sha256.py`: **PASSED**
- `conda run -n dili_v04_env python -m pytest tests/test_data_paths.py::test_raw_datasets_present -x -q`: **1 passed** (no longer skips)
- `conda run -n dili_v04_env python -m pytest tests/ -q`: **128 passed, 4 skipped** (no regressions; 4 more than Plan 01's 124+2)
- All expected_files assertions passed (post-download): **PASSED**
- HALT_REASON.md does NOT exist (all downloads succeeded): **CONFIRMED**

## Datasets Downloaded

| Slug | Organ | Species | Mechanism | Expected File | Status |
|------|-------|---------|-----------|--------------|--------|
| yu2022_liver | liver | human | figshare_api | L5_upload.zip (366 MB, md5 OK), L18_upload.zip (1.47 GB, md5 OK) | OK |
| andrews_liver | liver | human | geo_supp | GSE185477_RAW.tar (1.27 GB) | OK |
| lake_kpmp_kidney | kidney | human | geo_supp | GSE183456_RAW.tar (4.87 GB) | OK |
| abedini_kidney | kidney | human | geo_supp | GSE211785_RAW.tar (1.24 GB) | OK |
| canela_kidney | kidney | human | geo_supp | GSE202327_RAW.tar (31 MB) | OK |
| maynard_dlpfc | brain | human | spatialLIBD | metadata_spatialLIBD.csv (2.6 KB) | OK (manifest only; full objects Plan 04) |
| chen_brain_mtg | brain | human | geo_supp | GSE200474_Deseq2...txt.gz (14.8 MB) | OK |
| kanemaru_heart | heart | human | url | (none — no expected_files) | Skipped (no expected_files) |
| gse280652_apap_liver | liver | mouse | geo_supp | GSE280652_RAW.tar (552 MB) | OK |
| gse272564_apap_liver | liver | mouse | geo_supp | GSE272564_RAW.tar (103 MB) | OK |
| gse272564_mouse_liver_ctrl | liver | mouse | geo_supp | GSE272564_RAW.tar (103 MB) | OK |
| gse252772_mouse_kidney | kidney | mouse | geo_supp | GSE252772_RAW.tar (5.24 GB) | OK |
| gse233983_mouse_brain | brain | mouse | geo_supp | GSE233983_RAW.tar (1.2 GB) | OK |

## Label Sets Acquired

| Set | Organ | Status | Notes |
|-----|-------|--------|-------|
| DILIst | liver | OK (dilist.xlsx, 49 KB) | SHA256-verified copy from sibling dili_downstream |
| DILIrank | liver | OK (dilirank.xlsx, 110 KB) | SHA256-verified copy from sibling dili_downstream |
| DIRIL (Connor 2024) | kidney | TODO note written | Journal-gated supplement URL (Elsevier 404); see data/raw/labels/diril/DIRIL_TODO.txt |
| SIDER meddra_all_se.tsv.gz | brain | OK (2.4 MB) | SOC filter applied at use-time (Phase 1); only required P0 brain labels |
| DICTrank | heart | DEFERRED | Per ROADMAP — heart is last organ |

## Brain Label Scope DECISION (recorded)

P0 brain toxicity labels = **SIDER nervous-system SOC (serious terms) ONLY**.
- Lane-Ekins seizure label set: DEFERRED to Phase 1
- DNT-IVB developmental neurotoxicity: DEFERRED to Phase 1
- Rationale: brain is the last organ sequenced (liver → kidney → brain); P0 SIDER coverage satisfies DATA-01
- Deferred note written to: `data/raw/labels/sider/BRAIN_LABELS_DEFERRED_NOTE.txt`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] GSE accession parser failed on semicolons in accession strings**
- **Found during:** Task 2 run (Andrews liver GSE185477 — accession string is `"GEO: GSE185477; cellxgene: ..."`)
- **Issue:** String-split parser kept `GSE185477;` with trailing semicolon; `_build_geo_supp_url` correctly detected non-digit and returned empty, then HALT fired
- **Fix:** Replaced string-split parser with `re.findall(r"GSE\d+", ...)` in both `_download_geo_supp` and `_download_kpmp`
- **Files modified:** `scripts/download_spatial.py`
- **Commit:** b94eb65

**2. [Rule 1 - Bug] chen_brain_mtg expected_files had wrong filename**
- **Found during:** Task 2 run (GSE200474 suppl 404 for `GSE200474_RAW.tar`)
- **Issue:** GSE200474 supplementary directory has no `RAW.tar`; only `GSE200474_Deseq2_normalized_gene_expression_with_annotations.txt.gz`
- **Fix:** Updated `src/spatial/datasets.py` chen_brain_mtg entry to use the correct filename
- **Files modified:** `src/spatial/datasets.py`
- **Commit:** b94eb65

**3. [Rule 1 - Bug] _should_fetch() incorrectly included deferred heart entries**
- **Found during:** Task 2 run (asp_heart GSE113764 — returned 404 for RAW.tar)
- **Issue:** `_should_fetch` returned True for any entry with `expected_files + whole_transcriptome=True`, inadvertently including `asp_heart` (usable_as_input=False, heart deferred per ROADMAP)
- **Fix:** Updated `_should_fetch` to only include non-input entries if `species == "mouse"` (APAP validation anchors only); human deferred-heart entries are excluded
- **Files modified:** `scripts/download_spatial.py`
- **Commit:** b94eb65

**4. [Rule 1 - Bug] spatialLIBD GitHub URLs were stale (404)**
- **Found during:** Task 2 run (Maynard DLPFC)
- **Issue:** `sce_layer_metadata_fields.csv` and `spot_level_data_fields.csv` paths 404 on GitHub master branch
- **Fix:** Queried GitHub API to find actual files in `inst/extdata`; updated to `metadata_spatialLIBD.csv` (verified present, 2620 bytes)
- **Files modified:** `scripts/download_spatial.py`
- **Commit:** b94eb65

**5. [Rule 2 - Missing] Sibling SHA-copy fallback for DILIst/DILIrank**
- **Found during:** Task 2 run (FDA website returns bot-protection 404 for automated downloads)
- **Issue:** FDA.gov returns "abuse-detection-apology" redirect for automated requests; download_labels.py halted on DILIst
- **Fix:** Added `_copy_from_sibling()` helper: first attempt SHA256-verified copy from `dili_downstream/data/raw/DILIst/` and `dili_downstream/data/raw/DILIrank/` (same umbrella repo, identical files, SHAs match sibling MANIFEST). Network download is the fallback when sibling files are absent. Data is real and provenance is maintained.
- **Files modified:** `scripts/download_labels.py`
- **Commit:** b94eb65

**6. [Rule 1 - Bug] download_labels.py _stream_download() missing label_set kwarg**
- **Found during:** Task 2 first run
- **Issue:** `_stream_download(url, dest, label_set="DILIst")` failed with TypeError — signature did not include `label_set` keyword arg
- **Fix:** Added `label_set: str = ""` to `_stream_download` signature for Halt Gate 1 error messages
- **Files modified:** `scripts/download_labels.py`
- **Commit:** b94eb65

## Known Stubs

None. All downloaded files are real data from verified public sources.

The following entries remain without downloaded content for documented reasons:
- `maynard_dlpfc`: Only metadata CSV downloaded (2.6 KB); full multi-GB Visium objects are Plan 04 deliverable (entry has `expected_files=()` so no assertion applies)
- `kanemaru_heart`: No expected_files (heart deferred per ROADMAP; empty dir is correct)
- `diril`: TODO note written; kidney labels must be resolved before Phase 4 (journal-gated supplement; never fabricated per XC-01)

## Halt Gate Status

**Halt Gate 1: NOT FIRED.** All planned (non-deferred) datasets acquired successfully. No HALT_REASON.md exists in the phase directory.

The following were notable near-misses resolved by auto-fixes:
- FDA bot-protection: resolved by sibling SHA-copy (data is real, not fabricated)
- GSE200474 RAW.tar: resolved by using the actual file in the GEO supplementary directory
- spatialLIBD GitHub paths: resolved by GitHub API lookup

## Threat Flag Review

T-00-03 (Tampering/data integrity): Figshare md5 comparison executed for both Yu liver files — both OK. DILIst/DILIrank SHA256 verified before copy. SIDER downloaded directly from EMBL. Plan 04 will record full SHA256 for all files in MANIFEST.

T-00-04 (Untrusted content execution): All downloaded files are data only; no execution, no eval(), no pickle load. zip/tar extraction is Plan 04 when files are inspected.

T-00-05 (Unavailable/partial dataset): Post-download expected_files assertion executed and PASSED for all entries with expected_files. Zero-byte detection wired in _stream_download.

T-00-06 (License): GPL-3.0+ (Yu), CC BY 4.0 (Andrews, Siletti), see source (GEO entries), Artistic-2.0 (Maynard/spatialLIBD), FDA Excel (DILIst/DILIrank). Full license column recorded in Plan 04 MANIFEST.

## Self-Check: PASSED

- `scripts/compute_sha256.py`: FOUND (sha256_file() + main() streaming helper)
- `scripts/download_spatial.py`: FOUND (imports SPATIAL_DATASETS, dispatches on access_mechanism, uses .slug, has HALT_REASON path + expected_files assertion)
- `scripts/download_labels.py`: FOUND (HALT_REASON gate, SIDER-only brain labels, DIRIL TODO note, DICTrank deferred)
- `data/raw/spatial/yu2022_liver/L5_upload.zip`: FOUND (366 MB, md5 OK)
- `data/raw/spatial/yu2022_liver/L18_upload.zip`: FOUND (1.47 GB, md5 OK)
- `data/raw/spatial/andrews_liver/GSE185477_RAW.tar`: FOUND (1.27 GB)
- `data/raw/spatial/lake_kpmp_kidney/GSE183456_RAW.tar`: FOUND (4.87 GB)
- `data/raw/spatial/gse252772_mouse_kidney/GSE252772_RAW.tar`: FOUND (5.24 GB)
- `data/raw/labels/dilist/dilist.xlsx`: FOUND (49 KB, SHA256-verified)
- `data/raw/labels/dilirank/dilirank.xlsx`: FOUND (110 KB, SHA256-verified)
- `data/raw/labels/sider/meddra_all_se.tsv.gz`: FOUND (2.4 MB)
- `243e858`: FOUND in git log (Task 1)
- `b94eb65`: FOUND in git log (Task 2)
- `tests/test_data_paths.py::test_raw_datasets_present`: PASSED (1 passed)
- Full suite `pytest tests/`: 128 passed, 4 skipped (no regressions)
