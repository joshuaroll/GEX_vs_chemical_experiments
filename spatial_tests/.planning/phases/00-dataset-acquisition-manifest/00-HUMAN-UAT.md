---
status: resolved
phase: 00-dataset-acquisition-manifest
source: [00-VERIFICATION.md]
started: 2026-06-21T00:26:01Z
updated: 2026-06-21T00:26:01Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. maynard_dlpfc partial download
expected: Full Maynard DLPFC Visium objects on disk, OR explicit acceptance that only the 2.6 KB metadata CSV is needed for P0. Entry is `usable_as_input=True` with `expected_files=()`, so no assertion fired. Decide: accepted P0 partial, or pull full objects before Phase 1.
result: RESOLVED 2026-06-21 (soft-entry audit) — demoted to usable_as_input=False; only metadata on disk, and chen_brain_mtg (GSE220442) is now the verified human brain Visium input, so Maynard is not needed as a P0 input. Promote + fetch full spatialLIBD objects later if a second cortical reference is wanted. See the soft-entry audit in the Gaps section below.

### 2. chen_brain_mtg DE-results-only
expected: Confirm whether the DESeq2 normalized expression .txt.gz (15 MB) serves Phase 1, or whether the raw Visium feature matrix must be located. Entry is `usable_as_input=True`, `whole_transcriptome=True`.
result: RESOLVED 2026-06-21 — root cause was a WRONG-ACCESSION error, not a partial download. Prior GSE200474 is "Neurofilament accumulations in ALS patients' motor neurons" (bulk RNA-seq of iPSC motor neurons, GPL11154), unrelated to Chen MTG Visium. Located + applied the correct accession GSE220442 ("Spatially resolved transcriptomics ... vulnerability of MTG in AD", 10x Visium, GPL24676). Registry + MANIFEST updated; GSE220442_counts_and_images.tar.gz (451 MB, sha256 9c54b560...) downloaded and verified: 6 Space Ranger Visium sections (3 AD + 3 control), 36,601 genes, per-spot layer annotations (Layer 1–6 + White Matter) in GSE220442_metadata.csv.gz. 8/8 data-path tests pass.

### 3. DIRIL kidney labels TODO
expected: DIRIL kidney toxicity labels acquired, OR accepted as a recorded gap. Currently `data/raw/labels/diril/DIRIL_TODO.txt` exists; labels were not fetched (journal-gated Elsevier CDN). REQUIREMENTS.md DATA-01 lists DIRIL as required.
result: RESOLVED 2026-06-21 — ACQUIRED from the FDA, the gold public source (not journal-gated). Root cause of the prior failure: the driver's Elsevier-CDN candidate used the wrong article S-number (S1359644623003495). FDA hosts "Drug-Induced Renal Injury List (DIRIL) Dataset" = diril_dataset_508.xlsx (21.4 MB, sha256 602edfda...), sheet "A. DIRIL (317)": 317 drugs with SMILES + binary DIRI label ("My Findings (Toxicity)": 171 Nephrotoxic / 146 Non-Nephrotoxic, zero NaN). On disk at data/raw/labels/diril/diril_dataset_508.xlsx; DIRIL_TODO.txt removed. download_labels.py updated (FDA URL primary, correct dest filename, clears stale TODO, success on re-run). MANIFEST row updated. DATA-01 kidney label requirement now satisfied.

### 4. Latent download-driver bugs (00-REVIEW.md CR-01/02/03)
expected: Decide whether the 3 latent integrity bugs (CR-01 hardcoded Figshare article ID; CR-02 KPMP non-fatal path is dead code; CR-03 MD5 mismatch non-fatal) must be fixed before Phase 1, or accepted as deferred risk. All are LATENT for this run — the data on disk is correct.
result: RESOLVED 2026-06-21 — all three fixed in scripts/download_spatial.py + src/spatial/datasets.py. CR-01: article id now parsed from entry.accession via _figshare_article_id() (halts if unparseable; no hardcoded constant). CR-02: lake_kpmp_kidney access_mechanism geo_supp->kpmp, so a DUA/ToS-gated 404 logs the atlas.kpmp.org click-through and CONTINUES instead of aborting the run; content guard updated to skip empty (ToS-gated) dirs. CR-03: figshare md5 mismatch now removes the bad file and fires Halt Gate 1 (never accepts corrupt/substituted bytes). Regression tests added (test_data_validation.py: test_cr01_*, test_cr02_*, test_cr03_*). 149/149 tests pass.

## Summary

total: 4
passed: 4
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

### Soft-entry + input-counts audit (2026-06-21)
Triggered by the chen_brain_mtg wrong-accession finding. Audited all `expected_files=()` entries and content-verified every `usable_as_input=True` dataset on disk (counts present, not just images). Failure mode found in 3 forms: wrong study (chen, fixed), images-only tar (abedini), metadata-only (maynard). `expected_files` checks existence, never content/shape — the systemic hole.

Fixes applied (registry + MANIFEST):
- **abedini_kidney** → usable_as_input=False. GSE211785_RAW.tar is images(.tif)+spatial(.json) only; counts ship separately as GSE211785_EXPORT_ST_counts.rds.gz (not fetched). lake_kpmp is the verified primary kidney input (real Visium counts on disk, nested per-sample .h5).
- **canela_kidney** → usable_as_input=False. Long-read isoform data (.bb/SQANTI .gtf), no standard Visium matrix.
- **maynard_dlpfc** → usable_as_input=False (resolves UAT item 1). Metadata-only on disk; chen_brain_mtg (GSE220442) is the human brain input.
- **kanemaru_heart** → usable_as_input=False. Heart deferred; EGA controlled-access; dir empty.
- **muto_kidney_cosmx** → corrected. Prior GSE211785 accession was a misattributed duplicate of abedini (and not CosMx). No verified public Muto CosMx accession; annotation-only placeholder.
- **wu_moffitt_liver_merfish** → annotated. GSE210077 is the snRNA-seq companion (GPL18573), not the MERFISH data (Dryad DOI).
- **gse252772_mouse_kidney** → kept usable; counts present as Seurat .rds (needs R->anndata conversion in P1/P2).

Verified `usable_as_input=True` set (7), all with counts on disk: yu2022_liver, andrews_liver, lake_kpmp_kidney, chen_brain_mtg, gse272564_mouse_liver_ctrl, gse233983_mouse_brain, gse252772_mouse_kidney (.rds).

RESOLVED 2026-06-21: content/shape guard implemented. `src/spatial/data_validation.py` (`dataset_has_counts` / `validate_usable_inputs`) inspects each dataset dir for a real count artifact (10x triplet / Space Ranger .h5 / .h5ad / Seurat .rds, one level into per-sample archives). Enforced two ways: (1) `tests/test_data_validation.py` — 11 tests incl. a real integration check over all on-disk inputs (accept loose/nested/zip counts; reject images-only, metadata-only, DESeq2 table, long-read gtf); (2) post-download CONTENT assertion (Halt Gate 1) in `scripts/download_spatial.py` after the existence check. "Download succeeded" now means counts are actually present. Full suite 143 passed.
