---
status: partial
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
result: [pending]

### 2. chen_brain_mtg DE-results-only
expected: Confirm whether the DESeq2 normalized expression .txt.gz (15 MB) serves Phase 1, or whether the raw Visium feature matrix must be located. Entry is `usable_as_input=True`, `whole_transcriptome=True`.
result: RESOLVED 2026-06-21 — root cause was a WRONG-ACCESSION error, not a partial download. Prior GSE200474 is "Neurofilament accumulations in ALS patients' motor neurons" (bulk RNA-seq of iPSC motor neurons, GPL11154), unrelated to Chen MTG Visium. Located + applied the correct accession GSE220442 ("Spatially resolved transcriptomics ... vulnerability of MTG in AD", 10x Visium, GPL24676). Registry + MANIFEST updated; GSE220442_counts_and_images.tar.gz (451 MB, sha256 9c54b560...) downloaded and verified: 6 Space Ranger Visium sections (3 AD + 3 control), 36,601 genes, per-spot layer annotations (Layer 1–6 + White Matter) in GSE220442_metadata.csv.gz. 8/8 data-path tests pass.

### 3. DIRIL kidney labels TODO
expected: DIRIL kidney toxicity labels acquired, OR accepted as a recorded gap. Currently `data/raw/labels/diril/DIRIL_TODO.txt` exists; labels were not fetched (journal-gated Elsevier CDN). REQUIREMENTS.md DATA-01 lists DIRIL as required.
result: [pending]

### 4. Latent download-driver bugs (00-REVIEW.md CR-01/02/03)
expected: Decide whether the 3 latent integrity bugs (CR-01 hardcoded Figshare article ID; CR-02 KPMP non-fatal path is dead code; CR-03 MD5 mismatch non-fatal) must be fixed before Phase 1, or accepted as deferred risk. All are LATENT for this run — the data on disk is correct.
result: [pending]

## Summary

total: 4
passed: 1
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
