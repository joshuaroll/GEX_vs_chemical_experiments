# Phase 2 Reframe — Halt Gate 3 → "Fix the encoder" (re-open Phase 2)

**Date:** 2026-06-24
**Trigger:** Halt Gate 3 fired (pericentral predicted-vs-measured Pearson +0.0166 < 0.3). The frozen MultiDCP-CheMoE backbone produced no region-resolved DE on healthy-liver basals (per-zone predicted DE differs by ~5.96e-08, float32 epsilon).
**User decision (gate review):** **Fix the encoder — re-open Phase 2.** (Alternatives declined: proceed-as-null, pivot-to-writeup, defer.)

> This note scopes the re-opened Phase 2. It is NOT yet a plan. Start the next session with a discuss/spec pass (`/gsd-discuss-phase 2 --update` or a new `02.x` sub-phase) using this as input.

## Critical constraint to resolve first

"Fix the encoder" partially conflicts with **`DEC-frozen-baseline`** (PROJECT.md: "Out of scope — retraining / fine-tuning MultiDCP / CheMoE"). Fine-tuning the cell-context encoder would reverse that locked decision. The re-opened phase must either (a) stay within in-scope fixes, or (b) explicitly revisit `DEC-frozen-baseline` with professor sign-off. Decide this at the discuss pass.

## Mandatory first step — root-cause diagnostic (DO BEFORE any encoder work)

The epsilon-level per-zone difference has two distinct causes that demand different fixes. Isolate which one before committing effort:

- **D1 — Are the INPUT basals different?** Compute the periportal vs pericentral pseudobulk basal vectors actually fed to the model (post gene-alignment + rank-percentile normalization, 10,716 space). Report their Pearson/L2 and, specifically, canonical zonation markers: pericentral `Glul`, `Cyp2e1`, `Oat`, `Slc1a2`; periportal `Sds`, `Cyp2f2`, `Hal`, `Ass1`. **If these are ~identical → the problem is the basal pipeline, NOT the encoder** (input-flat).
- **D2 — Does the encoder respond at all?** Feed two clearly-different basals through the loaded `MultiDCP_CheMoE_AE` context encoder: (i) two real cancer-line basals it WAS trained on, (ii) the two healthy-liver zone basals. Compare the 50-d cell-context outputs. **If it moves for cancer lines but not healthy tissue → encoder OOD-saturation confirmed** (encoder-flat).
- **D3 — Did zonation markers survive the gene space?** Check the markers are present in 10,716 ∩ Visium coverage and not `fill_value=0`-zeroed (CON-gene-space ~30% absent → OOD factor). A zeroed marker is an input-flat cause masquerading as encoder failure.
- Sanity: confirm the rank-percentile normalization (introduced in 02-02 to fix the min-max 99%-floor collapse) did not itself flatten zonal contrast.

## Fix candidates (choose AFTER the diagnostic)

**If input-flat (likely cheapest, fully in-scope, no frozen-baseline conflict):**
- F1: stop over-averaging in pseudobulk; preserve zonal contrast. Revisit normalization (rank-percentile may flatten). Ensure zonation markers are in-coverage and not zero-filled. Re-cache and re-run the APAP gate.

**If encoder-flat, in-scope (no upstream retraining):**
- F3: change the INPUT representation — condition on the **delta/contrast basal** (e.g., zone basal − tissue-mean basal) so zonal difference is explicit in the input rather than relying on the encoder to extract it. No model weights change.
- F4: source a different EXISTING frozen checkpoint with a context encoder trained on broader/healthy basals (re-opens the S-B provenance hunt for a non-collapsed `MultiDCP_AE`, or another backbone).

**If encoder-flat, out-of-scope under current rules (requires revisiting `DEC-frozen-baseline`, professor sign-off):**
- F2: fine-tune ONLY the cell-context encoder on tissue basals while freezing the rest. Blocked by the lack of healthy-tissue drug-perturbed targets (no panel-wide healthy spatial perturbation data exists) — feasibility is doubtful; flag honestly.

## State
- Phase 3 remains BLOCKED until the diagnostic + chosen fix re-clear (or re-fire) Halt Gate 3.
- All Phase-2 engineering deliverables (forward path, 3-vector cache, tox_head, smoke-train) are intact and reusable — the fix targets the basal→context path, not the wiring.
- Artifacts: `HALT_REASON.md`, `results/tables/P2_apap_validation.md`, `02-VERIFICATION.md`, `02-02-SUMMARY.md` (zonal-invariance flag), `src/spatial/region_signature.py` (the wired seam), `scripts/cache_region_de.py` (the basal pipeline to inspect for D1/D3).
