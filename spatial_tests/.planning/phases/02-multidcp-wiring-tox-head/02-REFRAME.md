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

## Diagnostic result (2026-06-24) — VERDICT: ENCODER-FLAT (inert basal branch)

Ran the D1/D2/D3 diagnostic (`results/tables/P2_encoder_diagnostic.md`, `scripts/diagnose_encoder_zonal.py`, commit b00ef98). Result:
- **D1 (inputs differ):** periportal vs pericentral normalized basals Pearson 0.333 (human) / 0.867 (mouse), L2 30.4 / 13.6, ~9,000+/10,716 genes differ; zonation markers show correct zonal contrast raw AND post-normalization. **Basal pipeline is fine — NOT input-flat.**
- **D3 (markers survived):** 7/8 human, 8/8 mouse markers present + covered (not zero-filled). The lone gap (human CYP2F2) is a real mouse-specific gene; mouse Cyp2f2 → human CYP2F1. **Not a marker-dropout problem.**
- **D2 (encoder dead):** inputs from Pearson +0.33 down to −0.9999 change the output by ≤ 5.96e-08 and the 50-d cell-context by 2.98e-08 (finite-diff ratio 9.5e-09). At fixed basal, the DRUG branch moves the output 0.08–0.13. **The row-17 checkpoint reads (drug, dose) and ignores the basal entirely.**

**Implications:**
- F1 (basal-pipeline fix) is RULED OUT — no input change moves a model constant in its input.
- This is not just a spatial-arm problem: conditions B/C/S-B/S-C (all cell/tissue-conditioned predicted GEX) collapse to drug+dose with this checkpoint.
- **OPEN AMBIGUITY to resolve FIRST (order-of-magnitude cost difference):** is the basal branch inert in the checkpoint *weights*, or is our `_call_model` wiring not feeding the basal into `input_cell_gex` correctly? The diagnostic tested OUR wired forward. CHECK: run the upstream MultiDCP repo's OWN inference on two different basals (bypass our wiring) and/or inspect the context-encoder weights for near-zero. If wiring bug → trivial in-scope fix, milestone salvaged. If weights inert → see below.
- If genuinely inert weights: **F3** (condition the region feature on a delta/contrast basal computed OUTSIDE the model, e.g. `zone_basal − tissue_mean`) makes zonal contrast explicit with no weights changed — but it is NOT a *predicted* region response (it injects the observed basal contrast downstream), so it changes the scientific claim; report honestly. **F4** (source a non-collapsed checkpoint whose context encoder actually uses `input_cell_gex`) is the only path to a genuine predicted region-resolved signal under the frozen rule — provenance hunt (note both available checkpoints are non-functional: row-17 inert, row-18 collapsed). **F2** (fine-tune) remains blocked (frozen-baseline + no healthy-tissue perturbation targets).
- This materially strengthens the instrumented-negative framing (concept 4): the available frozen MultiDCP/CheMoE checkpoints do not condition predicted expression on cell/tissue context.

## State
- Phase 3 remains BLOCKED until the diagnostic + chosen fix re-clear (or re-fire) Halt Gate 3.
- All Phase-2 engineering deliverables (forward path, 3-vector cache, tox_head, smoke-train) are intact and reusable — the fix targets the basal→context path, not the wiring.
- Artifacts: `HALT_REASON.md`, `results/tables/P2_apap_validation.md`, `02-VERIFICATION.md`, `02-02-SUMMARY.md` (zonal-invariance flag), `src/spatial/region_signature.py` (the wired seam), `scripts/cache_region_de.py` (the basal pipeline to inspect for D1/D3).
