# Phase 2: MultiDCP wiring & toxicity head - Research

**Researched:** 2026-06-24
**Domain:** Frozen GEX-prediction backbone wiring (PyTorch state_dict load + forward), per-region DE caching, concat-MLP toxicity head, APAP per-zone Pearson validation
**Confidence:** HIGH on model I/O, checkpoint mechanics, DE semantics, and the architecture-mismatch landmine (all verified by loading the actual checkpoints and reading the training scripts). MEDIUM on the spatial basal→cell-context normalization path and the APAP zonation procedure (no code exists yet; verified the data shapes and the manifold normalization).

## Summary

The two `NotImplementedError` seams in `region_signature.py` (`load_model`, `_call_model`) are fillable, but **02-CONTEXT.md and MANIFEST.md point the planner at the wrong model classes and one broken checkpoint.** I loaded both MANIFEST checkpoints (rows 17-18) against the candidate classes and read the upstream training scripts. The findings reorder the phase plan:

1. **D-02 (rule B) is VALID.** The working backbone emits **absolute predicted treated expression** over 10,716 genes; DE is computed downstream as `pred_treated − basal`. The training label is the treated profile (`model.loss(lb, predict)`, `lb` = treated); `treated − diseased` appears only in diagnostics. `predictions.npz` from the upstream run stores `predictions`, `treated_test`, `diseased_test` as three separate arrays, confirming the model output is treated-space, not a native DE head. Rule B does **not** double-subtract. `[VERIFIED: loaded checkpoints + read src/multidcp_chemoe_ae_de_pdg.py + predictions.npz]`

2. **The MANIFEST "MultiDCP-PDG (S-B)" checkpoint (row 18, `chemoe_kpgt_MCF7_fold0/best_model.pt`) is unusable.** Its architecture is `CheMoE_PDG` (KPGT variant) living in `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/chemoe_pdg/model.py` — NOT `multidcp_pdg.py`. Worse, the checkpoint is **collapsed** (predictions are a near-constant 0.61, std 0.001, corr with both treated and DE ≈ 0.00). It also requires (a) a 2304-dim KPGT pretrained molecular embedding (a separate model, not the in-repo NeuralFingerprint), and (b) a categorical `cell_embedding` over 10 hardcoded cancer cell lines — which cannot represent a healthy-tissue region basal. **It is wrong on three independent axes.** `[VERIFIED: state_dict keys + predictions.npz degeneracy + read model.py]`

3. **The working backbone is MANIFEST row 17 (`MultiDCP_CheMoE_pdg/src/best_model.pt`).** It loads with **0 missing / 0 unexpected keys** into `MultiDCP_CheMoE_AE` (class `MultiDCP_CheMoEBase`) from `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/models/multidcp_chemoe_pdg.py`. Training diagnostics are healthy (DE top-20 R² ≈ 0.75, all-genes Pearson ≈ 0.67). It takes a **10,716-dim basal vector** as cell context (generalizes to any tissue), uses the in-repo NeuralFingerprint (SMILES→graph, atom=62/bond=6 feats), and outputs 10,716-gene treated expression. `[VERIFIED: strict load test + diagnostics CSV]`

**Primary recommendation:** Reverse D-04's ordering. Wire **MultiDCP-CheMoE (S-C) via `MultiDCP_CheMoE_AE` + `src/best_model.pt` FIRST** — it is the only working, tissue-context-capable frozen backbone on disk. Escalate the "MultiDCP-PDG (S-B)" path: either (a) source a working non-collapsed `MultiDCP_AE`/`MultiDCP_CheMoE` PDG checkpoint that takes a 10,716 basal, or (b) drop S-B from this phase and note it as a checkpoint-provenance blocker. Do not wire `CheMoE_PDG`/KPGT into the spatial path at all (categorical cell embedding is incompatible with region basal). This is a substantive deviation from D-04 and should be surfaced to the user at discuss/plan time (it does not contradict D-04's intent — "de-risk the seam first" — it just identifies that S-B's checkpoint is the risk, not the wiring).

<user_constraints>
## User Constraints (from 02-CONTEXT.md)

### Locked Decisions
- **D-01:** Cache predicted DE and feed the tox head in the **10,716-gene PDG space** (not 978). Generalize `region_signature.py`'s `N_LANDMARK=978` default to 10,716 without breaking 978 callers.
- **D-02:** Spatial DE = **rule B (bias-corrected):** `predicted_DE_region = predicted_treated(drug, region_basal) − predicted_control(region_basal)`. Subtract the model's own predicted control output, not the raw basal. **CONDITIONAL on the model emitting absolute predicted treated, not a native DE head.** → **RESEARCH CONFIRMS: model emits absolute treated expression. Rule B is valid.** (See Summary finding 1.)
- **D-03:** Persist **DE + intermediate predicted_treated AND predicted_control** vectors (not DE-only). Extend `RegionSignatureCache` (currently `de_array` only).
- **D-04:** Wire **MultiDCP-PDG (S-B) end-to-end first, then MultiDCP-CheMoE (S-C)** same phase if clean. → **RESEARCH RECOMMENDS REVERSING:** the S-B checkpoint is broken/incompatible; wire S-C first. See Summary finding 2-3 + Open Question 1.
- **D-05:** Fixed canonical reference dose (10 µM / 24 h) for the cached GEX signature. Gate on DE **pattern, not magnitude**. → **RESEARCH CAVEAT:** the working checkpoint's dose one-hot is 2-dim and the PDGrapher training data has no dose variation (single `pert_idose='x'`); "10 µM" has no clean bin. Use any fixed valid 2-dim one-hot; the signature is effectively dose-agnostic. See Open Question 3.
- **D-06:** Anchor = **GSE272564 primary + GSE280652 independent replication** (mouse liver, validation-only, never basal).
- **D-07:** Zones = published annotation if present, else canonical markers (pericentral Glul/Cyp2e1; periportal Sds/Cyp2f2). Leiden last resort.
- **D-08:** Halt Gate 3 = **Pearson of predicted vs measured DE across the 10,716 PDG genes** (∩ Visium coverage; absent genes flagged not silently zero'd), **per zone**, gate keyed to **pericentral**, threshold `< 0.3`.
- **D-09:** Accept rodent APAP anchor as direct validity stand-in; fired gate = **stop-and-REFRAME** (not abandon).

### Claude's Discretion
- basal → cell-context projection (region pseudobulk 978-or-10716 → model input + 50-d cell-context encoder).
- top-k size for region-pooled DE in the attention combiner.
- wandb project/run naming + smoke-train epoch count for condition A.
- pericentral/periportal marker thresholds + zone-assignment procedure.
- cache on-disk layout (npy + manifest JSON) for the extended multi-vector cache.

### Deferred Ideas (OUT OF SCOPE)
- CheMoE routing-permutation diagnostic (Halt Gate 6) — Phase 6.
- Dose-response channel (G/H, E-Hill) — later; D-05 fixes one dose for the GEX signature only.
- Compound-aware splits + leakage audits — Phase 3.
- Full multi-condition training (A/B/C/S-B/S-C/fusion, ≥3 seeds) — Phase 4.
- Other organs (kidney, brain, heart) — own phases; heart last.
- Targeted-panel promotion via Tangram — out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| WIRE-01 | Frozen-model forward path: resolve checkpoint paths, replace the `region_signature.py` `NotImplementedError` seam with the real call, cache per-region predicted DE (human first, both species). | Exact `_call_model` body resolved: load `MultiDCP_CheMoE_AE` from `src/best_model.pt`, build NeuralFingerprint drug input from SMILES, pass 10,716 basal + 2-dim dose one-hot, take `predictions` (treated), then rule-B subtract a predicted-control pass. Spatial DE rule corrected from rule A to rule B (D-02 confirmed valid). See Standard Stack, Code Examples, Pitfalls. |
| WIRE-02 | `tox_head.py` concat-MLP; confirm attention combiner feeds it; per-condition zero-tensor masking; smoke-train condition A; wandb logging. | `AttentionPoolCombiner(d=10716)` output → `proj_gex`; chem encoder choice (ChemBERTa 768 recommended); 3-layer concat-MLP (GELU/dropout/BN); condition A = zero GEX tensor (no cache needed). See Architecture Patterns + CON-tox-head. |
| WIRE-03 | APAP predicted-vs-measured per-zone Pearson on GSE272564/GSE280652. | DE-space comparison over 10,716 ∩ Visium genes per zone; measured DE = APAP_zone − control_zone; predicted DE from cache (rule B). GSE272564 has matched control + APAP arms (cleanest); GSE280652 APAP-only. Halt Gate 3 keyed to pericentral < 0.3. See Validation Architecture + Pitfalls. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SMILES → drug graph features | Data-prep (CPU) | — | `Molecules(smiles)` + `convert_smile_to_feature`; atom=62/bond=6; built once per drug, reused across regions |
| Frozen GEX inference (drug+basal+dose → treated 10,716) | Model-inference (GPU, `torch.no_grad`) | — | Frozen `MultiDCP_CheMoE_AE`; never trained; one GPU left free (Hard Rule 5) |
| Per-region rule-B DE (treated − predicted_control) | Pure post-processing (CPU/numpy) | — | `compute_de` already pure + tested; only the subtraction operand changes |
| Cache assembly + manifest | Pure (CPU/numpy) | Disk I/O | `assemble_cache`/`build_manifest` already pure; extend to 3-vector cache (D-03) |
| Region pooling (attention over zones) | Tox-head (GPU, trainable) | — | `AttentionPoolCombiner` is the only trainable component touching GEX |
| Concat-MLP tox head + smoke train | Tox-head (GPU, trainable) | wandb logging | The ONLY trained module this phase (DEC-frozen-baseline) |
| APAP zonation + measured DE + Pearson | Analysis (CPU, scanpy/squidpy/numpy) | — | Validation-only; reads Visium, never feeds the model |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | env `dili_v04_env` (already installed) | Load frozen state_dict, forward pass. Model is `.double()` — inputs must be float64. | The checkpoints are saved as `model.state_dict()` of a `nn.Module`. `[VERIFIED: loaded both .pt]` |
| MultiDCP_CheMoE_AE | repo SHA local (`MultiDCP_CheMoE_pdg`) | The working frozen backbone class. From `src/models/multidcp_chemoe_pdg.py`; alias of `MultiDCP_CheMoEBase`. | Loads `src/best_model.pt` with 0 missing/0 unexpected keys. `[VERIFIED: strict load]` |
| NeuralFingerprint (+ Molecules, molecule_utils) | repo `src/utils/` + `src/models/neural_fingerprint.py` | SMILES → atom/bond graph → 128-d drug embed inside the backbone. | The backbone's `drug_fp` is a NeuralFingerprint; the drug input dict is built by `convert_smile_to_feature` / `Molecules`. `[VERIFIED: read neural_fingerprint.py + molecules.py]` |
| scanpy / anndata | scanpy 1.11.5, anndata 0.12.10 (MANIFEST) | Load Visium APAP `.tar`/`.h5`, spot-level matrices for zonation + measured DE. | Already used in P0/P1. |
| squidpy | 1.8.2 (MANIFEST) | Optional: Moran's I / spatial-neighbor QC on zone assignment (CON-spatial-qc). | Already installed P0. |
| numpy / scipy.stats | installed | rule-B DE, per-zone Pearson (`scipy.stats.pearsonr`). | — |
| wandb | installed | Condition A smoke-train logging (WIRE-02). | Project precedent (`joshroll/...`). |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| transformers (ChemBERTa) | check availability | chem encoder for `proj_chem` (768-d). | WIRE-02 tox head. Condition A still needs SOME chem embedding (structure-only). Confirm the encoder used in P1's structure floor (ECFP4 was used there — see note below). |
| rdkit | env | SMILES canonicalization for the drug graph + ECFP4 fallback. | Drug prep; P1 used `smiles_to_ecfp4`. |

**Chem-encoder note (Claude's Discretion):** PROJECT.md `CON-tox-head` says `chem_emb → proj_chem` and PROJECT.md lists frozen chem encoders ChemBERTa(768)/MolFormer(768)/GIN(300)/UniMol(512). **But P1's structure floor used ECFP4 (2048-d Morgan), not a neural encoder** `[VERIFIED: STATE.md plan 01-03 "smiles_to_ecfp4 returns (n,2048)"]`. For condition A smoke-train the chem channel just needs to be *some* fixed structure embedding; ECFP4→`proj_chem` is the lowest-friction choice that matches existing P1 code and avoids adding a transformer dependency this phase. Recommend ECFP4 for the smoke-train and defer the ChemBERTa-vs-ECFP4 decision to P4 (it is a headline-model choice, not a wiring choice). `[ASSUMED]` that ECFP4 is acceptable for the smoke-train — confirm with user, since CON-tox-head implies a neural encoder.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `MultiDCP_CheMoE_AE` + `src/best_model.pt` (S-C) | `CheMoE_PDG` + `chemoe_kpgt_MCF7_fold0` (the MANIFEST "S-B" row) | REJECT: collapsed checkpoint (const output), needs KPGT 2304-d embeddings, categorical 10-cell-line embedding incompatible with tissue basal. |
| `MultiDCP_CheMoE_AE` (S-C, MoE) | `MultiDCP_AE` / `MultiDCPOriginal` (S-B, non-MoE) from `multidcp_pdg.py` | Plausible for a true S-B path IF a matching non-collapsed 10,716-basal checkpoint exists. None found on disk. Both `multidcp_pdg.py` variants take the same `(input_drug, input_gene, mask, input_cell_gex, input_pert_idose)` signature and output `[batch, num_gene]` treated. See Open Question 1. |

**Checkpoint provenance (corrected from MANIFEST):**
- Working S-C backbone: `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt` (SHA256 `fbee15f…`, 32 MB) → class `MultiDCP_CheMoE_AE`. `[VERIFIED]`
- Broken/incompatible "S-B": `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/trained_models/chemoe_kpgt_MCF7_fold0/best_model.pt` (SHA256 `8e9f0e4…`, 22 MB) → class `CheMoE_PDG` (KPGT) at `PDGrapher_Baseline_Models/models/chemoe_pdg/model.py`. DO NOT WIRE. `[VERIFIED]`
- Model code home: `MultiDCP_CheMoE_pdg/src/models/` (local; `multidcp_chemoe_pdg.py`, `multidcp_pdg.py`, `neural_fingerprint.py`). The dili_downstream `upstream/` clone is a *different vintage* (its `MultiDCP_AE` has no MoE) — the checkpoints do NOT load against it. Use the `MultiDCP_CheMoE_pdg/src/models/` copies. `[VERIFIED: key namespace mismatch]`

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────────────────────────────┐
  SMILES (drug) ──────────►  Molecules(smiles) → atom/bond graph (62/6)  │
                          │      convert_smile_to_feature → drug dict     │
                          └───────────────────┬─────────────────────────┘
                                              │ input_drug
  Region pseudobulk basal (Visium)           ▼
  → align_to_gene_space(→10,716, FLNC..)   ┌──────────────────────────────────────┐
  → 0-1 normalize to manifold range ───────► FROZEN MultiDCP_CheMoE_AE (no_grad)   │
        │  input_cell_gex [B,10716]         │  drug_fp + cell_encoder(10716→50)    │
        │                                   │  + dose_encoder(2→128) + 4 experts   │
  dose one-hot [B,2] (fixed) ───────────────►  top-2 gating → 10,716 treated       │
                                            └───────────────────┬──────────────────┘
                                       (A) treated(drug,basal)  │  (B) treated(CONTROL,basal)
                                                                ▼
                                          rule B: DE = treated_drug − treated_control   (per region)
                                                                │   [10,716]
                                              cache (D-03: DE + treated + control vecs)
                                                                │
                              ┌─────────────── per drug × per region (zones) ─────────────────┐
                              ▼                                                                 │
            stack regions → [B, n_regions, 10716] ──► AttentionPoolCombiner(d=10716)           │
                                                          → pooled [B,10716] + attn_weights     │
                                                                │ (top-k of 10,716)             │
   chem_emb (ECFP4/ChemBERTa) → proj_chem ─┐                    ▼ proj_gex                       │
   dose_response (zero this phase) → proj_dr ┼──► concat → 3-layer MLP (GELU/dropout/BN) → logit │
                                            └──────────────────────────────────────────────────┘
                                                          organ-tox logit  →  BCEWithLogits

  Condition A smoke-train: GEX channel = zeros([B,10716]); cache NOT needed; chem channel only.

  ── VALIDATION (WIRE-03, separate path, never feeds the model) ──
  APAP Visium (GSE272564 ctrl+APAP arms, GSE280652 APAP) → zone spots (markers/annotation)
    → measured DE = pseudobulk(APAP_zone) − pseudobulk(control_zone), align→10,716
    → per-zone Pearson(predicted_DE_rule_B, measured_DE) over 10,716 ∩ Visium genes
    → Halt Gate 3: pericentral Pearson < 0.3 → stop-and-reframe
```

### Recommended additions to `src/spatial/`
```
src/spatial/
├── region_signature.py   # EXTEND: fill load_model + _call_model; rule A→B; N_LANDMARK→support 10716
├── tox_head.py           # NEW (WIRE-02): concat-MLP + per-channel zero masking
├── apap_validation.py    # NEW (WIRE-03): zone assignment, measured DE, per-zone Pearson, Halt Gate 3
├── region_combiner.py    # REUSE: AttentionPoolCombiner(d=10716) already done
├── gene_alignment.py     # REUSE: align_to_gene_space(profile, source, target, missing=...)
├── pseudobulk.py         # REUSE: pseudobulk(matrix, spot_labels, gene_names, agg=...)
configs/
└── liver_p2.yaml         # NEW: checkpoint paths, gene-order file, dose one-hot, smoke-train epochs
scripts/
└── cache_region_de.py    # NEW: --gpu BEFORE import torch; drives RegionSignatureCacher
```

### Pattern 1: `_call_model` body (the seam fill)
**What:** One (drug, region) forward → absolute treated 10,716 vector. Subtraction stays in `compute_de`.
**When:** Inside `RegionSignatureCacher._call_model`, called per (pert_id, region) by `run()`.
**Example:** see Code Examples below. Key contract from the existing docstring: "Return the raw predicted treated GEX … Do NOT subtract the basal here." That contract is correct for rule B too — return absolute treated; the subtraction (now treated_drug − treated_control) is the caller's job.

### Pattern 2: rule B needs TWO forward passes per region
**What:** `predicted_control(region_basal)` = a forward pass with a **control/vehicle drug** in the same region context. There is no DMSO row in the PDGrapher data, so "control" must be defined. Options: (a) a designated vehicle SMILES if the training vocab had one (it did not — single `pert_idose='x'`, all `trt_cp`), (b) a zero/empty drug graph, (c) the per-region mean over all drugs (an empirical "average treated"). **Recommend (c) is wrong** (leaks drug signal). Recommend defining control as the model's output for a **canonical inert reference** — but absent a true vehicle in the training data, the cleanest principled control is **the model's prediction under an empty-perturbation drug input** if the NeuralFingerprint accepts it, else escalate. This is the single biggest unresolved design point for rule B — see Open Question 2.
**Anti-pattern:** subtracting the *raw observed basal* (that is rule A, which D-02 explicitly replaces) or subtracting `diseased` (that is the cell-line/v0.5 convention).

### Pattern 3: condition A smoke-train decouples from the cache
**What:** Condition A feeds a **zero tensor** to the GEX channel (`torch.zeros(B, 10716)`), so the smoke-train validates the head + loop + wandb **without any frozen-model inference**. Wire and test the head first; cache second; APAP gate last.
**Why:** de-risks WIRE-02 independently of the WIRE-01 checkpoint/control-pass questions.

### Anti-Patterns to Avoid
- **Wiring the kpgt/`CheMoE_PDG` checkpoint** (categorical 10-cell-line embedding cannot encode tissue basal; checkpoint is collapsed). 
- **Using the dili_downstream `upstream/` MultiDCP classes** — they are a different vintage; the checkpoints will not load (key namespace differs). Use `MultiDCP_CheMoE_pdg/src/models/`.
- **Feeding a float32 basal to a `.double()` model** — the checkpoint is double; cast inputs to float64 or the model will error / silently upcast.
- **Forgetting the basal normalization** — the manifold basal is min-max scaled to [0,1] (mean 0.62). A raw log-CPM Visium pseudobulk is OOD; normalize to the manifold range or the prediction is meaningless (see Pitfall 1).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SMILES → graph drug features | A custom atom/bond featurizer | `Molecules(smiles)` + `convert_smile_to_feature` from `MultiDCP_CheMoE_pdg/src/utils/` | atom=62/bond=6 must match the frozen NeuralFingerprint exactly; any mismatch = wrong drug embedding |
| Gene-space alignment | A custom reindex loop | `gene_alignment.align_to_gene_space(profile, source, target, missing='zero')` | Already handles duplicates, missing genes, order; tested |
| Region pseudobulk | A custom groupby-mean | `pseudobulk.pseudobulk(matrix, spot_labels, gene_names, agg='mean')` | NaN-aware, region_labels handling, tested |
| Attention pool over regions | A new attention module | `AttentionPoolCombiner(d=10716)` | Already emits attn_weights for P6 interpretability; tested |
| DE subtraction + cache assembly | New numpy stacking | `compute_de` + `assemble_cache` + `build_manifest` (extend for 3 vectors) | Pure + tested; only the operand and the cache schema change |
| state_dict load | Reconstructing the model from scratch | `MultiDCP_CheMoE_AE(device, registry).double(); m.load_state_dict(ckpt)` with `initialize_model_registry()` + `{num_gene:10716, pert_idose_input_dim:2}` | Verified to strict-load |

**Key insight:** Nearly every pure piece already exists and is tested. The genuinely new work is small: (1) the two-forward-pass `_call_model` with the correct drug featurization and basal normalization, (2) the control-pass definition for rule B, (3) `tox_head.py`, (4) APAP zonation + Pearson. The risk is provenance/normalization, not code volume.

## Runtime State Inventory

> Phase 2 is greenfield wiring (new code + new cache artifacts), not a rename/refactor. This section is included only to record the cross-project static dependencies the cache depends on, since a stale path silently fabricates wrong features.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None written yet (cache is a Phase-2 *output* under `data/processed/spatial/region_de_cache/`). | Define on-disk layout (D-03 multi-vector). |
| Live service config | None (no external service). | None — verified: pure local inference. |
| OS-registered state | None. | None. |
| Secrets/env vars | wandb API key (already configured for prior phases). | None — reuse. |
| Build artifacts / installed packages | The frozen checkpoint + the `MultiDCP_CheMoE_pdg/src/models/` + `src/utils/` Python modules are read-only cross-project imports (sys.path or copy). The 10,716 gene-order file `data/processed/spatial/multidcp_10716_symbols.txt` (P0 output) is the canonical column order. | Pin checkpoint SHA + gene-order file SHA in MANIFEST; assert at load. |

**Cross-project import landmine:** `region_signature.py` is currently a *pure* module ("no hardcoded /raid paths"). Filling `load_model`/`_call_model` will introduce a dependency on `MultiDCP_CheMoE_pdg/src/models/` + `src/utils/` (which use bare `import neural_fingerprint`-style imports, not package-relative). The planner must either (a) add those dirs to `sys.path` at load time inside `load_model`, or (b) vendor a thin loader. Keep the model-loading code in a *new* module or a clearly-marked impure method so the pure DE/cache functions stay testable in isolation.

## Common Pitfalls

### Pitfall 1: Basal-normalization OOD (the silent feature-corruption bug)
**What goes wrong:** Visium region pseudobulk is in log-CPM / lognorm units; the frozen model's `cell_encoder` was trained on a **[0,1] min-max-scaled** 10,716 basal (verified: manifold range [0.018, 1.000], mean 0.62). Feeding raw lognorm produces garbage predictions that *look* numerically fine (no NaN) but have no biological meaning — and WIRE-03's acceptance ("no NaNs") would still pass.
**Why:** The cell-context encoder is the only place tissue basal enters; its input distribution is fixed by training.
**How to avoid:** Normalize each region basal into the manifold's per-gene or global [0,1] range before `_call_model`. Document the exact transform. Compare a sanity prediction's value range against the manifold (~[0,1], mean ~0.6); if it is off-distribution, the normalization is wrong.
**Warning signs:** prediction std ≈ 0 (collapse like the kpgt checkpoint), or values far outside [0,1].

### Pitfall 2: float32/float64 mismatch
**What goes wrong:** Model is `.double()`; passing float32 basal/dose tensors errors or upcasts inconsistently.
**How to avoid:** `.double()` all model inputs (`input_cell_gex`, `input_pert_idose`) and the drug-feature tensors. The existing pure functions return float32 — cast at the model boundary, keep cache as float32 to save disk.

### Pitfall 3: rule-B control pass is undefined in the data
**What goes wrong:** There is no vehicle/DMSO drug in the PDGrapher training vocab (all `trt_cp`, single dose `'x'`). "predicted_control(region_basal)" has no canonical input.
**How to avoid:** Decide the control input explicitly (Open Question 2). Whatever is chosen, it must be applied identically to predicted and to the APAP measured comparison framing. The cache stores `predicted_control` (D-03) so the choice is auditable and re-deriveable.

### Pitfall 4: dose one-hot dimension is 2, not 6
**What goes wrong:** CON-model-io and D-05 assume a 6-way one-hot at "10 µM". The working checkpoint's `dose_encoder` is `(64, 2)` → `pert_idose_input_dim = 2`, and the training data has a single dose value. A 6-dim or wrong-dim dose tensor will shape-error.
**How to avoid:** Hardcode a fixed 2-dim one-hot (e.g., `[1.0, 0.0]`) for all drugs; document that dose is effectively constant for this backbone and that dose-dependence is the deferred G/H channel. Do NOT gate any result on dose magnitude (D-05).

### Pitfall 5: APAP zonation uses mouse gene symbols
**What goes wrong:** The 10,716 space is human symbols (FLNC, MAP2K4…). APAP anchors are mouse (GSE272564/GSE280652, ~32,245 mouse-symbol genes). Aligning mouse Visium DE to the human 10,716 needs the one-to-one ortholog map (P0 output `orthologs_h_m_r_one2one.tsv`, 15,956 pairs), and the per-zone Pearson is computed only over the 10,716 ∩ ortholog ∩ Visium-covered intersection (D-08: flag absent genes, do not silently zero).
**How to avoid:** Route mouse symbols → human via `orthology.py` before `align_to_gene_space`. Report the intersection size per zone (it bounds the Pearson's power).

### Pitfall 6: GSE272564 appears twice in the registry (same RAW.tar)
**What goes wrong:** `gse272564_apap_liver` and `gse272564_mouse_liver_ctrl` are the SAME download (`GSE272564_RAW.tar`, same SHA `0f6c15d3…`). The control arm is the basal-context candidate; the APAP arm is the validation anchor. Loading the wrong subset mixes treated into basal.
**How to avoid:** Split spots/samples by arm at load (control vs APAP) using the GEO sample metadata. GSE272564 carries BOTH arms → cleanest matched `treated − control` (D-06). GSE280652 is APAP-only → its control must come from elsewhere or the DE is APAP-vs-its-own-periportal (note the asymmetry in the report).

### Pitfall 7: CUDA hygiene
**What goes wrong:** Importing torch before setting `CUDA_VISIBLE_DEVICES` grabs the wrong/busy GPU; not leaving one GPU free violates Hard Rule 5 on the shared box.
**How to avoid:** In the caching/training scripts, parse `--gpu` and set `os.environ['CUDA_VISIBLE_DEVICES']` **before** `import torch`; auto-detect a free device via `nvidia-smi`; never take the last free GPU.

## Code Examples

### `load_model` (corrected, working backbone)
```python
# Source: VERIFIED strict-load of src/best_model.pt into MultiDCP_CheMoE_AE
# Inside RegionSignatureCacher.load_model (impure; keep pure DE/cache fns separate)
import os, sys
import numpy as np
# --- CUDA hygiene done by the *caller script* BEFORE torch import (Hard Rule 5) ---
import torch

MDCP_SRC = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
sys.path.insert(0, os.path.join(MDCP_SRC, "models"))
sys.path.insert(0, os.path.join(MDCP_SRC, "utils"))
from multidcp_ae_pdg_utils import initialize_model_registry           # registry defaults
import multidcp_chemoe_pdg as mc                                       # MultiDCP_CheMoE_AE

reg = initialize_model_registry()           # cell_id_input_dim=10716, num_gene=10716, etc.
reg.update({"num_gene": 10716,
            "pert_idose_input_dim": 2,      # VERIFIED from dose_encoder (64,2)
            "dropout": 0.3,
            "linear_encoder_flag": False})  # checkpoint has the transformer cell encoder
model = mc.MultiDCP_CheMoE_AE(device=self.device, model_param_registry=reg).double()
state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
missing, unexpected = model.load_state_dict(state, strict=True)   # expect 0/0
model.eval(); 
for p in model.parameters(): p.requires_grad_(False)
self._model = model
```

### `_call_model` (one absolute-treated forward, 10,716)
```python
# Source: VERIFIED forward signature in multidcp_chemoe_pdg.MultiDCP_CheMoEBase.forward
# Returns predictions [batch, num_gene] = ABSOLUTE predicted treated expression.
def _call_model(self, smiles: str, region_basal: np.ndarray) -> np.ndarray:
    # 1. SMILES -> drug graph dict (atom=62, bond=6) via repo utils (do NOT hand-roll)
    drug, mask = self._featurize_drug(smiles)         # builds input_drug dict + mask
    # 2. basal: align to 10,716 (done by caller) THEN normalize to manifold [0,1] range
    basal = torch.as_tensor(region_basal, dtype=torch.float64,
                            device=self.device).unsqueeze(0)         # [1, 10716]
    dose = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=self.device)  # fixed 2-dim
    with torch.no_grad():
        pred, _cell_hidden = self._model(
            input_cell_gex=basal, input_drug=drug, input_gene=self._gene_tensor,
            mask=mask, input_pert_idose=dose, job_id="perturbed", epoch=0)
    return pred.squeeze(0).float().cpu().numpy()      # [10716] absolute treated
```

### rule-B DE at the cache layer (D-02 correction to `assemble_cache`)
```python
# predicted_treated_map[(pert_id, region)] = _call_model(drug_smiles, basal)
# predicted_control_map[region]            = _call_model(CONTROL_INPUT, basal)   # Open Q2
de_vector = predicted_treated - predicted_control      # rule B (NOT - region_basal)
# manifest.de_convention must change from
#   "predicted_treated - region_basal"   (rule A, current)
# to
#   "predicted_treated(drug) - predicted_control(region_basal)"  (rule B, D-02)
```

### per-zone Pearson (WIRE-03 / Halt Gate 3)
```python
# Source: D-08; scipy.stats.pearsonr over the 10,716 ∩ ortholog ∩ Visium intersection
from scipy.stats import pearsonr
def zone_pearson(pred_de_10716, measured_de_10716, present_mask):
    idx = np.where(present_mask)[0]                    # flag-not-zero (D-08)
    r, p = pearsonr(pred_de_10716[idx], measured_de_10716[idx])
    return r, p, idx.size
# Halt Gate 3: r['pericentral'] < 0.3 -> write HALT_REASON.md, stop-and-reframe (D-09)
```

## State of the Art

| Old (assumed by CONTEXT/MANIFEST) | Actual (verified) | Impact |
|--------------|------------------|--------|
| S-B checkpoint = `multidcp_pdg.py` MultiDCP-PDG | row 18 is `CheMoE_PDG` (KPGT), collapsed | Reverse D-04 ordering; escalate S-B |
| Model emits DE natively (D-02 conditional risk) | Model emits absolute treated; DE is post-hoc | Rule B valid, no double-subtract |
| 6-way dose one-hot, 10 µM bin | 2-dim dose, single training dose `'x'` | Dose effectively constant; fixed one-hot |
| basal input 978 (seam `N_LANDMARK=978`) | basal input = **10,716** (`cell_id_input_dim=10716`) | No 978→10716 imputation at the model input boundary; align Visium straight to 10,716 |
| basal is lognorm | manifold basal is **min-max [0,1]** | Must renormalize region basal (Pitfall 1) |

**Deprecated/outdated:**
- dili_downstream `src/models/upstream/` MultiDCP classes for *this* checkpoint — wrong vintage (no MoE; key namespace mismatch). The README there is for v0.4 conditions B/C, not these spatial checkpoints.
- `region_signature.py` rule A (`predicted_treated − region_basal`) and its `de_convention` string — superseded by D-02 rule B.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ECFP4→proj_chem is acceptable for the condition-A smoke-train (vs a neural ChemBERTa encoder per CON-tox-head). | Standard Stack | Low for smoke-train (any fixed chem embedding validates the loop); if user wants ChemBERTa locked now, add the dependency. |
| A2 | The 10,716 basal must be 0-1 normalized to the manifold range before inference. | Pitfall 1 | HIGH — wrong normalization silently corrupts every cached feature and the APAP gate. Verify with a value-range sanity check on first inference; ideally reproduce a known cancer-line prediction. |
| A3 | "predicted_control" for rule B should be a defined inert/empty drug input (no vehicle exists in training data). | Pattern 2 / Open Q2 | HIGH — defines the entire DE quantity; wrong control = wrong DE and wrong APAP comparison. |
| A4 | Fixed 2-dim dose one-hot `[1,0]` is a valid, harmless conditioning for a backbone trained on a single dose. | Pitfall 4 | Low (dose deferred to G/H; D-05 says don't gate on magnitude), but the two bins' meaning is unverified. |
| A5 | The S-C CheMoE backbone (cancer-line trained) generalizes well enough to a healthy-tissue basal to be worth caching. | Summary | This IS the project's OOD hypothesis (CON-gene-space "OOD factor we instrument"); a low APAP Pearson is a *reportable* result (D-09), not a wiring bug. |

## Open Questions (RESOLVED 2026-06-24 via CONTEXT.md amendments)

> Q1→D-04 amendment (wire S-C only, descope S-B as a checkpoint-provenance blocker).
> Q2→D-02 amendment (control = inert/empty-drug control pass). Q3→D-05 (fixed 2-dim dose, dose-agnostic). Q4→D-06 (GSE272564 control is the gate reference; GSE280652 partial replication). All four are encoded in the Phase 2 plans.

1. **[RESOLVED→D-04] The "MultiDCP-PDG (S-B)" path has no working checkpoint.** (Blocks D-04 as written.)
   - What we know: row 18 (`chemoe_kpgt_…`) is `CheMoE_PDG`/KPGT, collapsed, categorical-cell-line, KPGT-dependent — unusable for tissue. The `multidcp_pdg.py` classes (`MultiDCP_AE`, `MultiDCPOriginal`) DO take a 10,716 basal and output treated with the same forward signature as CheMoE, so a *true* S-B is wireable IF a matching checkpoint exists.
   - What's unclear: whether a non-collapsed `MultiDCP_AE`-PDG checkpoint (10,716 basal) exists anywhere on disk. The healthy CheMoE training (`src/best_model.pt`) is the only verified-good one.
   - Recommendation: wire **S-C first** (working), then hunt for / re-derive an S-B checkpoint; if none, descope S-B to a later phase and note the provenance gap. Surface to user (deviation from D-04 ordering).

2. **rule-B "predicted_control" input is undefined.** (Blocks correct DE.)
   - What we know: no vehicle/DMSO drug in the training vocab; all `trt_cp`, single dose.
   - What's unclear: the principled control input — empty graph, a designated inert SMILES, or the basal-reconstruction (autoencoder `job_id != 'perturbed'`) path. Note `MultiDCP_CheMoE_AE.autoencoder` exists but reconstructs the basal (10,716), which is a *different* quantity than "predicted control treated".
   - Recommendation: decide explicitly at planning; cache `predicted_control` (D-03) so the choice is auditable and re-derivable. Likely the cleanest is the model's prediction for a fixed reference inert compound, OR define DE against the model's own autoencoder basal-reconstruction (would make rule B ≈ "treated − reconstructed basal", a coherent on-manifold control). Flag for user/professor.

3. **Dose bin semantics.** 2-dim one-hot, both bins from a single-dose training set — which (if either) is "10 µM"? 
   - Recommendation: treat as constant; pick `[1,0]`; do not gate on magnitude (D-05). Low risk.

4. **GSE280652 control source.** GSE280652 is APAP-only; its measured DE needs a control. 
   - Recommendation: use GSE272564's matched control arm as the primary gate; report GSE280652 as APAP-zone-vs-periportal-baseline replication with the caveat noted. Confirm at planning.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torch + CUDA | WIRE-01/02 inference + smoke-train | ✓ | `dili_v04_env` | — |
| `MultiDCP_CheMoE_pdg/src/models` + `src/utils` | model class + SMILES featurizer | ✓ | local repo | — |
| `src/best_model.pt` (S-C checkpoint) | WIRE-01 | ✓ | 32 MB, SHA `fbee15f…` | — |
| `chemoe_kpgt_…/best_model.pt` (S-B) | WIRE-01 S-B | ✓ on disk but **collapsed/incompatible** | 22 MB | descope S-B or source new ckpt (Open Q1) |
| scanpy / anndata / squidpy | WIRE-03 Visium load + QC | ✓ | 1.11.5 / 0.12.10 / 1.8.2 | — |
| ortholog map `orthologs_h_m_r_one2one.tsv` | WIRE-03 mouse→human | ✓ | P0 output, 15,956 pairs | — |
| `multidcp_10716_symbols.txt` (gene order) | align target order | ✓ | P0 output | — |
| APAP Visium `GSE272564_RAW.tar` / `GSE280652_RAW.tar` | WIRE-03 | ✓ | on disk (MANIFEST) | — |
| rdkit (SMILES canon/ECFP4) | drug prep + chem channel | ✓ (P1 used it) | env | — |
| wandb | WIRE-02 smoke-train logging | ✓ | env | dryrun mode |
| ChemBERTa/transformers | optional neural chem encoder | ✗ unconfirmed | — | ECFP4 (P1 precedent) |
| KPGT pretrained encoder (2304-d) | only if wiring kpgt ckpt | ✗ | — | N/A — do not wire kpgt |

**Missing with no fallback:** A working S-B (MultiDCP-PDG, 10,716-basal) checkpoint (Open Q1) — blocks the S-B half of D-04 only, not WIRE-01/02/03.

## Validation Architecture

> nyquist_validation: key absent in config.json → treated as ENABLED.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (168 tests currently green per STATE.md) |
| Config file | repo `tests/` (no pytest.ini observed; uses default discovery) |
| Quick run command | `conda run -n dili_v04_env pytest tests/spatial/test_region_signature.py -x -q` |
| Full suite command | `conda run -n dili_v04_env pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WIRE-01 | rule-B DE math (treated − control), 10,716 shape, no NaN | unit (fixture, inject synthetic treated/control) | `pytest tests/spatial/test_region_signature.py -k rule_b -x` | ❌ Wave 0 (extend existing file) |
| WIRE-01 | cache schema stores DE + treated + control (D-03), manifest de_convention updated | unit | `pytest tests/spatial/test_region_signature.py -k cache_three_vector -x` | ❌ Wave 0 |
| WIRE-01 | checkpoint loads strict 0/0 into MultiDCP_CheMoE_AE | integration (real ckpt, GPU/CPU) — CANNOT mock (Hard Rule 1) | `pytest tests/spatial/test_model_load.py -x` (real-inference smoke, mark `@pytest.mark.gpu`) | ❌ Wave 0 |
| WIRE-01 | one real forward produces finite [10716] in [~0,1] range (normalization sanity) | integration smoke (real model) | `pytest tests/spatial/test_model_load.py -k forward_sane -x` | ❌ Wave 0 |
| WIRE-02 | concat-MLP forward: zero GEX channel → finite logit; per-channel mask shapes | unit (fixture) | `pytest tests/spatial/test_tox_head.py -x` | ❌ Wave 0 |
| WIRE-02 | attention combiner feeds head; [B,n_regions,10716]→[B,10716]→logit | unit | `pytest tests/spatial/test_tox_head.py -k combiner_feed -x` | ❌ Wave 0 |
| WIRE-02 | smoke-train loop runs ≥1 epoch, loss decreases, wandb logged | integration smoke (tiny synthetic batch OK — head only, no frozen model) | manual / `scripts/smoke_train_condA.py` | ❌ Wave 0 |
| WIRE-03 | zone assignment from markers; pseudobulk per zone | unit (fixture AnnData) | `pytest tests/spatial/test_apap_validation.py -k zone -x` | ❌ Wave 0 |
| WIRE-03 | measured DE = APAP_zone − ctrl_zone; mouse→human ortholog align; intersection size reported | unit | `pytest tests/spatial/test_apap_validation.py -k measured_de -x` | ❌ Wave 0 |
| WIRE-03 | per-zone Pearson math + Halt-Gate-3 trigger (<0.3 pericentral) | unit | `pytest tests/spatial/test_apap_validation.py -k pearson_gate -x` | ❌ Wave 0 |

**Fixture-testable PURE logic (no real model):** rule-B subtraction, 3-vector cache assembly, manifest convention string, zone assignment, measured-DE subtraction, ortholog alignment, per-zone Pearson, Halt-Gate-3 threshold. These cover the bulk of WIRE-01/02/03 acceptance ("no NaNs, manifests align, working forward pass, Pearson reported") deterministically.

**Real-inference smoke (CANNOT be mocked, Hard Rule 1):** checkpoint strict-load, one forward producing a finite in-range 10,716 vector, normalization sanity. Mark `@pytest.mark.gpu`/slow; run once per wave merge, not per commit.

### Sampling Rate
- **Per task commit:** quick run on the touched test file (`pytest tests/spatial/test_<file>.py -x -q`).
- **Per wave merge:** full pure suite + the real-inference smoke (`pytest tests/ -q -m "not gpu"` then the gpu smoke once).
- **Phase gate:** full suite green + APAP per-zone Pearson reported (WIRE-03 acceptance) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/spatial/test_region_signature.py` — EXTEND for rule-B + 3-vector cache (file exists).
- [ ] `tests/spatial/test_model_load.py` — NEW, real-ckpt strict-load + forward sanity (gpu-marked).
- [ ] `tests/spatial/test_tox_head.py` — NEW, covers WIRE-02.
- [ ] `tests/spatial/test_apap_validation.py` — NEW, covers WIRE-03 + Halt Gate 3.
- [ ] `tests/spatial/conftest.py` — shared fixtures (tiny AnnData with zone markers; synthetic treated/control vectors; a fixed gene-order list slice).

## Security Domain

> security_enforcement: key absent → treated as enabled. This phase is a local research inference/training pipeline with no network-facing surface, no auth, no user input, no untrusted data ingestion at runtime. Standard ASVS web categories (V2/V3/V4) do not apply.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth surface) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | partial | SMILES are validated via rdkit `MolFromSmiles` (P1 precedent: invalid→zero vec + valid_mask=False); checkpoint paths asserted to exist + SHA-pinned (MANIFEST). |
| V6 Cryptography | no | — (no secrets handled beyond the existing wandb key) |

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Stale/wrong checkpoint silently loaded (wrong features) | Tampering / data-integrity | SHA-pin both checkpoints + gene-order file in MANIFEST; assert SHA at load (the kpgt-vs-CheMoE mixup is exactly this risk). |
| Mocked/fabricated model output | data-integrity (Hard Rule 1) | Keep the `NotImplementedError` discipline: pure code testable with fixtures; real inference never mocked; real-inference smoke gated. |
| `torch.load(weights_only=False)` on an untrusted pickle | code-exec | Checkpoints are first-party, SHA-pinned; acceptable. Do not load third-party .pt without SHA. |

## Sources

### Primary (HIGH confidence — verified by execution this session)
- `MultiDCP_CheMoE_pdg/src/best_model.pt` — strict-loaded into `MultiDCP_CheMoE_AE` (0 missing/0 unexpected); dose_encoder (64,2), gene_index_embed (10716,128), cell_encoder.cell_id_embed.0 (200,10716).
- `MultiDCP_CheMoE_pdg/trained_models/chemoe_kpgt_MCF7_fold0/{best_model.pt, predictions.npz}` — keys = `CheMoE_PDG`/KPGT; predictions std 0.001 (collapsed); predictions/treated/diseased shapes (6498,10716).
- `MultiDCP_CheMoE_pdg/src/models/multidcp_chemoe_pdg.py`, `multidcp_pdg.py`, `neural_fingerprint.py` — forward signatures + output `[batch, num_gene]` treated.
- `MultiDCP_CheMoE_pdg/src/multidcp_chemoe_ae_de_pdg.py` — `model.loss(lb, predict)` with lb=treated; GEX_SIZE=10716; `de_np = treated_np - diseased_np` (diagnostic only).
- `MultiDCP_CheMoE_pdg/src/utils/multidcp_ae_pdg_utils.py` — `initialize_model_registry()` (cell_id_input_dim=10716, num_gene=10716, drug_input_dim atom62/bond6).
- `MultiDCP_CheMoE_pdg/src/utils/data_utils_pdg.py` — pert_idose one-hot built from sorted unique dose strings; basal read from CSV per cell_id.
- `pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv` — basal manifold (10,10716), range [0.018,1.000], cols FLNC,MAP2K4,…
- `PDGrapher_Baseline_Models/models/chemoe_pdg/model.py` — `CheMoE_PDG` (KPGT/Morgan, categorical 10-cell-line embed) = the kpgt-checkpoint class.
- `src/spatial/{region_signature.py, region_combiner.py, datasets.py, gene_alignment.py, pseudobulk.py}` — existing pure APIs.
- `MANIFEST.md`, `02-CONTEXT.md`, `PROJECT.md`, `REQUIREMENTS.md`, `STATE.md`.

### Secondary (MEDIUM)
- Training diagnostics `src/diagnostics_de_MCF7_fold1_20260117_084753.csv` — CheMoE health (DE top-20 R²≈0.75, all-genes Pearson≈0.67).

### Tertiary (LOW)
- None — all load-bearing claims were verified by execution.

## Metadata

**Confidence breakdown:**
- Model I/O + checkpoint mechanics + DE semantics (D-02): HIGH — loaded the checkpoints, read the training loop, inspected predictions.npz.
- Architecture mismatch / broken S-B checkpoint: HIGH — three independent verifications (keys, collapse, class location).
- Standard stack + reuse map: HIGH — APIs read directly.
- basal normalization + rule-B control definition: MEDIUM — manifold range verified, but the spatial-side transform and the control input are unimplemented design choices (Open Q2, A2/A3).
- APAP zonation procedure: MEDIUM — data shapes/orthology verified; exact zone thresholds are Claude's Discretion + unwritten.

**Research date:** 2026-06-24
**Valid until:** 2026-07-24 (checkpoints are static first-party files; stable). Re-verify if `MultiDCP_CheMoE_pdg` is re-trained or the checkpoints are replaced.
