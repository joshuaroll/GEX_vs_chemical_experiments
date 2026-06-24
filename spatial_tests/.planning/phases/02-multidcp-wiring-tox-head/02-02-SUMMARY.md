---
phase: 02-multidcp-wiring-tox-head
plan: 02
subsystem: model-wiring
tags: [multidcp-chemoe, frozen-inference, region-de-cache, rule-b, pytorch, cuda]
provides:
  - "Filled region_signature.py seam: real strict-load of MultiDCP_CheMoE_AE (row-17) + real forward returning absolute treated [10716]"
  - "Rule-B DE (predicted_treated - predicted_control) with inert-drug control pass (D-02)"
  - "3-vector RegionSignatureCache (de/treated/control, D-03)"
  - "N_PDG=10716 model I/O + cache space (D-01), no 978-caller regression"
  - "configs/liver_p2.yaml (SHA-pinned checkpoint + gene order, dose, normalization, control, zones)"
  - "scripts/cache_region_de.py CUDA-hygiene driver"
  - "Liver 3-vector predicted-DE cache on disk for human + mouse (499 drugs x 2 zones x 10716)"
affects: [02-03-tox-head, 02-04-apap-gate, P4-per-organ-train, P5-translatability]
tech-stack:
  added: [torch (frozen inference), MultiDCP_CheMoE_pdg (cross-project static import), scipy.stats.rankdata]
  patterns: [pure/impure split (torch confined to model methods), rule-B bias-corrected DE, rank-percentile manifold normalization, CUDA-before-torch hygiene]
key-files:
  created: [configs/liver_p2.yaml, scripts/cache_region_de.py]
  modified: [src/spatial/region_signature.py, tests/spatial/test_region_signature.py]
key-decisions:
  - "S-B (row-18 KPGT checkpoint) descoped as a checkpoint-provenance blocker (D-04 amendment): collapsed/incompatible, NOT wired"
  - "Rule-B control = inert/empty-drug pass with INERT_CONTROL_SMILES='C' (methane), cached for auditability (D-02 amendment)"
  - "Rank-percentile basal normalization (not naive min-max) to match the DENSE training manifold (Pitfall 1 fix)"
duration: ~95min
completed: 2026-06-24
---

# Phase 2 Plan 02: WIRE-01 frozen forward path + liver DE cache Summary

**The two NotImplementedError seams in region_signature.py are now a real frozen MultiDCP-CheMoE forward (strict 0/0 load of row-17 best_model.pt), rule-B DE is wired with a cached inert-control pass, and the liver 3-vector predicted-DE cache is on disk for both human and mouse over the 10,716-gene PDG space.**

## Performance
- **Duration:** ~95 min
- **Tasks:** 3 / 3 complete
- **Files created:** 2 (config + driver); **modified:** 2 (region_signature.py source + its test file)

## Accomplishments
- `load_model` strict-loads `MultiDCP_CheMoE_AE` (alias `MultiDCP_CheMoEBase`) from the D-04 row-17 checkpoint with **0 missing / 0 unexpected keys**, SHA256-pinned to `fbee15f…` (asserted at load), `model.to(device)` before `.double()`, eval + frozen.
- `_call_model` runs one real forward via the repo's `convert_smile_to_feature`/`create_mask_feature` (atom62/bond6, no hand-rolled featurizer) and returns **absolute predicted treated [10716]** (min ~0.06, max ~0.99, mean ~0.62 — on-manifold).
- Rule A → rule B (D-02): `compute_de(treated, predicted_control)`; the per-region inert-control pass (`INERT_CONTROL_SMILES="C"`) is computed once per zone and cached.
- 3-vector cache (D-03): `de_array`, `treated_array`, `control_array`, with `de_array == treated_array - control_array` verified on disk.
- N_PDG=10716 added (D-01); N_LANDMARK=978 kept (no 978-caller regression — 143 pure spatial tests pass).
- Liver cache written for **human** (yu2022 L5, 10693/10716 genes covered) and **mouse** (GSE272564 control APAP0h, 9272/10716 via 15956 one2one orthologs), 499 drugs × 2 zones (periportal/pericentral) × 10716.
- gpu real-inference smoke (`test_model_load.py`) GREEN: strict 0/0 load + finite in-range forward, **not mocked** (Hard Rule 1).

## Task Commits
1. **Task 1: rule-B DE + 3-vector cache + N_PDG (pure)** — `0019e48`
2. **Task 2: fill load_model + _call_model + config** — `5694449`
3. **Task 3: CUDA-hygiene cache driver; write human+mouse cache** — `7504f44`

## Files Created/Modified
- `src/spatial/region_signature.py` — filled `load_model` (strict-load row-17, SHA-pinned), `_featurize_drug`, `_call_model` (absolute treated [10716]); rule-B `compute_de`; 3-vector `RegionSignatureCache`; `assemble_cache` takes a `(pert_id,region)`-keyed `predicted_control_map`; `N_PDG`, `INERT_CONTROL_SMILES`; `run()` does the per-region control pass; `device` kwarg. torch + sys.path confined to the impure methods (pure functions stay torch-free).
- `configs/liver_p2.yaml` — checkpoint path + SHA256, gene-order file + SHA256, dose one-hot `[1.0,0.0]`, basal normalization (manifold range), control-input definition, cache dir, organ/species, zone markers.
- `scripts/cache_region_de.py` — `--gpu` parsed + `CUDA_VISIBLE_DEVICES` set BEFORE torch; `nvidia-smi` auto-detect leaving one GPU free; per-species basal load → canonical-marker zoning → pseudobulk → ortholog map (mouse) → align → rank-normalize → frozen forward → 3-vector cache + manifest JSON; real-inference sanity assertion; MANIFEST provenance + checkpoint-SHA asserts.
- `tests/spatial/test_region_signature.py` — see Deviations (stale rule-A tests aligned to the locked rule-B/3-vector contract; RED contract tests untouched).
- `data/processed/spatial/region_de_cache/{human,mouse}/` — `de_array.npy`, `treated_array.npy`, `control_array.npy`, `manifest.json` (gitignored per umbrella `.gitignore` `*/data/processed/`).

## Decisions & Deviations

### S-B (row-18) checkpoint-provenance descope — D-04 amendment (recorded blocker)
The MANIFEST row-18 "MultiDCP-PDG (S-B)" checkpoint (`chemoe_kpgt_MCF7_fold0/best_model.pt`, SHA `8e9f0e4…`) is **NOT wired**. Research verified it is a `CheMoE_PDG`/KPGT model that is collapsed (constant output, corr≈0), requires a 2304-d KPGT embedding, and uses a categorical 10-cell-line embedding incompatible with a tissue basal. WIRE-01 is satisfied by **S-C alone** (row-17). The load path SHA-asserts to `fbee15f…` and refuses any other checkpoint (`grep chemoe_kpgt == 0`). **Blocker for a later phase:** source/re-derive a non-collapsed `MultiDCP_AE`-PDG checkpoint that takes a 10,716 basal before S-B can be reinstated.

### Rule-B control input — D-02 amendment
No vehicle/DMSO exists in the PDGrapher training vocab, so `predicted_control` is a forward pass with a designated inert reference. **Chosen: `INERT_CONTROL_SMILES="C"` (methane)** — the minimal valid molecular graph the NeuralFingerprint accepts. Its output is an extrapolation (never in training); it is cached (`control_array`, D-03) so the choice is auditable and rule A↔B is recomputable without re-running the model. The empty-graph alternative was avoided because the NeuralFingerprint needs at least one atom.

### Rule 1 (bug) — basal normalization fix (Pitfall 1)
- **Found during:** Task 3 first cache run.
- **Issue:** The naive "0-1 min-max" of a zero-filled aligned basal pinned 99% of genes to the floor (mean 0.018), silently OOD-corrupting every feature. The training manifold is **dense** (verified: mean 0.62, 85% of values > 0.5, almost nothing at the floor).
- **Fix:** rank-percentile normalization of the expressed genes into the manifold range, with missing (unmeasured) genes filled at the manifold midpoint (0.62) instead of 0. Result: human/mouse basals now have mean ~0.51–0.55, spread across [0,1], matching the manifold.
- **Files:** `scripts/cache_region_de.py` (`_normalize_to_manifold`). **Commit:** `7504f44`.

### Rule 3 (blocking) — contradictory Wave-0 test file
- **Found during:** Task 1.
- **Issue:** The Wave-0 commit (`73b9304`) left the stale rule-A tests (`de_convention == "predicted_treated - region_basal"`, region-keyed basal map, `region_basal must be 1-D` message) in the SAME file as the new RED rule-B tests. Rule B (the locked D-02 decision and a plan must_have) necessarily breaks the rule-A assertions — the two contracts are mutually exclusive.
- **Fix:** the superseded rule-A test assertions were corrected to the rule-B / 3-vector / `(pert_id,region)`-keyed-control schema; the RED contract tests (`test_rule_b_de_subtracts_control`, `test_cache_three_vector_schema`, `test_n_pdg_constant`) were left untouched and now pass. The NotImplementedError-stub tests were updated to the real-load contract (`FileNotFoundError` on a missing checkpoint, `RuntimeError` when the model isn't loaded) plus a new `chemoe_kpgt`-absence guard. No test was weakened or deleted to game a RED check.
- **Files:** `tests/spatial/test_region_signature.py`. **Commits:** `0019e48`, `5694449`.

## Known Stubs / Scientific Flags (for the WIRE-03 APAP gate, 02-04)

**threat_flag — near-zonal-invariance of the frozen backbone.** The cached per-zone treated vectors are **identical across pericentral vs periportal** (maxdiff ~6e-8, float32 epsilon), and human ≈ mouse, despite distinct rank-normalized basals (means 0.51 vs 0.55). The CheMoE cell-context encoder contributes little to the output for these healthy-liver basals; the prediction is dominated by drug + dose. The DE (treated − control) is still non-trivial per drug (abs-mean ~0.05, std ~0.02), and the rule-B identity holds. This is the project's instrumented OOD hypothesis (RESEARCH A5), **not a wiring bug** — but it directly threatens Halt Gate 3 (D-08): if predicted DE is zone-invariant, the **pericentral-vs-periportal** measured-DE contrast the APAP gate tests for cannot be reproduced from the predicted side. 02-04 should report the per-zone Pearson honestly and expect this to be the likely failure mode (stop-and-REFRAME per D-09, not a bug to chase).

Not a code stub: no hardcoded empty/placeholder values flow to the cache; the cache is real frozen inference over real Visium basals and real resolved SMILES.

## Self-Check: PASSED
- Files: `src/spatial/region_signature.py`, `configs/liver_p2.yaml`, `scripts/cache_region_de.py`, `data/processed/spatial/region_de_cache/{human,mouse}/de_array.npy` + `manifest.json` — all FOUND.
- Commits `0019e48`, `5694449`, `7504f44` — all FOUND in git log.
- gpu smoke (`test_model_load.py -m gpu`) 2 passed; pure spatial suite 143 passed (2 deselected: test_tox_head/test_apap_validation are RED files owned by 02-03/02-04, logged in `deferred-items.md`).

## Next Phase Readiness
- **02-03 (WIRE-02 tox head):** the cache + `AttentionPoolCombiner(d=10716)` feed are ready; condition-A smoke-train uses a zero GEX tensor and does not need the cache.
- **02-04 (WIRE-03 APAP gate):** the **mouse** liver 3-vector cache is on disk (the gate's predicted side); read the near-zonal-invariance flag above before computing the pericentral Pearson.
