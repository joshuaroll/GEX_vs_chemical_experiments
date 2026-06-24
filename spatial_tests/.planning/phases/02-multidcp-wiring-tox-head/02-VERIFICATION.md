---
phase: 02-multidcp-wiring-tox-head
verified: 2026-06-24T20:05:00Z
status: passed
score: 4/4 roadmap success criteria verified (WIRE-01/02/03 delivered); Halt Gate 3 fired + recorded as designed
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
  note: initial verification
downstream_gate_review: # NOT a gap — a scientific result for the human to review next (P1 precedent)
  - "Halt Gate 3 FIRED by design (pericentral predicted-vs-measured Pearson +0.0166 < 0.3). Per D-09 this is stop-and-REFRAME, not failure. The load-bearing finding — frozen CheMoE backbone is near-zonal-invariant on healthy-liver basals (verified: max |de[pericentral]-de[periportal]| = 5.96e-08 across 10,716 genes) — must be gate-reviewed before Phase 3 proceeds on the spatial-conditioning premise."
---

# Phase 2: MultiDCP wiring & toxicity head — Verification Report

**Phase Goal:** A working end-to-end forward path from drug + region basal to an organ-tox logit using the frozen baseline, validated against the APAP anchor.
**Verified:** 2026-06-24T20:05:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

This phase delivered every engineering must-have AND correctly fired and recorded its halt gate. Halt Gate 3 firing is the designed, correct outcome (D-09 stop-and-REFRAME), directly analogous to Phase 1's Halt Gate 2. It is a scientific result the human reviews at the next gate, not a wiring failure. All four ROADMAP success criteria are VERIFIED against the live codebase and on-disk artifacts; the negative validity result is reported honestly.

### Observable Truths (ROADMAP success criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `region_signature.py` `NotImplementedError` seam replaced with the real frozen row-17 CheMoE call; rule-B per-region DE cached for liver in both species | ✓ VERIFIED | `load_model` (region_signature.py:594-703) strict-loads `MultiDCP_CheMoE_AE`, SHA-pinned `fbee15f…`, no fallback; `_call_model` (731-788) runs one real `torch.no_grad` forward returning absolute treated [10716]. On disk: `data/processed/spatial/region_de_cache/{human,mouse}/{de,treated,control}_array.npy` each (499, 2, 10716); manifest `de_convention = "predicted_treated(drug) - predicted_control(region_basal)"`; `de == treated - control` confirmed (`np.allclose` True); human ≠ mouse arrays (real per-species inference, not a copy). |
| 2 | `tox_head.py` concat-MLP runs a working forward fed by the attention combiner, zero-tensor masking, no NaNs | ✓ VERIFIED | `ToxHead` (tox_head.py:102-255): proj_gex/proj_chem/proj_dr → 3-layer MLP (BatchNorm+GELU+Dropout) → single logit; zeroed channel projects to a finite bias (no special-casing). `test_tox_head.py::test_combiner_feed` wires `AttentionPoolCombiner(d=10716).pooled → ToxHead → finite logit`. Pure suite green. |
| 3 | Condition A smoke-trains with wandb confirmed; manifests align across regions and species | ✓ VERIFIED | `scripts/smoke_train_condA.py`: zero GEX tensor + ECFP4 chem, BCEWithLogits, real DILIrank labels (499 drugs, 62.7% positive). Verified run: loss 0.6540→0.0502 over 5 epochs, device=cuda, wandb offline synced, `smoke_exit=0` (02-03-SUMMARY:58). Both manifests share regions `[pericentral, periportal]`, 499 pert_ids, 10716 genes. |
| 4 | Predicted-vs-measured per-zone Pearson on the APAP anchor reported | ✓ VERIFIED | `results/tables/P2_apap_validation.md`: pericentral r=+0.0166 (n=9,272), periportal r=+0.0038 (n=9,272), GSE272564 matched arms (D-06 primary), present_mask intersection (D-08), rodent-anchor caveat stated. Halt Gate 3 verdict: FIRES → stop-and-REFRAME. |

**Score:** 4/4 ROADMAP success criteria verified.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| WIRE-01 | 02-02 | Frozen-model forward path (replace seam), cache per-region DE both species | ✓ SATISFIED | region_signature.py seam filled (strict 0/0 load, real forward); 3-vector human+mouse cache on disk (truth 1). Rule amended A→B (D-02) and gene space 978→10716 (D-01) per resolved decisions. |
| WIRE-02 | 02-03 | tox_head concat-MLP, combiner feed, zero-tensor masking, smoke-train condition A + wandb | ✓ SATISFIED | tox_head.py + smoke_train_condA.py; combiner feed tested; loss moved; wandb logged (truths 2,3). |
| WIRE-03 | 02-04 | APAP per-zone Pearson reported; Halt Gate 3 | ✓ SATISFIED | apap_validation.py + run_apap_validation.py; per-zone Pearson reported; gate fired and recorded (truth 4 + halt). Acceptance ("working forward pass; no NaNs; manifests align; per-zone Pearson reported") met. |

No orphaned requirements: REQUIREMENTS.md maps only WIRE-01/02/03 to P2, all claimed and delivered.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/spatial/region_signature.py` | filled load_model+_call_model, rule-B compute_de, 3-vector cache, N_PDG=10716 | ✓ VERIFIED | 883 lines; real strict-load, INERT_CONTROL_SMILES, N_PDG=10716 with N_LANDMARK=978 retained; pure functions torch-free. |
| `src/spatial/tox_head.py` | ToxHead concat-MLP + ToxHeadOutput, zero-channel masking | ✓ VERIFIED | 256 lines; concat-MLP, NamedTuple output, attn passthrough for P6. |
| `src/spatial/apap_validation.py` | zone assignment, measured_zone_de, zone_pearson, halt_gate_3_fires | ✓ VERIFIED | 425 lines; D-07 markers, present_mask flag-not-zero (D-08), pericentral-keyed gate. |
| `configs/liver_p2.yaml` | ckpt path+SHA, dose, normalization, control def, cache dir | ✓ VERIFIED | present (contains best_model.pt). |
| `scripts/cache_region_de.py` | CUDA-hygiene 3-vector cache driver | ✓ VERIFIED | --gpu before import torch (line 41-44); CUDA_VISIBLE_DEVICES set early. |
| `scripts/smoke_train_condA.py` | condition-A smoke-train driver (zero GEX, BCEWithLogits, wandb) | ✓ VERIFIED | CUDA hygiene; wandb; loss-moved assertion. |
| `scripts/run_apap_validation.py` | APAP gate driver, write report, fire Halt Gate 3 | ✓ VERIFIED | GSE272564 matched arms; HALT_REASON write path. |
| `results/tables/P2_apap_validation.md` | per-zone Pearson + intersection + verdict + caveat | ✓ VERIFIED | all present. |
| `tests/spatial/test_model_load.py` | gpu-marked real-ckpt strict-load + forward smoke | ✓ VERIFIED | 2 `@pytest.mark.gpu` tests; strict 0/0 load asserted, not mocked. |
| `pytest.ini` | gpu marker registration | ✓ VERIFIED | `gpu:` marker registered (line 3). |
| `HALT_REASON.md` | gate-fire record, stop-and-reframe framing | ✓ VERIFIED | per-zone table, mechanism, D-09 stop-and-REFRAME decision. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| scripts/cache_region_de.py | RegionSignatureCacher.load_model+run | row-17 best_model.pt, human+mouse | ✓ WIRED | cache on disk, both species distinct. |
| _call_model | MultiDCP_CheMoE_AE.forward | input_cell_gex + dose + no_grad | ✓ WIRED | real forward, absolute treated [10716]. |
| compute_de | de_array (rule B) | predicted_treated − predicted_control | ✓ WIRED | de == treated − control verified on disk. |
| AttentionPoolCombiner(d=10716) | ToxHead.forward (proj_gex) | .pooled feeds proj_gex; attn passthrough | ✓ WIRED | test_combiner_feed green. |
| smoke_train_condA.py | ToxHead | zero GEX + ECFP4; BCEWithLogits; wandb | ✓ WIRED | loss 0.654→0.050 logged. |
| run_apap_validation.py | mouse rule-B cache | predicted-DE source | ✓ WIRED | mouse cache loaded, acetaminophen present. |
| measured_zone_de | zone_pearson | APAP−ctrl, ortholog align, present_mask | ✓ WIRED | report: n=9,272 per zone. |
| halt_gate_3_fires | HALT_REASON.md | pericentral<0.3 → write + reframe | ✓ WIRED | HALT_REASON.md written, gate_fires True. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| mouse de cache | de_array | real CheMoE forward (rule B) | Yes — (499,2,10716) float, de==treated−control, drug-specific | ✓ FLOWING |
| human de cache | de_array | real CheMoE forward (rule B) | Yes — distinct from mouse, range [-0.40, 0.54] | ✓ FLOWING |
| P2_apap_validation.md | per-zone r | mouse cache vs GSE272564 measured | Yes — but predicted side is near-zonal-invariant (5.96e-08 zone delta) → low correlation BY FINDING, not by disconnection | ✓ FLOWING (negative result, instrumented) |
| smoke-train logit | loss curve | ToxHead on DILIrank labels | Yes — loss 0.654→0.050 | ✓ FLOWING |

The pericentral r≈0 is the project's instrumented OOD hypothesis surfacing (near-zonal-invariance), independently verified here (max inter-zone |DE| = 5.96e-08). This is a real, load-bearing scientific finding, not a hollow/disconnected pipeline.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Pure test suite (region_signature, tox_head, apap_validation) | pytest -m "not gpu" | 64 passed | ✓ PASS |
| Rule-B identity on disk | de == treated − control | np.allclose True | ✓ PASS |
| Cache shapes | load mouse/human de_array | (499, 2, 10716) both | ✓ PASS |
| Near-zonal-invariance (the finding) | max |de[z0]−de[z1]| | 5.96e-08 | ✓ PASS (confirms mechanism) |
| Human ≠ mouse cache | array_equal | False (real per-species) | ✓ PASS |
| gpu marker registered | grep pytest.ini | present | ✓ PASS |

GPU real-inference test (`test_model_load.py`) was NOT re-run here (requires the row-17 checkpoint + a free GPU); its prior GREEN status (strict 0/0 load, finite in-range forward) is corroborated by the on-disk cache, which could only have been produced by a successful real forward.

### Anti-Patterns Found

None blocking. The grep-flagged `return null`/empty patterns are not present in the delivered modules; the only "stub-shaped" item is `INERT_CONTROL_SMILES = "C"` (methane), which is an intentional, documented design choice (D-02 amendment), cached for auditability, not a placeholder. S-B/row-18 is explicitly DESCOPED (D-04 amendment) with a recorded provenance blocker — by instruction this is NOT a gap.

### Human Verification Required

None as an automated-verification blocker. The phase is engineering-complete.

The Halt Gate 3 result is a SCIENTIFIC decision for the next gate review (mirrors P1's negative-result gate), recorded under `downstream_gate_review` in the frontmatter rather than as a blocking human-test item:

- **Gate review before Phase 3:** Phase 3+ must not proceed on the spatial-conditioning premise until a human gate-reviews the near-zonal-invariance finding (frozen cancer-line CheMoE encoder contributes near-nothing for healthy-liver basals; prediction dominated by drug+dose). The reframe is already drafted in HALT_REASON.md + P2_apap_validation.md.

### Gaps Summary

No engineering gaps. All WIRE-01/02/03 deliverables exist, are substantive, are wired end-to-end, and real data flows through them. Rule B (D-02), N_PDG=10716 (D-01), 3-vector cache (D-03), and the S-C-only / S-B-descope (D-04 amendment) are all reflected in code and ROADMAP/CONTEXT. Halt Gate 3 fired correctly and is recorded with the prescribed stop-and-REFRAME framing (D-09), consistent with the P1 precedent. The phase goal — a working end-to-end forward path from drug + region basal to an organ-tox logit, validated against the APAP anchor — is achieved; the validation returned a negative (the predicted signature does not recover the pericentral APAP pattern under the frozen backbone), which is itself the reportable finding the human reviews next.

---

_Verified: 2026-06-24T20:05:00Z_
_Verifier: Claude (gsd-verifier)_
