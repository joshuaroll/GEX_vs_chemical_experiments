# ROADMAP: Spatial Cross-Species Toxicity Prediction (MultiDCP-CheMoE)

**Granularity:** coarse (8 phases mapped one-to-one from SPEC P0..P7).
**Coverage:** 19/19 requirements mapped. No orphans.
**Phase numbering:** SPEC phases P0..P7 map to GSD phases 0..7 directly. First active phase = Phase 0 (P0).

This arm EXTENDS the existing `src/spatial/` module (124 passing fixture tests). It does not rebuild it.

## Phases

- [x] **Phase 0: Dataset acquisition & MANIFEST** - Every input dataset on disk, versioned, with gene-space coverage + ortholog map; no model code. [HALT GATE 1 — NOT FIRED]
- [ ] **Phase 1: EDA (the bracket)** - Bound the achievable result before any model runs; floor, ceiling, region + cross-species diagnostics. [HALT GATE 2]
- [ ] **Phase 2: MultiDCP wiring & toxicity head** - End-to-end forward path from drug + region basal to organ-tox logit using the frozen baseline; APAP validity anchor. [HALT GATE 3]
- [ ] **Phase 3: Splits & no-leakage** - Compound-aware within-species + cross-species transfer splits with leakage audits. [HALT GATE 4]
- [ ] **Phase 4: Per-organ train/test (human first)** - Within-species headline condition matrix per organ, ≥3 seeds, key contrasts with bootstrap CIs. [HALT GATE 5]
- [ ] **Phase 5: Cross-species translatability** - Rodent↔human transfer and structure-vs-omics comparison; translatability gap reported honestly.
- [ ] **Phase 6: Confound control & interpretability** - Cytotoxicity/BBB confound control + regional attention attribution, GSEA, routing-permutation diagnostic. [HALT GATE 6]
- [ ] **Phase 7: Robustness & writeup** - Cluster-split robustness, hyperparameter sweep, publication artifacts mapped to the four concepts.

## Phase Details

### Phase 0: Dataset acquisition & MANIFEST
**Goal**: Every planned input dataset is on disk, versioned, with verified gene-space coverage and a usable cross-species ortholog map — and nothing else (no model code).
**Depends on**: Nothing (first phase).
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Every planned input dataset (whole-transcriptome Visium basal, rodent spatial, APAP validation, rodent toxicogenomics, per-organ tox labels) is on disk and versioned, with corrected accessions applied.
  2. `MANIFEST.md` records paths/versions/SHAs/licenses for all datasets and frozen MultiDCP/CheMoE checkpoints, and records the `squidpy` (+ optional `tangram-sc`) env additions.
  3. `src/spatial/orthology.py` produces a human-mouse-rat one-to-one ortholog map and reports the dropped many-to-many fraction.
  4. `tests/test_data_paths.py` is green and per-Visium-dataset coverage against the 10,716-gene space is reported.
**Halt gate**: HALT GATE 1 — any planned input dataset unavailable or not whole-transcriptome → write `HALT_REASON.md`, stop and re-plan source.
**Plans**: TBD

### Phase 1: EDA (the bracket)
**Goal**: Bound the achievable result and check leakage/species structure before any model runs, so we know the floor, the ceiling, and the gap between them.
**Depends on**: Phase 0.
**Requirements**: EDA-01, EDA-02, EDA-03
**Success Criteria** (what must be TRUE):
  1. A structure-only floor (logistic + RF) and label/class-balance/AUPRC base rates are reported per organ+species.
  2. A measured-biology ceiling (effective rank, per-gene MI, measured-signature-only baseline) is reported where measured data exists.
  3. Region diagnostics (basal-profile similarity, spatially-variable-gene retention via Moran's I, gene coverage) and cross-species diagnostics (ortholog overlap, human-vs-rodent basal correlation, OOD distance from the cancer-line manifold) are reported.
  4. `results/tables/P1_eda.md` states the floor-ceiling gap, region distinguishability, and human-rodent basal concordance.
**Halt gate**: HALT GATE 2 — floor-ceiling gap near zero for the organ → write `HALT_REASON.md`, reframe before training.
**Plans**: TBD

### Phase 2: MultiDCP wiring & toxicity head
**Goal**: A working end-to-end forward path from drug + region basal to an organ-tox logit using the frozen baseline, validated against the APAP anchor.
**Depends on**: Phase 1.
**Requirements**: WIRE-01, WIRE-02, WIRE-03
**Success Criteria** (what must be TRUE):
  1. The `region_signature.py` `NotImplementedError` seam is replaced with the real frozen-checkpoint call, and per-region predicted DE (spatial rule `predicted_treated(drug, region_basal) - region_basal`) is cached for the starting organ in both species.
  2. `src/spatial/tox_head.py` (concat-MLP) runs a working forward pass fed by the attention combiner, with per-condition zero-tensor masking and no NaNs.
  3. Condition A smoke-trains with wandb logging confirmed, and manifests align across regions and species.
  4. Predicted-vs-measured per-zone Pearson on the APAP anchor (GSE280652 / GSE272564) is reported.
**Halt gate**: HALT GATE 3 — predicted-vs-acetaminophen per-zone Pearson < 0.3 → write `HALT_REASON.md`, reframe spatial claim.
**Plans**: TBD
**UI hint**: no

### Phase 3: Splits & no-leakage
**Goal**: Compound-aware splits per organ and species, plus a cross-species transfer split, all leakage-audited and large enough to test on.
**Depends on**: Phase 2.
**Requirements**: SPLIT-01, SPLIT-02, SPLIT-03
**Success Criteria** (what must be TRUE):
  1. `src/spatial/splits.py` produces Murcko scaffold (80/10/10) and Tanimoto-0.4 cluster splits per organ+species with no drug crossing partitions and class balance within 5 points of the global rate.
  2. A cross-species transfer split holds the target species fully out, with shared drugs keeping the same partition role (no cross-boundary label leak).
  3. Leakage audits (scaffold overlap, LINCS Phase II time-leakage, ortholog-pairing consistency) pass and are documented.
  4. `results/tables/P3_splits.md` ships split JSONs, Tanimoto train-vs-test histograms, and per-organ drug-novel-slice counts (no silent truncation).
**Halt gate**: HALT GATE 4 — drug-novel slice < 30 for the organ → write `HALT_REASON.md`, adjust split or labels.
**Plans**: TBD

### Phase 4: Per-organ train/test (human first)
**Goal**: The within-species headline result per organ (human first, then rodent): does region-conditioned predicted GEX beat structure-only and cell-line predicted GEX?
**Depends on**: Phase 3.
**Requirements**: TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Conditions A, B/C, S-B/S-C, G/H, and fusion are each trained with ≥3 seeds, human first then rodent, via `src/spatial/train.py`/`eval.py`.
  2. AUROC/AUPRC/MCC on scaffold + cluster splits are reported with paired bootstrap CIs (10,000 resamples) for the B−A, S−B, and fusion−best-single-channel contrasts, plus per-mechanism breakdowns.
  3. Channel-ablation baselines (dose-response-only, GEX-only) are run every organ and fusion beats both.
  4. `results/tables/P4_headline_{organ}_{species}.md` exists with mean ± std across seeds.
**Halt gate**: HALT GATE 5 — B does not beat A by ≥ 0.02 AUROC; OR S does not beat/contextualize B → write `HALT_REASON.md`, consult before Phase 6+.
**Plans**: TBD

### Phase 5: Cross-species translatability
**Goal**: A quantified rodent↔human translatability number per organ, isolating the omics contribution against a structure-only transfer reference.
**Depends on**: Phase 4.
**Requirements**: XSPEC-01
**Success Criteria** (what must be TRUE):
  1. Train-rodent/test-human and the reverse are run per organ.
  2. A structure-only transfer baseline and an omics-based (ortholog-aligned) transfer result are both reported.
  3. `results/tables/P5_translatability_{organ}.md` reports within-species AUROC, transfer AUROC, and the translatability gap honestly (poor transfer is a valid result).
**Halt gate**: None (poor transfer is a reportable result per publishable concept 3/4).
**Plans**: TBD

### Phase 6: Confound control & interpretability
**Goal**: Demonstrate the result is not generic cytotoxicity (or a BBB artifact) and produce the mechanistic story.
**Depends on**: Phase 4 (needs trained headline models); informed by Phase 5.
**Requirements**: CONF-01, CONF-02
**Success Criteria** (what must be TRUE):
  1. A viability ablation (dose-response-only baseline + viability-masked variant) is reported per organ.
  2. For brain: a BBB subset evaluation and BBB-only-proxy ablation are reported (headline beats the proxy by ≥ 0.05 AUROC on BBB-permeable compounds).
  3. Regional attention attribution per organ plus GSEA on top attributed genes give a mechanistically plausible story in `results/tables/P6_*` and `results/figures/P6_*`.
  4. A routing-permutation diagnostic for CheMoE is reported against a control-layer permutation.
**Halt gate**: HALT GATE 6 — routing-permutation ΔAUROC not more negative than a control-layer permutation → write `HALT_REASON.md`, reframe CheMoE claim honestly.
**Plans**: TBD
**UI hint**: no

### Phase 7: Robustness & writeup
**Goal**: Confirm stability and assemble the publication artifacts mapped to the four publishable concepts.
**Depends on**: Phases 4, 5, 6.
**Requirements**: ROBUST-01, ROBUST-02
**Success Criteria** (what must be TRUE):
  1. Cluster-split robustness across all conditions and a hyperparameter sweep on the headline classifier are reported and show stability.
  2. `results/figures/` holds the publication figure set and a results narrative maps each finding to the four publishable concepts.
  3. The SPEC §9 write-up readiness checklist is satisfiable per organ (human then rodent), or a rigorously instrumented negative is documented as the publishable outcome.
**Halt gate**: None.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 0. Dataset acquisition & MANIFEST | 4/4 | Complete    | 2026-06-23 |
| 1. EDA (the bracket) | 0/0 | Not started | - |
| 2. MultiDCP wiring & toxicity head | 0/0 | Not started | - |
| 3. Splits & no-leakage | 0/0 | Not started | - |
| 4. Per-organ train/test (human first) | 0/0 | Not started | - |
| 5. Cross-species translatability | 0/0 | Not started | - |
| 6. Confound control & interpretability | 0/0 | Not started | - |
| 7. Robustness & writeup | 0/0 | Not started | - |

## Halt Gates Summary

| Gate | Phase | Trigger | Action |
|------|-------|---------|--------|
| 1 | 0 | Input dataset unavailable / not whole-transcriptome | Re-plan source |
| 2 | 1 | Floor-ceiling gap near zero | Reframe before training |
| 3 | 2 | Predicted-vs-APAP per-zone Pearson < 0.3 | Reframe spatial claim |
| 4 | 3 | Drug-novel slice < 30 | Adjust split or labels |
| 5 | 4 | B not beating A by ≥ 0.02 AUROC, OR S not beating/contextualizing B | Consult before P6+ |
| 6 | 6 | Routing-permutation ΔAUROC not below control-layer permutation | Reframe CheMoE claim |

Phases 5 and 7 have no halt gate. On any gate firing, write `HALT_REASON.md` into the phase directory and stop — do not bypass.

## Proposed Decisions to Confirm at Phase Planning

These three are PROPOSED (pending professor sign-off), NOT locked. Surface for confirmation at the relevant phase:
- **Spatial DE-rule divergence** (`predicted_treated - region_basal`) — confirm at P2/P4 planning.
- **APAP validity stand-in** (single-drug Visium anchor as per-region validation) — confirm at P2 planning.
- **Negative-result acceptability** (stop-and-reframe vs stop-and-abandon) — confirm before P1/P4 gates can be acted on.
