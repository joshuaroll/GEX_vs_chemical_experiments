# Phase 2: MultiDCP wiring & toxicity head - Context

**Gathered:** 2026-06-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the working end-to-end forward path: **drug + region basal → frozen
MultiDCP/CheMoE → per-region predicted DE (cached) → attention pool → concat-MLP
→ organ-tox logit**, then validate the predicted signature against the mouse APAP
anchor (Halt Gate 3). Concretely this phase fills the two `NotImplementedError`
seams in `src/spatial/region_signature.py` (`load_model`, `_call_model`), caches
per-region predicted DE for the starting organ (liver) in both species, implements
`src/spatial/tox_head.py` (concat-MLP fed by the attention combiner with
per-condition zero-tensor masking), and smoke-trains condition A with wandb.

Requirements in scope: **WIRE-01** (frozen forward path / replace seam + cache),
**WIRE-02** (tox head + smoke train condition A), **WIRE-03** (APAP per-zone
Pearson validity anchor → Halt Gate 3).

**Out of scope (own phases):** compound-aware splits (P3), full multi-condition
training (P4), cross-species transfer (P5), viability/BBB confounds + CheMoE
routing diagnostic (P6). No model retraining (frozen baseline only).

**Starting organ/species:** liver. Cache predicted DE for **both** human (the
eventual P4 headline) and mouse (required for the APAP gate — the only
drug-perturbed spatial anchor is rodent). Human-first ordering per D-01/P1.

</domain>

<decisions>
## Implementation Decisions

### Gene space
- **D-01:** Cache predicted DE and feed the tox head in the **10,716-gene PDG
  space**, not the 978 LINCS-landmark space. Matches `CON-gene-space` (head pools
  top-k of 10,716), the PDG/CheMoE checkpoints' native output, and
  `region_combiner.AttentionPoolCombiner(d=10716)` (N_PDG is already defined).
  The 978→10,716 imputation is the OOD factor we instrument, not avoid.
  `region_signature.py`'s `N_LANDMARK=978` default must be generalized to support
  10,716 without breaking 978 callers.

### Spatial DE rule (supersedes the seam's current convention)
- **D-02:** The per-region predicted signature is **rule B — bias-corrected:**
  `predicted_DE_region = predicted_treated(drug, region_basal) − predicted_control(region_basal)`.
  Subtract the model's *own* predicted vehicle/control output for the same region
  context (not the raw observed basal). Rationale: keeps both terms in the model's
  output manifold (cancels basal-reconstruction error / observed-vs-predicted
  mismatch) and makes predicted DE the **same kind of quantity** as the measured
  `treated − control` the APAP gate tests against. Cost: +1 control forward pass
  per region (cheap). This **replaces** `region_signature.py`'s current
  `compute_de` rule A (`predicted_treated − region_basal`) and the manifest
  `de_convention` string. Locks `DEC-de-rule-spatial-divergence` with the
  bias-corrected variant. **Conditional:** the researcher MUST confirm the
  checkpoint emits *absolute predicted treated expression* (so a control pass is
  meaningful). If the model emits a DE head natively, escalate before wiring — do
  not double-subtract.
  **POST-RESEARCH AMENDMENT (2026-06-24):** research loaded the working checkpoint
  and CONFIRMED it emits absolute predicted treated (not a DE head) — rule B is
  valid, no double-subtract. BUT there is no vehicle/DMSO in the training vocab, so
  `predicted_control` has no canonical input. **User decision: control = an
  inert/empty-drug control pass** — a forward pass with a designated inert reference
  (empty drug graph or a fixed DMSO-like SMILES) in the same region basal context.
  Caveat: that reference was never in training, so its output is an extrapolation;
  cache `predicted_control` (D-03) so the choice is auditable/re-derivable.
  (Autoencoder basal-reconstruction and revert-to-rule-A were the alternatives.)

### Cache contents
- **D-03:** Persist the **final DE plus the intermediate predicted_treated and
  predicted_control vectors** (not DE-only). Enables auditing reconstruction
  error, recomputing rule A↔B without re-running the frozen model, and the APAP
  comparison in either DE or absolute form. ~2–3× cache size, acceptable for one
  organ. (Current `RegionSignatureCache` stores `de_array` only — extend it.)

### Frozen-backbone scope this phase
- **D-04 (AMENDED 2026-06-24 post-research):** Wire **MultiDCP-CheMoE (condition
  S-C) end-to-end first** via `MultiDCP_CheMoE_AE` + the **row-17** checkpoint
  `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt` (SHA
  `fbee15f…`) — research VERIFIED it as the only working, tissue-basal-capable
  frozen backbone (0 missing keys, 10,716 basal, DE R²≈0.75). **S-B is DESCOPED
  from Phase 2:** the MANIFEST row-18 "MultiDCP-PDG" checkpoint
  (`chemoe_kpgt_MCF7_fold0/best_model.pt`) is actually a `CheMoE_PDG`/KPGT model,
  is **collapsed** (constant output, corr≈0), needs KPGT 2304-d embeddings, and uses
  a categorical 10-cell-line embedding incompatible with a tissue basal — DO NOT
  WIRE. No working PDG/10,716-basal checkpoint exists on disk, so S-B is a
  **checkpoint-provenance blocker** recorded for a later phase (source/re-derive a
  non-collapsed `MultiDCP_AE` checkpoint). This reverses D-04's original ordering
  but honors its intent ("de-risk the seam first" — S-B's checkpoint IS the risk).
  WIRE-01/02/03 are satisfied by S-C alone.

### Post-research implementation notes (2026-06-24; planner-level, no user decision)
- **Basal normalization landmine:** the model's basal input must be **0-1 min-max
  normalized** to the training manifold range (≈[0.018, 1.000], mean 0.62) before the
  forward pass, or every cached feature is silently OOD-corrupted. Apply on the
  spatial side after Visium→10,716 alignment.
- **Dose is 2-dim, not 6-way** (contradicts CON-model-io's "6-way"): the working
  checkpoint trained on a single dose. Use a fixed valid 2-dim one-hot; the cached
  signature is effectively dose-agnostic (consistent with D-05 "don't gate on
  magnitude").
- **APAP control arm:** GSE280652 has **no matched control arm** → use GSE272564's
  control as the primary gate reference (D-06/D-08); GSE280652 becomes a weaker
  partial replication, not an independent matched-DE anchor.

### Dose conditioning for the GEX signature
- **D-05:** Condition the cached per-region GEX signature on a **fixed canonical
  reference dose/time (10 µM / 24 h, LINCS standard)** for all drugs. The
  dose-*response* channel (conditions G/H) handles dose-dependence separately and
  later. Caveat: in-vivo APAP dose ≠ in-vitro µM, so the APAP Pearson tests the DE
  **pattern across genes, not magnitude** — a fixed dose is fine and must not be
  gated on magnitude.

### APAP validity anchor + Halt Gate 3
- **D-06:** **Anchor = GSE272564 primary, GSE280652 independent replication.**
  GSE272564 carries matched control + APAP arms in one series (same tissue as the
  basal context) → cleanest measured `treated − control` DE, pairing naturally with
  rule B's `model(APAP,basal) − model(control,basal)`. GSE280652 reported as a
  second, independent anchor. Both are validation-only (never basal input).
- **D-07:** **Zone definition = published zonation annotation if present, else
  derive periportal/pericentral from canonical zonation markers** (pericentral
  Glul/Cyp2e1; periportal Sds/Cyp2f2). Unsupervised Leiden only as last resort.
  (Matches `DEC-region-granularity-published`.)
- **D-08:** **Halt Gate 3 metric = Pearson of predicted vs measured DE across the
  10,716 PDG genes** (intersection with Visium coverage; `fill_value=0` /absent
  genes flagged, not silently included), computed **per zone**. Report all zones;
  the gate (`< 0.3`) is **keyed to the pericentral zone** — where APAP injury
  classically acts — because nailing the right regional effect is the actual claim.
- **D-09:** **Validity stance:** accept the **rodent APAP anchor as the direct
  validity stand-in** (no human drug-perturbed spatial data exists as of mid-2026);
  per-region predicted DE remains the **primary, indirect** validity path. Locks
  `DEC-indirect-validity-plus-apap`. A fired Halt Gate 3 = **stop-and-REFRAME** the
  spatial claim (not abandon), consistent with P1's D-02 precedent and
  `DEC-negative-result-acceptable`.

### Claude's Discretion (research/planner decide)
- basal → cell-context projection: how the region pseudobulk basal (978 or 10,716)
  is mapped to the model's expected input + 50-d cell-context encoder.
- top-k size for the region-pooled DE (top-k of 10,716) in the attention combiner.
- wandb project/run naming and smoke-train epoch count for condition A.
- exact pericentral/periportal marker thresholds and the zone-assignment procedure.
- cache on-disk layout (npy + manifest JSON) for the extended multi-vector cache.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase spec + requirements (source of truth)
- `/raid/home/joshua/projects/0_project_documents/downstream_spatial_xspecies_MDCPMoE_06172026.md` §"Phase 2 — MultiDCP wiring and the toxicity head" — tasks, deliverable, Halt Gate 3.
- `.planning/ROADMAP.md` §"Phase 2: MultiDCP wiring & toxicity head" — success criteria + Halt Gate 3.
- `.planning/REQUIREMENTS.md` — WIRE-01, WIRE-02, WIRE-03 (definitions + acceptance).
- `.planning/PROJECT.md` — `CON-model-io`, `CON-tox-head`, `CON-gene-space`, `CON-splits`, `CON-spatial-qc`, `CON-code-module-layout`; `DEC-*` decisions; Hard Rules.

### Channel architecture + resolved decisions (DOCs)
- `/raid/home/joshua/projects/0_project_documents/downstream_tasks/06_three_channel_pipeline.md` — channel-to-code map, three-channel architecture (chem + region GEX + dose-response).
- `/raid/home/joshua/projects/0_project_documents/downstream_tasks/09_spatial_decisions.md` — open items #1 (negative-result stance) and #3 (DE-rule divergence). **Now resolved here: #3 → D-02 (rule B bias-corrected); #1 → D-09 (rodent anchor accepted, stop-and-reframe).**

### Frozen checkpoints + upstream model code (provenance)
- `MANIFEST.md` rows 17–19 — MultiDCP-CheMoE `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt` (SHA `fbee15f…`); MultiDCP-PDG `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/trained_models/chemoe_kpgt_MCF7_fold0/best_model.pt` (SHA `8e9f0e4…`); MultiDCP repo SHA `871b8de`. **Both .pt files confirmed on disk (32 MB / 22 MB).**
- `../dili_downstream/src/models/upstream/multidcp_pdg.py` — MultiDCP-PDG class (Condition S-B), SHA-pinned `871b8de`.
- `../dili_downstream/src/models/upstream/multidcp_pdgrapher_fusion.py` — CheMoE fusion class (Condition S-C).

### Code seams to fill / extend (this repo)
- `src/spatial/region_signature.py` — the `NotImplementedError` seams (`load_model`, `_call_model`); pure parts (`compute_de`, `build_manifest`, `make_cache_key`, `assemble_cache`) already tested. **Its `load_model`/`_call_model` TODO references STALE checkpoint paths — use MANIFEST's resolved paths above.**
- `src/spatial/region_combiner.py` — `AttentionPoolCombiner(d=10716)`, emits per-region attention weights.
- `src/spatial/datasets.py` — APAP registry entries (`gse272564_apap_liver`, `gse280652_apap_liver`, validation-only).
- `src/spatial/gene_alignment.py` / `pseudobulk.py` — Visium→10,716 alignment + region pseudobulk basal.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `region_signature.py`: pure DE/cache machinery done (`compute_de`, `build_manifest`, `make_cache_key`, `assemble_cache`, `RegionSignatureCacher`). Phase 2 fills `load_model` + `_call_model` and updates the DE convention per D-02.
- `region_combiner.py`: `AttentionPoolCombiner(d)` already supports `d=10716`; the per-region attention weights are the interpretability output (used in P6).
- `gene_alignment.py` / `pseudobulk.py`: align Visium genes to the 10,716 space and pseudobulk region basal.
- `datasets.py`: APAP anchors registered (validation-only; must never be used as basal input).

### Established Patterns
- Real-data-only / provenance-before-progress: checkpoints + APAP data are in MANIFEST with SHAs; confirmed on disk.
- Cross-project static reuse (as in P1): import upstream model classes from `../dili_downstream/src/models/upstream/` as a read-only dependency; do not touch `/raid/home/joshua/.planning`.
- DE-rule discipline: spatial DE only; raw expression as feature/metric forbidden; top-k retained.

### Integration Points
- Cached DE (10,716, rule B) → `region_combiner` attention pool (top-k) → `tox_head.py` concat-MLP → organ-tox logit. Inactive channel = zero tensor of identical shape (condition A smoke-train uses a zero GEX tensor and does NOT need the cache).
- APAP gate reads predicted DE (from cache) vs measured DE (from GSE272564/GSE280652 Visium) per zone.

### Landmines
- Seam TODO points at stale `dili_downstream/trained_models/...best.pt` paths — use MANIFEST's `MultiDCP_CheMoE_pdg/...` paths.
- `N_LANDMARK=978` hardcoded; D-01 needs 10,716 — generalize without breaking the 978 default.
- `compute_de` implements rule A; D-02 selects rule B — update the subtraction + the `de_convention` manifest string; confirm the model emits absolute treated (not a DE head) first.
- CUDA hygiene (Hard Rule 5): set `--gpu`/`CUDA_VISIBLE_DEVICES` BEFORE `import torch`; auto-detect; always leave one GPU free.
- APAP magnitude caveat (D-05): gate on DE pattern, not magnitude.

</code_context>

<specifics>
## Specific Ideas

- The APAP gate is the phase's only direct-validity evidence and it is **rodent**;
  the headline organ is human. This asymmetry is accepted (D-09) and must be stated
  honestly wherever the gate result is reported.
- Smoke-train scope (WIRE-02) = condition **A** (structure-only, zero GEX tensor):
  it validates the head, training loop, and wandb logging, independent of the cache.
- Cache predicted DE for liver in BOTH human and mouse (D-01 scope): human for the
  P4 headline, mouse for the APAP gate.

</specifics>

<deferred>
## Deferred Ideas

- CheMoE routing-permutation diagnostic (Halt Gate 6) — Phase 6.
- Dose-response channel (conditions G/H, E-Hill) — later; D-05 fixes a single dose for the GEX signature only.
- Compound-aware splits + leakage audits — Phase 3.
- Full multi-condition training (A/B/C/S-B/S-C/fusion, ≥3 seeds) — Phase 4.
- Other organs (kidney, brain, heart) — their own training phases; heart last.
- Promoting the targeted-panel datasets via Tangram — out of scope (Visium-only basal input, DEC-platform-visium-only).

None of the above is scope creep — all are explicit later-phase sequencing.

</deferred>

---

*Phase: 2-multidcp-wiring-tox-head*
*Context gathered: 2026-06-24*
