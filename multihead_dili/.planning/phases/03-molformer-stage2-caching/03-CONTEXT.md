# Phase 3: MolFormer download + Stage-2 feature caching - Context

**Gathered:** 2026-05-20
**Status:** Ready for planning (executable after P1+P2 checkpoints land)
**Mode:** Auto-generated (discuss skipped via workflow.skip_discuss)

<domain>
## Phase Boundary

Download the frozen MolFormer chemical encoder (HuggingFace `ibm/MoLFormer-XL-both-10pct`) and compute Stage-2 features for all 1,118 DILIst drugs. For each drug, query MODEL_DOSE and MODEL_GEX at all 10 MultiDCP LINCS training cells and mean-pool the outputs, then concat with the frozen MolFormer embedding. Emit `data/processed/dili_features.parquet` ready for Phase 4 (DILI consumer).

### Cell Count Deviation from Design Doc (IMPORTANT)

The design doc specified 9 training cells: A375, A549, BT20, HELA, HT29, MCF7, MDAMB231, PC3, VCAP.

Phase 2 planning discovered the actual PDG-filtered LINCS dataset has **10 cells**, including **HA1E** (a kidney epithelial line). This Phase 3 plan uses **10 cells** (matching training distribution exactly) rather than the 9 cited in the design doc. This deviation is intentional: using the same cell set as training avoids a distribution mismatch at inference time.

**Cells used for inference (10):** A375, A549, BT20, HA1E, HELA, HT29, MCF7, MDAMB231, PC3, VCAP.

### Feature Dimension Confirmation

Design doc cited 1,747-dim total:
- feat_dose: 1-dim (scalar E-Hill prediction, mean-pooled over 10 cells)
- feat_gex: 978-dim (landmark-gene DE vector, mean-pooled over 10 cells)
- feat_embed: 768-dim (MolFormer frozen embedding)
- Total: 1 + 978 + 768 = **1,747-dim** ✓

The 978-dim output from MODEL_GEX requires the **same landmark-gene filter applied at inference as was applied during Phase 2 training**. The plan must load the landmark gene index from the same source used in Phase 2 (likely `data/processed/landmark_genes.json` or the landmark index embedded in the LINCS parquet). Mismatched filter = wrong dimension = Phase 4 crash.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices at Claude's discretion. Use:
- ROADMAP Phase 3 goal + Requirements EMBED-01..05
- Frozen MolFormer (no training, no fine-tuning); `transformers >= 4.40`, may need `trust_remote_code=True` (audit before enabling)
- 10-cell mean-pool inference (deviation from design doc's 9-cell — document in plan)
- Apply same landmark-gene filter at inference as in Phase 2 training so feat_gex stays 978-dim
- Canonical dose for inference: use the same dose convention as MODEL_DOSE/GEX training (likely `'x'` aggregated dose since that's what training data used; confirm in plan)
- Cache output as parquet at `data/processed/dili_features.parquet`
- WandB run group `multihead_dili` (Phase 3 sub-group `stage2_cache`)

### trust_remote_code Audit

Per design doc OQ-5, `trust_remote_code=True` must be audited before any HuggingFace download:
- Review `ibm/MoLFormer-XL-both-10pct` model card and source files on HuggingFace hub
- Confirm the model uses standard `transformers` tokenizer/model classes
- If standard classes are sufficient, use `trust_remote_code=False` (preferred)
- Only set `trust_remote_code=True` if required by the model, and document the reason in the plan

### Canonical Dose for Inference (OQ-2 Resolution)

Determine what dose value MODEL_DOSE was trained to predict and use the identical convention at inference:
- Inspect `data/processed/ehill_train_safe.parquet` dose column(s)
- Inspect Phase 1 training script for what dose is fed to the model
- Most likely: single canonical dose per drug (e.g., 10 μM) or the aggregated `'x'` dose used in E-Hill aggregation
- Document resolution in plan

### Cell Baseline Vectors for MODEL_GEX Inference

MODEL_GEX (MultiDCP-AE fork) requires a per-cell baseline expression vector as context input. The 10 LINCS training cells have baselines available in the same PDG pickle used for training. Phase 3 must:
- Load cell baselines from `data/processed/lincs_train_safe.parquet` or the PDG pickle
- Extract one baseline vector per cell (10 vectors × 978 landmark genes)
- Use these consistently across all 1,118 drug × 10 cell inference calls

</decisions>

<code_context>
## Existing Code Insights

P1 checkpoint (when ready): `multihead_dili/results/checkpoints/chkpt_dose.pt` (MultiDCP-AE forked, trained on E-Hill scalar Hill target).
P2 checkpoint (when ready): `multihead_dili/results/checkpoints/chkpt_gex.pt` (MultiDCP-AE forked, trained on LINCS DE vector; 978-dim post-landmark-filter).
DILIst canonical: `../dili_downstream/data/processed/dili_canonical.csv` (1,118 drugs with SMILES and DILI labels).
Cell baselines for inference: MultiDCP's cell encoder takes a baseline expression vector per cell. The 10 LINCS training cells have baselines available in the same PDG pickle used for training (need to extract per-cell baseline vectors).

Existing Phase 0 code relevant to Phase 3:
- `src/data/scaffold_split.py` — scaffold splitting (not needed for caching, but useful to verify drug identity columns)
- `tests/test_data_paths.py` — sanity test harness to extend for Phase 3 outputs

Phase 3 must NOT modify Phase 1 or Phase 2 directories or checkpoints.

</code_context>

<specifics>
## Specific Ideas

- Implement `src/embed/molformer_wrapper.py` — frozen HF inference, single SMILES → 768-dim. Validate on 10 sample DILIst SMILES.
- Implement `src/stage2/cache_dili_features.py` — load P1+P2 checkpoints, iterate 1,118 drugs × 10 cells × 2 models = 22,360 forward passes; mean-pool per drug; concat with MolFormer embedding; write parquet.
- Audit `trust_remote_code=True` for MolFormer (security check per design doc OQ-5)
- Determine canonical dose for inference (design doc OQ-2 — was "10 μM" tentative; resolve based on what training actually used)
- Write `results/tables/P3_stage2_summary.md` with feature stats (mean, std per pathway), MolFormer download SHA
- Extend `tests/test_data_paths.py` (or write `tests/test_stage2_features.py`) to assert parquet shape, no NaNs, and correct feature dimension

</specifics>

<deferred>
## Deferred Ideas

None — discuss phase skipped.

</deferred>
