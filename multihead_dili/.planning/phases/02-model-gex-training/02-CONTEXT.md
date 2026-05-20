# Phase 2: MODEL_GEX training - Context

**Gathered:** 2026-05-20
**Status:** Ready for planning
**Mode:** Auto-generated (discuss skipped via workflow.skip_discuss)

<domain>
## Phase Boundary

Train a MultiDCP-AE on the leakage-filtered LINCS PDG-filtered data (`data/processed/lincs_train_safe.parquet`, 164,516 rows, 978-dim DE vector target per (drug, cell, dose) tuple). The model is the SECOND of three independent pathways feeding the downstream DILI consumer. MODEL_GEX's chemical+cell+dose encoder learns to predict the differential gene expression vector. Training uses the canonical DE-rule (top-k by |true_DE|).

This phase runs INDEPENDENTLY of Phase 1 (MODEL_DOSE) — they share no parameters and can train on different GPUs in parallel.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices at Claude's discretion. Use:
- ROADMAP Phase 2 goal + Requirements GEX-01..05 + halt gate HG2
- Existing MultiDCP-AE training infrastructure as the starting point (fork the canonical AE script into `src/train/train_model_gex.py`)
- Leakage-filtered training data at `data/processed/lincs_train_safe.parquet`
- Vendored canonical DE-rule evaluator at `src/eval/de_evaluator.py` (copy from PDGrapher_Baseline_Models with provenance comment)
- WandB project `joshroll/MultiDCP_multihead_dili`, run group `model_gex`
- CUDA hygiene per CLAUDE.md hard rules (argparse --gpu before torch import; always leave one GPU free; this phase uses a DIFFERENT GPU from Phase 1 to allow parallel training)

</decisions>

<code_context>
## Existing Code Insights

Pinned MultiDCP fork: `/raid/home/joshua/projects/MultiDCP/` (SHA to record in MANIFEST.md during plan-phase research).
Canonical MultiDCP-AE training entrypoint: TBD by planner — likely `MultiDCP/MultiDCP/models/multidcp_ae_balanceloss.py` or a sibling. Verify in plan-phase research step.
PDG-filtered LINCS pickle: 182,246 rows in original; 164,516 after leakage filter. Contains `sig_id`, `idx`, `pert_id`, `pert_type`, `cell_id`, `pert_idose`, plus 978 gene landmark columns.
Cell lines in LINCS: A375, A549, BT20, HELA, HT29, MCF7, MDAMB231, PC3, VCAP (9 total).
LINCS dev/test sets: need to be carved from the pickle (no separate dev/test files exist; plan should specify the split strategy — likely a held-out fraction of training data, NOT cell-blind or chemical-blind since those would conflict with the leakage filter's design).

</code_context>

<specifics>
## Specific Ideas

- Vendor `TopKEvaluator` and `train_bl_pdg_de.py` conventions from `PDGrapher_Baseline_Models/Biolord/pdgrapher_experiments/` into `multihead_dili/src/eval/de_evaluator.py` with provenance comment pointing back to source
- The DE rule is LOAD-BEARING (project hard rule 2): all R²/regression metrics on expression MUST be on differential expression (treated − diseased), top-k by |true_DE|. NO inline `r2_score(treated, pred)` allowed.
- Compute per-cell dev Pearson + averaged across cells; halt-gate check on mean Pearson ≥ 0.2
- Run training to convergence; cache `chkpt_gex.pt` in `multihead_dili/results/checkpoints/`
- Write `results/tables/P2_model_gex_summary.md` with training curves, per-cell + mean dev Pearson, halt-gate status
- SHA-pin MultiDCP fork in MANIFEST.md

</specifics>

<deferred>
## Deferred Ideas

None — discuss phase skipped.

</deferred>
