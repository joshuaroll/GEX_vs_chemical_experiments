# Phase 1: MODEL_DOSE training - Context

**Gathered:** 2026-05-20
**Status:** Ready for planning
**Mode:** Auto-generated (discuss skipped via workflow.skip_discuss)

<domain>
## Phase Boundary

Train a MultiDCP-AE-based MODEL_DOSE on the leakage-filtered E-Hill dataset (`data/processed/ehill_train_safe.parquet`, 73,955 rows, scalar Hill target per (drug, cell, dose) tuple). The model is the FIRST of three independent pathways feeding the downstream DILI consumer (the others are MODEL_GEX in Phase 2 and frozen MolFormer in Phase 3). MODEL_DOSE's chemical+cell+dose encoder learns to predict a scalar Hill response.

**Training data schema:**
- `ehill_train_safe.parquet`: columns [sig_id, pert_id, pert_type, cell_id, pert_idose, ehill] — 73,955 rows
- `ehill` is a float64 scalar Hill coefficient (dose-response curve fitting output)
- `pert_idose` values are strings like "1.11 um", "10.0 um" — 6 dose levels
- Leakage discipline already applied: DILIst test-set scaffolds and drug names excluded

**Dev data:**
- Source: `/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_dev.csv`
- Same leakage filter (scaffold + drug_name exclusion from dili_split.json `scaffolds_in_test`) must be applied → emit `data/processed/ehill_dev_safe.parquet`

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices at Claude's discretion. Use:
- ROADMAP Phase 1 goal + Requirements DOSE-01..05 + halt gate HG1
- Existing script `MultiDCP/ehill_multidcp_pretrain.py` as the starting point (fork into `src/train/train_model_dose.py`)
- Leakage-filtered training data at `data/processed/ehill_train_safe.parquet`
- WandB project `joshroll/MultiDCP_multihead_dili`, run group `model_dose`
- CUDA hygiene per CLAUDE.md hard rules (argparse --gpu before torch import; always leave one GPU free)

### Key architectural facts from upstream
- Model class: `multidcp.MultiDCPEhillPretraining` from `/raid/home/joshua/projects/MultiDCP/MultiDCP/models/multidcp.py`
- Data loader: `datareader.EhillDataLoader` — currently reads CSV; fork must add parquet support
- Gene feature: loaded separately from a gene file; `num_gene=978`
- Drug graph: loaded from a `--drug_file` (SMILES → atom/bond graph features); `drug_input_dim = {atom: 62, bond: 6}`
- Cell embedding: `cell_id_input_dim=978`, `cell_feature_emb_dim=32`; cell GE loaded from `--cell_ge_file`
- Dose embedding: `pert_idose_emb_dim=4`; input dim = len(dose_levels) = 6
- Optimizer: Adam lr=0.0002; early stopping on best dev Pearson (Spearman used for tracking)
- MultiDCP SHA: `871b8de` (pinned)

### Checkpoint output
- Save full `model.sub_multidcp.state_dict()` + model param registry to `results/checkpoints/chkpt_dose.pt`
- Use `torch.save({'state_dict': ..., 'model_params': ..., 'best_epoch': ..., 'dev_rmse': ..., 'baseline_rmse': ...}, path)` so halt gate verdict is embedded

</decisions>

<code_context>
## Existing Code Insights

Pinned MultiDCP fork: `/raid/home/joshua/projects/MultiDCP/` (SHA: 871b8de — record in MANIFEST.md during plan-phase).

Existing E-Hill training: `MultiDCP/ehill_multidcp_pretrain.py` uses `multidcp.MultiDCPEhillPretraining` with point-wise MSE loss. It currently hard-codes `device = torch.device("cuda")` at module import time (CUDA hygiene violation). Fork must:
1. Move `--gpu` argparse flag BEFORE any torch import (per hard rule)
2. Replace `device = torch.device("cuda")` with device selected from `--gpu` argparse value
3. Add `--safe-parquet` flag: when set, use `ehill_train_safe.parquet` and `ehill_dev_safe.parquet` instead of CSV files
4. Modify save path to `results/checkpoints/chkpt_dose.pt`
5. Log to WandB project `joshroll/MultiDCP_multihead_dili`, group `model_dose`

The EhillDataLoader in `datareader.py` reads from CSV files referenced by `--hill_train_file`, `--hill_dev_file`, `--hill_test_file`. A thin adapter that writes the parquet to a temp CSV (or subclass that reads parquet) is acceptable given the complexity of forking the full DataLoader.

Existing E-Hill dev/test: `/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_{dev,test}.csv` — apply same leakage filter (from `data/processed/dili_split.json` → `scaffolds_in_test` list + drug names in test set) before passing to training.

Supporting data files (must pass as argparse args to the fork):
- `--drug_file`: `/raid/home/joshua/data/MultiDCP/data/drug.csv` (or equivalent)
- `--gene_file`: gene identifier file
- `--cell_ge_file`: cell line GE mapping
- `--all_cells`: `pretrain_cell_list_ehill.p` pickle

</code_context>

<specifics>
## Specific Ideas

- Apply leakage filter to E-Hill dev set (same scaffold/name mask from `dili_split.json`) — emit `data/processed/ehill_dev_safe.parquet`
- Run training to convergence (max_epoch=50, early stopping patience=10 on dev Pearson); cache `results/checkpoints/chkpt_dose.pt`
- Record dev RMSE per epoch; compare final dev RMSE to `np.std(dev_targets)` for halt gate 1
- Write `results/tables/P1_model_dose_summary.md` with training curves, dev metrics, halt-gate status
- SHA-pin MultiDCP fork (871b8de) in MANIFEST.md
- Batch size: 32 (matching upstream default; ehill data is not large — 73K rows fits)
- Save checkpoint in rich dict format so downstream Phase 3 can load model params without re-specifying them

</specifics>

<deferred>
## Deferred Ideas

None — discuss phase skipped.

</deferred>
