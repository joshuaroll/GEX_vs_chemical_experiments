# Phase 2: MODEL_GEX Training — Research

**Researched:** 2026-05-20
**Researcher:** Orchestrator (inline — canonical entrypoints confirmed via direct inspection)

---

## RESEARCH COMPLETE

---

## 1. Canonical MultiDCP-AE Entrypoint

**Confirmed script:** `/raid/home/joshua/projects/MultiDCP/MultiDCP/models/multidcp_ae_balanceloss.py` (339 lines)

This is the canonical AE training entrypoint used in the upstream pipeline. It:
- Jointly trains an autoencoder (AE) path and a perturbed-GEX prediction path per epoch
- Uses `multidcp_balanceloss.MultiDCP_AE` (class in `/raid/home/joshua/projects/MultiDCP/MultiDCP/models/multidcp_balanceloss.py`)
- Argparse entrypoint with `--drug_file`, `--gene_file`, `--train_file`, `--dev_file`, `--test_file`, `--batch_size`, `--max_epoch`, `--seed`, `--dropout`, `--saved_model_name`
- WandB logging is already integrated (`wandb.init`, `wandb.log`, `wandb.watch`)
- **PROBLEM:** GPU is set via `device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")` at module level — no `--gpu` argparse. This violates CLAUDE.md hard rule 5. Must be fixed in fork.

**Most recent PDG-AE script with correct GPU hygiene:**
`/raid/home/joshua/projects/MultiDCP/MultiDCP/multidcp_ae_de_pdg_12192025.py`

This more recent script has:
- Correct `--gpu` argparse pattern: parse `--gpu` BEFORE `import torch`, sets `CUDA_VISIBLE_DEVICES`
- DE-aware evaluation (same methodology as `train_bl_pdg_de.py`)
- `--safe-parquet` is not present in either — must add in our fork

**Strategy:** Fork `multidcp_ae_balanceloss.py` (simpler, cleaner structure) but apply:
1. GPU hygiene from `multidcp_ae_de_pdg_12192025.py` (parse `--gpu` before torch import)
2. `--safe-parquet` flag to accept `lincs_train_safe.parquet` directly
3. DE-rule evaluation metrics from vendored `de_evaluator.py`

**MultiDCP SHA:** `871b8de` (HEAD of `/raid/home/joshua/projects/MultiDCP/`)

---

## 2. MultiDCP_AE Model Architecture

From `multidcp_balanceloss.py` (lines 378–440):

```python
class MultiDCP_AE(MultiDCPBase):
    def __init__(self, device, model_param_registry):
        self.guassian_noise = GaussianNoise(device=device)
        self.relu = nn.ReLU()
        self.linear_final = nn.Linear(model_param_registry['hid_dim'], 1)
        self.decoder_linear = nn.Sequential(nn.Linear(50, 200), nn.Linear(200, model_param_registry['cell_decoder_dim']))
        self.fusion_type = model_param_registry.get('fusion_type', 'concat')
```

- Dual-path: `job_id='ae'` (encoder→decoder on cell GEX) vs `job_id='perturbed'` (full MultiDCP forward)
- Output shape: `[batch, num_gene=978]`
- Input: cell gene expression (978-dim), drug SMILES graph, gene topology, mask, dose
- Saved via `torch.save(model.state_dict(), args.saved_model_name)` when dev Pearson improves

**model_param_registry keys required:**
- `num_gene`: 978 (landmark genes)
- `pert_idose_input_dim`: number of dose levels (6 in original DATA_FILTER)
- `dropout`: float (default 0.3)
- `linear_encoder_flag`: bool
- `fusion_type`: str (default 'concat')

---

## 3. LINCS Training Data Schema

**File:** `data/processed/lincs_train_safe.parquet` (164,516 rows × 10,722 columns — confirmed 2026-05-20)

**Metadata columns:** `sig_id`, `idx`, `pert_id`, `pert_type`, `cell_id`, `pert_idose`

**Gene columns:** 10,716 total (full L1000 landmark + inferred space, named by gene symbol e.g. `FLNC`, `MAP2K4`, `SGCD`). These are NAMED, not integer-indexed.

**IMPORTANT:** MultiDCP uses only 978 LANDMARK genes from `gene_vector.csv`. The training fork must filter the parquet to these 978 genes before creating DataLoader tensors.

**Cell lines in data:** A375, A549, BT20, **HA1E**, HELA, HT29, MCF7, MDAMB231, PC3, VCAP — **10 cells** (includes HA1E which was not in the CONTEXT.md list of 9).

**`pert_idose` values:** all `'x'` (dose-aggregated LINCS signatures, not raw dose-resolved). This means `pert_idose_input_dim` in the model param registry may need to be 1 instead of 6. The executor should inspect `datareader.py` to confirm how `pert_idose='x'` is handled.

**No separate dev/test parquets exist.** Must carve from the 164,516-row parquet.

**Dev/test split strategy (decided):** 10% random hold-out for dev, 5% for test (all sampled BEFORE loading into DataLoader, using `sklearn.model_selection.train_test_split` with `random_state=343`). This is NOT cell-blind or chemical-blind — those strategies would conflict with the leakage filter design.

**Parquet loading:** Use `pd.read_parquet(args.safe_parquet)` when `--safe-parquet` flag is given; subset to landmark genes using `gene_vector.csv` index; write temp CSVs for datareader compatibility.

---

## 4. Canonical DE Evaluator (Vendoring Plan)

**Source:** `/raid/home/joshua/projects/PDGrapher_Baseline_Models/Biolord/pdgrapher_experiments/train_bl_pdg_de.py` (1016 lines)

**Functions to vendor** (extract into `src/eval/de_evaluator.py`):
- `compute_topk_by_differential_expression(true_de, k=20)` — returns `[n_samples, k]` indices
- `compute_global_metrics(true_de, pred_de)` — Pearson/Spearman/R²/RMSE/MAE over all values
- `compute_persample_metrics(true_de, pred_de, gene_indices=None)` — per-sample Pearson/R² aggregated
- `compute_prediction_bias(true_de, pred_de)` — systematic bias diagnostics

**DE Rule (LOAD-BEARING):** `true_de = treated − diseased`. For LINCS, `diseased` = baseline cell GEX (the AE reconstruction target is the perturbed expression; DE = perturbed − baseline). Top-k selected by `|true_de|`.

**Halt gate 2 metric:** `mean_pearson_across_cells ≥ 0.2` where each cell's Pearson is `compute_persample_metrics(true_de_cell, pred_de_cell)['pearson_mean']`.

**Per-cell evaluation:** Group dev set by `cell_id`, compute Pearson per cell, then mean. 9 cells total.

---

## 5. WandB Integration

**Project:** `joshroll/MultiDCP_multihead_dili`
**Run group:** `model_gex`
**Run name pattern:** `model_gex_seed{seed}_bs{batch_size}_ep{max_epoch}`

Log per epoch:
- `train/ae_loss`
- `train/perturbed_loss`
- `dev/ae_pearson_mean` (per-sample, all genes)
- `dev/perturbed_pearson_per_cell` (dict keyed by cell_id)
- `dev/perturbed_pearson_mean` (halt gate 2 metric)
- `dev/top80_pearson_mean`

---

## 6. CUDA Hygiene Pattern (from `multidcp_ae_de_pdg_12192025.py`)

```python
def get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

gpu_arg = get_gpu_arg()
if gpu_arg is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_arg

# THEN import torch
import torch
```

This pattern MUST be at the top of `train_model_gex.py` (before any torch import), per CLAUDE.md hard rule 5.

**GPU assignment:** Phase 2 uses a DIFFERENT GPU from Phase 1 (MODEL_DOSE) to allow parallel training. Always leave one GPU free on the shared box.

---

## 7. Checkpoint and Output Files

- `results/checkpoints/chkpt_gex.pt` — saved via `torch.save(model.state_dict(), ...)` at best dev perturbed Pearson
- `results/tables/P2_model_gex_summary.md` — architecture, HP, per-cell Pearson, mean Pearson, halt gate 2 verdict, MultiDCP SHA
- `src/eval/de_evaluator.py` — vendored DE evaluation functions with provenance comment
- `src/train/train_model_gex.py` — forked AE training script

---

## 8. Existing Code Analogs in multihead_dili/

`src/train/` and `src/models/` directories exist but contain only `__init__.py` stubs. No prior training code to conflict with.

`src/embed/` and `src/stage2/` likewise are stubs.

The fork will be the first substantial code in `src/train/`.

---

## 9. Validation Architecture

**Phase 2 has a halt gate (HG2):** Dev predicted-vs-measured Pearson ≥ 0.2 averaged across 9 LINCS cells.

**Verification commands:**
```bash
# Checkpoint loadable
python -c "import torch; ckpt = torch.load('results/checkpoints/chkpt_gex.pt', map_location='cpu'); print(list(ckpt.keys())[:5])"

# Summary exists and contains halt gate verdict
grep "HG2" results/tables/P2_model_gex_summary.md
grep "mean_pearson" results/tables/P2_model_gex_summary.md

# DE evaluator vendored correctly
grep "provenance\|train_bl_pdg_de" src/eval/de_evaluator.py
```

**DataLoader num_workers:** Cap at 4–6 per CLAUDE.md memory note (CPU bottleneck on shared box).

---

## 10. Dependency Notes

- `dili_v04_env` has PyTorch, pandas, numpy, scipy, wandb, tqdm installed
- `multidcp_balanceloss.py` requires `multidcp_balanceloss` module (in same `models/` dir), `datareader`, `metric`, `multidcp_ae_utils` — these live in the upstream repo; our fork must `sys.path.append` to the MultiDCP repo or copy the needed modules
- **Recommended approach:** Keep `sys.path.append('/raid/home/joshua/projects/MultiDCP/MultiDCP/models')` and `sys.path.append('/raid/home/joshua/projects/MultiDCP/MultiDCP/utils')` in the training script header with a SHA-pinned comment
