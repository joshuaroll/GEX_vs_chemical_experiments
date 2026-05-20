# Phase 1: MODEL_DOSE Training — Research

**Phase:** 01 — MODEL_DOSE training  
**Researched:** 2026-05-20  
**Researcher:** gsd-phase-researcher (inline, automated context)

---

## RESEARCH COMPLETE

---

## Summary

Phase 1 forks `ehill_multidcp_pretrain.py` into `src/train/train_model_dose.py`, applies a leakage filter to the dev split, runs training to convergence, and records halt gate 1. The key insight is that the upstream script, data paths, and model class are all well-understood — the main work is (a) CUDA hygiene refactor, (b) a thin leakage-filter pass on the dev CSV, and (c) a checkpoint format that embeds the halt-gate verdict.

---

## 1. Upstream Script Analysis

**File:** `/raid/home/joshua/projects/MultiDCP/MultiDCP/ehill_multidcp_pretrain.py`  
**SHA pin:** `871b8de` (MultiDCP HEAD)

### Critical CUDA Hygiene Violation (Hard Rule 4)
The upstream script runs `device = torch.device("cuda")` at module import time, **before** argparse parsing. The fork must restructure so:
1. `argparse.ArgumentParser` adds `--gpu INT` as the **first** parse (in `__main__` block, before any `import torch`)
2. `CUDA_VISIBLE_DEVICES` set from `--gpu` before `import torch`
3. `device = torch.device("cuda:0")` set after imports

### Required Argparse Additions
| New Flag | Type | Purpose |
|----------|------|---------|
| `--gpu` | int | GPU index; set `CUDA_VISIBLE_DEVICES` before torch import |
| `--safe-parquet` | store_true | Use leakage-filtered parquet files instead of upstream CSVs |
| `--wandb-project` | str | Default: `joshroll/MultiDCP_multihead_dili` |
| `--wandb-group` | str | Default: `model_dose` |
| `--checkpoint-path` | str | Default: `results/checkpoints/chkpt_dose.pt` |
| `--patience` | int | Default: 10 (early stopping on dev Pearson) |

### WandB Change
Upstream hard-codes `wandb.init(project="MultiDCP_AE_ehill")`. Fork changes to:
```python
wandb.init(project=args.wandb_project, group=args.wandb_group, config=vars(args))
```

### Checkpoint Format
Upstream saves only `model.sub_multidcp.state_dict()` with a fixed filename. Fork saves:
```python
torch.save({
    'state_dict': model.sub_multidcp.state_dict(),
    'model_params': dict(model_param_registry),
    'best_epoch': best_epoch,
    'dev_rmse': best_dev_rmse,
    'baseline_rmse': predict_mean_baseline_rmse,
    'halt_gate_1_pass': best_dev_rmse < predict_mean_baseline_rmse,
    'multidcp_sha': '871b8de',
}, args.checkpoint_path)
```

---

## 2. Data Paths — Canonical Reference

All data paths confirmed by reading `train_multidcp_ehill_pretraining.sh`:

| Arg | Path |
|-----|------|
| `--drug_file` | `/raid/home/joshua/data/MultiDCP/data/all_drugs_l1000.csv` |
| `--gene_file` | `/raid/home/joshua/data/MultiDCP/data/gene_vector.csv` (129 cols each: gene_id + 128 floats) |
| `--cell_ge_file` | `/raid/home/joshua/data/MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv` |
| `--all_cells` | `/raid/home/joshua/data/MultiDCP/data/ehill_data/pretrain_cell_list_ehill.p` |
| `--hill_train_file` | (replaced by `ehill_train_safe.parquet` when `--safe-parquet`) |
| `--hill_dev_file` | (replaced by `ehill_dev_safe.parquet` when `--safe-parquet`) |
| `--hill_test_file` | `/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_test.csv` (unfiltered — test set untouched) |
| `--train_file` | `/raid/home/joshua/data/MultiDCP/data/pert_transcriptom/signature_train_cell_2.csv` (PerturbedDataLoader — needed for gene tensor) |
| `--dev_file` | same pattern as train, dev suffix |
| `--test_file` | same pattern as train, test suffix |

**Important:** The upstream script uses both `EhillDataLoader` (for ehill labels) and `PerturbedDataLoader` (for the gene tensor). The gene tensor (`hill_data.gene`) has shape `[978, 128]`. Both data loaders must be initialized even if PerturbedDataLoader's train/dev/test splits aren't used in ehill training.

---

## 3. Dev Leakage Filter

### Source Dev CSV
`/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_dev.csv`  
- 19,061 rows (same schema as train: sig_id, pert_id, pert_type, cell_id, pert_idose, ehill)

### Filter Logic (from `run_leakage_filter.py` and `leakage_report.md`)
The leakage filter uses:
1. **test_drug_names**: set of lowercased `pert_id` values in DILIst test partition
2. **test_scaffolds**: 30 non-empty Murcko scaffold strings from `dili_split.json['scaffolds_in_test']`

For the dev CSV, we apply the same name+scaffold exclusion. However, a key simplification: the dev set filter **only needs drug-name matching** without DrugBank SMILES lookup if we're willing to exclude only exact name matches. For a robust filter, we can use the `drugbank_smiles_index` already in `src/data/` to resolve scaffolds.

**Simpler approach for dev filter (accepted):** Since `dili_split.json` already has `scaffolds_in_test` (30 scaffold strings) precomputed, and E-Hill's `pert_id` column IS the drug name, we can do:
```python
# Load dili_split.json
test_drug_names = {d['drug_name'].lower() for d in ... }  # from dili_canonical test pert_ids
test_scaffolds = set(dili_split['scaffolds_in_test'])

# For each row in dev CSV: exclude if pert_id.lower() in test_drug_names
# AND/OR if SMILES-resolved scaffold in test_scaffolds
# For simplicity: name-only filter is sufficient for dev (test contamination is the concern)
dev_safe = dev_df[~dev_df['pert_id'].str.lower().isin(test_drug_names)]
```

**Note:** The leakage report shows that for ehill_train, ALL 2,578 dropped rows were `Both name+scaffold` (0 scaffold-only, 0 name-only). This suggests name-only filtering on dev is sufficient — any drug whose scaffold is in test_scaffolds will also have its name in test_drug_names, for the ehill dataset.

**Canonical approach:** Write `scripts/build_ehill_dev_safe.py` that:
1. Loads `dili_split.json` → extracts test_drug_names set
2. Filters `high_confident_data_dev.csv` on `pert_id.lower()` not in test_drug_names
3. Saves `data/processed/ehill_dev_safe.parquet`

---

## 4. Model Architecture (MultiDCPEhillPretraining)

From `/raid/home/joshua/projects/MultiDCP/MultiDCP/models/multidcp.py`:

```python
class MultiDCPEhillPretraining(MultiDCPBase):
    # task_linear: hid_dim→hid_dim//2→hid_dim//2→1 (ReLU between each)
    # genes_linear: 978→489→489→1 (ReLU between each)
    # forward (job_id='pretraining'): task_linear(out) → squeeze → genes_linear → scalar
```

**Key**: `job_id='pretraining'` is what routes through `task_linear` + `genes_linear` → scalar ehill prediction.

**Model param registry (locked):**
```python
drug_input_dim = {'atom': 62, 'bond': 6}
drug_emb_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_emb_dim = 128
gene_input_dim = 128
cell_id_input_dim = 978
cell_feature_emb_dim = 32
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
loss_type = 'point_wise_mse'
initializer = torch.nn.init.kaiming_uniform_
```

**`pert_idose_input_dim`** = `len(DATA_FILTER['pert_idose'])` = 6 (the 6 dose levels).

**`linear_encoder_flag`**: Upstream script uses `--linear_encoder_flag`. Include this flag in the fork; default to the same value used in the original pretraining script.

---

## 5. DATA_FILTER Configuration

```python
DATA_FILTER = {
    "time": "24H",
    "pert_id": ['BRD-U41416256', 'BRD-U60236422'],  # 2 excluded drugs
    "pert_type": ["trt_cp"],
    "cell_feature": all_cells,  # loaded from pretrain_cell_list_ehill.p
    "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]
}
```

**Critical**: `data_utils.read_data()` checks `filter["time"] in line[0]` where `line[0]` is `sig_id`. The `sig_id` in `ehill_train_safe.parquet` follows the pattern `Vorinostat_HCC1428_1.11_24H` — it contains "24H" but NOT the `line[0]` format expected by `read_data()`. 

**The parquet adapter must write a temp CSV** in the exact same column format as `high_confident_data_train.csv` before passing to `EhillDataLoader`. This is the cleanest approach — write a `pandas.DataFrame.to_csv()` to a temp file, pass the temp file path as `--hill_train_file`.

---

## 6. Halt Gate 1 Implementation

```python
# After training, compute baseline and dev RMSE:
baseline_rmse = np.std(dev_labels_np)  # predict-mean baseline = std(targets)
final_dev_rmse = metrics_summary['rmse_list_dev_ehill'][best_epoch]
halt_gate_1_pass = final_dev_rmse < baseline_rmse
```

**Report**: Write to `results/tables/P1_model_dose_summary.md` with:
- MultiDCP SHA: `871b8de`
- Architecture: `MultiDCPEhillPretraining`, `hid_dim=128`, `num_gene=978`, `dropout={dropout}`
- Training data: 73,955 rows (`ehill_train_safe.parquet`)
- Dev data: `ehill_dev_safe.parquet` (N rows after filter)
- Optimizer: Adam, lr=0.0002
- Batch size: 64 (upstream default — 32 acceptable but 64 matches original)
- Max epochs: 100 (upstream default)
- Early stopping: patience=10 on dev Pearson
- Dev RMSE (best epoch): {value}
- Predict-mean baseline RMSE: std(dev_ehill) = {value}
- HALT GATE 1: PASS / FAIL

---

## 7. Training Hyperparameters (from upstream shell script)

```
--dropout 0.1
--batch_size 64
--max_epoch 100
--linear_encoder_flag  (use for fork; same as upstream default)
```

---

## 8. File System Layout for Phase 1

```
multihead_dili/
├── scripts/
│   └── build_ehill_dev_safe.py          # NEW: filters dev CSV → ehill_dev_safe.parquet
├── src/train/
│   └── train_model_dose.py              # NEW: fork of ehill_multidcp_pretrain.py
├── data/processed/
│   └── ehill_dev_safe.parquet           # NEW: filtered dev set
├── results/
│   ├── checkpoints/
│   │   └── chkpt_dose.pt                # NEW: rich checkpoint dict
│   └── tables/
│       └── P1_model_dose_summary.md     # NEW: summary + halt gate verdict
└── MANIFEST.md                          # UPDATE: pin MultiDCP SHA 871b8de
```

---

## 9. Pitfalls and Landmines

1. **`EhillDataLoader.setup()` bug**: The upstream `setup()` method only initializes `self.test_data` (train and dev are commented out). The fork must fix this to initialize all three splits when `--safe-parquet` is active. The temp-CSV approach handles this cleanly.

2. **PerturbedDataLoader dependency**: `ehill_multidcp_pretrain.py` instantiates `PerturbedDataLoader` in addition to `EhillDataLoader`. Even though perturbed data isn't used in the ehill training loop, the gene tensor is retrieved from `hill_data.gene`, not `data.gene`. So PerturbedDataLoader must still be instantiated (for model_param_registry `num_gene`). If the upstream `--train_file` etc. are not available, the fork can compute `num_gene=978` directly and skip PerturbedDataLoader.

3. **`read_data()` filter on sig_id format**: The `sig_id` in the parquet file contains `_24H` in the pattern `DrugName_CellLine_Dose_24H`. The `data_utils.read_data()` checks `filter["time"] in line[0]` — this will correctly match "24H" in the sig_id string.

4. **`model.sub_multidcp` vs `model` state_dict**: The upstream saves `model.sub_multidcp.state_dict()`. The fork should save the full model `state_dict()` to include the `task_linear` and `genes_linear` heads, which are needed for Phase 3 inference (querying MODEL_DOSE at 9 LINCS cells). Use `model.state_dict()`.

5. **GPU free policy**: With 8 × V100-32GB available (all ~32GB free), leave GPU 7 free. Default `--gpu 0` for MODEL_DOSE (Phase 1), `--gpu 1` for MODEL_GEX (Phase 2 parallel).

---

## 10. MANIFEST.md Update

Add to Code section:
```markdown
| MultiDCP upstream | `/raid/home/joshua/projects/MultiDCP` | 871b8de | 2026-05-20 |
| `ehill_multidcp_pretrain.py` | `MultiDCP/MultiDCP/ehill_multidcp_pretrain.py` | 871b8de | 2026-05-20 |
```

---

## Validation Architecture

Not applicable — Phase 1 is a training script fork; validation is the halt gate 1 metric comparison, not a unit-test suite.
