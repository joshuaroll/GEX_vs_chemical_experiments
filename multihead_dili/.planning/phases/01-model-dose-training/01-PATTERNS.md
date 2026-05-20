# Phase 1: MODEL_DOSE Training — Pattern Map

**Phase:** 01  
**Generated:** 2026-05-20

---

## PATTERN MAPPING COMPLETE

---

## Files to Create

### 1. `scripts/build_ehill_dev_safe.py` (NEW)

**Role:** Data preparation script — filters E-Hill dev CSV → leakage-safe parquet.  
**Closest analog:** `scripts/run_leakage_filter.py` (same project)

**Key pattern from analog:**
```python
# From scripts/run_leakage_filter.py lines 1-30:
# 1. Parse args (argparse)
# 2. Load dili_split.json -> test_drug_names set
# 3. Load dev CSV
# 4. Apply name-only filter: dev_df[~dev_df['pert_id'].str.lower().isin(test_drug_names)]
# 5. Save to parquet
```

---

### 2. `src/train/train_model_dose.py` (NEW)

**Role:** Training script (fork of upstream pretrain with CUDA hygiene + WandB + checkpoint).  
**Closest analog:** `/raid/home/joshua/projects/MultiDCP/MultiDCP/ehill_multidcp_pretrain.py`

**Pattern excerpt (argparse structure — restructured for CUDA hygiene):**
```python
# MUST come before any torch import
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiDCP Ehill MODEL_DOSE training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index; leaves one GPU free')
    # ... other args ...
    args, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
# ... rest of imports ...
```

**WandB init pattern (from upstream, modified):**
```python
wandb.init(
    project=args.wandb_project,   # 'joshroll/MultiDCP_multihead_dili'
    group=args.wandb_group,       # 'model_dose'
    config=vars(args)
)
```

**Checkpoint save pattern:**
```python
torch.save({
    'state_dict': model.state_dict(),
    'model_params': dict(model_param_registry),
    'best_epoch': best_dev_epoch,
    'dev_rmse': best_dev_rmse,
    'baseline_rmse': predict_mean_baseline_rmse,
    'halt_gate_1_pass': bool(best_dev_rmse < predict_mean_baseline_rmse),
    'multidcp_sha': '871b8de',
}, args.checkpoint_path)
```

---

### 3. `results/tables/P1_model_dose_summary.md` (NEW)

**Role:** Halt gate verdict + training summary.  
**Closest analog:** `.planning/phases/01-model-dose-training/01-CONTEXT.md` (structure reference only)

**Pattern:**
```markdown
# P1: MODEL_DOSE Training Summary

| Item | Value |
|------|-------|
| MultiDCP SHA | 871b8de |
| Training rows | 73,955 |
| Dev rows | N |
| Best epoch | {epoch} |
| Dev RMSE | {rmse:.4f} |
| Predict-mean baseline RMSE | {std:.4f} |
| **HALT GATE 1** | **PASS / FAIL** |
```

---

### 4. `data/processed/ehill_dev_safe.parquet` (NEW)

**Role:** Leakage-safe dev split for MODEL_DOSE training.  
**Closest analog:** `data/processed/ehill_train_safe.parquet` (same schema)

Schema: [sig_id, pert_id, pert_type, cell_id, pert_idose, ehill] (float64)

---

### 5. `results/checkpoints/chkpt_dose.pt` (NEW)

**Role:** Saved MODEL_DOSE checkpoint with embedded halt gate verdict.  
**Closest analog:** `model.sub_multidcp.state_dict()` save pattern from upstream (modified to full model + rich dict).

---

## Files to Modify

### 6. `MANIFEST.md`

**Change:** Add MultiDCP SHA pin (871b8de) to Code table.  
**Pattern:** Existing table in MANIFEST.md — add row with `| MultiDCP upstream | /raid/home/joshua/projects/MultiDCP | 871b8de | 2026-05-20 |`
