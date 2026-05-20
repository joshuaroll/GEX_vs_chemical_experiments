# Phase 1: MODEL_DOSE Training Summary

**Date:** 2026-05-20
**Phase:** 1 — MODEL_DOSE Training
**MultiDCP SHA:** 871b8de (/raid/home/joshua/projects/MultiDCP/)
**Checkpoint:** results/checkpoints/chkpt_dose.pt
**WandB run:** dwi0qqx2 (project joshroll/MultiDCP_multihead_dili, group model_dose)
**WandB URL:** https://wandb.ai/joshroll/MultiDCP_multihead_dili/runs/dwi0qqx2

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Model class | `MultiDCP` (E-Hill regression head) |
| linear_encoder_flag | True |
| dropout | 0.1 |
| batch_size | 64 |
| max_epoch | 100 |
| patience | 10 |
| optimizer | Adam |
| learning_rate | 0.0002 |
| GPU | 0 (CUDA available) |

## Data

| Item | Value |
|------|-------|
| Training parquet | data/processed/ehill_train_safe.parquet (73,955 rows) |
| Dev parquet | data/processed/ehill_dev_safe.parquet (18,397 rows) |
| Train hill data | 35,972 samples |
| Dev hill data | 8,895 samples |
| Test hill data | 11,337 samples |
| Cell lines (train) | A375, A549, BT20, HA1E, HELA, HT29, JURKAT, MCF7, MDAMB231, SKBR3 (10) |
| Additional cells | HEPG2, PC3 (2), HCC515, HS578T, YAPC (3) |
| Doses | 6 dose levels (0.04, 0.12, 0.37, 1.11, 3.33, 10.0 um) |

## Training Results

| Metric | Value |
|--------|-------|
| Best epoch | 21 |
| Early stop epoch | 32 (patience=10 since epoch 21) |
| Best dev RMSE | 19.455 |
| Best dev Pearson | 0.8182 |
| Predict-mean baseline RMSE | 33.641 |

## Halt Gate 1

**Metric:** Dev RMSE vs predict-mean baseline RMSE
**Threshold:** Dev RMSE << baseline RMSE (model must beat trivial baseline)
**Result:** 19.455 << 33.641
**Status: PASS**

## Notes

- Post-training metadata save initially failed due to PyTorch 2.6 `weights_only=True` default at
  `src/train/train_model_dose.py` line 522 — patched manually post-training via metadata re-save.
- `baseline_rmse` and `halt_gate_1_pass` were `None` in initial checkpoint; patched to correct values.
- Training script bug fix needed: change `torch.load(...)` at line 522 to use `weights_only=False`.
- `chkpt_dose.pt` now has all fields: `best_epoch`, `dev_rmse`, `dev_pearson`, `baseline_rmse`,
  `halt_gate_1_pass`, `multidcp_sha`, `state_dict`, `model_params`.
