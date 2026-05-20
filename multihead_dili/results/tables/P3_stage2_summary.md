# Phase 3 Summary — MolFormer + Stage-2 Feature Caching

**Date:** 2026-05-20
**Phase:** P3 (molformer-stage2-caching)
**Status:** COMPLETE — 0 NaN, EMBED-04 PASS

---

## Output: dili_features.parquet

| Field | Value |
|-------|-------|
| Path | `data/processed/dili_features.parquet` |
| Rows | 1118 |
| Total columns | 1694 |
| Feature columns | 1688 |
| NaN in features | 0 (EMBED-04 PASS) |
| Class balance | DILI=1: 685, DILI=0: 433 |
| Runtime | ~219 s (3.7 min, GPU 1, V100-PCIE-32GB) |
| WandB run | `joshroll/MultiDCP_multihead_dili/runs/8eruui80` |

---

## Feature Breakdown

| Feature group | Columns | Source |
|---|---|---|
| `feat_dose` | 1 | MODEL_DOSE mean E-Hill prediction over 10 LINCS cells |
| `feat_gex_0..918` | 919 | MODEL_GEX mean DE (predicted_perturbed − diseased) over 10 cells |
| `feat_embed_0..767` | 768 | MolFormer frozen encoder (ibm-research/MoLFormer-XL-both-10pct) |

**Total feature dim:** 1 + 919 + 768 = 1688

---

## Models Used

### MODEL_DOSE (MultiDCPEhillPretraining)
- Module: `multidcp.py` (original concat fusion, not MoE)
- Checkpoint: `results/checkpoints/chkpt_dose.pt`
- num_gene: 978 (gene_vector.csv via `data_utils.read_gene`)
- Cell baselines: CCLE file (`adjusted_ccle_tcga_ad_tpm_log2.csv`, 978-dim)
- Output: E-Hill scalar per drug-cell pair, mean-pooled over 10 cells
- feat_dose stats: mean=55.88, std=20.78

### MODEL_GEX (MultiDCP_AE / multidcp_balanceloss)
- Module: `multidcp_balanceloss.py` (sparse_moe fusion)
- Checkpoint: `results/checkpoints/chkpt_gex.pt`
- num_gene: 919 (landmark genes overlapping gene_vector.csv and lincs_train_safe.parquet)
- Cell baselines: PDG diseased file (`pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv`, 919-dim)
- DE rule: `predicted_perturbed − diseased_baseline` (train_bl_pdg_de.py convention)
- Output: [919] DE vector per drug-cell pair, mean-pooled over 10 cells

### MolFormer (frozen encoder)
- HF model: `ibm-research/MoLFormer-XL-both-10pct`
- HF commit SHA: `7b12d946c181a37f6012b9dc3b002275de070314`
- Output: 768-dim pooler_output per SMILES, encoded once per drug (no cell loop)
- Wrapper: `src/embed/molformer_wrapper.py`
- Key fix: `_reinitialize_rotary_embeddings()` before `.to(device)` to prevent transformers 5.x NaN bug

---

## Cells Used (10 cells)

A375, A549, BT20, HA1E, HELA, HT29, MCF7, MDAMB231, PC3, VCAP

Note: Design doc cited 9 cells (omitting HA1E). Actual training used 10. CELL COUNT DEVIATION logged.

---

## Known Issues / Deviations

| Issue | Resolution |
|---|---|
| SMILES featurization failure for nitroprusside (`N#C[Fe-2](C#N)...`) — iron coordination compound with atom degree 6, unsupported by MultiDCP degree encoding [0–5] | Zeros used for feat_dose and feat_gex; MolFormer embedding retained |
| 351 NaN in `dili_severity` column | Expected — 351/1118 drugs have no severity metadata. Not a feature column; EMBED-04 checks feature cols only |
| multidcp_balanceloss.py rejected fusion_type='concat' for MODEL_DOSE | MODEL_DOSE was trained with `multidcp.py` (original concat fusion). Fixed by importing `multidcp` (not `multidcp_balanceloss`) for DOSE loading |
| gene tensor shape: training uses 2D [num_gene, 128] | Model internally adds batch dim via `unsqueeze(0)`. Fixed caching script to not pre-unsqueeze |

---

## CI Tests

All 9 tests in `tests/test_stage2_features.py` pass:
- test_parquet_exists
- test_row_count
- test_feature_col_count
- test_no_nan_in_feature_cols (EMBED-04 hard gate)
- test_class_balance
- test_feat_dose_sanity
- test_feat_gex_sanity
- test_feat_embed_sanity
- test_molformer_sha_pinned

---

## transformers 5.x Compatibility Patches

The following HF model files were patched for transformers 5.8.1 compatibility:

| File | Patch |
|---|---|
| `configuration_molformer.py` | try/except for removed `transformers.onnx.OnnxConfig` |
| `modeling_molformer.py` | try/except for removed `find_pruneable_heads_and_indices`; added `get_head_mask()` stub |
| `src/embed/molformer_wrapper.py` | `_reinitialize_rotary_embeddings()` to fix NaN from uninitialized non-persistent buffers |
