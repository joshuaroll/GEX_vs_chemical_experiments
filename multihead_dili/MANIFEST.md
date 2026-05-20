# MANIFEST — Multi-Head MultiDCP DILI

Data sources, model checkpoints, and code SHAs pinned for reproducibility.

## Data

| Item | Path | SHA256 | Date pinned |
|---|---|---|---|
| E-Hill train | `/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_train.csv` | _Phase 0: record_ | _TBD_ |
| E-Hill dev (raw) | `/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_dev.csv` | _Phase 1: record_ | 2026-05-20 |
| E-Hill dev (safe) | `data/processed/ehill_dev_safe.parquet` | _Phase 1: record_ | 2026-05-20 |
| E-Hill test | `.../high_confident_data_test.csv` | _Phase 0: record_ | _TBD_ |
| LINCS PDG-filtered | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl` | _Phase 0: record_ | _TBD_ |
| DILIst canonical | `../dili_downstream/data/processed/dili_canonical.csv` | _Phase 0: record_ | _TBD_ |
| MolFormer model | HF `ibm-research/MoLFormer-XL-both-10pct` | commit `7b12d946c181a37f6012b9dc3b002275de070314` | 2026-05-20 |
| Feature cache | `data/processed/dili_features.parquet` | 1118 rows × 1694 cols (1688 feat) | 2026-05-20 |
| CCLE HepG2 baseline | `/raid/home/joshua/data/MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv` | MODEL_DOSE cell baselines (978-dim) | 2026-05-20 |

## Code

| Item | Path | SHA / version | Date pinned |
|---|---|---|---|
| MultiDCP upstream | `/raid/home/joshua/projects/MultiDCP` | `871b8de` | 2026-05-20 |
| `ehill_multidcp_pretrain.py` | `MultiDCP/MultiDCP/ehill_multidcp_pretrain.py` | `871b8de` | 2026-05-20 |
| `multidcp_ae_balanceloss.py` | `MultiDCP/MultiDCP/models/multidcp_ae_balanceloss.py` | (same SHA) | 2026-05-20 |
| MolFormer (HF) | `ibm-research/MoLFormer-XL-both-10pct` | commit `7b12d946c181a37f6012b9dc3b002275de070314` | 2026-05-20 |
| `dili_v04_env` | conda env | _Phase 0: record `conda list` snapshot_ | _TBD_ |

## MultiDCP (Stage-1 model backbone)

| Item | Value |
|------|-------|
| Repo path | `/raid/home/joshua/projects/MultiDCP/` |
| SHA | `871b8de` |
| SHA confirmed | 2026-05-20 (Phase 2 planning) |
| Canonical AE script | `MultiDCP/MultiDCP/models/multidcp_ae_balanceloss.py` |
| Fork target | `multihead_dili/src/train/train_model_gex.py` |
| Model class | `multidcp_balanceloss.MultiDCP_AE` |

## Feature Vector (Stage-2 cache)

| Item | Value |
|------|-------|
| Path | `data/processed/dili_features.parquet` |
| Shape | 1118 rows × 1694 cols |
| Feature cols | 1688 (1 feat_dose + 919 feat_gex + 768 feat_embed) |
| Metadata cols | 6 (drug_name, pert_id, dili_binary, dili_severity, in_lincs, in_pdg) |
| NaN in features | 0 |
| Class balance | {DILI=1: 685, DILI=0: 433} |
| feat_dose source | MODEL_DOSE (MultiDCPEhillPretraining), mean E-Hill over 10 cells |
| feat_gex source | MODEL_GEX (MultiDCP_AE), mean DE (perturbed − diseased) over 10 cells, 919 landmark genes |
| feat_embed source | MolFormer frozen encoder (ibm-research/MoLFormer-XL-both-10pct), 768-dim |
| Cells used | A375, A549, BT20, HA1E, HELA, HT29, MCF7, MDAMB231, PC3, VCAP (10 cells) |
| SMILES failures | 1 (nitroprusside — iron coordination compound, atom degree 6 unsupported; zeros used) |
| Date | 2026-05-20 |

## Checkpoints

| Item | Path | Date | Halt gate? |
|---|---|---|---|
| MODEL_DOSE | `multihead_dili/results/checkpoints/chkpt_dose.pt` | _Phase 1: record_ | HG1 |
| MODEL_GEX | `multihead_dili/results/checkpoints/chkpt_gex.pt` | _Phase 2: record_ | HG2 |
